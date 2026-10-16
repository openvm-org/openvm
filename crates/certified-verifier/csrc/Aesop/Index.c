// Lean compiler output
// Module: Aesop.Index
// Imports: public import Init public meta import Init public import Aesop.Index.Basic public import Aesop.Index.DiscrTreeConfig public import Aesop.Index.RulePattern public import Aesop.Rule.Basic public import Batteries.Lean.Meta.InstantiateMVars import Batteries.Lean.Meta.DiscrTree import Batteries.Lean.PersistentHashSet
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Rule_instHashable___lam__0___boxed(lean_object*);
lean_object* lp_aesop_Aesop_Rule_instBEq___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint64_t l_Lean_Meta_DiscrTree_Key_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Meta_DiscrTree_instBEqKey_beq(lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_instMonadExceptOfEIO(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(lean_object*);
lean_object* l_Lean_instMonadAlwaysExceptReaderT___redArg(lean_object*);
extern lean_object* l_Lean_Meta_instMonadMCtxMetaM;
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_debug;
lean_object* lp_batteries_Lean_MVarId_instantiateMVars___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getUnify___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_Lean_MVarId_withContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_PersistentArray_forIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_index(lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instExceptToTraceResult___lam__0___boxed(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_of_nat(lean_object*);
double lean_float_div(double, double);
lean_object* l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_trace_profiler;
extern lean_object* l_Lean_Core_instMonadRefCoreM;
extern lean_object* l_Lean_Core_instAddMessageContextCoreM;
lean_object* l_String_compare___boxed(lean_object*, lean_object*);
lean_object* l_Array_qsortOrd___redArg(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_empty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Bool_Internal_not___boxed(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_filterDiscrTree___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_insertKeyValue___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0;
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedIndex_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedIndex_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedIndex_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedIndex_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedIndex_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedIndex_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedIndex_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedIndex_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndex_default(lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedIndex___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedIndex___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndex(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_String_compare___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__4_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__5_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__6_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__7_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__8_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__9_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__10_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__11_value;
static const lean_ctor_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__5_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__6_value)}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__12_value;
static const lean_ctor_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__12_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__7_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__8_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__9_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__10_value)}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__13_value;
static const lean_ctor_object lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__13_value),((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__11_value)}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Index_trace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__3;
static const lean_closure_object lp_aesop_Aesop_Index_trace___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Index_trace___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Index_trace___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Index_trace___redArg___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_Index_trace___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instExceptToTraceResult___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Index_trace___redArg___closed__7;
static const lean_array_object lp_aesop_Aesop_Index_trace___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Index_trace___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Unindexed"};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_Index_trace___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__11;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__12;
static const lean_string_object lp_aesop_Aesop_Index_trace___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__13 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__13_value;
static const lean_string_object lp_aesop_Aesop_Index_trace___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_Index_trace___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__14_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__15 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__15_value;
static const lean_closure_object lp_aesop_Aesop_Index_trace___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Index_trace___redArg___lam__2___boxed, .m_arity = 5, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14_value),((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__4_value)} };
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__16 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__16_value;
static const lean_string_object lp_aesop_Aesop_Index_trace___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Indexed by hypotheses"};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_Index_trace___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__18 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__18_value;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__19;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__20;
static const lean_string_object lp_aesop_Aesop_Index_trace___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Indexed by target"};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__21 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__21_value;
static const lean_ctor_object lp_aesop_Aesop_Index_trace___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__21_value)}};
static const lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__22 = (const lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__23;
static lean_once_cell_t lp_aesop_Aesop_Index_trace___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_trace___redArg___closed__24;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Index_instEmptyCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rule_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_instEmptyCollection___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_instEmptyCollection___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Index_instEmptyCollection___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Rule_instHashable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_instEmptyCollection___closed__1 = (const lean_object*)&lp_aesop_Aesop_Index_instEmptyCollection___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Index_instEmptyCollection___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_instEmptyCollection___closed__2;
static lean_once_cell_t lp_aesop_Aesop_Index_instEmptyCollection___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_instEmptyCollection___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_instEmptyCollection(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___closed__0 = (const lean_object*)&lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Index_merge___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Index_merge___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_merge___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_merge___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16(lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_instEmptyCollection___closed__0_value),((lean_object*)&lp_aesop_Aesop_Index_instEmptyCollection___closed__1_value)} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_Internal_not___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_unindex___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Index_unindex___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Index_unindex___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_unindex___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_unindex___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_fold___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___closed__0 = (const lean_object*)&lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__12(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__12___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__3(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__4(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__2(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__13(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__13___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__3;
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__5_value;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__6;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__7;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__8;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__9;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__10;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__11;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__12;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__13;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__14;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__15;
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Index_trace___redArg___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__16 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__16_value;
static const lean_closure_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__16_value)} };
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__17_value;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "selected rules:"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__18 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__18_value;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__19;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__20 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__20_value;
static const lean_ctor_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__20_value)}};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__21 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__21_value;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__22;
static const lean_string_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "rule selection"};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__23 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__23_value;
static const lean_ctor_object lp_aesop_Aesop_Index_applicableRules___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__23_value)}};
static const lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__24 = (const lean_object*)&lp_aesop_Aesop_Index_applicableRules___redArg___closed__24_value;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__25;
static lean_once_cell_t lp_aesop_Aesop_Index_applicableRules___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___closed__26;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__0);
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1, &lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1_once, _init_lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0___closed__1);
return v___x_6_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__0(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_7_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__1(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex_default___closed__0, &lp_aesop_Aesop_instInhabitedIndex_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__0);
v___x_9_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
return v___x_9_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__2(void){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_aesop_Lean_PersistentHashMap_empty___at___00Aesop_instInhabitedIndex_default_spec__0(lean_box(0), lean_box(0));
return v___x_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__3(void){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex_default___closed__2, &lp_aesop_Aesop_instInhabitedIndex_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__2);
v___x_12_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex_default___closed__1, &lp_aesop_Aesop_instInhabitedIndex_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__1);
v___x_13_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_12_);
lean_ctor_set(v___x_13_, 2, v___x_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndex_default(lean_object* v_00_u03b1_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex_default___closed__3, &lp_aesop_Aesop_instInhabitedIndex_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__3);
return v___x_15_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedIndex___closed__0(void){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_aesop_Aesop_instInhabitedIndex_default(lean_box(0));
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedIndex(lean_object* v_a_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex___closed__0, &lp_aesop_Aesop_instInhabitedIndex___closed__0_once, _init_lp_aesop_Aesop_instInhabitedIndex___closed__0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_x_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_apply_1(v_inst_19_, v_x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1(lean_object* v_traceOpt_22_, lean_object* v___x_23_, lean_object* v___x_24_, lean_object* v___x_25_, lean_object* v___x_26_, lean_object* v_x_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_){
_start:
{
lean_object* v_traceClass_32_; lean_object* v___x_33_; lean_object* v___x_236__overap_34_; lean_object* v___x_35_; 
v_traceClass_32_ = lean_ctor_get(v_traceOpt_22_, 0);
lean_inc(v_traceClass_32_);
lean_dec_ref(v_traceOpt_22_);
v___x_33_ = l_Lean_stringToMessageData(v___y_28_);
v___x_236__overap_34_ = l_Lean_addTrace___redArg(v___x_23_, v___x_24_, v___x_25_, v___x_26_, v_traceClass_32_, v___x_33_);
lean_inc(v___y_30_);
lean_inc_ref(v___y_29_);
v___x_35_ = lean_apply_3(v___x_236__overap_34_, v___y_29_, v___y_30_, lean_box(0));
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1___boxed(lean_object* v_traceOpt_36_, lean_object* v___x_37_, lean_object* v___x_38_, lean_object* v___x_39_, lean_object* v___x_40_, lean_object* v_x_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1(v_traceOpt_36_, v___x_37_, v___x_38_, v___x_39_, v___x_40_, v_x_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
return v_res_46_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0(void){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = l_instMonadEIO(lean_box(0));
return v___x_47_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1(void){
_start:
{
lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_48_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__0);
v___x_49_ = l_StateRefT_x27_instMonad___redArg(v___x_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(lean_object* v_inst_72_, lean_object* v_traceOpt_73_, lean_object* v_as_74_, lean_object* v_a_75_, lean_object* v_a_76_){
_start:
{
lean_object* v___x_78_; lean_object* v_toApplicative_79_; lean_object* v_toFunctor_80_; lean_object* v_toSeq_81_; lean_object* v_toSeqLeft_82_; lean_object* v_toSeqRight_83_; lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___f_87_; lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___f_90_; lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; size_t v_sz_99_; size_t v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; uint8_t v___x_106_; 
v___x_78_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_79_ = lean_ctor_get(v___x_78_, 0);
v_toFunctor_80_ = lean_ctor_get(v_toApplicative_79_, 0);
v_toSeq_81_ = lean_ctor_get(v_toApplicative_79_, 2);
v_toSeqLeft_82_ = lean_ctor_get(v_toApplicative_79_, 3);
v_toSeqRight_83_ = lean_ctor_get(v_toApplicative_79_, 4);
v___f_84_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__0), 2, 1);
lean_closure_set(v___f_84_, 0, v_inst_72_);
v___f_85_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_86_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_80_, 2);
v___f_87_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_87_, 0, v_toFunctor_80_);
v___f_88_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_88_, 0, v_toFunctor_80_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___f_87_);
lean_ctor_set(v___x_89_, 1, v___f_88_);
lean_inc(v_toSeqRight_83_);
v___f_90_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_90_, 0, v_toSeqRight_83_);
lean_inc(v_toSeqLeft_82_);
v___f_91_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_91_, 0, v_toSeqLeft_82_);
lean_inc(v_toSeq_81_);
v___f_92_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_92_, 0, v_toSeq_81_);
v___x_93_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_93_, 0, v___x_89_);
lean_ctor_set(v___x_93_, 1, v___f_85_);
lean_ctor_set(v___x_93_, 2, v___f_92_);
lean_ctor_set(v___x_93_, 3, v___f_91_);
lean_ctor_set(v___x_93_, 4, v___f_90_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_93_);
lean_ctor_set(v___x_94_, 1, v___f_86_);
v___x_95_ = l_Lean_Core_instMonadTraceCoreM;
v___x_96_ = l_Lean_Core_instMonadRefCoreM;
v___x_97_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__4));
v___x_98_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v_sz_99_ = lean_array_size(v_as_74_);
v___x_100_ = ((size_t)0ULL);
v___x_101_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_98_, v___f_84_, v_sz_99_, v___x_100_, v_as_74_);
v___x_102_ = l_Array_qsortOrd___redArg(v___x_97_, v___x_101_);
v___x_103_ = lean_unsigned_to_nat(0u);
v___x_104_ = lean_array_get_size(v___x_102_);
v___x_105_ = lean_box(0);
v___x_106_ = lean_nat_dec_lt(v___x_103_, v___x_104_);
if (v___x_106_ == 0)
{
lean_object* v___x_107_; 
lean_dec_ref(v___x_102_);
lean_dec_ref_known(v___x_94_, 2);
lean_dec_ref(v_traceOpt_73_);
v___x_107_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_107_, 0, v___x_105_);
return v___x_107_;
}
else
{
lean_object* v___x_108_; lean_object* v___f_109_; uint8_t v___x_110_; 
v___x_108_ = l_Lean_Core_instAddMessageContextCoreM;
lean_inc_ref(v___x_94_);
v___f_109_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___lam__1___boxed), 10, 5);
lean_closure_set(v___f_109_, 0, v_traceOpt_73_);
lean_closure_set(v___f_109_, 1, v___x_94_);
lean_closure_set(v___f_109_, 2, v___x_95_);
lean_closure_set(v___f_109_, 3, v___x_96_);
lean_closure_set(v___f_109_, 4, v___x_108_);
v___x_110_ = lean_nat_dec_le(v___x_104_, v___x_104_);
if (v___x_110_ == 0)
{
if (v___x_106_ == 0)
{
lean_object* v___x_111_; 
lean_dec_ref(v___f_109_);
lean_dec_ref(v___x_102_);
lean_dec_ref_known(v___x_94_, 2);
v___x_111_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_111_, 0, v___x_105_);
return v___x_111_;
}
else
{
size_t v___x_112_; lean_object* v___x_186__overap_113_; lean_object* v___x_114_; 
v___x_112_ = lean_usize_of_nat(v___x_104_);
v___x_186__overap_113_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_94_, v___f_109_, v___x_102_, v___x_100_, v___x_112_, v___x_105_);
lean_inc(v_a_76_);
lean_inc_ref(v_a_75_);
v___x_114_ = lean_apply_3(v___x_186__overap_113_, v_a_75_, v_a_76_, lean_box(0));
return v___x_114_;
}
}
else
{
size_t v___x_115_; lean_object* v___x_191__overap_116_; lean_object* v___x_117_; 
v___x_115_ = lean_usize_of_nat(v___x_104_);
v___x_191__overap_116_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_94_, v___f_109_, v___x_102_, v___x_100_, v___x_115_, v___x_105_);
lean_inc(v_a_76_);
lean_inc_ref(v_a_75_);
v___x_117_ = lean_apply_3(v___x_191__overap_116_, v_a_75_, v_a_76_, lean_box(0));
return v___x_117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___boxed(lean_object* v_inst_118_, lean_object* v_traceOpt_119_, lean_object* v_as_120_, lean_object* v_a_121_, lean_object* v_a_122_, lean_object* v_a_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_118_, v_traceOpt_119_, v_as_120_, v_a_121_, v_a_122_);
lean_dec(v_a_122_);
lean_dec_ref(v_a_121_);
return v_res_124_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_, lean_object* v_traceOpt_127_, lean_object* v_as_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_126_, v_traceOpt_127_, v_as_128_, v_a_129_, v_a_130_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___boxed(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_, lean_object* v_traceOpt_135_, lean_object* v_as_136_, lean_object* v_a_137_, lean_object* v_a_138_, lean_object* v_a_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray(v_00_u03b1_133_, v_inst_134_, v_traceOpt_135_, v_as_136_, v_a_137_, v_a_138_);
lean_dec(v_a_138_);
lean_dec_ref(v_a_137_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__0(lean_object* v_x1_141_, lean_object* v_x2_142_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lean_array_push(v_x1_141_, v_x2_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__1(lean_object* v_d_144_, lean_object* v_a_145_, lean_object* v_x_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lean_array_push(v_d_144_, v_a_145_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__3(lean_object* v___x_148_, lean_object* v_x_149_, lean_object* v___y_150_, lean_object* v___y_151_){
_start:
{
lean_object* v___x_153_; 
v___x_153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_153_, 0, v___x_148_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__3___boxed(lean_object* v___x_154_, lean_object* v_x_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_aesop_Aesop_Index_trace___redArg___lam__3(v___x_154_, v_x_155_, v___y_156_, v___y_157_);
lean_dec(v___y_157_);
lean_dec_ref(v___y_156_);
lean_dec_ref(v_x_155_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__2(lean_object* v___x_160_, lean_object* v___f_161_, lean_object* v_s_162_, lean_object* v_x_163_, lean_object* v_t_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(v___x_160_, v___f_161_, v_s_162_, v_t_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___lam__2___boxed(lean_object* v___x_166_, lean_object* v___f_167_, lean_object* v_s_168_, lean_object* v_x_169_, lean_object* v_t_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_aesop_Aesop_Index_trace___redArg___lam__2(v___x_166_, v___f_167_, v_s_168_, v_x_169_, v_t_170_);
lean_dec(v_x_169_);
return v_res_171_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__1(void){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = l_instMonadExceptOfEIO(lean_box(0));
return v___x_173_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__2(void){
_start:
{
lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_174_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__1, &lp_aesop_Aesop_Index_trace___redArg___closed__1_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__1);
v___x_175_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_174_);
return v___x_175_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__3(void){
_start:
{
lean_object* v___x_176_; lean_object* v___x_177_; 
v___x_176_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__2, &lp_aesop_Aesop_Index_trace___redArg___closed__2_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__2);
v___x_177_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_176_);
return v___x_177_;
}
}
static double _init_lp_aesop_Aesop_Index_trace___redArg___closed__7(void){
_start:
{
lean_object* v___x_181_; double v___x_182_; 
v___x_181_ = lean_unsigned_to_nat(1000000000u);
v___x_182_ = lean_float_of_nat(v___x_181_);
return v___x_182_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__11(void){
_start:
{
lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_188_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__10));
v___x_189_ = l_Lean_MessageData_ofFormat(v___x_188_);
return v___x_189_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__12(void){
_start:
{
lean_object* v___x_190_; lean_object* v___f_191_; 
v___x_190_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__11, &lp_aesop_Aesop_Index_trace___redArg___closed__11_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__11);
v___f_191_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_trace___redArg___lam__3___boxed), 5, 1);
lean_closure_set(v___f_191_, 0, v___x_190_);
return v___f_191_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__19(void){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_202_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__18));
v___x_203_ = l_Lean_MessageData_ofFormat(v___x_202_);
return v___x_203_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__20(void){
_start:
{
lean_object* v___x_204_; lean_object* v___f_205_; 
v___x_204_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__19, &lp_aesop_Aesop_Index_trace___redArg___closed__19_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__19);
v___f_205_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_trace___redArg___lam__3___boxed), 5, 1);
lean_closure_set(v___f_205_, 0, v___x_204_);
return v___f_205_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__23(void){
_start:
{
lean_object* v___x_209_; lean_object* v___x_210_; 
v___x_209_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__22));
v___x_210_ = l_Lean_MessageData_ofFormat(v___x_209_);
return v___x_210_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_trace___redArg___closed__24(void){
_start:
{
lean_object* v___x_211_; lean_object* v___f_212_; 
v___x_211_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__23, &lp_aesop_Aesop_Index_trace___redArg___closed__23_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__23);
v___f_212_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_trace___redArg___lam__3___boxed), 5, 1);
lean_closure_set(v___f_212_, 0, v___x_211_);
return v___f_212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg(lean_object* v_inst_213_, lean_object* v_ri_214_, lean_object* v_traceOpt_215_, lean_object* v_a_216_, lean_object* v_a_217_){
_start:
{
lean_object* v___x_219_; lean_object* v_toApplicative_220_; lean_object* v_toFunctor_221_; lean_object* v_toSeq_222_; lean_object* v_toSeqLeft_223_; lean_object* v_toSeqRight_224_; lean_object* v___f_225_; lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___f_230_; lean_object* v___f_231_; lean_object* v___f_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___f_235_; lean_object* v___x_8234__overap_236_; lean_object* v___x_237_; 
v___x_219_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_220_ = lean_ctor_get(v___x_219_, 0);
v_toFunctor_221_ = lean_ctor_get(v_toApplicative_220_, 0);
v_toSeq_222_ = lean_ctor_get(v_toApplicative_220_, 2);
v_toSeqLeft_223_ = lean_ctor_get(v_toApplicative_220_, 3);
v_toSeqRight_224_ = lean_ctor_get(v_toApplicative_220_, 4);
v___f_225_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_226_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_221_, 2);
v___f_227_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_227_, 0, v_toFunctor_221_);
v___f_228_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_228_, 0, v_toFunctor_221_);
v___x_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_229_, 0, v___f_227_);
lean_ctor_set(v___x_229_, 1, v___f_228_);
lean_inc(v_toSeqRight_224_);
v___f_230_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_230_, 0, v_toSeqRight_224_);
lean_inc(v_toSeqLeft_223_);
v___f_231_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_231_, 0, v_toSeqLeft_223_);
lean_inc(v_toSeq_222_);
v___f_232_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_232_, 0, v_toSeq_222_);
v___x_233_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_233_, 0, v___x_229_);
lean_ctor_set(v___x_233_, 1, v___f_225_);
lean_ctor_set(v___x_233_, 2, v___f_232_);
lean_ctor_set(v___x_233_, 3, v___f_231_);
lean_ctor_set(v___x_233_, 4, v___f_230_);
v___x_234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___f_226_);
v___f_235_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__0));
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v___x_234_);
v___x_8234__overap_236_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_234_, v___f_235_, v_traceOpt_215_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_237_ = lean_apply_3(v___x_8234__overap_236_, v_a_216_, v_a_217_, lean_box(0));
if (lean_obj_tag(v___x_237_) == 0)
{
lean_object* v_a_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_589_; 
v_a_238_ = lean_ctor_get(v___x_237_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_237_);
if (v_isSharedCheck_589_ == 0)
{
v___x_240_ = v___x_237_;
v_isShared_241_ = v_isSharedCheck_589_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_a_238_);
lean_dec(v___x_237_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_589_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
uint8_t v___x_242_; 
v___x_242_ = lean_unbox(v_a_238_);
if (v___x_242_ == 0)
{
lean_object* v___x_243_; lean_object* v___x_245_; 
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_ri_214_);
lean_dec_ref(v_inst_213_);
v___x_243_ = lean_box(0);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 0, v___x_243_);
v___x_245_ = v___x_240_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
else
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v_byTarget_250_; lean_object* v_byHyp_251_; lean_object* v_unindexed_252_; lean_object* v___x_253_; lean_object* v_options_254_; lean_object* v_inheritedTraceOptions_255_; uint8_t v_hasTrace_256_; lean_object* v___f_257_; lean_object* v___x_258_; lean_object* v___f_259_; lean_object* v___y_261_; lean_object* v___y_262_; lean_object* v___y_263_; lean_object* v___y_264_; lean_object* v___y_265_; lean_object* v___y_266_; uint8_t v___y_267_; lean_object* v_a_268_; lean_object* v___y_280_; lean_object* v___y_281_; lean_object* v___y_282_; lean_object* v___y_283_; lean_object* v___y_284_; lean_object* v___y_285_; uint8_t v___y_286_; lean_object* v_a_287_; lean_object* v___y_302_; lean_object* v___y_303_; lean_object* v___y_304_; lean_object* v___y_305_; uint8_t v___y_306_; lean_object* v___y_307_; lean_object* v___y_360_; lean_object* v___y_376_; uint8_t v___y_377_; lean_object* v___y_378_; lean_object* v___y_379_; lean_object* v___y_380_; lean_object* v___y_381_; lean_object* v___y_382_; lean_object* v_a_383_; lean_object* v___y_398_; lean_object* v___y_399_; uint8_t v___y_400_; lean_object* v___y_401_; lean_object* v___y_402_; lean_object* v___y_403_; lean_object* v___y_404_; lean_object* v_a_405_; uint8_t v___y_417_; lean_object* v___y_418_; lean_object* v___y_419_; lean_object* v___y_420_; lean_object* v___y_421_; lean_object* v___y_422_; lean_object* v___y_475_; lean_object* v___x_491_; lean_object* v___f_492_; lean_object* v___x_493_; 
lean_del_object(v___x_240_);
v___x_247_ = l_Lean_Core_instMonadTraceCoreM;
v___x_248_ = l_Lean_Core_instMonadRefCoreM;
v___x_249_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__3, &lp_aesop_Aesop_Index_trace___redArg___closed__3_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__3);
v_byTarget_250_ = lean_ctor_get(v_ri_214_, 0);
lean_inc_ref(v_byTarget_250_);
v_byHyp_251_ = lean_ctor_get(v_ri_214_, 1);
lean_inc_ref(v_byHyp_251_);
v_unindexed_252_ = lean_ctor_get(v_ri_214_, 2);
lean_inc_ref(v_unindexed_252_);
lean_dec_ref(v_ri_214_);
v___x_253_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v_options_254_ = lean_ctor_get(v_a_216_, 2);
v_inheritedTraceOptions_255_ = lean_ctor_get(v_a_216_, 13);
v_hasTrace_256_ = lean_ctor_get_uint8(v_options_254_, sizeof(void*)*1);
v___f_257_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__5));
v___x_258_ = l_Lean_Core_instAddMessageContextCoreM;
v___f_259_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__6));
v___x_491_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__8));
v___f_492_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__16));
v___x_493_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_253_, v___f_492_, v_byTarget_250_, v___x_491_);
if (v_hasTrace_256_ == 0)
{
lean_object* v___x_494_; 
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_494_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_493_, v_a_216_, v_a_217_);
v___y_475_ = v___x_494_;
goto v___jp_474_;
}
else
{
lean_object* v_traceClass_495_; lean_object* v___f_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; uint8_t v___x_500_; lean_object* v___y_502_; lean_object* v___y_503_; lean_object* v_a_504_; lean_object* v___y_519_; lean_object* v___y_520_; lean_object* v_a_521_; 
v_traceClass_495_ = lean_ctor_get(v_traceOpt_215_, 0);
v___f_496_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__24, &lp_aesop_Aesop_Index_trace___redArg___closed__24_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__24);
v___x_497_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__13));
v___x_498_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__15));
lean_inc(v_traceClass_495_);
v___x_499_ = l_Lean_Name_append(v___x_498_, v_traceClass_495_);
v___x_500_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_255_, v_options_254_, v___x_499_);
lean_dec(v___x_499_);
if (v___x_500_ == 0)
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; uint8_t v___x_587_; 
v___x_584_ = l_Lean_KVMap_instValueBool;
v___x_585_ = l_Lean_trace_profiler;
v___x_586_ = l_Lean_Option_get___redArg(v___x_584_, v_options_254_, v___x_585_);
v___x_587_ = lean_unbox(v___x_586_);
lean_dec(v___x_586_);
if (v___x_587_ == 0)
{
lean_object* v___x_588_; 
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_588_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_493_, v_a_216_, v_a_217_);
v___y_475_ = v___x_588_;
goto v___jp_474_;
}
else
{
goto v___jp_532_;
}
}
else
{
goto v___jp_532_;
}
v___jp_501_:
{
lean_object* v___x_505_; double v___x_506_; double v___x_507_; double v___x_508_; double v___x_509_; double v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; uint8_t v___x_515_; lean_object* v___x_11627__overap_516_; lean_object* v___x_517_; 
v___x_505_ = lean_io_mono_nanos_now();
v___x_506_ = lean_float_of_nat(v___y_502_);
v___x_507_ = lean_float_once(&lp_aesop_Aesop_Index_trace___redArg___closed__7, &lp_aesop_Aesop_Index_trace___redArg___closed__7_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__7);
v___x_508_ = lean_float_div(v___x_506_, v___x_507_);
v___x_509_ = lean_float_of_nat(v___x_505_);
v___x_510_ = lean_float_div(v___x_509_, v___x_507_);
v___x_511_ = lean_box_float(v___x_508_);
v___x_512_ = lean_box_float(v___x_510_);
v___x_513_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_511_);
lean_ctor_set(v___x_513_, 1, v___x_512_);
v___x_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_514_, 0, v_a_504_);
lean_ctor_set(v___x_514_, 1, v___x_513_);
v___x_515_ = lean_unbox(v_a_238_);
lean_inc(v_traceClass_495_);
lean_inc_ref(v___x_234_);
v___x_11627__overap_516_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v_traceClass_495_, v___x_515_, v___x_497_, v_options_254_, v___x_500_, v___y_503_, v___f_496_, v___x_514_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_517_ = lean_apply_3(v___x_11627__overap_516_, v_a_216_, v_a_217_, lean_box(0));
v___y_475_ = v___x_517_;
goto v___jp_474_;
}
v___jp_518_:
{
lean_object* v___x_522_; double v___x_523_; double v___x_524_; lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; uint8_t v___x_529_; lean_object* v___x_11648__overap_530_; lean_object* v___x_531_; 
v___x_522_ = lean_io_get_num_heartbeats();
v___x_523_ = lean_float_of_nat(v___y_519_);
v___x_524_ = lean_float_of_nat(v___x_522_);
v___x_525_ = lean_box_float(v___x_523_);
v___x_526_ = lean_box_float(v___x_524_);
v___x_527_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_527_, 0, v___x_525_);
lean_ctor_set(v___x_527_, 1, v___x_526_);
v___x_528_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_528_, 0, v_a_521_);
lean_ctor_set(v___x_528_, 1, v___x_527_);
v___x_529_ = lean_unbox(v_a_238_);
lean_inc(v_traceClass_495_);
lean_inc_ref(v___x_234_);
v___x_11648__overap_530_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v_traceClass_495_, v___x_529_, v___x_497_, v_options_254_, v___x_500_, v___y_520_, v___f_496_, v___x_528_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_531_ = lean_apply_3(v___x_11648__overap_530_, v_a_216_, v_a_217_, lean_box(0));
v___y_475_ = v___x_531_;
goto v___jp_474_;
}
v___jp_532_:
{
lean_object* v___x_11604__overap_533_; lean_object* v___x_534_; 
lean_inc_ref(v___x_234_);
v___x_11604__overap_533_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_234_, v___x_247_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_534_ = lean_apply_3(v___x_11604__overap_533_, v_a_216_, v_a_217_, lean_box(0));
if (lean_obj_tag(v___x_534_) == 0)
{
lean_object* v_a_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; uint8_t v___x_539_; 
v_a_535_ = lean_ctor_get(v___x_534_, 0);
lean_inc(v_a_535_);
lean_dec_ref_known(v___x_534_, 1);
v___x_536_ = l_Lean_KVMap_instValueBool;
v___x_537_ = l_Lean_trace_profiler_useHeartbeats;
v___x_538_ = l_Lean_Option_get___redArg(v___x_536_, v_options_254_, v___x_537_);
v___x_539_ = lean_unbox(v___x_538_);
lean_dec(v___x_538_);
if (v___x_539_ == 0)
{
lean_object* v___x_540_; lean_object* v___x_541_; 
v___x_540_ = lean_io_mono_nanos_now();
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_541_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_493_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_541_) == 0)
{
lean_object* v_a_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_549_; 
v_a_542_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_549_ == 0)
{
v___x_544_ = v___x_541_;
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_a_542_);
lean_dec(v___x_541_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___x_547_; 
if (v_isShared_545_ == 0)
{
lean_ctor_set_tag(v___x_544_, 1);
v___x_547_ = v___x_544_;
goto v_reusejp_546_;
}
else
{
lean_object* v_reuseFailAlloc_548_; 
v_reuseFailAlloc_548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_548_, 0, v_a_542_);
v___x_547_ = v_reuseFailAlloc_548_;
goto v_reusejp_546_;
}
v_reusejp_546_:
{
v___y_502_ = v___x_540_;
v___y_503_ = v_a_535_;
v_a_504_ = v___x_547_;
goto v___jp_501_;
}
}
}
else
{
lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_557_; 
v_a_550_ = lean_ctor_get(v___x_541_, 0);
v_isSharedCheck_557_ = !lean_is_exclusive(v___x_541_);
if (v_isSharedCheck_557_ == 0)
{
v___x_552_ = v___x_541_;
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_541_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
lean_object* v___x_555_; 
if (v_isShared_553_ == 0)
{
lean_ctor_set_tag(v___x_552_, 0);
v___x_555_ = v___x_552_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_556_; 
v_reuseFailAlloc_556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_556_, 0, v_a_550_);
v___x_555_ = v_reuseFailAlloc_556_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
v___y_502_ = v___x_540_;
v___y_503_ = v_a_535_;
v_a_504_ = v___x_555_;
goto v___jp_501_;
}
}
}
}
else
{
lean_object* v___x_558_; lean_object* v___x_559_; 
v___x_558_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_559_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_493_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_559_) == 0)
{
lean_object* v_a_560_; lean_object* v___x_562_; uint8_t v_isShared_563_; uint8_t v_isSharedCheck_567_; 
v_a_560_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_567_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_567_ == 0)
{
v___x_562_ = v___x_559_;
v_isShared_563_ = v_isSharedCheck_567_;
goto v_resetjp_561_;
}
else
{
lean_inc(v_a_560_);
lean_dec(v___x_559_);
v___x_562_ = lean_box(0);
v_isShared_563_ = v_isSharedCheck_567_;
goto v_resetjp_561_;
}
v_resetjp_561_:
{
lean_object* v___x_565_; 
if (v_isShared_563_ == 0)
{
lean_ctor_set_tag(v___x_562_, 1);
v___x_565_ = v___x_562_;
goto v_reusejp_564_;
}
else
{
lean_object* v_reuseFailAlloc_566_; 
v_reuseFailAlloc_566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_566_, 0, v_a_560_);
v___x_565_ = v_reuseFailAlloc_566_;
goto v_reusejp_564_;
}
v_reusejp_564_:
{
v___y_519_ = v___x_558_;
v___y_520_ = v_a_535_;
v_a_521_ = v___x_565_;
goto v___jp_518_;
}
}
}
else
{
lean_object* v_a_568_; lean_object* v___x_570_; uint8_t v_isShared_571_; uint8_t v_isSharedCheck_575_; 
v_a_568_ = lean_ctor_get(v___x_559_, 0);
v_isSharedCheck_575_ = !lean_is_exclusive(v___x_559_);
if (v_isSharedCheck_575_ == 0)
{
v___x_570_ = v___x_559_;
v_isShared_571_ = v_isSharedCheck_575_;
goto v_resetjp_569_;
}
else
{
lean_inc(v_a_568_);
lean_dec(v___x_559_);
v___x_570_ = lean_box(0);
v_isShared_571_ = v_isSharedCheck_575_;
goto v_resetjp_569_;
}
v_resetjp_569_:
{
lean_object* v___x_573_; 
if (v_isShared_571_ == 0)
{
lean_ctor_set_tag(v___x_570_, 0);
v___x_573_ = v___x_570_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_574_; 
v_reuseFailAlloc_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_574_, 0, v_a_568_);
v___x_573_ = v_reuseFailAlloc_574_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
v___y_519_ = v___x_558_;
v___y_520_ = v_a_535_;
v_a_521_ = v___x_573_;
goto v___jp_518_;
}
}
}
}
}
else
{
lean_object* v_a_576_; lean_object* v___x_578_; uint8_t v_isShared_579_; uint8_t v_isSharedCheck_583_; 
lean_dec(v___x_493_);
lean_dec_ref(v_unindexed_252_);
lean_dec_ref(v_byHyp_251_);
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_inst_213_);
v_a_576_ = lean_ctor_get(v___x_534_, 0);
v_isSharedCheck_583_ = !lean_is_exclusive(v___x_534_);
if (v_isSharedCheck_583_ == 0)
{
v___x_578_ = v___x_534_;
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
else
{
lean_inc(v_a_576_);
lean_dec(v___x_534_);
v___x_578_ = lean_box(0);
v_isShared_579_ = v_isSharedCheck_583_;
goto v_resetjp_577_;
}
v_resetjp_577_:
{
lean_object* v___x_581_; 
if (v_isShared_579_ == 0)
{
v___x_581_ = v___x_578_;
goto v_reusejp_580_;
}
else
{
lean_object* v_reuseFailAlloc_582_; 
v_reuseFailAlloc_582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_582_, 0, v_a_576_);
v___x_581_ = v_reuseFailAlloc_582_;
goto v_reusejp_580_;
}
v_reusejp_580_:
{
return v___x_581_;
}
}
}
}
}
v___jp_260_:
{
lean_object* v___x_269_; double v___x_270_; double v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; uint8_t v___x_276_; lean_object* v___x_11528__overap_277_; lean_object* v___x_278_; 
v___x_269_ = lean_io_get_num_heartbeats();
v___x_270_ = lean_float_of_nat(v___y_265_);
v___x_271_ = lean_float_of_nat(v___x_269_);
v___x_272_ = lean_box_float(v___x_270_);
v___x_273_ = lean_box_float(v___x_271_);
v___x_274_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_274_, 0, v___x_272_);
lean_ctor_set(v___x_274_, 1, v___x_273_);
v___x_275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_275_, 0, v_a_268_);
lean_ctor_set(v___x_275_, 1, v___x_274_);
v___x_276_ = lean_unbox(v_a_238_);
lean_dec(v_a_238_);
lean_inc_ref(v___y_263_);
lean_inc_ref(v___y_266_);
v___x_11528__overap_277_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v___y_262_, v___x_276_, v___y_266_, v___y_261_, v___y_267_, v___y_264_, v___y_263_, v___x_275_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_278_ = lean_apply_3(v___x_11528__overap_277_, v_a_216_, v_a_217_, lean_box(0));
return v___x_278_;
}
v___jp_279_:
{
lean_object* v___x_288_; double v___x_289_; double v___x_290_; double v___x_291_; double v___x_292_; double v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; uint8_t v___x_298_; lean_object* v___x_11507__overap_299_; lean_object* v___x_300_; 
v___x_288_ = lean_io_mono_nanos_now();
v___x_289_ = lean_float_of_nat(v___y_285_);
v___x_290_ = lean_float_once(&lp_aesop_Aesop_Index_trace___redArg___closed__7, &lp_aesop_Aesop_Index_trace___redArg___closed__7_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__7);
v___x_291_ = lean_float_div(v___x_289_, v___x_290_);
v___x_292_ = lean_float_of_nat(v___x_288_);
v___x_293_ = lean_float_div(v___x_292_, v___x_290_);
v___x_294_ = lean_box_float(v___x_291_);
v___x_295_ = lean_box_float(v___x_293_);
v___x_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_296_, 0, v___x_294_);
lean_ctor_set(v___x_296_, 1, v___x_295_);
v___x_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_297_, 0, v_a_287_);
lean_ctor_set(v___x_297_, 1, v___x_296_);
v___x_298_ = lean_unbox(v_a_238_);
lean_dec(v_a_238_);
lean_inc_ref(v___y_282_);
lean_inc_ref(v___y_284_);
v___x_11507__overap_299_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v___y_281_, v___x_298_, v___y_284_, v___y_280_, v___y_286_, v___y_283_, v___y_282_, v___x_297_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_300_ = lean_apply_3(v___x_11507__overap_299_, v_a_216_, v_a_217_, lean_box(0));
return v___x_300_;
}
v___jp_301_:
{
lean_object* v___x_11484__overap_308_; lean_object* v___x_309_; 
lean_inc_ref(v___x_234_);
v___x_11484__overap_308_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_234_, v___x_247_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_309_ = lean_apply_3(v___x_11484__overap_308_, v_a_216_, v_a_217_, lean_box(0));
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_a_310_);
lean_dec_ref_known(v___x_309_, 1);
v___x_311_ = l_Lean_KVMap_instValueBool;
v___x_312_ = l_Lean_trace_profiler_useHeartbeats;
v___x_313_ = l_Lean_Option_get___redArg(v___x_311_, v___y_302_, v___x_312_);
v___x_314_ = lean_unbox(v___x_313_);
lean_dec(v___x_313_);
if (v___x_314_ == 0)
{
lean_object* v___x_315_; lean_object* v___x_316_; 
v___x_315_ = lean_io_mono_nanos_now();
v___x_316_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___y_307_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_316_) == 0)
{
lean_object* v_a_317_; lean_object* v___x_319_; uint8_t v_isShared_320_; uint8_t v_isSharedCheck_324_; 
v_a_317_ = lean_ctor_get(v___x_316_, 0);
v_isSharedCheck_324_ = !lean_is_exclusive(v___x_316_);
if (v_isSharedCheck_324_ == 0)
{
v___x_319_ = v___x_316_;
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
else
{
lean_inc(v_a_317_);
lean_dec(v___x_316_);
v___x_319_ = lean_box(0);
v_isShared_320_ = v_isSharedCheck_324_;
goto v_resetjp_318_;
}
v_resetjp_318_:
{
lean_object* v___x_322_; 
if (v_isShared_320_ == 0)
{
lean_ctor_set_tag(v___x_319_, 1);
v___x_322_ = v___x_319_;
goto v_reusejp_321_;
}
else
{
lean_object* v_reuseFailAlloc_323_; 
v_reuseFailAlloc_323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_323_, 0, v_a_317_);
v___x_322_ = v_reuseFailAlloc_323_;
goto v_reusejp_321_;
}
v_reusejp_321_:
{
v___y_280_ = v___y_302_;
v___y_281_ = v___y_303_;
v___y_282_ = v___y_304_;
v___y_283_ = v_a_310_;
v___y_284_ = v___y_305_;
v___y_285_ = v___x_315_;
v___y_286_ = v___y_306_;
v_a_287_ = v___x_322_;
goto v___jp_279_;
}
}
}
else
{
lean_object* v_a_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_332_; 
v_a_325_ = lean_ctor_get(v___x_316_, 0);
v_isSharedCheck_332_ = !lean_is_exclusive(v___x_316_);
if (v_isSharedCheck_332_ == 0)
{
v___x_327_ = v___x_316_;
v_isShared_328_ = v_isSharedCheck_332_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_a_325_);
lean_dec(v___x_316_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_332_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
lean_object* v___x_330_; 
if (v_isShared_328_ == 0)
{
lean_ctor_set_tag(v___x_327_, 0);
v___x_330_ = v___x_327_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_331_; 
v_reuseFailAlloc_331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_331_, 0, v_a_325_);
v___x_330_ = v_reuseFailAlloc_331_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
v___y_280_ = v___y_302_;
v___y_281_ = v___y_303_;
v___y_282_ = v___y_304_;
v___y_283_ = v_a_310_;
v___y_284_ = v___y_305_;
v___y_285_ = v___x_315_;
v___y_286_ = v___y_306_;
v_a_287_ = v___x_330_;
goto v___jp_279_;
}
}
}
}
else
{
lean_object* v___x_333_; lean_object* v___x_334_; 
v___x_333_ = lean_io_get_num_heartbeats();
v___x_334_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___y_307_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_334_) == 0)
{
lean_object* v_a_335_; lean_object* v___x_337_; uint8_t v_isShared_338_; uint8_t v_isSharedCheck_342_; 
v_a_335_ = lean_ctor_get(v___x_334_, 0);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_334_);
if (v_isSharedCheck_342_ == 0)
{
v___x_337_ = v___x_334_;
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
else
{
lean_inc(v_a_335_);
lean_dec(v___x_334_);
v___x_337_ = lean_box(0);
v_isShared_338_ = v_isSharedCheck_342_;
goto v_resetjp_336_;
}
v_resetjp_336_:
{
lean_object* v___x_340_; 
if (v_isShared_338_ == 0)
{
lean_ctor_set_tag(v___x_337_, 1);
v___x_340_ = v___x_337_;
goto v_reusejp_339_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v_a_335_);
v___x_340_ = v_reuseFailAlloc_341_;
goto v_reusejp_339_;
}
v_reusejp_339_:
{
v___y_261_ = v___y_302_;
v___y_262_ = v___y_303_;
v___y_263_ = v___y_304_;
v___y_264_ = v_a_310_;
v___y_265_ = v___x_333_;
v___y_266_ = v___y_305_;
v___y_267_ = v___y_306_;
v_a_268_ = v___x_340_;
goto v___jp_260_;
}
}
}
else
{
lean_object* v_a_343_; lean_object* v___x_345_; uint8_t v_isShared_346_; uint8_t v_isSharedCheck_350_; 
v_a_343_ = lean_ctor_get(v___x_334_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v___x_334_);
if (v_isSharedCheck_350_ == 0)
{
v___x_345_ = v___x_334_;
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
else
{
lean_inc(v_a_343_);
lean_dec(v___x_334_);
v___x_345_ = lean_box(0);
v_isShared_346_ = v_isSharedCheck_350_;
goto v_resetjp_344_;
}
v_resetjp_344_:
{
lean_object* v___x_348_; 
if (v_isShared_346_ == 0)
{
lean_ctor_set_tag(v___x_345_, 0);
v___x_348_ = v___x_345_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v_a_343_);
v___x_348_ = v_reuseFailAlloc_349_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
v___y_261_ = v___y_302_;
v___y_262_ = v___y_303_;
v___y_263_ = v___y_304_;
v___y_264_ = v_a_310_;
v___y_265_ = v___x_333_;
v___y_266_ = v___y_305_;
v___y_267_ = v___y_306_;
v_a_268_ = v___x_348_;
goto v___jp_260_;
}
}
}
}
}
else
{
lean_object* v_a_351_; lean_object* v___x_353_; uint8_t v_isShared_354_; uint8_t v_isSharedCheck_358_; 
lean_dec(v___y_307_);
lean_dec(v___y_303_);
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_inst_213_);
v_a_351_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_358_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_358_ == 0)
{
v___x_353_ = v___x_309_;
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
else
{
lean_inc(v_a_351_);
lean_dec(v___x_309_);
v___x_353_ = lean_box(0);
v_isShared_354_ = v_isSharedCheck_358_;
goto v_resetjp_352_;
}
v_resetjp_352_:
{
lean_object* v___x_356_; 
if (v_isShared_354_ == 0)
{
v___x_356_ = v___x_353_;
goto v_reusejp_355_;
}
else
{
lean_object* v_reuseFailAlloc_357_; 
v_reuseFailAlloc_357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_357_, 0, v_a_351_);
v___x_356_ = v_reuseFailAlloc_357_;
goto v_reusejp_355_;
}
v_reusejp_355_:
{
return v___x_356_;
}
}
}
}
v___jp_359_:
{
if (lean_obj_tag(v___y_360_) == 0)
{
lean_object* v___x_361_; lean_object* v___x_362_; 
lean_dec_ref_known(v___y_360_, 1);
v___x_361_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__8));
v___x_362_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_253_, v___f_257_, v_unindexed_252_, v___x_361_);
if (v_hasTrace_256_ == 0)
{
lean_object* v___x_363_; 
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
v___x_363_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_362_, v_a_216_, v_a_217_);
return v___x_363_;
}
else
{
lean_object* v_traceClass_364_; lean_object* v___f_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; uint8_t v___x_369_; 
v_traceClass_364_ = lean_ctor_get(v_traceOpt_215_, 0);
v___f_365_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__12, &lp_aesop_Aesop_Index_trace___redArg___closed__12_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__12);
v___x_366_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__13));
v___x_367_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__15));
lean_inc(v_traceClass_364_);
v___x_368_ = l_Lean_Name_append(v___x_367_, v_traceClass_364_);
v___x_369_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_255_, v_options_254_, v___x_368_);
lean_dec(v___x_368_);
if (v___x_369_ == 0)
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_370_ = l_Lean_KVMap_instValueBool;
v___x_371_ = l_Lean_trace_profiler;
v___x_372_ = l_Lean_Option_get___redArg(v___x_370_, v_options_254_, v___x_371_);
v___x_373_ = lean_unbox(v___x_372_);
lean_dec(v___x_372_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; 
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
v___x_374_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_362_, v_a_216_, v_a_217_);
return v___x_374_;
}
else
{
lean_inc(v_traceClass_364_);
v___y_302_ = v_options_254_;
v___y_303_ = v_traceClass_364_;
v___y_304_ = v___f_365_;
v___y_305_ = v___x_366_;
v___y_306_ = v___x_369_;
v___y_307_ = v___x_362_;
goto v___jp_301_;
}
}
else
{
lean_inc(v_traceClass_364_);
v___y_302_ = v_options_254_;
v___y_303_ = v_traceClass_364_;
v___y_304_ = v___f_365_;
v___y_305_ = v___x_366_;
v___y_306_ = v___x_369_;
v___y_307_ = v___x_362_;
goto v___jp_301_;
}
}
}
else
{
lean_dec_ref(v_unindexed_252_);
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_inst_213_);
return v___y_360_;
}
}
v___jp_375_:
{
lean_object* v___x_384_; double v___x_385_; double v___x_386_; double v___x_387_; double v___x_388_; double v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; uint8_t v___x_394_; lean_object* v___x_11567__overap_395_; lean_object* v___x_396_; 
v___x_384_ = lean_io_mono_nanos_now();
v___x_385_ = lean_float_of_nat(v___y_378_);
v___x_386_ = lean_float_once(&lp_aesop_Aesop_Index_trace___redArg___closed__7, &lp_aesop_Aesop_Index_trace___redArg___closed__7_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__7);
v___x_387_ = lean_float_div(v___x_385_, v___x_386_);
v___x_388_ = lean_float_of_nat(v___x_384_);
v___x_389_ = lean_float_div(v___x_388_, v___x_386_);
v___x_390_ = lean_box_float(v___x_387_);
v___x_391_ = lean_box_float(v___x_389_);
v___x_392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_392_, 0, v___x_390_);
lean_ctor_set(v___x_392_, 1, v___x_391_);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v_a_383_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
v___x_394_ = lean_unbox(v_a_238_);
lean_inc_ref(v___y_381_);
lean_inc_ref(v___y_379_);
lean_inc_ref(v___x_234_);
v___x_11567__overap_395_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v___y_380_, v___x_394_, v___y_379_, v___y_382_, v___y_377_, v___y_376_, v___y_381_, v___x_393_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_396_ = lean_apply_3(v___x_11567__overap_395_, v_a_216_, v_a_217_, lean_box(0));
v___y_360_ = v___x_396_;
goto v___jp_359_;
}
v___jp_397_:
{
lean_object* v___x_406_; double v___x_407_; double v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; uint8_t v___x_413_; lean_object* v___x_11588__overap_414_; lean_object* v___x_415_; 
v___x_406_ = lean_io_get_num_heartbeats();
v___x_407_ = lean_float_of_nat(v___y_399_);
v___x_408_ = lean_float_of_nat(v___x_406_);
v___x_409_ = lean_box_float(v___x_407_);
v___x_410_ = lean_box_float(v___x_408_);
v___x_411_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_411_, 0, v___x_409_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
v___x_412_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_412_, 0, v_a_405_);
lean_ctor_set(v___x_412_, 1, v___x_411_);
v___x_413_ = lean_unbox(v_a_238_);
lean_inc_ref(v___y_403_);
lean_inc_ref(v___y_401_);
lean_inc_ref(v___x_234_);
v___x_11588__overap_414_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_234_, v___x_247_, v___x_248_, v___x_258_, lean_box(0), v___x_249_, v___f_259_, v___y_402_, v___x_413_, v___y_401_, v___y_404_, v___y_400_, v___y_398_, v___y_403_, v___x_412_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_415_ = lean_apply_3(v___x_11588__overap_414_, v_a_216_, v_a_217_, lean_box(0));
v___y_360_ = v___x_415_;
goto v___jp_359_;
}
v___jp_416_:
{
lean_object* v___x_11544__overap_423_; lean_object* v___x_424_; 
lean_inc_ref(v___x_234_);
v___x_11544__overap_423_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_234_, v___x_247_);
lean_inc(v_a_217_);
lean_inc_ref(v_a_216_);
v___x_424_ = lean_apply_3(v___x_11544__overap_423_, v_a_216_, v_a_217_, lean_box(0));
if (lean_obj_tag(v___x_424_) == 0)
{
lean_object* v_a_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; uint8_t v___x_429_; 
v_a_425_ = lean_ctor_get(v___x_424_, 0);
lean_inc(v_a_425_);
lean_dec_ref_known(v___x_424_, 1);
v___x_426_ = l_Lean_KVMap_instValueBool;
v___x_427_ = l_Lean_trace_profiler_useHeartbeats;
v___x_428_ = l_Lean_Option_get___redArg(v___x_426_, v___y_422_, v___x_427_);
v___x_429_ = lean_unbox(v___x_428_);
lean_dec(v___x_428_);
if (v___x_429_ == 0)
{
lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_430_ = lean_io_mono_nanos_now();
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_431_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___y_419_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_431_) == 0)
{
lean_object* v_a_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_439_; 
v_a_432_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_439_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_439_ == 0)
{
v___x_434_ = v___x_431_;
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_a_432_);
lean_dec(v___x_431_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_439_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_437_; 
if (v_isShared_435_ == 0)
{
lean_ctor_set_tag(v___x_434_, 1);
v___x_437_ = v___x_434_;
goto v_reusejp_436_;
}
else
{
lean_object* v_reuseFailAlloc_438_; 
v_reuseFailAlloc_438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_438_, 0, v_a_432_);
v___x_437_ = v_reuseFailAlloc_438_;
goto v_reusejp_436_;
}
v_reusejp_436_:
{
v___y_376_ = v_a_425_;
v___y_377_ = v___y_417_;
v___y_378_ = v___x_430_;
v___y_379_ = v___y_418_;
v___y_380_ = v___y_420_;
v___y_381_ = v___y_421_;
v___y_382_ = v___y_422_;
v_a_383_ = v___x_437_;
goto v___jp_375_;
}
}
}
else
{
lean_object* v_a_440_; lean_object* v___x_442_; uint8_t v_isShared_443_; uint8_t v_isSharedCheck_447_; 
v_a_440_ = lean_ctor_get(v___x_431_, 0);
v_isSharedCheck_447_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_447_ == 0)
{
v___x_442_ = v___x_431_;
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
else
{
lean_inc(v_a_440_);
lean_dec(v___x_431_);
v___x_442_ = lean_box(0);
v_isShared_443_ = v_isSharedCheck_447_;
goto v_resetjp_441_;
}
v_resetjp_441_:
{
lean_object* v___x_445_; 
if (v_isShared_443_ == 0)
{
lean_ctor_set_tag(v___x_442_, 0);
v___x_445_ = v___x_442_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_446_; 
v_reuseFailAlloc_446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_446_, 0, v_a_440_);
v___x_445_ = v_reuseFailAlloc_446_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
v___y_376_ = v_a_425_;
v___y_377_ = v___y_417_;
v___y_378_ = v___x_430_;
v___y_379_ = v___y_418_;
v___y_380_ = v___y_420_;
v___y_381_ = v___y_421_;
v___y_382_ = v___y_422_;
v_a_383_ = v___x_445_;
goto v___jp_375_;
}
}
}
}
else
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = lean_io_get_num_heartbeats();
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_449_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___y_419_, v_a_216_, v_a_217_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; lean_object* v___x_452_; uint8_t v_isShared_453_; uint8_t v_isSharedCheck_457_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_457_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_457_ == 0)
{
v___x_452_ = v___x_449_;
v_isShared_453_ = v_isSharedCheck_457_;
goto v_resetjp_451_;
}
else
{
lean_inc(v_a_450_);
lean_dec(v___x_449_);
v___x_452_ = lean_box(0);
v_isShared_453_ = v_isSharedCheck_457_;
goto v_resetjp_451_;
}
v_resetjp_451_:
{
lean_object* v___x_455_; 
if (v_isShared_453_ == 0)
{
lean_ctor_set_tag(v___x_452_, 1);
v___x_455_ = v___x_452_;
goto v_reusejp_454_;
}
else
{
lean_object* v_reuseFailAlloc_456_; 
v_reuseFailAlloc_456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_456_, 0, v_a_450_);
v___x_455_ = v_reuseFailAlloc_456_;
goto v_reusejp_454_;
}
v_reusejp_454_:
{
v___y_398_ = v_a_425_;
v___y_399_ = v___x_448_;
v___y_400_ = v___y_417_;
v___y_401_ = v___y_418_;
v___y_402_ = v___y_420_;
v___y_403_ = v___y_421_;
v___y_404_ = v___y_422_;
v_a_405_ = v___x_455_;
goto v___jp_397_;
}
}
}
else
{
lean_object* v_a_458_; lean_object* v___x_460_; uint8_t v_isShared_461_; uint8_t v_isSharedCheck_465_; 
v_a_458_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_465_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_465_ == 0)
{
v___x_460_ = v___x_449_;
v_isShared_461_ = v_isSharedCheck_465_;
goto v_resetjp_459_;
}
else
{
lean_inc(v_a_458_);
lean_dec(v___x_449_);
v___x_460_ = lean_box(0);
v_isShared_461_ = v_isSharedCheck_465_;
goto v_resetjp_459_;
}
v_resetjp_459_:
{
lean_object* v___x_463_; 
if (v_isShared_461_ == 0)
{
lean_ctor_set_tag(v___x_460_, 0);
v___x_463_ = v___x_460_;
goto v_reusejp_462_;
}
else
{
lean_object* v_reuseFailAlloc_464_; 
v_reuseFailAlloc_464_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_464_, 0, v_a_458_);
v___x_463_ = v_reuseFailAlloc_464_;
goto v_reusejp_462_;
}
v_reusejp_462_:
{
v___y_398_ = v_a_425_;
v___y_399_ = v___x_448_;
v___y_400_ = v___y_417_;
v___y_401_ = v___y_418_;
v___y_402_ = v___y_420_;
v___y_403_ = v___y_421_;
v___y_404_ = v___y_422_;
v_a_405_ = v___x_463_;
goto v___jp_397_;
}
}
}
}
}
else
{
lean_object* v_a_466_; lean_object* v___x_468_; uint8_t v_isShared_469_; uint8_t v_isSharedCheck_473_; 
lean_dec(v___y_420_);
lean_dec(v___y_419_);
lean_dec_ref(v_unindexed_252_);
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_inst_213_);
v_a_466_ = lean_ctor_get(v___x_424_, 0);
v_isSharedCheck_473_ = !lean_is_exclusive(v___x_424_);
if (v_isSharedCheck_473_ == 0)
{
v___x_468_ = v___x_424_;
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
else
{
lean_inc(v_a_466_);
lean_dec(v___x_424_);
v___x_468_ = lean_box(0);
v_isShared_469_ = v_isSharedCheck_473_;
goto v_resetjp_467_;
}
v_resetjp_467_:
{
lean_object* v___x_471_; 
if (v_isShared_469_ == 0)
{
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
}
v___jp_474_:
{
if (lean_obj_tag(v___y_475_) == 0)
{
lean_object* v___x_476_; lean_object* v___f_477_; lean_object* v___x_478_; 
lean_dec_ref_known(v___y_475_, 1);
v___x_476_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__8));
v___f_477_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__16));
v___x_478_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_253_, v___f_477_, v_byHyp_251_, v___x_476_);
if (v_hasTrace_256_ == 0)
{
lean_object* v___x_479_; 
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_479_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_478_, v_a_216_, v_a_217_);
v___y_360_ = v___x_479_;
goto v___jp_359_;
}
else
{
lean_object* v_traceClass_480_; lean_object* v___f_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; uint8_t v___x_485_; 
v_traceClass_480_ = lean_ctor_get(v_traceOpt_215_, 0);
v___f_481_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__20, &lp_aesop_Aesop_Index_trace___redArg___closed__20_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__20);
v___x_482_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__13));
v___x_483_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__15));
lean_inc(v_traceClass_480_);
v___x_484_ = l_Lean_Name_append(v___x_483_, v_traceClass_480_);
v___x_485_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_255_, v_options_254_, v___x_484_);
lean_dec(v___x_484_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; uint8_t v___x_489_; 
v___x_486_ = l_Lean_KVMap_instValueBool;
v___x_487_ = l_Lean_trace_profiler;
v___x_488_ = l_Lean_Option_get___redArg(v___x_486_, v_options_254_, v___x_487_);
v___x_489_ = lean_unbox(v___x_488_);
lean_dec(v___x_488_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; 
lean_inc_ref(v_traceOpt_215_);
lean_inc_ref(v_inst_213_);
v___x_490_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg(v_inst_213_, v_traceOpt_215_, v___x_478_, v_a_216_, v_a_217_);
v___y_360_ = v___x_490_;
goto v___jp_359_;
}
else
{
lean_inc(v_traceClass_480_);
v___y_417_ = v___x_485_;
v___y_418_ = v___x_482_;
v___y_419_ = v___x_478_;
v___y_420_ = v_traceClass_480_;
v___y_421_ = v___f_481_;
v___y_422_ = v_options_254_;
goto v___jp_416_;
}
}
else
{
lean_inc(v_traceClass_480_);
v___y_417_ = v___x_485_;
v___y_418_ = v___x_482_;
v___y_419_ = v___x_478_;
v___y_420_ = v_traceClass_480_;
v___y_421_ = v___f_481_;
v___y_422_ = v_options_254_;
goto v___jp_416_;
}
}
}
else
{
lean_dec_ref(v_unindexed_252_);
lean_dec_ref(v_byHyp_251_);
lean_dec(v_a_238_);
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_inst_213_);
return v___y_475_;
}
}
}
}
}
else
{
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
lean_dec_ref_known(v___x_234_, 2);
lean_dec_ref(v_traceOpt_215_);
lean_dec_ref(v_ri_214_);
lean_dec_ref(v_inst_213_);
v_a_590_ = lean_ctor_get(v___x_237_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_237_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_237_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_237_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_590_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___redArg___boxed(lean_object* v_inst_598_, lean_object* v_ri_599_, lean_object* v_traceOpt_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_){
_start:
{
lean_object* v_res_604_; 
v_res_604_ = lp_aesop_Aesop_Index_trace___redArg(v_inst_598_, v_ri_599_, v_traceOpt_600_, v_a_601_, v_a_602_);
lean_dec(v_a_602_);
lean_dec_ref(v_a_601_);
return v_res_604_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace(lean_object* v_00_u03b1_605_, lean_object* v_inst_606_, lean_object* v_ri_607_, lean_object* v_traceOpt_608_, lean_object* v_a_609_, lean_object* v_a_610_){
_start:
{
lean_object* v___x_612_; 
v___x_612_ = lp_aesop_Aesop_Index_trace___redArg(v_inst_606_, v_ri_607_, v_traceOpt_608_, v_a_609_, v_a_610_);
return v___x_612_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_trace___boxed(lean_object* v_00_u03b1_613_, lean_object* v_inst_614_, lean_object* v_ri_615_, lean_object* v_traceOpt_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_aesop_Aesop_Index_trace(v_00_u03b1_613_, v_inst_614_, v_ri_615_, v_traceOpt_616_, v_a_617_, v_a_618_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
return v_res_620_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_instEmptyCollection___closed__2(void){
_start:
{
lean_object* v___f_623_; lean_object* v___f_624_; lean_object* v___x_625_; 
v___f_623_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__1));
v___f_624_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__0));
v___x_625_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___f_624_, v___f_623_);
return v___x_625_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_instEmptyCollection___closed__3(void){
_start:
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; 
v___x_626_ = lean_obj_once(&lp_aesop_Aesop_Index_instEmptyCollection___closed__2, &lp_aesop_Aesop_Index_instEmptyCollection___closed__2_once, _init_lp_aesop_Aesop_Index_instEmptyCollection___closed__2);
v___x_627_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedIndex_default___closed__1, &lp_aesop_Aesop_instInhabitedIndex_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedIndex_default___closed__1);
v___x_628_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_628_, 0, v___x_627_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
lean_ctor_set(v___x_628_, 2, v___x_626_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_instEmptyCollection(lean_object* v_00_u03b1_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lean_obj_once(&lp_aesop_Aesop_Index_instEmptyCollection___closed__3, &lp_aesop_Aesop_Index_instEmptyCollection___closed__3_once, _init_lp_aesop_Aesop_Index_instEmptyCollection___closed__3);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg(lean_object* v_keys_631_, lean_object* v_vals_632_, lean_object* v_i_633_, lean_object* v_k_634_){
_start:
{
lean_object* v___x_635_; uint8_t v___x_636_; 
v___x_635_ = lean_array_get_size(v_keys_631_);
v___x_636_ = lean_nat_dec_lt(v_i_633_, v___x_635_);
if (v___x_636_ == 0)
{
lean_object* v___x_637_; 
lean_dec(v_i_633_);
v___x_637_ = lean_box(0);
return v___x_637_;
}
else
{
lean_object* v_k_x27_638_; uint8_t v___x_639_; 
v_k_x27_638_ = lean_array_fget_borrowed(v_keys_631_, v_i_633_);
v___x_639_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_k_634_, v_k_x27_638_);
if (v___x_639_ == 0)
{
lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_640_ = lean_unsigned_to_nat(1u);
v___x_641_ = lean_nat_add(v_i_633_, v___x_640_);
lean_dec(v_i_633_);
v_i_633_ = v___x_641_;
goto _start;
}
else
{
lean_object* v___x_643_; lean_object* v___x_644_; 
v___x_643_ = lean_array_fget_borrowed(v_vals_632_, v_i_633_);
lean_dec(v_i_633_);
lean_inc(v___x_643_);
v___x_644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_644_, 0, v___x_643_);
return v___x_644_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_keys_645_, lean_object* v_vals_646_, lean_object* v_i_647_, lean_object* v_k_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg(v_keys_645_, v_vals_646_, v_i_647_, v_k_648_);
lean_dec(v_k_648_);
lean_dec_ref(v_vals_646_);
lean_dec_ref(v_keys_645_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg(lean_object* v_x_650_, size_t v_x_651_, lean_object* v_x_652_){
_start:
{
if (lean_obj_tag(v_x_650_) == 0)
{
lean_object* v_es_653_; lean_object* v___x_654_; size_t v___x_655_; size_t v___x_656_; lean_object* v_j_657_; lean_object* v___x_658_; 
v_es_653_ = lean_ctor_get(v_x_650_, 0);
v___x_654_ = lean_box(2);
v___x_655_ = ((size_t)31ULL);
v___x_656_ = lean_usize_land(v_x_651_, v___x_655_);
v_j_657_ = lean_usize_to_nat(v___x_656_);
v___x_658_ = lean_array_get_borrowed(v___x_654_, v_es_653_, v_j_657_);
lean_dec(v_j_657_);
switch(lean_obj_tag(v___x_658_))
{
case 0:
{
lean_object* v_key_659_; lean_object* v_val_660_; uint8_t v___x_661_; 
v_key_659_ = lean_ctor_get(v___x_658_, 0);
v_val_660_ = lean_ctor_get(v___x_658_, 1);
v___x_661_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_652_, v_key_659_);
if (v___x_661_ == 0)
{
lean_object* v___x_662_; 
v___x_662_ = lean_box(0);
return v___x_662_;
}
else
{
lean_object* v___x_663_; 
lean_inc(v_val_660_);
v___x_663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_663_, 0, v_val_660_);
return v___x_663_;
}
}
case 1:
{
lean_object* v_node_664_; size_t v___x_665_; size_t v___x_666_; 
v_node_664_ = lean_ctor_get(v___x_658_, 0);
v___x_665_ = ((size_t)5ULL);
v___x_666_ = lean_usize_shift_right(v_x_651_, v___x_665_);
v_x_650_ = v_node_664_;
v_x_651_ = v___x_666_;
goto _start;
}
default: 
{
lean_object* v___x_668_; 
v___x_668_ = lean_box(0);
return v___x_668_;
}
}
}
else
{
lean_object* v_ks_669_; lean_object* v_vs_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v_ks_669_ = lean_ctor_get(v_x_650_, 0);
v_vs_670_ = lean_ctor_get(v_x_650_, 1);
v___x_671_ = lean_unsigned_to_nat(0u);
v___x_672_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg(v_ks_669_, v_vs_670_, v___x_671_, v_x_652_);
return v___x_672_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg___boxed(lean_object* v_x_673_, lean_object* v_x_674_, lean_object* v_x_675_){
_start:
{
size_t v_x_2964__boxed_676_; lean_object* v_res_677_; 
v_x_2964__boxed_676_ = lean_unbox_usize(v_x_674_);
lean_dec(v_x_674_);
v_res_677_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg(v_x_673_, v_x_2964__boxed_676_, v_x_675_);
lean_dec(v_x_675_);
lean_dec_ref(v_x_673_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg(lean_object* v_x_678_, lean_object* v_x_679_){
_start:
{
uint64_t v___x_680_; size_t v___x_681_; lean_object* v___x_682_; 
v___x_680_ = l_Lean_Meta_DiscrTree_Key_hash(v_x_679_);
v___x_681_ = lean_uint64_to_usize(v___x_680_);
v___x_682_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg(v_x_678_, v___x_681_, v_x_679_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg___boxed(lean_object* v_x_683_, lean_object* v_x_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg(v_x_683_, v_x_684_);
lean_dec(v_x_684_);
lean_dec_ref(v_x_683_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7___redArg(lean_object* v_x_686_, lean_object* v_x_687_, lean_object* v_x_688_, lean_object* v_x_689_){
_start:
{
lean_object* v_ks_690_; lean_object* v_vs_691_; lean_object* v___x_693_; uint8_t v_isShared_694_; uint8_t v_isSharedCheck_715_; 
v_ks_690_ = lean_ctor_get(v_x_686_, 0);
v_vs_691_ = lean_ctor_get(v_x_686_, 1);
v_isSharedCheck_715_ = !lean_is_exclusive(v_x_686_);
if (v_isSharedCheck_715_ == 0)
{
v___x_693_ = v_x_686_;
v_isShared_694_ = v_isSharedCheck_715_;
goto v_resetjp_692_;
}
else
{
lean_inc(v_vs_691_);
lean_inc(v_ks_690_);
lean_dec(v_x_686_);
v___x_693_ = lean_box(0);
v_isShared_694_ = v_isSharedCheck_715_;
goto v_resetjp_692_;
}
v_resetjp_692_:
{
lean_object* v___x_695_; uint8_t v___x_696_; 
v___x_695_ = lean_array_get_size(v_ks_690_);
v___x_696_ = lean_nat_dec_lt(v_x_687_, v___x_695_);
if (v___x_696_ == 0)
{
lean_object* v___x_697_; lean_object* v___x_698_; lean_object* v___x_700_; 
lean_dec(v_x_687_);
v___x_697_ = lean_array_push(v_ks_690_, v_x_688_);
v___x_698_ = lean_array_push(v_vs_691_, v_x_689_);
if (v_isShared_694_ == 0)
{
lean_ctor_set(v___x_693_, 1, v___x_698_);
lean_ctor_set(v___x_693_, 0, v___x_697_);
v___x_700_ = v___x_693_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v___x_697_);
lean_ctor_set(v_reuseFailAlloc_701_, 1, v___x_698_);
v___x_700_ = v_reuseFailAlloc_701_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
return v___x_700_;
}
}
else
{
lean_object* v_k_x27_702_; uint8_t v___x_703_; 
v_k_x27_702_ = lean_array_fget_borrowed(v_ks_690_, v_x_687_);
v___x_703_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_688_, v_k_x27_702_);
if (v___x_703_ == 0)
{
lean_object* v___x_705_; 
if (v_isShared_694_ == 0)
{
v___x_705_ = v___x_693_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_ks_690_);
lean_ctor_set(v_reuseFailAlloc_709_, 1, v_vs_691_);
v___x_705_ = v_reuseFailAlloc_709_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_706_ = lean_unsigned_to_nat(1u);
v___x_707_ = lean_nat_add(v_x_687_, v___x_706_);
lean_dec(v_x_687_);
v_x_686_ = v___x_705_;
v_x_687_ = v___x_707_;
goto _start;
}
}
else
{
lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_713_; 
v___x_710_ = lean_array_fset(v_ks_690_, v_x_687_, v_x_688_);
v___x_711_ = lean_array_fset(v_vs_691_, v_x_687_, v_x_689_);
lean_dec(v_x_687_);
if (v_isShared_694_ == 0)
{
lean_ctor_set(v___x_693_, 1, v___x_711_);
lean_ctor_set(v___x_693_, 0, v___x_710_);
v___x_713_ = v___x_693_;
goto v_reusejp_712_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v___x_710_);
lean_ctor_set(v_reuseFailAlloc_714_, 1, v___x_711_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4___redArg(lean_object* v_n_716_, lean_object* v_k_717_, lean_object* v_v_718_){
_start:
{
lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_719_ = lean_unsigned_to_nat(0u);
v___x_720_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7___redArg(v_n_716_, v___x_719_, v_k_717_, v_v_718_);
return v___x_720_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_721_; 
v___x_721_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(lean_object* v_x_722_, size_t v_x_723_, size_t v_x_724_, lean_object* v_x_725_, lean_object* v_x_726_){
_start:
{
if (lean_obj_tag(v_x_722_) == 0)
{
lean_object* v_es_727_; size_t v___x_728_; size_t v___x_729_; lean_object* v_j_730_; lean_object* v___x_731_; uint8_t v___x_732_; 
v_es_727_ = lean_ctor_get(v_x_722_, 0);
v___x_728_ = ((size_t)31ULL);
v___x_729_ = lean_usize_land(v_x_723_, v___x_728_);
v_j_730_ = lean_usize_to_nat(v___x_729_);
v___x_731_ = lean_array_get_size(v_es_727_);
v___x_732_ = lean_nat_dec_lt(v_j_730_, v___x_731_);
if (v___x_732_ == 0)
{
lean_dec(v_j_730_);
lean_dec(v_x_726_);
lean_dec(v_x_725_);
return v_x_722_;
}
else
{
lean_object* v___x_734_; uint8_t v_isShared_735_; uint8_t v_isSharedCheck_771_; 
lean_inc_ref(v_es_727_);
v_isSharedCheck_771_ = !lean_is_exclusive(v_x_722_);
if (v_isSharedCheck_771_ == 0)
{
lean_object* v_unused_772_; 
v_unused_772_ = lean_ctor_get(v_x_722_, 0);
lean_dec(v_unused_772_);
v___x_734_ = v_x_722_;
v_isShared_735_ = v_isSharedCheck_771_;
goto v_resetjp_733_;
}
else
{
lean_dec(v_x_722_);
v___x_734_ = lean_box(0);
v_isShared_735_ = v_isSharedCheck_771_;
goto v_resetjp_733_;
}
v_resetjp_733_:
{
lean_object* v_v_736_; lean_object* v___x_737_; lean_object* v_xs_x27_738_; lean_object* v___y_740_; 
v_v_736_ = lean_array_fget(v_es_727_, v_j_730_);
v___x_737_ = lean_box(0);
v_xs_x27_738_ = lean_array_fset(v_es_727_, v_j_730_, v___x_737_);
switch(lean_obj_tag(v_v_736_))
{
case 0:
{
lean_object* v_key_745_; lean_object* v_val_746_; lean_object* v___x_748_; uint8_t v_isShared_749_; uint8_t v_isSharedCheck_756_; 
v_key_745_ = lean_ctor_get(v_v_736_, 0);
v_val_746_ = lean_ctor_get(v_v_736_, 1);
v_isSharedCheck_756_ = !lean_is_exclusive(v_v_736_);
if (v_isSharedCheck_756_ == 0)
{
v___x_748_ = v_v_736_;
v_isShared_749_ = v_isSharedCheck_756_;
goto v_resetjp_747_;
}
else
{
lean_inc(v_val_746_);
lean_inc(v_key_745_);
lean_dec(v_v_736_);
v___x_748_ = lean_box(0);
v_isShared_749_ = v_isSharedCheck_756_;
goto v_resetjp_747_;
}
v_resetjp_747_:
{
uint8_t v___x_750_; 
v___x_750_ = l_Lean_Meta_DiscrTree_instBEqKey_beq(v_x_725_, v_key_745_);
if (v___x_750_ == 0)
{
lean_object* v___x_751_; lean_object* v___x_752_; 
lean_del_object(v___x_748_);
v___x_751_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_745_, v_val_746_, v_x_725_, v_x_726_);
v___x_752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_752_, 0, v___x_751_);
v___y_740_ = v___x_752_;
goto v___jp_739_;
}
else
{
lean_object* v___x_754_; 
lean_dec(v_val_746_);
lean_dec(v_key_745_);
if (v_isShared_749_ == 0)
{
lean_ctor_set(v___x_748_, 1, v_x_726_);
lean_ctor_set(v___x_748_, 0, v_x_725_);
v___x_754_ = v___x_748_;
goto v_reusejp_753_;
}
else
{
lean_object* v_reuseFailAlloc_755_; 
v_reuseFailAlloc_755_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_755_, 0, v_x_725_);
lean_ctor_set(v_reuseFailAlloc_755_, 1, v_x_726_);
v___x_754_ = v_reuseFailAlloc_755_;
goto v_reusejp_753_;
}
v_reusejp_753_:
{
v___y_740_ = v___x_754_;
goto v___jp_739_;
}
}
}
}
case 1:
{
lean_object* v_node_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_769_; 
v_node_757_ = lean_ctor_get(v_v_736_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v_v_736_);
if (v_isSharedCheck_769_ == 0)
{
v___x_759_ = v_v_736_;
v_isShared_760_ = v_isSharedCheck_769_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_node_757_);
lean_dec(v_v_736_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_769_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
size_t v___x_761_; size_t v___x_762_; size_t v___x_763_; size_t v___x_764_; lean_object* v___x_765_; lean_object* v___x_767_; 
v___x_761_ = ((size_t)5ULL);
v___x_762_ = lean_usize_shift_right(v_x_723_, v___x_761_);
v___x_763_ = ((size_t)1ULL);
v___x_764_ = lean_usize_add(v_x_724_, v___x_763_);
v___x_765_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(v_node_757_, v___x_762_, v___x_764_, v_x_725_, v_x_726_);
if (v_isShared_760_ == 0)
{
lean_ctor_set(v___x_759_, 0, v___x_765_);
v___x_767_ = v___x_759_;
goto v_reusejp_766_;
}
else
{
lean_object* v_reuseFailAlloc_768_; 
v_reuseFailAlloc_768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_768_, 0, v___x_765_);
v___x_767_ = v_reuseFailAlloc_768_;
goto v_reusejp_766_;
}
v_reusejp_766_:
{
v___y_740_ = v___x_767_;
goto v___jp_739_;
}
}
}
default: 
{
lean_object* v___x_770_; 
v___x_770_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_770_, 0, v_x_725_);
lean_ctor_set(v___x_770_, 1, v_x_726_);
v___y_740_ = v___x_770_;
goto v___jp_739_;
}
}
v___jp_739_:
{
lean_object* v___x_741_; lean_object* v___x_743_; 
v___x_741_ = lean_array_fset(v_xs_x27_738_, v_j_730_, v___y_740_);
lean_dec(v_j_730_);
if (v_isShared_735_ == 0)
{
lean_ctor_set(v___x_734_, 0, v___x_741_);
v___x_743_ = v___x_734_;
goto v_reusejp_742_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v___x_741_);
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
}
else
{
lean_object* v_ks_773_; lean_object* v_vs_774_; lean_object* v___x_776_; uint8_t v_isShared_777_; uint8_t v_isSharedCheck_794_; 
v_ks_773_ = lean_ctor_get(v_x_722_, 0);
v_vs_774_ = lean_ctor_get(v_x_722_, 1);
v_isSharedCheck_794_ = !lean_is_exclusive(v_x_722_);
if (v_isSharedCheck_794_ == 0)
{
v___x_776_ = v_x_722_;
v_isShared_777_ = v_isSharedCheck_794_;
goto v_resetjp_775_;
}
else
{
lean_inc(v_vs_774_);
lean_inc(v_ks_773_);
lean_dec(v_x_722_);
v___x_776_ = lean_box(0);
v_isShared_777_ = v_isSharedCheck_794_;
goto v_resetjp_775_;
}
v_resetjp_775_:
{
lean_object* v___x_779_; 
if (v_isShared_777_ == 0)
{
v___x_779_ = v___x_776_;
goto v_reusejp_778_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v_ks_773_);
lean_ctor_set(v_reuseFailAlloc_793_, 1, v_vs_774_);
v___x_779_ = v_reuseFailAlloc_793_;
goto v_reusejp_778_;
}
v_reusejp_778_:
{
lean_object* v_newNode_780_; uint8_t v___y_782_; size_t v___x_788_; uint8_t v___x_789_; 
v_newNode_780_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4___redArg(v___x_779_, v_x_725_, v_x_726_);
v___x_788_ = ((size_t)7ULL);
v___x_789_ = lean_usize_dec_le(v___x_788_, v_x_724_);
if (v___x_789_ == 0)
{
lean_object* v___x_790_; lean_object* v___x_791_; uint8_t v___x_792_; 
v___x_790_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_780_);
v___x_791_ = lean_unsigned_to_nat(4u);
v___x_792_ = lean_nat_dec_lt(v___x_790_, v___x_791_);
lean_dec(v___x_790_);
v___y_782_ = v___x_792_;
goto v___jp_781_;
}
else
{
v___y_782_ = v___x_789_;
goto v___jp_781_;
}
v___jp_781_:
{
if (v___y_782_ == 0)
{
lean_object* v_ks_783_; lean_object* v_vs_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v___x_787_; 
v_ks_783_ = lean_ctor_get(v_newNode_780_, 0);
lean_inc_ref(v_ks_783_);
v_vs_784_ = lean_ctor_get(v_newNode_780_, 1);
lean_inc_ref(v_vs_784_);
lean_dec_ref(v_newNode_780_);
v___x_785_ = lean_unsigned_to_nat(0u);
v___x_786_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___closed__0);
v___x_787_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg(v_x_724_, v_ks_783_, v_vs_784_, v___x_785_, v___x_786_);
lean_dec_ref(v_vs_784_);
lean_dec_ref(v_ks_783_);
return v___x_787_;
}
else
{
return v_newNode_780_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg(size_t v_depth_795_, lean_object* v_keys_796_, lean_object* v_vals_797_, lean_object* v_i_798_, lean_object* v_entries_799_){
_start:
{
lean_object* v___x_800_; uint8_t v___x_801_; 
v___x_800_ = lean_array_get_size(v_keys_796_);
v___x_801_ = lean_nat_dec_lt(v_i_798_, v___x_800_);
if (v___x_801_ == 0)
{
lean_dec(v_i_798_);
return v_entries_799_;
}
else
{
lean_object* v_k_802_; lean_object* v_v_803_; uint64_t v___x_804_; size_t v_h_805_; size_t v___x_806_; lean_object* v___x_807_; size_t v___x_808_; size_t v___x_809_; size_t v___x_810_; size_t v_h_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v_k_802_ = lean_array_fget_borrowed(v_keys_796_, v_i_798_);
v_v_803_ = lean_array_fget_borrowed(v_vals_797_, v_i_798_);
v___x_804_ = l_Lean_Meta_DiscrTree_Key_hash(v_k_802_);
v_h_805_ = lean_uint64_to_usize(v___x_804_);
v___x_806_ = ((size_t)5ULL);
v___x_807_ = lean_unsigned_to_nat(1u);
v___x_808_ = ((size_t)1ULL);
v___x_809_ = lean_usize_sub(v_depth_795_, v___x_808_);
v___x_810_ = lean_usize_mul(v___x_806_, v___x_809_);
v_h_811_ = lean_usize_shift_right(v_h_805_, v___x_810_);
v___x_812_ = lean_nat_add(v_i_798_, v___x_807_);
lean_dec(v_i_798_);
lean_inc(v_v_803_);
lean_inc(v_k_802_);
v___x_813_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(v_entries_799_, v_h_811_, v_depth_795_, v_k_802_, v_v_803_);
v_i_798_ = v___x_812_;
v_entries_799_ = v___x_813_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg___boxed(lean_object* v_depth_815_, lean_object* v_keys_816_, lean_object* v_vals_817_, lean_object* v_i_818_, lean_object* v_entries_819_){
_start:
{
size_t v_depth_boxed_820_; lean_object* v_res_821_; 
v_depth_boxed_820_ = lean_unbox_usize(v_depth_815_);
lean_dec(v_depth_815_);
v_res_821_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg(v_depth_boxed_820_, v_keys_816_, v_vals_817_, v_i_818_, v_entries_819_);
lean_dec_ref(v_vals_817_);
lean_dec_ref(v_keys_816_);
return v_res_821_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg___boxed(lean_object* v_x_822_, lean_object* v_x_823_, lean_object* v_x_824_, lean_object* v_x_825_, lean_object* v_x_826_){
_start:
{
size_t v_x_3099__boxed_827_; size_t v_x_3100__boxed_828_; lean_object* v_res_829_; 
v_x_3099__boxed_827_ = lean_unbox_usize(v_x_823_);
lean_dec(v_x_823_);
v_x_3100__boxed_828_ = lean_unbox_usize(v_x_824_);
lean_dec(v_x_824_);
v_res_829_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(v_x_822_, v_x_3099__boxed_827_, v_x_3100__boxed_828_, v_x_825_, v_x_826_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1___redArg(lean_object* v_x_830_, lean_object* v_x_831_, lean_object* v_x_832_){
_start:
{
uint64_t v___x_833_; size_t v___x_834_; size_t v___x_835_; lean_object* v___x_836_; 
v___x_833_ = l_Lean_Meta_DiscrTree_Key_hash(v_x_831_);
v___x_834_ = lean_uint64_to_usize(v___x_833_);
v___x_835_ = ((size_t)1ULL);
v___x_836_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(v_x_830_, v___x_834_, v___x_835_, v_x_831_, v_x_832_);
return v___x_836_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge___redArg___lam__0(lean_object* v_map_837_, lean_object* v_k_838_, lean_object* v_v_u2082_839_){
_start:
{
lean_object* v___x_840_; 
v___x_840_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg(v_map_837_, v_k_838_);
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v___x_841_; 
v___x_841_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1___redArg(v_map_837_, v_k_838_, v_v_u2082_839_);
return v___x_841_;
}
else
{
lean_object* v_val_842_; lean_object* v___x_843_; lean_object* v___x_844_; 
v_val_842_ = lean_ctor_get(v___x_840_, 0);
lean_inc(v_val_842_);
lean_dec_ref_known(v___x_840_, 1);
v___x_843_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(v_val_842_, v_v_u2082_839_);
v___x_844_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1___redArg(v_map_837_, v_k_838_, v___x_843_);
return v___x_844_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg(lean_object* v_f_845_, lean_object* v_keys_846_, lean_object* v_vals_847_, lean_object* v_i_848_, lean_object* v_acc_849_){
_start:
{
lean_object* v___x_850_; uint8_t v___x_851_; 
v___x_850_ = lean_array_get_size(v_keys_846_);
v___x_851_ = lean_nat_dec_lt(v_i_848_, v___x_850_);
if (v___x_851_ == 0)
{
lean_dec(v_i_848_);
lean_dec(v_f_845_);
return v_acc_849_;
}
else
{
lean_object* v_k_852_; lean_object* v_v_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; 
v_k_852_ = lean_array_fget_borrowed(v_keys_846_, v_i_848_);
v_v_853_ = lean_array_fget_borrowed(v_vals_847_, v_i_848_);
lean_inc(v_f_845_);
lean_inc(v_v_853_);
lean_inc(v_k_852_);
v___x_854_ = lean_apply_3(v_f_845_, v_acc_849_, v_k_852_, v_v_853_);
v___x_855_ = lean_unsigned_to_nat(1u);
v___x_856_ = lean_nat_add(v_i_848_, v___x_855_);
lean_dec(v_i_848_);
v_i_848_ = v___x_856_;
v_acc_849_ = v___x_854_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg___boxed(lean_object* v_f_858_, lean_object* v_keys_859_, lean_object* v_vals_860_, lean_object* v_i_861_, lean_object* v_acc_862_){
_start:
{
lean_object* v_res_863_; 
v_res_863_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg(v_f_858_, v_keys_859_, v_vals_860_, v_i_861_, v_acc_862_);
lean_dec_ref(v_vals_860_);
lean_dec_ref(v_keys_859_);
return v_res_863_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(lean_object* v_f_864_, lean_object* v_x_865_, lean_object* v_x_866_){
_start:
{
if (lean_obj_tag(v_x_865_) == 0)
{
lean_object* v_es_867_; lean_object* v___x_868_; lean_object* v___x_869_; uint8_t v___x_870_; 
v_es_867_ = lean_ctor_get(v_x_865_, 0);
v___x_868_ = lean_unsigned_to_nat(0u);
v___x_869_ = lean_array_get_size(v_es_867_);
v___x_870_ = lean_nat_dec_lt(v___x_868_, v___x_869_);
if (v___x_870_ == 0)
{
lean_dec(v_f_864_);
return v_x_866_;
}
else
{
uint8_t v___x_871_; 
v___x_871_ = lean_nat_dec_le(v___x_869_, v___x_869_);
if (v___x_871_ == 0)
{
if (v___x_870_ == 0)
{
lean_dec(v_f_864_);
return v_x_866_;
}
else
{
size_t v___x_872_; size_t v___x_873_; lean_object* v___x_874_; 
v___x_872_ = ((size_t)0ULL);
v___x_873_ = lean_usize_of_nat(v___x_869_);
v___x_874_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(v_f_864_, v_es_867_, v___x_872_, v___x_873_, v_x_866_);
return v___x_874_;
}
}
else
{
size_t v___x_875_; size_t v___x_876_; lean_object* v___x_877_; 
v___x_875_ = ((size_t)0ULL);
v___x_876_ = lean_usize_of_nat(v___x_869_);
v___x_877_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(v_f_864_, v_es_867_, v___x_875_, v___x_876_, v_x_866_);
return v___x_877_;
}
}
}
else
{
lean_object* v_ks_878_; lean_object* v_vs_879_; lean_object* v___x_880_; lean_object* v___x_881_; 
v_ks_878_ = lean_ctor_get(v_x_865_, 0);
v_vs_879_ = lean_ctor_get(v_x_865_, 1);
v___x_880_ = lean_unsigned_to_nat(0u);
v___x_881_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg(v_f_864_, v_ks_878_, v_vs_879_, v___x_880_, v_x_866_);
return v___x_881_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(lean_object* v_f_882_, lean_object* v_as_883_, size_t v_i_884_, size_t v_stop_885_, lean_object* v_b_886_){
_start:
{
lean_object* v___y_888_; uint8_t v___x_892_; 
v___x_892_ = lean_usize_dec_eq(v_i_884_, v_stop_885_);
if (v___x_892_ == 0)
{
lean_object* v___x_893_; 
v___x_893_ = lean_array_uget_borrowed(v_as_883_, v_i_884_);
switch(lean_obj_tag(v___x_893_))
{
case 0:
{
lean_object* v_key_894_; lean_object* v_val_895_; lean_object* v___x_896_; 
v_key_894_ = lean_ctor_get(v___x_893_, 0);
v_val_895_ = lean_ctor_get(v___x_893_, 1);
lean_inc(v_f_882_);
lean_inc(v_val_895_);
lean_inc(v_key_894_);
v___x_896_ = lean_apply_3(v_f_882_, v_b_886_, v_key_894_, v_val_895_);
v___y_888_ = v___x_896_;
goto v___jp_887_;
}
case 1:
{
lean_object* v_node_897_; lean_object* v___x_898_; 
v_node_897_ = lean_ctor_get(v___x_893_, 0);
lean_inc(v_f_882_);
v___x_898_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v_f_882_, v_node_897_, v_b_886_);
v___y_888_ = v___x_898_;
goto v___jp_887_;
}
default: 
{
v___y_888_ = v_b_886_;
goto v___jp_887_;
}
}
}
else
{
lean_dec(v_f_882_);
return v_b_886_;
}
v___jp_887_:
{
size_t v___x_889_; size_t v___x_890_; 
v___x_889_ = ((size_t)1ULL);
v___x_890_ = lean_usize_add(v_i_884_, v___x_889_);
v_i_884_ = v___x_890_;
v_b_886_ = v___y_888_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg___boxed(lean_object* v_f_899_, lean_object* v_as_900_, lean_object* v_i_901_, lean_object* v_stop_902_, lean_object* v_b_903_){
_start:
{
size_t v_i_boxed_904_; size_t v_stop_boxed_905_; lean_object* v_res_906_; 
v_i_boxed_904_ = lean_unbox_usize(v_i_901_);
lean_dec(v_i_901_);
v_stop_boxed_905_ = lean_unbox_usize(v_stop_902_);
lean_dec(v_stop_902_);
v_res_906_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(v_f_899_, v_as_900_, v_i_boxed_904_, v_stop_boxed_905_, v_b_903_);
lean_dec_ref(v_as_900_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg___boxed(lean_object* v_f_907_, lean_object* v_x_908_, lean_object* v_x_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v_f_907_, v_x_908_, v_x_909_);
lean_dec_ref(v_x_908_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg___lam__0(lean_object* v_f_911_, lean_object* v_x1_912_, lean_object* v_x2_913_, lean_object* v_x3_914_){
_start:
{
lean_object* v___x_915_; 
v___x_915_ = lean_apply_3(v_f_911_, v_x1_912_, v_x2_913_, v_x3_914_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(lean_object* v_map_916_, lean_object* v_f_917_, lean_object* v_init_918_){
_start:
{
lean_object* v___f_919_; lean_object* v___x_920_; 
v___f_919_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg___lam__0), 4, 1);
lean_closure_set(v___f_919_, 0, v_f_917_);
v___x_920_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v___f_919_, v_map_916_, v_init_918_);
return v___x_920_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg___boxed(lean_object* v_map_921_, lean_object* v_f_922_, lean_object* v_init_923_){
_start:
{
lean_object* v_res_924_; 
v_res_924_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(v_map_921_, v_f_922_, v_init_923_);
lean_dec_ref(v_map_921_);
return v_res_924_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17___redArg(lean_object* v_x_925_, lean_object* v_x_926_, lean_object* v_x_927_, lean_object* v_x_928_){
_start:
{
lean_object* v_ks_929_; lean_object* v_vs_930_; lean_object* v___x_932_; uint8_t v_isShared_933_; uint8_t v_isSharedCheck_971_; 
v_ks_929_ = lean_ctor_get(v_x_925_, 0);
v_vs_930_ = lean_ctor_get(v_x_925_, 1);
v_isSharedCheck_971_ = !lean_is_exclusive(v_x_925_);
if (v_isSharedCheck_971_ == 0)
{
v___x_932_ = v_x_925_;
v_isShared_933_ = v_isSharedCheck_971_;
goto v_resetjp_931_;
}
else
{
lean_inc(v_vs_930_);
lean_inc(v_ks_929_);
lean_dec(v_x_925_);
v___x_932_ = lean_box(0);
v_isShared_933_ = v_isSharedCheck_971_;
goto v_resetjp_931_;
}
v_resetjp_931_:
{
uint8_t v___y_942_; lean_object* v___x_946_; uint8_t v___x_947_; 
v___x_946_ = lean_array_get_size(v_ks_929_);
v___x_947_ = lean_nat_dec_lt(v_x_926_, v___x_946_);
if (v___x_947_ == 0)
{
lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; 
lean_del_object(v___x_932_);
lean_dec(v_x_926_);
v___x_948_ = lean_array_push(v_ks_929_, v_x_927_);
v___x_949_ = lean_array_push(v_vs_930_, v_x_928_);
v___x_950_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_950_, 0, v___x_948_);
lean_ctor_set(v___x_950_, 1, v___x_949_);
return v___x_950_;
}
else
{
lean_object* v_name_951_; lean_object* v_k_x27_952_; lean_object* v_name_953_; lean_object* v_name_954_; uint8_t v_builder_955_; uint8_t v_phase_956_; uint8_t v_scope_957_; uint64_t v_hash_958_; lean_object* v_name_959_; uint8_t v_builder_960_; uint8_t v_phase_961_; uint8_t v_scope_962_; uint64_t v_hash_963_; uint8_t v___y_965_; uint8_t v___x_969_; 
v_name_951_ = lean_ctor_get(v_x_927_, 0);
v_k_x27_952_ = lean_array_fget_borrowed(v_ks_929_, v_x_926_);
v_name_953_ = lean_ctor_get(v_k_x27_952_, 0);
v_name_954_ = lean_ctor_get(v_name_951_, 0);
v_builder_955_ = lean_ctor_get_uint8(v_name_951_, sizeof(void*)*1 + 8);
v_phase_956_ = lean_ctor_get_uint8(v_name_951_, sizeof(void*)*1 + 9);
v_scope_957_ = lean_ctor_get_uint8(v_name_951_, sizeof(void*)*1 + 10);
v_hash_958_ = lean_ctor_get_uint64(v_name_951_, sizeof(void*)*1);
v_name_959_ = lean_ctor_get(v_name_953_, 0);
v_builder_960_ = lean_ctor_get_uint8(v_name_953_, sizeof(void*)*1 + 8);
v_phase_961_ = lean_ctor_get_uint8(v_name_953_, sizeof(void*)*1 + 9);
v_scope_962_ = lean_ctor_get_uint8(v_name_953_, sizeof(void*)*1 + 10);
v_hash_963_ = lean_ctor_get_uint64(v_name_953_, sizeof(void*)*1);
v___x_969_ = lean_uint64_dec_eq(v_hash_958_, v_hash_963_);
if (v___x_969_ == 0)
{
v___y_965_ = v___x_969_;
goto v___jp_964_;
}
else
{
uint8_t v___x_970_; 
v___x_970_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_955_, v_builder_960_);
v___y_965_ = v___x_970_;
goto v___jp_964_;
}
v___jp_964_:
{
if (v___y_965_ == 0)
{
goto v___jp_934_;
}
else
{
uint8_t v___x_966_; 
v___x_966_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_956_, v_phase_961_);
if (v___x_966_ == 0)
{
v___y_942_ = v___x_966_;
goto v___jp_941_;
}
else
{
uint8_t v___x_967_; 
v___x_967_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_957_, v_scope_962_);
if (v___x_967_ == 0)
{
v___y_942_ = v___x_967_;
goto v___jp_941_;
}
else
{
uint8_t v___x_968_; 
v___x_968_ = lean_name_eq(v_name_954_, v_name_959_);
v___y_942_ = v___x_968_;
goto v___jp_941_;
}
}
}
}
}
v___jp_934_:
{
lean_object* v___x_936_; 
if (v_isShared_933_ == 0)
{
v___x_936_ = v___x_932_;
goto v_reusejp_935_;
}
else
{
lean_object* v_reuseFailAlloc_940_; 
v_reuseFailAlloc_940_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_940_, 0, v_ks_929_);
lean_ctor_set(v_reuseFailAlloc_940_, 1, v_vs_930_);
v___x_936_ = v_reuseFailAlloc_940_;
goto v_reusejp_935_;
}
v_reusejp_935_:
{
lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_937_ = lean_unsigned_to_nat(1u);
v___x_938_ = lean_nat_add(v_x_926_, v___x_937_);
lean_dec(v_x_926_);
v_x_925_ = v___x_936_;
v_x_926_ = v___x_938_;
goto _start;
}
}
v___jp_941_:
{
if (v___y_942_ == 0)
{
goto v___jp_934_;
}
else
{
lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; 
lean_del_object(v___x_932_);
v___x_943_ = lean_array_fset(v_ks_929_, v_x_926_, v_x_927_);
v___x_944_ = lean_array_fset(v_vs_930_, v_x_926_, v_x_928_);
lean_dec(v_x_926_);
v___x_945_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_945_, 0, v___x_943_);
lean_ctor_set(v___x_945_, 1, v___x_944_);
return v___x_945_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15___redArg(lean_object* v_n_972_, lean_object* v_k_973_, lean_object* v_v_974_){
_start:
{
lean_object* v___x_975_; lean_object* v___x_976_; 
v___x_975_ = lean_unsigned_to_nat(0u);
v___x_976_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17___redArg(v_n_972_, v___x_975_, v_k_973_, v_v_974_);
return v___x_976_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_977_; 
v___x_977_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_977_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(lean_object* v_x_978_, size_t v_x_979_, size_t v_x_980_, lean_object* v_x_981_, lean_object* v_x_982_){
_start:
{
if (lean_obj_tag(v_x_978_) == 0)
{
lean_object* v_es_983_; size_t v___x_984_; size_t v___x_985_; lean_object* v_j_986_; lean_object* v___x_987_; uint8_t v___x_988_; 
v_es_983_ = lean_ctor_get(v_x_978_, 0);
v___x_984_ = ((size_t)31ULL);
v___x_985_ = lean_usize_land(v_x_979_, v___x_984_);
v_j_986_ = lean_usize_to_nat(v___x_985_);
v___x_987_ = lean_array_get_size(v_es_983_);
v___x_988_ = lean_nat_dec_lt(v_j_986_, v___x_987_);
if (v___x_988_ == 0)
{
lean_dec(v_j_986_);
lean_dec(v_x_982_);
lean_dec_ref(v_x_981_);
return v_x_978_;
}
else
{
lean_object* v___x_990_; uint8_t v_isShared_991_; uint8_t v_isSharedCheck_1048_; 
lean_inc_ref(v_es_983_);
v_isSharedCheck_1048_ = !lean_is_exclusive(v_x_978_);
if (v_isSharedCheck_1048_ == 0)
{
lean_object* v_unused_1049_; 
v_unused_1049_ = lean_ctor_get(v_x_978_, 0);
lean_dec(v_unused_1049_);
v___x_990_ = v_x_978_;
v_isShared_991_ = v_isSharedCheck_1048_;
goto v_resetjp_989_;
}
else
{
lean_dec(v_x_978_);
v___x_990_ = lean_box(0);
v_isShared_991_ = v_isSharedCheck_1048_;
goto v_resetjp_989_;
}
v_resetjp_989_:
{
lean_object* v_v_992_; lean_object* v___x_993_; lean_object* v_xs_x27_994_; lean_object* v___y_996_; 
v_v_992_ = lean_array_fget(v_es_983_, v_j_986_);
v___x_993_ = lean_box(0);
v_xs_x27_994_ = lean_array_fset(v_es_983_, v_j_986_, v___x_993_);
switch(lean_obj_tag(v_v_992_))
{
case 0:
{
lean_object* v_key_1001_; lean_object* v_val_1002_; lean_object* v___x_1004_; uint8_t v_isShared_1005_; uint8_t v_isSharedCheck_1033_; 
v_key_1001_ = lean_ctor_get(v_v_992_, 0);
v_val_1002_ = lean_ctor_get(v_v_992_, 1);
v_isSharedCheck_1033_ = !lean_is_exclusive(v_v_992_);
if (v_isSharedCheck_1033_ == 0)
{
v___x_1004_ = v_v_992_;
v_isShared_1005_ = v_isSharedCheck_1033_;
goto v_resetjp_1003_;
}
else
{
lean_inc(v_val_1002_);
lean_inc(v_key_1001_);
lean_dec(v_v_992_);
v___x_1004_ = lean_box(0);
v_isShared_1005_ = v_isSharedCheck_1033_;
goto v_resetjp_1003_;
}
v_resetjp_1003_:
{
uint8_t v___y_1010_; lean_object* v_name_1014_; lean_object* v_name_1015_; lean_object* v_name_1016_; uint8_t v_builder_1017_; uint8_t v_phase_1018_; uint8_t v_scope_1019_; uint64_t v_hash_1020_; lean_object* v_name_1021_; uint8_t v_builder_1022_; uint8_t v_phase_1023_; uint8_t v_scope_1024_; uint64_t v_hash_1025_; uint8_t v___y_1027_; uint8_t v___x_1031_; 
v_name_1014_ = lean_ctor_get(v_x_981_, 0);
v_name_1015_ = lean_ctor_get(v_key_1001_, 0);
v_name_1016_ = lean_ctor_get(v_name_1014_, 0);
v_builder_1017_ = lean_ctor_get_uint8(v_name_1014_, sizeof(void*)*1 + 8);
v_phase_1018_ = lean_ctor_get_uint8(v_name_1014_, sizeof(void*)*1 + 9);
v_scope_1019_ = lean_ctor_get_uint8(v_name_1014_, sizeof(void*)*1 + 10);
v_hash_1020_ = lean_ctor_get_uint64(v_name_1014_, sizeof(void*)*1);
v_name_1021_ = lean_ctor_get(v_name_1015_, 0);
v_builder_1022_ = lean_ctor_get_uint8(v_name_1015_, sizeof(void*)*1 + 8);
v_phase_1023_ = lean_ctor_get_uint8(v_name_1015_, sizeof(void*)*1 + 9);
v_scope_1024_ = lean_ctor_get_uint8(v_name_1015_, sizeof(void*)*1 + 10);
v_hash_1025_ = lean_ctor_get_uint64(v_name_1015_, sizeof(void*)*1);
v___x_1031_ = lean_uint64_dec_eq(v_hash_1020_, v_hash_1025_);
if (v___x_1031_ == 0)
{
v___y_1027_ = v___x_1031_;
goto v___jp_1026_;
}
else
{
uint8_t v___x_1032_; 
v___x_1032_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_1017_, v_builder_1022_);
v___y_1027_ = v___x_1032_;
goto v___jp_1026_;
}
v___jp_1006_:
{
lean_object* v___x_1007_; lean_object* v___x_1008_; 
v___x_1007_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1001_, v_val_1002_, v_x_981_, v_x_982_);
v___x_1008_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1008_, 0, v___x_1007_);
v___y_996_ = v___x_1008_;
goto v___jp_995_;
}
v___jp_1009_:
{
if (v___y_1010_ == 0)
{
lean_del_object(v___x_1004_);
goto v___jp_1006_;
}
else
{
lean_object* v___x_1012_; 
lean_dec(v_val_1002_);
lean_dec(v_key_1001_);
if (v_isShared_1005_ == 0)
{
lean_ctor_set(v___x_1004_, 1, v_x_982_);
lean_ctor_set(v___x_1004_, 0, v_x_981_);
v___x_1012_ = v___x_1004_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1013_; 
v_reuseFailAlloc_1013_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1013_, 0, v_x_981_);
lean_ctor_set(v_reuseFailAlloc_1013_, 1, v_x_982_);
v___x_1012_ = v_reuseFailAlloc_1013_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
v___y_996_ = v___x_1012_;
goto v___jp_995_;
}
}
}
v___jp_1026_:
{
if (v___y_1027_ == 0)
{
lean_del_object(v___x_1004_);
goto v___jp_1006_;
}
else
{
uint8_t v___x_1028_; 
v___x_1028_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_1018_, v_phase_1023_);
if (v___x_1028_ == 0)
{
v___y_1010_ = v___x_1028_;
goto v___jp_1009_;
}
else
{
uint8_t v___x_1029_; 
v___x_1029_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_1019_, v_scope_1024_);
if (v___x_1029_ == 0)
{
v___y_1010_ = v___x_1029_;
goto v___jp_1009_;
}
else
{
uint8_t v___x_1030_; 
v___x_1030_ = lean_name_eq(v_name_1016_, v_name_1021_);
v___y_1010_ = v___x_1030_;
goto v___jp_1009_;
}
}
}
}
}
}
case 1:
{
lean_object* v_node_1034_; lean_object* v___x_1036_; uint8_t v_isShared_1037_; uint8_t v_isSharedCheck_1046_; 
v_node_1034_ = lean_ctor_get(v_v_992_, 0);
v_isSharedCheck_1046_ = !lean_is_exclusive(v_v_992_);
if (v_isSharedCheck_1046_ == 0)
{
v___x_1036_ = v_v_992_;
v_isShared_1037_ = v_isSharedCheck_1046_;
goto v_resetjp_1035_;
}
else
{
lean_inc(v_node_1034_);
lean_dec(v_v_992_);
v___x_1036_ = lean_box(0);
v_isShared_1037_ = v_isSharedCheck_1046_;
goto v_resetjp_1035_;
}
v_resetjp_1035_:
{
size_t v___x_1038_; size_t v___x_1039_; size_t v___x_1040_; size_t v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1044_; 
v___x_1038_ = ((size_t)5ULL);
v___x_1039_ = lean_usize_shift_right(v_x_979_, v___x_1038_);
v___x_1040_ = ((size_t)1ULL);
v___x_1041_ = lean_usize_add(v_x_980_, v___x_1040_);
v___x_1042_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(v_node_1034_, v___x_1039_, v___x_1041_, v_x_981_, v_x_982_);
if (v_isShared_1037_ == 0)
{
lean_ctor_set(v___x_1036_, 0, v___x_1042_);
v___x_1044_ = v___x_1036_;
goto v_reusejp_1043_;
}
else
{
lean_object* v_reuseFailAlloc_1045_; 
v_reuseFailAlloc_1045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1045_, 0, v___x_1042_);
v___x_1044_ = v_reuseFailAlloc_1045_;
goto v_reusejp_1043_;
}
v_reusejp_1043_:
{
v___y_996_ = v___x_1044_;
goto v___jp_995_;
}
}
}
default: 
{
lean_object* v___x_1047_; 
v___x_1047_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1047_, 0, v_x_981_);
lean_ctor_set(v___x_1047_, 1, v_x_982_);
v___y_996_ = v___x_1047_;
goto v___jp_995_;
}
}
v___jp_995_:
{
lean_object* v___x_997_; lean_object* v___x_999_; 
v___x_997_ = lean_array_fset(v_xs_x27_994_, v_j_986_, v___y_996_);
lean_dec(v_j_986_);
if (v_isShared_991_ == 0)
{
lean_ctor_set(v___x_990_, 0, v___x_997_);
v___x_999_ = v___x_990_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v___x_997_);
v___x_999_ = v_reuseFailAlloc_1000_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
return v___x_999_;
}
}
}
}
}
else
{
lean_object* v_ks_1050_; lean_object* v_vs_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1071_; 
v_ks_1050_ = lean_ctor_get(v_x_978_, 0);
v_vs_1051_ = lean_ctor_get(v_x_978_, 1);
v_isSharedCheck_1071_ = !lean_is_exclusive(v_x_978_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1053_ = v_x_978_;
v_isShared_1054_ = v_isSharedCheck_1071_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_vs_1051_);
lean_inc(v_ks_1050_);
lean_dec(v_x_978_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1071_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1056_; 
if (v_isShared_1054_ == 0)
{
v___x_1056_ = v___x_1053_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_ks_1050_);
lean_ctor_set(v_reuseFailAlloc_1070_, 1, v_vs_1051_);
v___x_1056_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
lean_object* v_newNode_1057_; uint8_t v___y_1059_; size_t v___x_1065_; uint8_t v___x_1066_; 
v_newNode_1057_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15___redArg(v___x_1056_, v_x_981_, v_x_982_);
v___x_1065_ = ((size_t)7ULL);
v___x_1066_ = lean_usize_dec_le(v___x_1065_, v_x_980_);
if (v___x_1066_ == 0)
{
lean_object* v___x_1067_; lean_object* v___x_1068_; uint8_t v___x_1069_; 
v___x_1067_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1057_);
v___x_1068_ = lean_unsigned_to_nat(4u);
v___x_1069_ = lean_nat_dec_lt(v___x_1067_, v___x_1068_);
lean_dec(v___x_1067_);
v___y_1059_ = v___x_1069_;
goto v___jp_1058_;
}
else
{
v___y_1059_ = v___x_1066_;
goto v___jp_1058_;
}
v___jp_1058_:
{
if (v___y_1059_ == 0)
{
lean_object* v_ks_1060_; lean_object* v_vs_1061_; lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; 
v_ks_1060_ = lean_ctor_get(v_newNode_1057_, 0);
lean_inc_ref(v_ks_1060_);
v_vs_1061_ = lean_ctor_get(v_newNode_1057_, 1);
lean_inc_ref(v_vs_1061_);
lean_dec_ref(v_newNode_1057_);
v___x_1062_ = lean_unsigned_to_nat(0u);
v___x_1063_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___closed__0);
v___x_1064_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg(v_x_980_, v_ks_1060_, v_vs_1061_, v___x_1062_, v___x_1063_);
lean_dec_ref(v_vs_1061_);
lean_dec_ref(v_ks_1060_);
return v___x_1064_;
}
else
{
return v_newNode_1057_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg(size_t v_depth_1072_, lean_object* v_keys_1073_, lean_object* v_vals_1074_, lean_object* v_i_1075_, lean_object* v_entries_1076_){
_start:
{
lean_object* v___x_1077_; uint8_t v___x_1078_; 
v___x_1077_ = lean_array_get_size(v_keys_1073_);
v___x_1078_ = lean_nat_dec_lt(v_i_1075_, v___x_1077_);
if (v___x_1078_ == 0)
{
lean_dec(v_i_1075_);
return v_entries_1076_;
}
else
{
lean_object* v_k_1079_; lean_object* v_name_1080_; uint64_t v_hash_1081_; lean_object* v_v_1082_; size_t v_h_1083_; size_t v___x_1084_; lean_object* v___x_1085_; size_t v___x_1086_; size_t v___x_1087_; size_t v___x_1088_; size_t v_h_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; 
v_k_1079_ = lean_array_fget_borrowed(v_keys_1073_, v_i_1075_);
v_name_1080_ = lean_ctor_get(v_k_1079_, 0);
v_hash_1081_ = lean_ctor_get_uint64(v_name_1080_, sizeof(void*)*1);
v_v_1082_ = lean_array_fget_borrowed(v_vals_1074_, v_i_1075_);
v_h_1083_ = lean_uint64_to_usize(v_hash_1081_);
v___x_1084_ = ((size_t)5ULL);
v___x_1085_ = lean_unsigned_to_nat(1u);
v___x_1086_ = ((size_t)1ULL);
v___x_1087_ = lean_usize_sub(v_depth_1072_, v___x_1086_);
v___x_1088_ = lean_usize_mul(v___x_1084_, v___x_1087_);
v_h_1089_ = lean_usize_shift_right(v_h_1083_, v___x_1088_);
v___x_1090_ = lean_nat_add(v_i_1075_, v___x_1085_);
lean_dec(v_i_1075_);
lean_inc(v_v_1082_);
lean_inc(v_k_1079_);
v___x_1091_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(v_entries_1076_, v_h_1089_, v_depth_1072_, v_k_1079_, v_v_1082_);
v_i_1075_ = v___x_1090_;
v_entries_1076_ = v___x_1091_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg___boxed(lean_object* v_depth_1093_, lean_object* v_keys_1094_, lean_object* v_vals_1095_, lean_object* v_i_1096_, lean_object* v_entries_1097_){
_start:
{
size_t v_depth_boxed_1098_; lean_object* v_res_1099_; 
v_depth_boxed_1098_ = lean_unbox_usize(v_depth_1093_);
lean_dec(v_depth_1093_);
v_res_1099_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg(v_depth_boxed_1098_, v_keys_1094_, v_vals_1095_, v_i_1096_, v_entries_1097_);
lean_dec_ref(v_vals_1095_);
lean_dec_ref(v_keys_1094_);
return v_res_1099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg___boxed(lean_object* v_x_1100_, lean_object* v_x_1101_, lean_object* v_x_1102_, lean_object* v_x_1103_, lean_object* v_x_1104_){
_start:
{
size_t v_x_3448__boxed_1105_; size_t v_x_3449__boxed_1106_; lean_object* v_res_1107_; 
v_x_3448__boxed_1105_ = lean_unbox_usize(v_x_1101_);
lean_dec(v_x_1101_);
v_x_3449__boxed_1106_ = lean_unbox_usize(v_x_1102_);
lean_dec(v_x_1102_);
v_res_1107_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(v_x_1100_, v_x_3448__boxed_1105_, v_x_3449__boxed_1106_, v_x_1103_, v_x_1104_);
return v_res_1107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6___redArg(lean_object* v_x_1108_, lean_object* v_x_1109_, lean_object* v_x_1110_){
_start:
{
lean_object* v_name_1111_; uint64_t v_hash_1112_; size_t v___x_1113_; size_t v___x_1114_; lean_object* v___x_1115_; 
v_name_1111_ = lean_ctor_get(v_x_1109_, 0);
v_hash_1112_ = lean_ctor_get_uint64(v_name_1111_, sizeof(void*)*1);
v___x_1113_ = lean_uint64_to_usize(v_hash_1112_);
v___x_1114_ = ((size_t)1ULL);
v___x_1115_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(v_x_1108_, v___x_1113_, v___x_1114_, v_x_1109_, v_x_1110_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___lam__0(lean_object* v___x_1116_, lean_object* v_x_1117_, lean_object* v_____s_1118_){
_start:
{
lean_object* v_fst_1119_; lean_object* v_snd_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1130_; 
v_fst_1119_ = lean_ctor_get(v_x_1117_, 0);
lean_inc(v_fst_1119_);
lean_dec_ref(v_x_1117_);
v_snd_1120_ = lean_ctor_get(v_____s_1118_, 1);
v_isSharedCheck_1130_ = !lean_is_exclusive(v_____s_1118_);
if (v_isSharedCheck_1130_ == 0)
{
lean_object* v_unused_1131_; 
v_unused_1131_ = lean_ctor_get(v_____s_1118_, 0);
lean_dec(v_unused_1131_);
v___x_1122_ = v_____s_1118_;
v_isShared_1123_ = v_isSharedCheck_1130_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_snd_1120_);
lean_dec(v_____s_1118_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1130_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
lean_object* v___x_1124_; lean_object* v_s_1125_; lean_object* v___x_1127_; 
v___x_1124_ = lean_box(0);
v_s_1125_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6___redArg(v_snd_1120_, v_fst_1119_, v___x_1124_);
if (v_isShared_1123_ == 0)
{
lean_ctor_set(v___x_1122_, 1, v_s_1125_);
lean_ctor_set(v___x_1122_, 0, v___x_1116_);
v___x_1127_ = v___x_1122_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1129_; 
v_reuseFailAlloc_1129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1129_, 0, v___x_1116_);
lean_ctor_set(v_reuseFailAlloc_1129_, 1, v_s_1125_);
v___x_1127_ = v_reuseFailAlloc_1129_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
lean_object* v___x_1128_; 
v___x_1128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1127_);
return v___x_1128_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg___lam__0(lean_object* v_f_1132_, lean_object* v_s_1133_, lean_object* v_a_1134_, lean_object* v_b_1135_){
_start:
{
lean_object* v___x_1136_; lean_object* v___x_1137_; 
v___x_1136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1136_, 0, v_a_1134_);
lean_ctor_set(v___x_1136_, 1, v_b_1135_);
v___x_1137_ = lean_apply_2(v_f_1132_, v___x_1136_, v_s_1133_);
if (lean_obj_tag(v___x_1137_) == 0)
{
lean_object* v_a_1138_; lean_object* v___x_1140_; uint8_t v_isShared_1141_; uint8_t v_isSharedCheck_1145_; 
v_a_1138_ = lean_ctor_get(v___x_1137_, 0);
v_isSharedCheck_1145_ = !lean_is_exclusive(v___x_1137_);
if (v_isSharedCheck_1145_ == 0)
{
v___x_1140_ = v___x_1137_;
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
else
{
lean_inc(v_a_1138_);
lean_dec(v___x_1137_);
v___x_1140_ = lean_box(0);
v_isShared_1141_ = v_isSharedCheck_1145_;
goto v_resetjp_1139_;
}
v_resetjp_1139_:
{
lean_object* v___x_1143_; 
if (v_isShared_1141_ == 0)
{
v___x_1143_ = v___x_1140_;
goto v_reusejp_1142_;
}
else
{
lean_object* v_reuseFailAlloc_1144_; 
v_reuseFailAlloc_1144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1144_, 0, v_a_1138_);
v___x_1143_ = v_reuseFailAlloc_1144_;
goto v_reusejp_1142_;
}
v_reusejp_1142_:
{
return v___x_1143_;
}
}
}
else
{
lean_object* v_a_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1153_; 
v_a_1146_ = lean_ctor_get(v___x_1137_, 0);
v_isSharedCheck_1153_ = !lean_is_exclusive(v___x_1137_);
if (v_isSharedCheck_1153_ == 0)
{
v___x_1148_ = v___x_1137_;
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_a_1146_);
lean_dec(v___x_1137_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1153_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
lean_object* v___x_1151_; 
if (v_isShared_1149_ == 0)
{
v___x_1151_ = v___x_1148_;
goto v_reusejp_1150_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v_a_1146_);
v___x_1151_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1150_;
}
v_reusejp_1150_:
{
return v___x_1151_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg(lean_object* v_f_1154_, lean_object* v_keys_1155_, lean_object* v_vals_1156_, lean_object* v_i_1157_, lean_object* v_acc_1158_){
_start:
{
lean_object* v___x_1159_; uint8_t v___x_1160_; 
v___x_1159_ = lean_array_get_size(v_keys_1155_);
v___x_1160_ = lean_nat_dec_lt(v_i_1157_, v___x_1159_);
if (v___x_1160_ == 0)
{
lean_object* v___x_1161_; 
lean_dec(v_i_1157_);
lean_dec_ref(v_f_1154_);
v___x_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1161_, 0, v_acc_1158_);
return v___x_1161_;
}
else
{
lean_object* v_k_1162_; lean_object* v_v_1163_; lean_object* v___x_1164_; 
v_k_1162_ = lean_array_fget_borrowed(v_keys_1155_, v_i_1157_);
v_v_1163_ = lean_array_fget_borrowed(v_vals_1156_, v_i_1157_);
lean_inc_ref(v_f_1154_);
lean_inc(v_v_1163_);
lean_inc(v_k_1162_);
v___x_1164_ = lean_apply_3(v_f_1154_, v_acc_1158_, v_k_1162_, v_v_1163_);
if (lean_obj_tag(v___x_1164_) == 0)
{
lean_dec(v_i_1157_);
lean_dec_ref(v_f_1154_);
return v___x_1164_;
}
else
{
lean_object* v_a_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; 
v_a_1165_ = lean_ctor_get(v___x_1164_, 0);
lean_inc(v_a_1165_);
lean_dec_ref_known(v___x_1164_, 1);
v___x_1166_ = lean_unsigned_to_nat(1u);
v___x_1167_ = lean_nat_add(v_i_1157_, v___x_1166_);
lean_dec(v_i_1157_);
v_i_1157_ = v___x_1167_;
v_acc_1158_ = v_a_1165_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg___boxed(lean_object* v_f_1169_, lean_object* v_keys_1170_, lean_object* v_vals_1171_, lean_object* v_i_1172_, lean_object* v_acc_1173_){
_start:
{
lean_object* v_res_1174_; 
v_res_1174_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg(v_f_1169_, v_keys_1170_, v_vals_1171_, v_i_1172_, v_acc_1173_);
lean_dec_ref(v_vals_1171_);
lean_dec_ref(v_keys_1170_);
return v_res_1174_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(lean_object* v_f_1175_, lean_object* v_x_1176_, lean_object* v_x_1177_){
_start:
{
if (lean_obj_tag(v_x_1176_) == 0)
{
lean_object* v_es_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1198_; 
v_es_1178_ = lean_ctor_get(v_x_1176_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v_x_1176_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1180_ = v_x_1176_;
v_isShared_1181_ = v_isSharedCheck_1198_;
goto v_resetjp_1179_;
}
else
{
lean_inc(v_es_1178_);
lean_dec(v_x_1176_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1198_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v___x_1182_; lean_object* v___x_1183_; uint8_t v___x_1184_; 
v___x_1182_ = lean_unsigned_to_nat(0u);
v___x_1183_ = lean_array_get_size(v_es_1178_);
v___x_1184_ = lean_nat_dec_lt(v___x_1182_, v___x_1183_);
if (v___x_1184_ == 0)
{
lean_object* v___x_1186_; 
lean_dec_ref(v_es_1178_);
lean_dec_ref(v_f_1175_);
if (v_isShared_1181_ == 0)
{
lean_ctor_set_tag(v___x_1180_, 1);
lean_ctor_set(v___x_1180_, 0, v_x_1177_);
v___x_1186_ = v___x_1180_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1187_; 
v_reuseFailAlloc_1187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1187_, 0, v_x_1177_);
v___x_1186_ = v_reuseFailAlloc_1187_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
return v___x_1186_;
}
}
else
{
uint8_t v___x_1188_; 
v___x_1188_ = lean_nat_dec_le(v___x_1183_, v___x_1183_);
if (v___x_1188_ == 0)
{
if (v___x_1184_ == 0)
{
lean_object* v___x_1190_; 
lean_dec_ref(v_es_1178_);
lean_dec_ref(v_f_1175_);
if (v_isShared_1181_ == 0)
{
lean_ctor_set_tag(v___x_1180_, 1);
lean_ctor_set(v___x_1180_, 0, v_x_1177_);
v___x_1190_ = v___x_1180_;
goto v_reusejp_1189_;
}
else
{
lean_object* v_reuseFailAlloc_1191_; 
v_reuseFailAlloc_1191_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1191_, 0, v_x_1177_);
v___x_1190_ = v_reuseFailAlloc_1191_;
goto v_reusejp_1189_;
}
v_reusejp_1189_:
{
return v___x_1190_;
}
}
else
{
size_t v___x_1192_; size_t v___x_1193_; lean_object* v___x_1194_; 
lean_del_object(v___x_1180_);
v___x_1192_ = ((size_t)0ULL);
v___x_1193_ = lean_usize_of_nat(v___x_1183_);
v___x_1194_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(v_f_1175_, v_es_1178_, v___x_1192_, v___x_1193_, v_x_1177_);
lean_dec_ref(v_es_1178_);
return v___x_1194_;
}
}
else
{
size_t v___x_1195_; size_t v___x_1196_; lean_object* v___x_1197_; 
lean_del_object(v___x_1180_);
v___x_1195_ = ((size_t)0ULL);
v___x_1196_ = lean_usize_of_nat(v___x_1183_);
v___x_1197_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(v_f_1175_, v_es_1178_, v___x_1195_, v___x_1196_, v_x_1177_);
lean_dec_ref(v_es_1178_);
return v___x_1197_;
}
}
}
}
else
{
lean_object* v_ks_1199_; lean_object* v_vs_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; 
v_ks_1199_ = lean_ctor_get(v_x_1176_, 0);
lean_inc_ref(v_ks_1199_);
v_vs_1200_ = lean_ctor_get(v_x_1176_, 1);
lean_inc_ref(v_vs_1200_);
lean_dec_ref_known(v_x_1176_, 2);
v___x_1201_ = lean_unsigned_to_nat(0u);
v___x_1202_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg(v_f_1175_, v_ks_1199_, v_vs_1200_, v___x_1201_, v_x_1177_);
lean_dec_ref(v_vs_1200_);
lean_dec_ref(v_ks_1199_);
return v___x_1202_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(lean_object* v_f_1203_, lean_object* v_as_1204_, size_t v_i_1205_, size_t v_stop_1206_, lean_object* v_b_1207_){
_start:
{
lean_object* v_a_1209_; lean_object* v___y_1214_; uint8_t v___x_1216_; 
v___x_1216_ = lean_usize_dec_eq(v_i_1205_, v_stop_1206_);
if (v___x_1216_ == 0)
{
lean_object* v___x_1217_; 
v___x_1217_ = lean_array_uget_borrowed(v_as_1204_, v_i_1205_);
switch(lean_obj_tag(v___x_1217_))
{
case 0:
{
lean_object* v_key_1218_; lean_object* v_val_1219_; lean_object* v___x_1220_; 
v_key_1218_ = lean_ctor_get(v___x_1217_, 0);
v_val_1219_ = lean_ctor_get(v___x_1217_, 1);
lean_inc_ref(v_f_1203_);
lean_inc(v_val_1219_);
lean_inc(v_key_1218_);
v___x_1220_ = lean_apply_3(v_f_1203_, v_b_1207_, v_key_1218_, v_val_1219_);
v___y_1214_ = v___x_1220_;
goto v___jp_1213_;
}
case 1:
{
lean_object* v_node_1221_; lean_object* v___x_1222_; 
v_node_1221_ = lean_ctor_get(v___x_1217_, 0);
lean_inc(v_node_1221_);
lean_inc_ref(v_f_1203_);
v___x_1222_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(v_f_1203_, v_node_1221_, v_b_1207_);
v___y_1214_ = v___x_1222_;
goto v___jp_1213_;
}
default: 
{
v_a_1209_ = v_b_1207_;
goto v___jp_1208_;
}
}
}
else
{
lean_object* v___x_1223_; 
lean_dec_ref(v_f_1203_);
v___x_1223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1223_, 0, v_b_1207_);
return v___x_1223_;
}
v___jp_1208_:
{
size_t v___x_1210_; size_t v___x_1211_; 
v___x_1210_ = ((size_t)1ULL);
v___x_1211_ = lean_usize_add(v_i_1205_, v___x_1210_);
v_i_1205_ = v___x_1211_;
v_b_1207_ = v_a_1209_;
goto _start;
}
v___jp_1213_:
{
if (lean_obj_tag(v___y_1214_) == 0)
{
lean_dec_ref(v_f_1203_);
return v___y_1214_;
}
else
{
lean_object* v_a_1215_; 
v_a_1215_ = lean_ctor_get(v___y_1214_, 0);
lean_inc(v_a_1215_);
lean_dec_ref_known(v___y_1214_, 1);
v_a_1209_ = v_a_1215_;
goto v___jp_1208_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg___boxed(lean_object* v_f_1224_, lean_object* v_as_1225_, lean_object* v_i_1226_, lean_object* v_stop_1227_, lean_object* v_b_1228_){
_start:
{
size_t v_i_boxed_1229_; size_t v_stop_boxed_1230_; lean_object* v_res_1231_; 
v_i_boxed_1229_ = lean_unbox_usize(v_i_1226_);
lean_dec(v_i_1226_);
v_stop_boxed_1230_ = lean_unbox_usize(v_stop_1227_);
lean_dec(v_stop_1227_);
v_res_1231_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(v_f_1224_, v_as_1225_, v_i_boxed_1229_, v_stop_boxed_1230_, v_b_1228_);
lean_dec_ref(v_as_1225_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg(lean_object* v_map_1232_, lean_object* v_init_1233_, lean_object* v_f_1234_){
_start:
{
lean_object* v___f_1235_; lean_object* v___x_1236_; lean_object* v_a_1237_; 
v___f_1235_ = lean_alloc_closure((void*)(lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1235_, 0, v_f_1234_);
lean_inc_ref(v_map_1232_);
v___x_1236_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(v___f_1235_, v_map_1232_, v_init_1233_);
v_a_1237_ = lean_ctor_get(v___x_1236_, 0);
lean_inc(v_a_1237_);
lean_dec_ref(v___x_1236_);
return v_a_1237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg___boxed(lean_object* v_map_1238_, lean_object* v_init_1239_, lean_object* v_f_1240_){
_start:
{
lean_object* v_res_1241_; 
v_res_1241_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg(v_map_1238_, v_init_1239_, v_f_1240_);
lean_dec_ref(v_map_1238_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg(lean_object* v_s_1244_, lean_object* v_as_1245_){
_start:
{
lean_object* v___x_1246_; lean_object* v___f_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v_fst_1250_; 
v___x_1246_ = lean_box(0);
v___f_1247_ = ((lean_object*)(lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___closed__0));
v___x_1248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1248_, 0, v___x_1246_);
lean_ctor_set(v___x_1248_, 1, v_s_1244_);
v___x_1249_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg(v_as_1245_, v___x_1248_, v___f_1247_);
v_fst_1250_ = lean_ctor_get(v___x_1249_, 0);
lean_inc(v_fst_1250_);
if (lean_obj_tag(v_fst_1250_) == 0)
{
lean_object* v_snd_1251_; 
v_snd_1251_ = lean_ctor_get(v___x_1249_, 1);
lean_inc(v_snd_1251_);
lean_dec(v___x_1249_);
return v_snd_1251_;
}
else
{
lean_object* v_val_1252_; 
lean_dec(v___x_1249_);
v_val_1252_ = lean_ctor_get(v_fst_1250_, 0);
lean_inc(v_val_1252_);
lean_dec_ref_known(v_fst_1250_, 1);
return v_val_1252_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg___boxed(lean_object* v_s_1253_, lean_object* v_as_1254_){
_start:
{
lean_object* v_res_1255_; 
v_res_1255_ = lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg(v_s_1253_, v_as_1254_);
lean_dec_ref(v_as_1254_);
return v_res_1255_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge___redArg(lean_object* v_ri_u2081_1257_, lean_object* v_ri_u2082_1258_){
_start:
{
lean_object* v_byTarget_1259_; lean_object* v_byHyp_1260_; lean_object* v_unindexed_1261_; lean_object* v_byTarget_1262_; lean_object* v_byHyp_1263_; lean_object* v_unindexed_1264_; lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1275_; 
v_byTarget_1259_ = lean_ctor_get(v_ri_u2081_1257_, 0);
lean_inc_ref(v_byTarget_1259_);
v_byHyp_1260_ = lean_ctor_get(v_ri_u2081_1257_, 1);
lean_inc_ref(v_byHyp_1260_);
v_unindexed_1261_ = lean_ctor_get(v_ri_u2081_1257_, 2);
lean_inc_ref(v_unindexed_1261_);
lean_dec_ref(v_ri_u2081_1257_);
v_byTarget_1262_ = lean_ctor_get(v_ri_u2082_1258_, 0);
v_byHyp_1263_ = lean_ctor_get(v_ri_u2082_1258_, 1);
v_unindexed_1264_ = lean_ctor_get(v_ri_u2082_1258_, 2);
v_isSharedCheck_1275_ = !lean_is_exclusive(v_ri_u2082_1258_);
if (v_isSharedCheck_1275_ == 0)
{
v___x_1266_ = v_ri_u2082_1258_;
v_isShared_1267_ = v_isSharedCheck_1275_;
goto v_resetjp_1265_;
}
else
{
lean_inc(v_unindexed_1264_);
lean_inc(v_byHyp_1263_);
lean_inc(v_byTarget_1262_);
lean_dec(v_ri_u2082_1258_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1275_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___f_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1273_; 
v___f_1268_ = ((lean_object*)(lp_aesop_Aesop_Index_merge___redArg___closed__0));
v___x_1269_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(v_byTarget_1262_, v___f_1268_, v_byTarget_1259_);
lean_dec_ref(v_byTarget_1262_);
v___x_1270_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(v_byHyp_1263_, v___f_1268_, v_byHyp_1260_);
lean_dec_ref(v_byHyp_1263_);
v___x_1271_ = lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg(v_unindexed_1261_, v_unindexed_1264_);
lean_dec_ref(v_unindexed_1264_);
if (v_isShared_1267_ == 0)
{
lean_ctor_set(v___x_1266_, 2, v___x_1271_);
lean_ctor_set(v___x_1266_, 1, v___x_1270_);
lean_ctor_set(v___x_1266_, 0, v___x_1269_);
v___x_1273_ = v___x_1266_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v___x_1269_);
lean_ctor_set(v_reuseFailAlloc_1274_, 1, v___x_1270_);
lean_ctor_set(v_reuseFailAlloc_1274_, 2, v___x_1271_);
v___x_1273_ = v_reuseFailAlloc_1274_;
goto v_reusejp_1272_;
}
v_reusejp_1272_:
{
return v___x_1273_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_merge(lean_object* v_00_u03b1_1276_, lean_object* v_ri_u2081_1277_, lean_object* v_ri_u2082_1278_){
_start:
{
lean_object* v___x_1279_; 
v___x_1279_ = lp_aesop_Aesop_Index_merge___redArg(v_ri_u2081_1277_, v_ri_u2082_1278_);
return v___x_1279_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0(lean_object* v_00_u03b2_1280_, lean_object* v_x_1281_, lean_object* v_x_1282_){
_start:
{
lean_object* v___x_1283_; 
v___x_1283_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___redArg(v_x_1281_, v_x_1282_);
return v___x_1283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0___boxed(lean_object* v_00_u03b2_1284_, lean_object* v_x_1285_, lean_object* v_x_1286_){
_start:
{
lean_object* v_res_1287_; 
v_res_1287_ = lp_aesop_Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0(v_00_u03b2_1284_, v_x_1285_, v_x_1286_);
lean_dec(v_x_1286_);
lean_dec_ref(v_x_1285_);
return v_res_1287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1(lean_object* v_00_u03b2_1288_, lean_object* v_x_1289_, lean_object* v_x_1290_, lean_object* v_x_1291_){
_start:
{
lean_object* v___x_1292_; 
v___x_1292_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1___redArg(v_x_1289_, v_x_1290_, v_x_1291_);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2(lean_object* v_00_u03c3_1293_, lean_object* v_00_u03b2_1294_, lean_object* v_map_1295_, lean_object* v_f_1296_, lean_object* v_init_1297_){
_start:
{
lean_object* v___x_1298_; 
v___x_1298_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___redArg(v_map_1295_, v_f_1296_, v_init_1297_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2___boxed(lean_object* v_00_u03c3_1299_, lean_object* v_00_u03b2_1300_, lean_object* v_map_1301_, lean_object* v_f_1302_, lean_object* v_init_1303_){
_start:
{
lean_object* v_res_1304_; 
v_res_1304_ = lp_aesop_Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2(v_00_u03c3_1299_, v_00_u03b2_1300_, v_map_1301_, v_f_1302_, v_init_1303_);
lean_dec_ref(v_map_1301_);
return v_res_1304_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3(lean_object* v_00_u03b1_1305_, lean_object* v_s_1306_, lean_object* v_as_1307_){
_start:
{
lean_object* v___x_1308_; 
v___x_1308_ = lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___redArg(v_s_1306_, v_as_1307_);
return v___x_1308_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3___boxed(lean_object* v_00_u03b1_1309_, lean_object* v_s_1310_, lean_object* v_as_1311_){
_start:
{
lean_object* v_res_1312_; 
v_res_1312_ = lp_aesop_Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3(v_00_u03b1_1309_, v_s_1310_, v_as_1311_);
lean_dec_ref(v_as_1311_);
return v_res_1312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0(lean_object* v_00_u03b2_1313_, lean_object* v_x_1314_, size_t v_x_1315_, lean_object* v_x_1316_){
_start:
{
lean_object* v___x_1317_; 
v___x_1317_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___redArg(v_x_1314_, v_x_1315_, v_x_1316_);
return v___x_1317_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1318_, lean_object* v_x_1319_, lean_object* v_x_1320_, lean_object* v_x_1321_){
_start:
{
size_t v_x_3871__boxed_1322_; lean_object* v_res_1323_; 
v_x_3871__boxed_1322_ = lean_unbox_usize(v_x_1320_);
lean_dec(v_x_1320_);
v_res_1323_ = lp_aesop_Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0(v_00_u03b2_1318_, v_x_1319_, v_x_3871__boxed_1322_, v_x_1321_);
lean_dec(v_x_1321_);
lean_dec_ref(v_x_1319_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2(lean_object* v_00_u03b2_1324_, lean_object* v_x_1325_, size_t v_x_1326_, size_t v_x_1327_, lean_object* v_x_1328_, lean_object* v_x_1329_){
_start:
{
lean_object* v___x_1330_; 
v___x_1330_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___redArg(v_x_1325_, v_x_1326_, v_x_1327_, v_x_1328_, v_x_1329_);
return v___x_1330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2___boxed(lean_object* v_00_u03b2_1331_, lean_object* v_x_1332_, lean_object* v_x_1333_, lean_object* v_x_1334_, lean_object* v_x_1335_, lean_object* v_x_1336_){
_start:
{
size_t v_x_3882__boxed_1337_; size_t v_x_3883__boxed_1338_; lean_object* v_res_1339_; 
v_x_3882__boxed_1337_ = lean_unbox_usize(v_x_1333_);
lean_dec(v_x_1333_);
v_x_3883__boxed_1338_ = lean_unbox_usize(v_x_1334_);
lean_dec(v_x_1334_);
v_res_1339_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2(v_00_u03b2_1331_, v_x_1332_, v_x_3882__boxed_1337_, v_x_3883__boxed_1338_, v_x_1335_, v_x_1336_);
return v_res_1339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___redArg(lean_object* v_map_1340_, lean_object* v_f_1341_, lean_object* v_init_1342_){
_start:
{
lean_object* v___x_1343_; 
v___x_1343_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v_f_1341_, v_map_1340_, v_init_1342_);
return v___x_1343_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___redArg___boxed(lean_object* v_map_1344_, lean_object* v_f_1345_, lean_object* v_init_1346_){
_start:
{
lean_object* v_res_1347_; 
v_res_1347_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___redArg(v_map_1344_, v_f_1345_, v_init_1346_);
lean_dec_ref(v_map_1344_);
return v_res_1347_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4(lean_object* v_00_u03c3_1348_, lean_object* v_00_u03b2_1349_, lean_object* v_map_1350_, lean_object* v_f_1351_, lean_object* v_init_1352_){
_start:
{
lean_object* v___x_1353_; 
v___x_1353_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v_f_1351_, v_map_1350_, v_init_1352_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4___boxed(lean_object* v_00_u03c3_1354_, lean_object* v_00_u03b2_1355_, lean_object* v_map_1356_, lean_object* v_f_1357_, lean_object* v_init_1358_){
_start:
{
lean_object* v_res_1359_; 
v_res_1359_ = lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4(v_00_u03c3_1354_, v_00_u03b2_1355_, v_map_1356_, v_f_1357_, v_init_1358_);
lean_dec_ref(v_map_1356_);
return v_res_1359_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6(lean_object* v_00_u03b1_1360_, lean_object* v_00_u03b2_1361_, lean_object* v_x_1362_, lean_object* v_x_1363_, lean_object* v_x_1364_){
_start:
{
lean_object* v___x_1365_; 
v___x_1365_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6___redArg(v_x_1362_, v_x_1363_, v_x_1364_);
return v___x_1365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7(lean_object* v_00_u03b1_1366_, lean_object* v_00_u03c3_1367_, lean_object* v_00_u03b2_1368_, lean_object* v_map_1369_, lean_object* v_init_1370_, lean_object* v_f_1371_){
_start:
{
lean_object* v___x_1372_; 
v___x_1372_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___redArg(v_map_1369_, v_init_1370_, v_f_1371_);
return v___x_1372_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7___boxed(lean_object* v_00_u03b1_1373_, lean_object* v_00_u03c3_1374_, lean_object* v_00_u03b2_1375_, lean_object* v_map_1376_, lean_object* v_init_1377_, lean_object* v_f_1378_){
_start:
{
lean_object* v_res_1379_; 
v_res_1379_ = lp_aesop_Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7(v_00_u03b1_1373_, v_00_u03c3_1374_, v_00_u03b2_1375_, v_map_1376_, v_init_1377_, v_f_1378_);
lean_dec_ref(v_map_1376_);
return v_res_1379_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_1380_, lean_object* v_keys_1381_, lean_object* v_vals_1382_, lean_object* v_heq_1383_, lean_object* v_i_1384_, lean_object* v_k_1385_){
_start:
{
lean_object* v___x_1386_; 
v___x_1386_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___redArg(v_keys_1381_, v_vals_1382_, v_i_1384_, v_k_1385_);
return v___x_1386_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_1387_, lean_object* v_keys_1388_, lean_object* v_vals_1389_, lean_object* v_heq_1390_, lean_object* v_i_1391_, lean_object* v_k_1392_){
_start:
{
lean_object* v_res_1393_; 
v_res_1393_ = lp_aesop_Lean_PersistentHashMap_findAtAux___at___00Lean_PersistentHashMap_findAux___at___00Lean_PersistentHashMap_find_x3f___at___00Aesop_Index_merge_spec__0_spec__0_spec__1(v_00_u03b2_1387_, v_keys_1388_, v_vals_1389_, v_heq_1390_, v_i_1391_, v_k_1392_);
lean_dec(v_k_1392_);
lean_dec_ref(v_vals_1389_);
lean_dec_ref(v_keys_1388_);
return v_res_1393_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4(lean_object* v_00_u03b2_1394_, lean_object* v_n_1395_, lean_object* v_k_1396_, lean_object* v_v_1397_){
_start:
{
lean_object* v___x_1398_; 
v___x_1398_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4___redArg(v_n_1395_, v_k_1396_, v_v_1397_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_1399_, size_t v_depth_1400_, lean_object* v_keys_1401_, lean_object* v_vals_1402_, lean_object* v_heq_1403_, lean_object* v_i_1404_, lean_object* v_entries_1405_){
_start:
{
lean_object* v___x_1406_; 
v___x_1406_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___redArg(v_depth_1400_, v_keys_1401_, v_vals_1402_, v_i_1404_, v_entries_1405_);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5___boxed(lean_object* v_00_u03b2_1407_, lean_object* v_depth_1408_, lean_object* v_keys_1409_, lean_object* v_vals_1410_, lean_object* v_heq_1411_, lean_object* v_i_1412_, lean_object* v_entries_1413_){
_start:
{
size_t v_depth_boxed_1414_; lean_object* v_res_1415_; 
v_depth_boxed_1414_ = lean_unbox_usize(v_depth_1408_);
lean_dec(v_depth_1408_);
v_res_1415_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__5(v_00_u03b2_1407_, v_depth_boxed_1414_, v_keys_1409_, v_vals_1410_, v_heq_1411_, v_i_1412_, v_entries_1413_);
lean_dec_ref(v_vals_1410_);
lean_dec_ref(v_keys_1409_);
return v_res_1415_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8(lean_object* v_00_u03c3_1416_, lean_object* v_00_u03b1_1417_, lean_object* v_00_u03b2_1418_, lean_object* v_f_1419_, lean_object* v_x_1420_, lean_object* v_x_1421_){
_start:
{
lean_object* v___x_1422_; 
v___x_1422_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___redArg(v_f_1419_, v_x_1420_, v_x_1421_);
return v___x_1422_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8___boxed(lean_object* v_00_u03c3_1423_, lean_object* v_00_u03b1_1424_, lean_object* v_00_u03b2_1425_, lean_object* v_f_1426_, lean_object* v_x_1427_, lean_object* v_x_1428_){
_start:
{
lean_object* v_res_1429_; 
v_res_1429_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8(v_00_u03c3_1423_, v_00_u03b1_1424_, v_00_u03b2_1425_, v_f_1426_, v_x_1427_, v_x_1428_);
lean_dec_ref(v_x_1427_);
return v_res_1429_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11(lean_object* v_00_u03b1_1430_, lean_object* v_00_u03b2_1431_, lean_object* v_x_1432_, size_t v_x_1433_, size_t v_x_1434_, lean_object* v_x_1435_, lean_object* v_x_1436_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___redArg(v_x_1432_, v_x_1433_, v_x_1434_, v_x_1435_, v_x_1436_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11___boxed(lean_object* v_00_u03b1_1438_, lean_object* v_00_u03b2_1439_, lean_object* v_x_1440_, lean_object* v_x_1441_, lean_object* v_x_1442_, lean_object* v_x_1443_, lean_object* v_x_1444_){
_start:
{
size_t v_x_3927__boxed_1445_; size_t v_x_3928__boxed_1446_; lean_object* v_res_1447_; 
v_x_3927__boxed_1445_ = lean_unbox_usize(v_x_1441_);
lean_dec(v_x_1441_);
v_x_3928__boxed_1446_ = lean_unbox_usize(v_x_1442_);
lean_dec(v_x_1442_);
v_res_1447_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11(v_00_u03b1_1438_, v_00_u03b2_1439_, v_x_1440_, v_x_3927__boxed_1445_, v_x_3928__boxed_1446_, v_x_1443_, v_x_1444_);
return v_res_1447_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13___redArg(lean_object* v_map_1448_, lean_object* v_f_1449_, lean_object* v_init_1450_){
_start:
{
lean_object* v___x_1451_; 
v___x_1451_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(v_f_1449_, v_map_1448_, v_init_1450_);
return v___x_1451_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13(lean_object* v_00_u03c3_1452_, lean_object* v_00_u03b1_1453_, lean_object* v_00_u03c3_1454_, lean_object* v_00_u03b2_1455_, lean_object* v_map_1456_, lean_object* v_f_1457_, lean_object* v_init_1458_){
_start:
{
lean_object* v___x_1459_; 
v___x_1459_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(v_f_1457_, v_map_1456_, v_init_1458_);
return v___x_1459_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7(lean_object* v_00_u03b2_1460_, lean_object* v_x_1461_, lean_object* v_x_1462_, lean_object* v_x_1463_, lean_object* v_x_1464_){
_start:
{
lean_object* v___x_1465_; 
v___x_1465_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Aesop_Index_merge_spec__1_spec__2_spec__4_spec__7___redArg(v_x_1461_, v_x_1462_, v_x_1463_, v_x_1464_);
return v___x_1465_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11(lean_object* v_00_u03b1_1466_, lean_object* v_00_u03b2_1467_, lean_object* v_00_u03c3_1468_, lean_object* v_f_1469_, lean_object* v_as_1470_, size_t v_i_1471_, size_t v_stop_1472_, lean_object* v_b_1473_){
_start:
{
lean_object* v___x_1474_; 
v___x_1474_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___redArg(v_f_1469_, v_as_1470_, v_i_1471_, v_stop_1472_, v_b_1473_);
return v___x_1474_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11___boxed(lean_object* v_00_u03b1_1475_, lean_object* v_00_u03b2_1476_, lean_object* v_00_u03c3_1477_, lean_object* v_f_1478_, lean_object* v_as_1479_, lean_object* v_i_1480_, lean_object* v_stop_1481_, lean_object* v_b_1482_){
_start:
{
size_t v_i_boxed_1483_; size_t v_stop_boxed_1484_; lean_object* v_res_1485_; 
v_i_boxed_1483_ = lean_unbox_usize(v_i_1480_);
lean_dec(v_i_1480_);
v_stop_boxed_1484_ = lean_unbox_usize(v_stop_1481_);
lean_dec(v_stop_1481_);
v_res_1485_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__11(v_00_u03b1_1475_, v_00_u03b2_1476_, v_00_u03c3_1477_, v_f_1478_, v_as_1479_, v_i_boxed_1483_, v_stop_boxed_1484_, v_b_1482_);
lean_dec_ref(v_as_1479_);
return v_res_1485_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12(lean_object* v_00_u03c3_1486_, lean_object* v_00_u03b1_1487_, lean_object* v_00_u03b2_1488_, lean_object* v_f_1489_, lean_object* v_keys_1490_, lean_object* v_vals_1491_, lean_object* v_heq_1492_, lean_object* v_i_1493_, lean_object* v_acc_1494_){
_start:
{
lean_object* v___x_1495_; 
v___x_1495_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___redArg(v_f_1489_, v_keys_1490_, v_vals_1491_, v_i_1493_, v_acc_1494_);
return v___x_1495_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12___boxed(lean_object* v_00_u03c3_1496_, lean_object* v_00_u03b1_1497_, lean_object* v_00_u03b2_1498_, lean_object* v_f_1499_, lean_object* v_keys_1500_, lean_object* v_vals_1501_, lean_object* v_heq_1502_, lean_object* v_i_1503_, lean_object* v_acc_1504_){
_start:
{
lean_object* v_res_1505_; 
v_res_1505_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Aesop_Index_merge_spec__2_spec__4_spec__8_spec__12(v_00_u03c3_1496_, v_00_u03b1_1497_, v_00_u03b2_1498_, v_f_1499_, v_keys_1500_, v_vals_1501_, v_heq_1502_, v_i_1503_, v_acc_1504_);
lean_dec_ref(v_vals_1501_);
lean_dec_ref(v_keys_1500_);
return v_res_1505_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15(lean_object* v_00_u03b1_1506_, lean_object* v_00_u03b2_1507_, lean_object* v_n_1508_, lean_object* v_k_1509_, lean_object* v_v_1510_){
_start:
{
lean_object* v___x_1511_; 
v___x_1511_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15___redArg(v_n_1508_, v_k_1509_, v_v_1510_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16(lean_object* v_00_u03b1_1512_, lean_object* v_00_u03b2_1513_, size_t v_depth_1514_, lean_object* v_keys_1515_, lean_object* v_vals_1516_, lean_object* v_heq_1517_, lean_object* v_i_1518_, lean_object* v_entries_1519_){
_start:
{
lean_object* v___x_1520_; 
v___x_1520_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___redArg(v_depth_1514_, v_keys_1515_, v_vals_1516_, v_i_1518_, v_entries_1519_);
return v___x_1520_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16___boxed(lean_object* v_00_u03b1_1521_, lean_object* v_00_u03b2_1522_, lean_object* v_depth_1523_, lean_object* v_keys_1524_, lean_object* v_vals_1525_, lean_object* v_heq_1526_, lean_object* v_i_1527_, lean_object* v_entries_1528_){
_start:
{
size_t v_depth_boxed_1529_; lean_object* v_res_1530_; 
v_depth_boxed_1529_ = lean_unbox_usize(v_depth_1523_);
lean_dec(v_depth_1523_);
v_res_1530_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__16(v_00_u03b1_1521_, v_00_u03b2_1522_, v_depth_boxed_1529_, v_keys_1524_, v_vals_1525_, v_heq_1526_, v_i_1527_, v_entries_1528_);
lean_dec_ref(v_vals_1525_);
lean_dec_ref(v_keys_1524_);
return v_res_1530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19(lean_object* v_00_u03c3_1531_, lean_object* v_00_u03c3_1532_, lean_object* v_00_u03b1_1533_, lean_object* v_00_u03b2_1534_, lean_object* v_f_1535_, lean_object* v_x_1536_, lean_object* v_x_1537_){
_start:
{
lean_object* v___x_1538_; 
v___x_1538_ = lp_aesop_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19___redArg(v_f_1535_, v_x_1536_, v_x_1537_);
return v___x_1538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17(lean_object* v_00_u03b1_1539_, lean_object* v_00_u03b2_1540_, lean_object* v_x_1541_, lean_object* v_x_1542_, lean_object* v_x_1543_, lean_object* v_x_1544_){
_start:
{
lean_object* v___x_1545_; 
v___x_1545_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6_spec__11_spec__15_spec__17___redArg(v_x_1541_, v_x_1542_, v_x_1543_, v_x_1544_);
return v___x_1545_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21(lean_object* v_00_u03b1_1546_, lean_object* v_00_u03b2_1547_, lean_object* v_00_u03c3_1548_, lean_object* v_00_u03c3_1549_, lean_object* v_f_1550_, lean_object* v_as_1551_, size_t v_i_1552_, size_t v_stop_1553_, lean_object* v_b_1554_){
_start:
{
lean_object* v___x_1555_; 
v___x_1555_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___redArg(v_f_1550_, v_as_1551_, v_i_1552_, v_stop_1553_, v_b_1554_);
return v___x_1555_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21___boxed(lean_object* v_00_u03b1_1556_, lean_object* v_00_u03b2_1557_, lean_object* v_00_u03c3_1558_, lean_object* v_00_u03c3_1559_, lean_object* v_f_1560_, lean_object* v_as_1561_, lean_object* v_i_1562_, lean_object* v_stop_1563_, lean_object* v_b_1564_){
_start:
{
size_t v_i_boxed_1565_; size_t v_stop_boxed_1566_; lean_object* v_res_1567_; 
v_i_boxed_1565_ = lean_unbox_usize(v_i_1562_);
lean_dec(v_i_1562_);
v_stop_boxed_1566_ = lean_unbox_usize(v_stop_1563_);
lean_dec(v_stop_1563_);
v_res_1567_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__21(v_00_u03b1_1556_, v_00_u03b2_1557_, v_00_u03c3_1558_, v_00_u03c3_1559_, v_f_1560_, v_as_1561_, v_i_boxed_1565_, v_stop_boxed_1566_, v_b_1564_);
lean_dec_ref(v_as_1561_);
return v_res_1567_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22(lean_object* v_00_u03c3_1568_, lean_object* v_00_u03c3_1569_, lean_object* v_00_u03b1_1570_, lean_object* v_00_u03b2_1571_, lean_object* v_f_1572_, lean_object* v_keys_1573_, lean_object* v_vals_1574_, lean_object* v_heq_1575_, lean_object* v_i_1576_, lean_object* v_acc_1577_){
_start:
{
lean_object* v___x_1578_; 
v___x_1578_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___redArg(v_f_1572_, v_keys_1573_, v_vals_1574_, v_i_1576_, v_acc_1577_);
return v___x_1578_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22___boxed(lean_object* v_00_u03c3_1579_, lean_object* v_00_u03c3_1580_, lean_object* v_00_u03b1_1581_, lean_object* v_00_u03b2_1582_, lean_object* v_f_1583_, lean_object* v_keys_1584_, lean_object* v_vals_1585_, lean_object* v_heq_1586_, lean_object* v_i_1587_, lean_object* v_acc_1588_){
_start:
{
lean_object* v_res_1589_; 
v_res_1589_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_forIn___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__7_spec__13_spec__19_spec__22(v_00_u03c3_1579_, v_00_u03c3_1580_, v_00_u03b1_1581_, v_00_u03b2_1582_, v_f_1583_, v_keys_1584_, v_vals_1585_, v_heq_1586_, v_i_1587_, v_acc_1588_);
lean_dec_ref(v_vals_1585_);
lean_dec_ref(v_keys_1584_);
return v_res_1589_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add___redArg(lean_object* v_r_1590_, lean_object* v_imode_1591_, lean_object* v_ri_1592_){
_start:
{
lean_object* v___f_1593_; 
v___f_1593_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__0));
switch(lean_obj_tag(v_imode_1591_))
{
case 0:
{
lean_object* v_byTarget_1594_; lean_object* v_byHyp_1595_; lean_object* v_unindexed_1596_; lean_object* v___x_1598_; uint8_t v_isShared_1599_; uint8_t v_isSharedCheck_1606_; 
v_byTarget_1594_ = lean_ctor_get(v_ri_1592_, 0);
v_byHyp_1595_ = lean_ctor_get(v_ri_1592_, 1);
v_unindexed_1596_ = lean_ctor_get(v_ri_1592_, 2);
v_isSharedCheck_1606_ = !lean_is_exclusive(v_ri_1592_);
if (v_isSharedCheck_1606_ == 0)
{
v___x_1598_ = v_ri_1592_;
v_isShared_1599_ = v_isSharedCheck_1606_;
goto v_resetjp_1597_;
}
else
{
lean_inc(v_unindexed_1596_);
lean_inc(v_byHyp_1595_);
lean_inc(v_byTarget_1594_);
lean_dec(v_ri_1592_);
v___x_1598_ = lean_box(0);
v_isShared_1599_ = v_isSharedCheck_1606_;
goto v_resetjp_1597_;
}
v_resetjp_1597_:
{
lean_object* v___f_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; lean_object* v___x_1604_; 
v___f_1600_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__1));
v___x_1601_ = lean_box(0);
v___x_1602_ = l_Lean_PersistentHashMap_insert___redArg(v___f_1593_, v___f_1600_, v_unindexed_1596_, v_r_1590_, v___x_1601_);
if (v_isShared_1599_ == 0)
{
lean_ctor_set(v___x_1598_, 2, v___x_1602_);
v___x_1604_ = v___x_1598_;
goto v_reusejp_1603_;
}
else
{
lean_object* v_reuseFailAlloc_1605_; 
v_reuseFailAlloc_1605_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1605_, 0, v_byTarget_1594_);
lean_ctor_set(v_reuseFailAlloc_1605_, 1, v_byHyp_1595_);
lean_ctor_set(v_reuseFailAlloc_1605_, 2, v___x_1602_);
v___x_1604_ = v_reuseFailAlloc_1605_;
goto v_reusejp_1603_;
}
v_reusejp_1603_:
{
return v___x_1604_;
}
}
}
case 1:
{
lean_object* v_keys_1607_; lean_object* v_byTarget_1608_; lean_object* v_byHyp_1609_; lean_object* v_unindexed_1610_; lean_object* v___x_1612_; uint8_t v_isShared_1613_; uint8_t v_isSharedCheck_1618_; 
v_keys_1607_ = lean_ctor_get(v_imode_1591_, 0);
lean_inc_ref(v_keys_1607_);
lean_dec_ref_known(v_imode_1591_, 1);
v_byTarget_1608_ = lean_ctor_get(v_ri_1592_, 0);
v_byHyp_1609_ = lean_ctor_get(v_ri_1592_, 1);
v_unindexed_1610_ = lean_ctor_get(v_ri_1592_, 2);
v_isSharedCheck_1618_ = !lean_is_exclusive(v_ri_1592_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1612_ = v_ri_1592_;
v_isShared_1613_ = v_isSharedCheck_1618_;
goto v_resetjp_1611_;
}
else
{
lean_inc(v_unindexed_1610_);
lean_inc(v_byHyp_1609_);
lean_inc(v_byTarget_1608_);
lean_dec(v_ri_1592_);
v___x_1612_ = lean_box(0);
v_isShared_1613_ = v_isSharedCheck_1618_;
goto v_resetjp_1611_;
}
v_resetjp_1611_:
{
lean_object* v___x_1614_; lean_object* v___x_1616_; 
v___x_1614_ = l_Lean_Meta_DiscrTree_insertKeyValue___redArg(v___f_1593_, v_byTarget_1608_, v_keys_1607_, v_r_1590_);
if (v_isShared_1613_ == 0)
{
lean_ctor_set(v___x_1612_, 0, v___x_1614_);
v___x_1616_ = v___x_1612_;
goto v_reusejp_1615_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v___x_1614_);
lean_ctor_set(v_reuseFailAlloc_1617_, 1, v_byHyp_1609_);
lean_ctor_set(v_reuseFailAlloc_1617_, 2, v_unindexed_1610_);
v___x_1616_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1615_;
}
v_reusejp_1615_:
{
return v___x_1616_;
}
}
}
case 2:
{
lean_object* v_keys_1619_; lean_object* v_byTarget_1620_; lean_object* v_byHyp_1621_; lean_object* v_unindexed_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1630_; 
v_keys_1619_ = lean_ctor_get(v_imode_1591_, 0);
lean_inc_ref(v_keys_1619_);
lean_dec_ref_known(v_imode_1591_, 1);
v_byTarget_1620_ = lean_ctor_get(v_ri_1592_, 0);
v_byHyp_1621_ = lean_ctor_get(v_ri_1592_, 1);
v_unindexed_1622_ = lean_ctor_get(v_ri_1592_, 2);
v_isSharedCheck_1630_ = !lean_is_exclusive(v_ri_1592_);
if (v_isSharedCheck_1630_ == 0)
{
v___x_1624_ = v_ri_1592_;
v_isShared_1625_ = v_isSharedCheck_1630_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_unindexed_1622_);
lean_inc(v_byHyp_1621_);
lean_inc(v_byTarget_1620_);
lean_dec(v_ri_1592_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1630_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1626_; lean_object* v___x_1628_; 
v___x_1626_ = l_Lean_Meta_DiscrTree_insertKeyValue___redArg(v___f_1593_, v_byHyp_1621_, v_keys_1619_, v_r_1590_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set(v___x_1624_, 1, v___x_1626_);
v___x_1628_ = v___x_1624_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v_byTarget_1620_);
lean_ctor_set(v_reuseFailAlloc_1629_, 1, v___x_1626_);
lean_ctor_set(v_reuseFailAlloc_1629_, 2, v_unindexed_1622_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
}
default: 
{
lean_object* v_imodes_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; uint8_t v___x_1635_; 
v_imodes_1631_ = lean_ctor_get(v_imode_1591_, 0);
lean_inc_ref(v_imodes_1631_);
lean_dec_ref_known(v_imode_1591_, 1);
v___x_1632_ = lean_unsigned_to_nat(0u);
v___x_1633_ = lean_array_get_size(v_imodes_1631_);
v___x_1634_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_1635_ = lean_nat_dec_lt(v___x_1632_, v___x_1633_);
if (v___x_1635_ == 0)
{
lean_dec_ref(v_imodes_1631_);
lean_dec_ref(v_r_1590_);
return v_ri_1592_;
}
else
{
lean_object* v___f_1636_; uint8_t v___x_1637_; 
v___f_1636_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_add___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1636_, 0, v_r_1590_);
v___x_1637_ = lean_nat_dec_le(v___x_1633_, v___x_1633_);
if (v___x_1637_ == 0)
{
if (v___x_1635_ == 0)
{
lean_dec_ref(v___f_1636_);
lean_dec_ref(v_imodes_1631_);
return v_ri_1592_;
}
else
{
size_t v___x_1638_; size_t v___x_1639_; lean_object* v___x_1640_; 
v___x_1638_ = ((size_t)0ULL);
v___x_1639_ = lean_usize_of_nat(v___x_1633_);
v___x_1640_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1634_, v___f_1636_, v_imodes_1631_, v___x_1638_, v___x_1639_, v_ri_1592_);
return v___x_1640_;
}
}
else
{
size_t v___x_1641_; size_t v___x_1642_; lean_object* v___x_1643_; 
v___x_1641_ = ((size_t)0ULL);
v___x_1642_ = lean_usize_of_nat(v___x_1633_);
v___x_1643_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_1634_, v___f_1636_, v_imodes_1631_, v___x_1641_, v___x_1642_, v_ri_1592_);
return v___x_1643_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add___redArg___lam__0(lean_object* v_r_1644_, lean_object* v_x1_1645_, lean_object* v_x2_1646_){
_start:
{
lean_object* v___x_1647_; 
v___x_1647_ = lp_aesop_Aesop_Index_add___redArg(v_r_1644_, v_x2_1646_, v_x1_1645_);
return v___x_1647_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_add(lean_object* v_00_u03b1_1648_, lean_object* v_r_1649_, lean_object* v_imode_1650_, lean_object* v_ri_1651_){
_start:
{
lean_object* v___x_1652_; 
v___x_1652_ = lp_aesop_Aesop_Index_add___redArg(v_r_1649_, v_imode_1650_, v_ri_1651_);
return v___x_1652_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___lam__0(lean_object* v___f_1653_, lean_object* v___f_1654_, lean_object* v_unindexed_1655_, lean_object* v_v_1656_){
_start:
{
lean_object* v___x_1657_; lean_object* v___x_1658_; 
v___x_1657_ = lean_box(0);
v___x_1658_ = l_Lean_PersistentHashMap_insert___redArg(v___f_1653_, v___f_1654_, v_unindexed_1655_, v_v_1656_, v___x_1657_);
return v___x_1658_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1(void){
_start:
{
lean_object* v___f_1662_; lean_object* v___f_1663_; lean_object* v___x_1664_; 
v___f_1662_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__1));
v___f_1663_ = ((lean_object*)(lp_aesop_Aesop_Index_instEmptyCollection___closed__0));
v___x_1664_ = l_Lean_PersistentHashMap_empty(lean_box(0), lean_box(0), v___f_1663_, v___f_1662_);
return v___x_1664_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg(lean_object* v_p_1666_, lean_object* v_unindexed_1667_, lean_object* v_t_1668_){
_start:
{
lean_object* v___f_1669_; lean_object* v___x_1670_; lean_object* v___x_1671_; lean_object* v___x_1672_; lean_object* v___x_1673_; 
v___f_1669_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__0));
v___x_1670_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1);
v___x_1671_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__2));
v___x_1672_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_1672_, 0, lean_box(0));
lean_closure_set(v___x_1672_, 1, lean_box(0));
lean_closure_set(v___x_1672_, 2, lean_box(0));
lean_closure_set(v___x_1672_, 3, v___x_1671_);
lean_closure_set(v___x_1672_, 4, v_p_1666_);
v___x_1673_ = lp_aesop_Aesop_filterDiscrTree___redArg(v___x_1670_, v___x_1672_, v___f_1669_, v_unindexed_1667_, v_t_1668_);
return v___x_1673_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27(lean_object* v_00_u03b1_1674_, lean_object* v_p_1675_, lean_object* v_unindexed_1676_, lean_object* v_t_1677_){
_start:
{
lean_object* v___f_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___x_1682_; 
v___f_1678_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__0));
v___x_1679_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1);
v___x_1680_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__2));
v___x_1681_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_1681_, 0, lean_box(0));
lean_closure_set(v___x_1681_, 1, lean_box(0));
lean_closure_set(v___x_1681_, 2, lean_box(0));
lean_closure_set(v___x_1681_, 3, v___x_1680_);
lean_closure_set(v___x_1681_, 4, v_p_1675_);
v___x_1682_ = lp_aesop_Aesop_filterDiscrTree___redArg(v___x_1679_, v___x_1681_, v___f_1678_, v_unindexed_1676_, v_t_1677_);
return v___x_1682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg___lam__0(lean_object* v_unindexed_1683_, lean_object* v_v_1684_){
_start:
{
lean_object* v___x_1685_; lean_object* v___x_1686_; 
v___x_1685_ = lean_box(0);
v___x_1686_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_PersistentHashSet_insertMany___at___00Aesop_Index_merge_spec__3_spec__6___redArg(v_unindexed_1683_, v_v_1684_, v___x_1685_);
return v___x_1686_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_unindex___redArg___lam__1(lean_object* v_p_1687_, lean_object* v___y_1688_){
_start:
{
lean_object* v___x_1689_; uint8_t v___x_1690_; 
v___x_1689_ = lean_apply_1(v_p_1687_, v___y_1688_);
v___x_1690_ = lean_unbox(v___x_1689_);
if (v___x_1690_ == 0)
{
uint8_t v___x_1691_; 
v___x_1691_ = 1;
return v___x_1691_;
}
else
{
uint8_t v___x_1692_; 
v___x_1692_ = 0;
return v___x_1692_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg___lam__1___boxed(lean_object* v_p_1693_, lean_object* v___y_1694_){
_start:
{
uint8_t v_res_1695_; lean_object* v_r_1696_; 
v_res_1695_ = lp_aesop_Aesop_Index_unindex___redArg___lam__1(v_p_1693_, v___y_1694_);
v_r_1696_ = lean_box(v_res_1695_);
return v_r_1696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex___redArg(lean_object* v_ri_1698_, lean_object* v_p_1699_){
_start:
{
lean_object* v_byTarget_1700_; lean_object* v_byHyp_1701_; lean_object* v_unindexed_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1718_; 
v_byTarget_1700_ = lean_ctor_get(v_ri_1698_, 0);
v_byHyp_1701_ = lean_ctor_get(v_ri_1698_, 1);
v_unindexed_1702_ = lean_ctor_get(v_ri_1698_, 2);
v_isSharedCheck_1718_ = !lean_is_exclusive(v_ri_1698_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1704_ = v_ri_1698_;
v_isShared_1705_ = v_isSharedCheck_1718_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_unindexed_1702_);
lean_inc(v_byHyp_1701_);
lean_inc(v_byTarget_1700_);
lean_dec(v_ri_1698_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1718_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___f_1706_; lean_object* v___f_1707_; lean_object* v___x_1708_; lean_object* v___x_1709_; lean_object* v_fst_1710_; lean_object* v_snd_1711_; lean_object* v___x_1712_; lean_object* v_fst_1713_; lean_object* v_snd_1714_; lean_object* v___x_1716_; 
v___f_1706_ = ((lean_object*)(lp_aesop_Aesop_Index_unindex___redArg___closed__0));
v___f_1707_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_unindex___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_1707_, 0, v_p_1699_);
v___x_1708_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_unindex_filterDiscrTree_x27___redArg___closed__1);
lean_inc_ref(v___f_1707_);
v___x_1709_ = lp_aesop_Aesop_filterDiscrTree___redArg(v___x_1708_, v___f_1707_, v___f_1706_, v_unindexed_1702_, v_byTarget_1700_);
v_fst_1710_ = lean_ctor_get(v___x_1709_, 0);
lean_inc(v_fst_1710_);
v_snd_1711_ = lean_ctor_get(v___x_1709_, 1);
lean_inc(v_snd_1711_);
lean_dec_ref(v___x_1709_);
v___x_1712_ = lp_aesop_Aesop_filterDiscrTree___redArg(v___x_1708_, v___f_1707_, v___f_1706_, v_snd_1711_, v_byHyp_1701_);
v_fst_1713_ = lean_ctor_get(v___x_1712_, 0);
lean_inc(v_fst_1713_);
v_snd_1714_ = lean_ctor_get(v___x_1712_, 1);
lean_inc(v_snd_1714_);
lean_dec_ref(v___x_1712_);
if (v_isShared_1705_ == 0)
{
lean_ctor_set(v___x_1704_, 2, v_snd_1714_);
lean_ctor_set(v___x_1704_, 1, v_fst_1713_);
lean_ctor_set(v___x_1704_, 0, v_fst_1710_);
v___x_1716_ = v___x_1704_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_fst_1710_);
lean_ctor_set(v_reuseFailAlloc_1717_, 1, v_fst_1713_);
lean_ctor_set(v_reuseFailAlloc_1717_, 2, v_snd_1714_);
v___x_1716_ = v_reuseFailAlloc_1717_;
goto v_reusejp_1715_;
}
v_reusejp_1715_:
{
return v___x_1716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_unindex(lean_object* v_00_u03b1_1719_, lean_object* v_ri_1720_, lean_object* v_p_1721_){
_start:
{
lean_object* v___x_1722_; 
v___x_1722_ = lp_aesop_Aesop_Index_unindex___redArg(v_ri_1720_, v_p_1721_);
return v___x_1722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__0(lean_object* v_inst_1723_, lean_object* v_f_1724_, lean_object* v_s_1725_, lean_object* v_x_1726_, lean_object* v_t_1727_){
_start:
{
lean_object* v___x_1728_; 
v___x_1728_ = l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(v_inst_1723_, v_f_1724_, v_s_1725_, v_t_1727_);
return v___x_1728_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__0___boxed(lean_object* v_inst_1729_, lean_object* v_f_1730_, lean_object* v_s_1731_, lean_object* v_x_1732_, lean_object* v_t_1733_){
_start:
{
lean_object* v_res_1734_; 
v_res_1734_ = lp_aesop_Aesop_Index_foldM___redArg___lam__0(v_inst_1729_, v_f_1730_, v_s_1731_, v_x_1732_, v_t_1733_);
lean_dec(v_x_1732_);
return v_res_1734_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__2(lean_object* v_f_1735_, lean_object* v_d_1736_, lean_object* v_a_1737_, lean_object* v_x_1738_){
_start:
{
lean_object* v___x_1739_; 
v___x_1739_ = lean_apply_2(v_f_1735_, v_d_1736_, v_a_1737_);
return v___x_1739_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__1(lean_object* v_inst_1740_, lean_object* v___f_1741_, lean_object* v_unindexed_1742_, lean_object* v_s_1743_){
_start:
{
lean_object* v___x_1744_; 
v___x_1744_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1740_, v___f_1741_, v_unindexed_1742_, v_s_1743_);
return v___x_1744_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg___lam__3(lean_object* v_inst_1745_, lean_object* v___f_1746_, lean_object* v_byTarget_1747_, lean_object* v_toBind_1748_, lean_object* v___f_1749_, lean_object* v_s_1750_){
_start:
{
lean_object* v___x_1751_; lean_object* v___x_1752_; 
v___x_1751_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1745_, v___f_1746_, v_byTarget_1747_, v_s_1750_);
v___x_1752_ = lean_apply_4(v_toBind_1748_, lean_box(0), lean_box(0), v___x_1751_, v___f_1749_);
return v___x_1752_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM___redArg(lean_object* v_inst_1753_, lean_object* v_ri_1754_, lean_object* v_f_1755_, lean_object* v_init_1756_){
_start:
{
lean_object* v_toBind_1757_; lean_object* v_byTarget_1758_; lean_object* v_byHyp_1759_; lean_object* v_unindexed_1760_; lean_object* v___f_1761_; lean_object* v___f_1762_; lean_object* v___f_1763_; lean_object* v___f_1764_; lean_object* v___x_1765_; lean_object* v___x_1766_; 
v_toBind_1757_ = lean_ctor_get(v_inst_1753_, 1);
lean_inc_n(v_toBind_1757_, 2);
v_byTarget_1758_ = lean_ctor_get(v_ri_1754_, 0);
lean_inc_ref(v_byTarget_1758_);
v_byHyp_1759_ = lean_ctor_get(v_ri_1754_, 1);
lean_inc_ref(v_byHyp_1759_);
v_unindexed_1760_ = lean_ctor_get(v_ri_1754_, 2);
lean_inc_ref(v_unindexed_1760_);
lean_dec_ref(v_ri_1754_);
lean_inc(v_f_1755_);
lean_inc_ref_n(v_inst_1753_, 3);
v___f_1761_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_foldM___redArg___lam__0___boxed), 5, 2);
lean_closure_set(v___f_1761_, 0, v_inst_1753_);
lean_closure_set(v___f_1761_, 1, v_f_1755_);
v___f_1762_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_foldM___redArg___lam__2), 4, 1);
lean_closure_set(v___f_1762_, 0, v_f_1755_);
v___f_1763_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_foldM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_1763_, 0, v_inst_1753_);
lean_closure_set(v___f_1763_, 1, v___f_1762_);
lean_closure_set(v___f_1763_, 2, v_unindexed_1760_);
lean_inc_ref(v___f_1761_);
v___f_1764_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_foldM___redArg___lam__3), 6, 5);
lean_closure_set(v___f_1764_, 0, v_inst_1753_);
lean_closure_set(v___f_1764_, 1, v___f_1761_);
lean_closure_set(v___f_1764_, 2, v_byTarget_1758_);
lean_closure_set(v___f_1764_, 3, v_toBind_1757_);
lean_closure_set(v___f_1764_, 4, v___f_1763_);
v___x_1765_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v_inst_1753_, v___f_1761_, v_byHyp_1759_, v_init_1756_);
v___x_1766_ = lean_apply_4(v_toBind_1757_, lean_box(0), lean_box(0), v___x_1765_, v___f_1764_);
return v___x_1766_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_foldM(lean_object* v_00_u03b1_1767_, lean_object* v_m_1768_, lean_object* v_00_u03c3_1769_, lean_object* v_inst_1770_, lean_object* v_ri_1771_, lean_object* v_f_1772_, lean_object* v_init_1773_){
_start:
{
lean_object* v___x_1774_; 
v___x_1774_ = lp_aesop_Aesop_Index_foldM___redArg(v_inst_1770_, v_ri_1771_, v_f_1772_, v_init_1773_);
return v___x_1774_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_fold___redArg(lean_object* v_ri_1775_, lean_object* v_f_1776_, lean_object* v_init_1777_){
_start:
{
lean_object* v___x_1778_; lean_object* v___x_1779_; 
v___x_1778_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_1779_ = lp_aesop_Aesop_Index_foldM___redArg(v___x_1778_, v_ri_1775_, v_f_1776_, v_init_1777_);
return v___x_1779_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_fold(lean_object* v_00_u03b1_1780_, lean_object* v_00_u03c3_1781_, lean_object* v_ri_1782_, lean_object* v_f_1783_, lean_object* v_init_1784_){
_start:
{
lean_object* v___x_1785_; lean_object* v___x_1786_; 
v___x_1785_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_1786_ = lp_aesop_Aesop_Index_foldM___redArg(v___x_1785_, v_ri_1782_, v_f_1783_, v_init_1784_);
return v___x_1786_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0(lean_object* v_include_x3f_1791_, lean_object* v_a_1792_, lean_object* v_x_1793_, lean_object* v___y_1794_, lean_object* v___y_1795_, lean_object* v___y_1796_, lean_object* v___y_1797_, lean_object* v___y_1798_){
_start:
{
lean_object* v___x_1800_; uint8_t v___x_1801_; 
lean_inc_ref(v_a_1792_);
v___x_1800_ = lean_apply_1(v_include_x3f_1791_, v_a_1792_);
v___x_1801_ = lean_unbox(v___x_1800_);
if (v___x_1801_ == 0)
{
lean_object* v___x_1802_; lean_object* v___x_1803_; 
lean_dec_ref(v_a_1792_);
v___x_1802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1802_, 0, v___y_1794_);
v___x_1803_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1803_, 0, v___x_1802_);
return v___x_1803_;
}
else
{
lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; 
v___x_1804_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___closed__0));
v___x_1805_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1805_, 0, v_a_1792_);
lean_ctor_set(v___x_1805_, 1, v___x_1804_);
v___x_1806_ = lean_array_push(v___y_1794_, v___x_1805_);
v___x_1807_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1807_, 0, v___x_1806_);
v___x_1808_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1808_, 0, v___x_1807_);
return v___x_1808_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed(lean_object* v_include_x3f_1809_, lean_object* v_a_1810_, lean_object* v_x_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_){
_start:
{
lean_object* v_res_1818_; 
v_res_1818_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0(v_include_x3f_1809_, v_a_1810_, v_x_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_, v___y_1816_);
lean_dec(v___y_1816_);
lean_dec_ref(v___y_1815_);
lean_dec(v___y_1814_);
lean_dec_ref(v___y_1813_);
return v_res_1818_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1(lean_object* v_goal_1819_, lean_object* v_ri_1820_, lean_object* v___x_1821_, lean_object* v___f_1822_, lean_object* v___y_1823_, lean_object* v___y_1824_, lean_object* v___y_1825_, lean_object* v___y_1826_){
_start:
{
lean_object* v___x_1828_; 
v___x_1828_ = l_Lean_MVarId_getType(v_goal_1819_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1828_) == 0)
{
lean_object* v_a_1829_; lean_object* v_byTarget_1830_; lean_object* v___x_1831_; 
v_a_1829_ = lean_ctor_get(v___x_1828_, 0);
lean_inc(v_a_1829_);
lean_dec_ref_known(v___x_1828_, 1);
v_byTarget_1830_ = lean_ctor_get(v_ri_1820_, 0);
lean_inc_ref(v_byTarget_1830_);
lean_dec_ref(v_ri_1820_);
v___x_1831_ = lp_aesop_Aesop_getUnify___redArg(v_byTarget_1830_, v_a_1829_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_);
if (lean_obj_tag(v___x_1831_) == 0)
{
lean_object* v_a_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; size_t v_sz_1835_; size_t v___x_1836_; lean_object* v___x_869__overap_1837_; lean_object* v___x_1838_; 
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
lean_inc(v_a_1832_);
lean_dec_ref_known(v___x_1831_, 1);
v___x_1833_ = lean_array_get_size(v_a_1832_);
v___x_1834_ = lean_mk_empty_array_with_capacity(v___x_1833_);
v_sz_1835_ = lean_array_size(v_a_1832_);
v___x_1836_ = ((size_t)0ULL);
v___x_869__overap_1837_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_1821_, v_a_1832_, v___f_1822_, v_sz_1835_, v___x_1836_, v___x_1834_);
v___x_1838_ = lean_apply_5(v___x_869__overap_1837_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_, lean_box(0));
return v___x_1838_;
}
else
{
lean_object* v_a_1839_; lean_object* v___x_1841_; uint8_t v_isShared_1842_; uint8_t v_isSharedCheck_1846_; 
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec_ref(v___f_1822_);
lean_dec_ref(v___x_1821_);
v_a_1839_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1846_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1846_ == 0)
{
v___x_1841_ = v___x_1831_;
v_isShared_1842_ = v_isSharedCheck_1846_;
goto v_resetjp_1840_;
}
else
{
lean_inc(v_a_1839_);
lean_dec(v___x_1831_);
v___x_1841_ = lean_box(0);
v_isShared_1842_ = v_isSharedCheck_1846_;
goto v_resetjp_1840_;
}
v_resetjp_1840_:
{
lean_object* v___x_1844_; 
if (v_isShared_1842_ == 0)
{
v___x_1844_ = v___x_1841_;
goto v_reusejp_1843_;
}
else
{
lean_object* v_reuseFailAlloc_1845_; 
v_reuseFailAlloc_1845_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1845_, 0, v_a_1839_);
v___x_1844_ = v_reuseFailAlloc_1845_;
goto v_reusejp_1843_;
}
v_reusejp_1843_:
{
return v___x_1844_;
}
}
}
}
else
{
lean_object* v_a_1847_; lean_object* v___x_1849_; uint8_t v_isShared_1850_; uint8_t v_isSharedCheck_1854_; 
lean_dec(v___y_1826_);
lean_dec_ref(v___y_1825_);
lean_dec(v___y_1824_);
lean_dec_ref(v___y_1823_);
lean_dec_ref(v___f_1822_);
lean_dec_ref(v___x_1821_);
lean_dec_ref(v_ri_1820_);
v_a_1847_ = lean_ctor_get(v___x_1828_, 0);
v_isSharedCheck_1854_ = !lean_is_exclusive(v___x_1828_);
if (v_isSharedCheck_1854_ == 0)
{
v___x_1849_ = v___x_1828_;
v_isShared_1850_ = v_isSharedCheck_1854_;
goto v_resetjp_1848_;
}
else
{
lean_inc(v_a_1847_);
lean_dec(v___x_1828_);
v___x_1849_ = lean_box(0);
v_isShared_1850_ = v_isSharedCheck_1854_;
goto v_resetjp_1848_;
}
v_resetjp_1848_:
{
lean_object* v___x_1852_; 
if (v_isShared_1850_ == 0)
{
v___x_1852_ = v___x_1849_;
goto v_reusejp_1851_;
}
else
{
lean_object* v_reuseFailAlloc_1853_; 
v_reuseFailAlloc_1853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1853_, 0, v_a_1847_);
v___x_1852_ = v_reuseFailAlloc_1853_;
goto v_reusejp_1851_;
}
v_reusejp_1851_:
{
return v___x_1852_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed(lean_object* v_goal_1855_, lean_object* v_ri_1856_, lean_object* v___x_1857_, lean_object* v___f_1858_, lean_object* v___y_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_){
_start:
{
lean_object* v_res_1864_; 
v_res_1864_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1(v_goal_1855_, v_ri_1856_, v___x_1857_, v___f_1858_, v___y_1859_, v___y_1860_, v___y_1861_, v___y_1862_);
return v_res_1864_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg(lean_object* v_ri_1867_, lean_object* v_goal_1868_, lean_object* v_include_x3f_1869_, lean_object* v_a_1870_, lean_object* v_a_1871_, lean_object* v_a_1872_, lean_object* v_a_1873_){
_start:
{
lean_object* v___x_1875_; lean_object* v_toApplicative_1876_; lean_object* v_toFunctor_1877_; lean_object* v_toSeq_1878_; lean_object* v_toSeqLeft_1879_; lean_object* v_toSeqRight_1880_; lean_object* v___f_1881_; lean_object* v___f_1882_; lean_object* v___f_1883_; lean_object* v___f_1884_; lean_object* v___x_1885_; lean_object* v___f_1886_; lean_object* v___f_1887_; lean_object* v___f_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v_toApplicative_1894_; lean_object* v_toFunctor_1895_; lean_object* v_toSeq_1896_; lean_object* v_toSeqLeft_1897_; lean_object* v_toSeqRight_1898_; lean_object* v___f_1899_; lean_object* v___f_1900_; lean_object* v___x_1901_; lean_object* v___f_1902_; lean_object* v___f_1903_; lean_object* v___f_1904_; lean_object* v___x_1905_; lean_object* v___x_1906_; lean_object* v___x_1907_; lean_object* v_toApplicative_1908_; lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1939_; 
v___x_1875_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_1876_ = lean_ctor_get(v___x_1875_, 0);
v_toFunctor_1877_ = lean_ctor_get(v_toApplicative_1876_, 0);
v_toSeq_1878_ = lean_ctor_get(v_toApplicative_1876_, 2);
v_toSeqLeft_1879_ = lean_ctor_get(v_toApplicative_1876_, 3);
v_toSeqRight_1880_ = lean_ctor_get(v_toApplicative_1876_, 4);
v___f_1881_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_1882_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_1877_, 2);
v___f_1883_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1883_, 0, v_toFunctor_1877_);
v___f_1884_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1884_, 0, v_toFunctor_1877_);
v___x_1885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1885_, 0, v___f_1883_);
lean_ctor_set(v___x_1885_, 1, v___f_1884_);
lean_inc(v_toSeqRight_1880_);
v___f_1886_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1886_, 0, v_toSeqRight_1880_);
lean_inc(v_toSeqLeft_1879_);
v___f_1887_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1887_, 0, v_toSeqLeft_1879_);
lean_inc(v_toSeq_1878_);
v___f_1888_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1888_, 0, v_toSeq_1878_);
v___x_1889_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1889_, 0, v___x_1885_);
lean_ctor_set(v___x_1889_, 1, v___f_1881_);
lean_ctor_set(v___x_1889_, 2, v___f_1888_);
lean_ctor_set(v___x_1889_, 3, v___f_1887_);
lean_ctor_set(v___x_1889_, 4, v___f_1886_);
v___x_1890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1890_, 0, v___x_1889_);
lean_ctor_set(v___x_1890_, 1, v___f_1882_);
v___x_1891_ = l_StateRefT_x27_instMonad___redArg(v___x_1890_);
v___x_1892_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_1892_, 0, lean_box(0));
lean_closure_set(v___x_1892_, 1, lean_box(0));
lean_closure_set(v___x_1892_, 2, v___x_1891_);
v___x_1893_ = l_instMonadControlTOfPure___redArg(v___x_1892_);
v_toApplicative_1894_ = lean_ctor_get(v___x_1875_, 0);
v_toFunctor_1895_ = lean_ctor_get(v_toApplicative_1894_, 0);
v_toSeq_1896_ = lean_ctor_get(v_toApplicative_1894_, 2);
v_toSeqLeft_1897_ = lean_ctor_get(v_toApplicative_1894_, 3);
v_toSeqRight_1898_ = lean_ctor_get(v_toApplicative_1894_, 4);
lean_inc_ref_n(v_toFunctor_1895_, 2);
v___f_1899_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1899_, 0, v_toFunctor_1895_);
v___f_1900_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1900_, 0, v_toFunctor_1895_);
v___x_1901_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1901_, 0, v___f_1899_);
lean_ctor_set(v___x_1901_, 1, v___f_1900_);
lean_inc(v_toSeqRight_1898_);
v___f_1902_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1902_, 0, v_toSeqRight_1898_);
lean_inc(v_toSeqLeft_1897_);
v___f_1903_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1903_, 0, v_toSeqLeft_1897_);
lean_inc(v_toSeq_1896_);
v___f_1904_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1904_, 0, v_toSeq_1896_);
v___x_1905_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1905_, 0, v___x_1901_);
lean_ctor_set(v___x_1905_, 1, v___f_1881_);
lean_ctor_set(v___x_1905_, 2, v___f_1904_);
lean_ctor_set(v___x_1905_, 3, v___f_1903_);
lean_ctor_set(v___x_1905_, 4, v___f_1902_);
v___x_1906_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1906_, 0, v___x_1905_);
lean_ctor_set(v___x_1906_, 1, v___f_1882_);
v___x_1907_ = l_StateRefT_x27_instMonad___redArg(v___x_1906_);
v_toApplicative_1908_ = lean_ctor_get(v___x_1907_, 0);
v_isSharedCheck_1939_ = !lean_is_exclusive(v___x_1907_);
if (v_isSharedCheck_1939_ == 0)
{
lean_object* v_unused_1940_; 
v_unused_1940_ = lean_ctor_get(v___x_1907_, 1);
lean_dec(v_unused_1940_);
v___x_1910_ = v___x_1907_;
v_isShared_1911_ = v_isSharedCheck_1939_;
goto v_resetjp_1909_;
}
else
{
lean_inc(v_toApplicative_1908_);
lean_dec(v___x_1907_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1939_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v_toFunctor_1912_; lean_object* v_toSeq_1913_; lean_object* v_toSeqLeft_1914_; lean_object* v_toSeqRight_1915_; lean_object* v___x_1917_; uint8_t v_isShared_1918_; uint8_t v_isSharedCheck_1937_; 
v_toFunctor_1912_ = lean_ctor_get(v_toApplicative_1908_, 0);
v_toSeq_1913_ = lean_ctor_get(v_toApplicative_1908_, 2);
v_toSeqLeft_1914_ = lean_ctor_get(v_toApplicative_1908_, 3);
v_toSeqRight_1915_ = lean_ctor_get(v_toApplicative_1908_, 4);
v_isSharedCheck_1937_ = !lean_is_exclusive(v_toApplicative_1908_);
if (v_isSharedCheck_1937_ == 0)
{
lean_object* v_unused_1938_; 
v_unused_1938_ = lean_ctor_get(v_toApplicative_1908_, 1);
lean_dec(v_unused_1938_);
v___x_1917_ = v_toApplicative_1908_;
v_isShared_1918_ = v_isSharedCheck_1937_;
goto v_resetjp_1916_;
}
else
{
lean_inc(v_toSeqRight_1915_);
lean_inc(v_toSeqLeft_1914_);
lean_inc(v_toSeq_1913_);
lean_inc(v_toFunctor_1912_);
lean_dec(v_toApplicative_1908_);
v___x_1917_ = lean_box(0);
v_isShared_1918_ = v_isSharedCheck_1937_;
goto v_resetjp_1916_;
}
v_resetjp_1916_:
{
lean_object* v___f_1919_; lean_object* v___f_1920_; lean_object* v___f_1921_; lean_object* v___f_1922_; lean_object* v___f_1923_; lean_object* v___x_1924_; lean_object* v___f_1925_; lean_object* v___f_1926_; lean_object* v___f_1927_; lean_object* v___x_1929_; 
v___f_1919_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_1919_, 0, v_include_x3f_1869_);
v___f_1920_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0));
v___f_1921_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1));
lean_inc_ref(v_toFunctor_1912_);
v___f_1922_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1922_, 0, v_toFunctor_1912_);
v___f_1923_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1923_, 0, v_toFunctor_1912_);
v___x_1924_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1924_, 0, v___f_1922_);
lean_ctor_set(v___x_1924_, 1, v___f_1923_);
v___f_1925_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1925_, 0, v_toSeqRight_1915_);
v___f_1926_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1926_, 0, v_toSeqLeft_1914_);
v___f_1927_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1927_, 0, v_toSeq_1913_);
if (v_isShared_1918_ == 0)
{
lean_ctor_set(v___x_1917_, 4, v___f_1925_);
lean_ctor_set(v___x_1917_, 3, v___f_1926_);
lean_ctor_set(v___x_1917_, 2, v___f_1927_);
lean_ctor_set(v___x_1917_, 1, v___f_1920_);
lean_ctor_set(v___x_1917_, 0, v___x_1924_);
v___x_1929_ = v___x_1917_;
goto v_reusejp_1928_;
}
else
{
lean_object* v_reuseFailAlloc_1936_; 
v_reuseFailAlloc_1936_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1936_, 0, v___x_1924_);
lean_ctor_set(v_reuseFailAlloc_1936_, 1, v___f_1920_);
lean_ctor_set(v_reuseFailAlloc_1936_, 2, v___f_1927_);
lean_ctor_set(v_reuseFailAlloc_1936_, 3, v___f_1926_);
lean_ctor_set(v_reuseFailAlloc_1936_, 4, v___f_1925_);
v___x_1929_ = v_reuseFailAlloc_1936_;
goto v_reusejp_1928_;
}
v_reusejp_1928_:
{
lean_object* v___x_1931_; 
if (v_isShared_1911_ == 0)
{
lean_ctor_set(v___x_1910_, 1, v___f_1921_);
lean_ctor_set(v___x_1910_, 0, v___x_1929_);
v___x_1931_ = v___x_1910_;
goto v_reusejp_1930_;
}
else
{
lean_object* v_reuseFailAlloc_1935_; 
v_reuseFailAlloc_1935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1935_, 0, v___x_1929_);
lean_ctor_set(v_reuseFailAlloc_1935_, 1, v___f_1921_);
v___x_1931_ = v_reuseFailAlloc_1935_;
goto v_reusejp_1930_;
}
v_reusejp_1930_:
{
lean_object* v___f_1932_; lean_object* v___x_51__overap_1933_; lean_object* v___x_1934_; 
lean_inc_ref(v___x_1931_);
lean_inc(v_goal_1868_);
v___f_1932_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_1932_, 0, v_goal_1868_);
lean_closure_set(v___f_1932_, 1, v_ri_1867_);
lean_closure_set(v___f_1932_, 2, v___x_1931_);
lean_closure_set(v___f_1932_, 3, v___f_1919_);
v___x_51__overap_1933_ = l_Lean_MVarId_withContext___redArg(v___x_1893_, v___x_1931_, v_goal_1868_, v___f_1932_);
lean_inc(v_a_1873_);
lean_inc_ref(v_a_1872_);
lean_inc(v_a_1871_);
lean_inc_ref(v_a_1870_);
v___x_1934_ = lean_apply_5(v___x_51__overap_1933_, v_a_1870_, v_a_1871_, v_a_1872_, v_a_1873_, lean_box(0));
return v___x_1934_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___boxed(lean_object* v_ri_1941_, lean_object* v_goal_1942_, lean_object* v_include_x3f_1943_, lean_object* v_a_1944_, lean_object* v_a_1945_, lean_object* v_a_1946_, lean_object* v_a_1947_, lean_object* v_a_1948_){
_start:
{
lean_object* v_res_1949_; 
v_res_1949_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg(v_ri_1941_, v_goal_1942_, v_include_x3f_1943_, v_a_1944_, v_a_1945_, v_a_1946_, v_a_1947_);
lean_dec(v_a_1947_);
lean_dec_ref(v_a_1946_);
lean_dec(v_a_1945_);
lean_dec_ref(v_a_1944_);
return v_res_1949_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules(lean_object* v_00_u03b1_1950_, lean_object* v_ri_1951_, lean_object* v_goal_1952_, lean_object* v_include_x3f_1953_, lean_object* v_a_1954_, lean_object* v_a_1955_, lean_object* v_a_1956_, lean_object* v_a_1957_){
_start:
{
lean_object* v___x_1959_; lean_object* v_toApplicative_1960_; lean_object* v_toFunctor_1961_; lean_object* v_toSeq_1962_; lean_object* v_toSeqLeft_1963_; lean_object* v_toSeqRight_1964_; lean_object* v___f_1965_; lean_object* v___f_1966_; lean_object* v___f_1967_; lean_object* v___f_1968_; lean_object* v___x_1969_; lean_object* v___f_1970_; lean_object* v___f_1971_; lean_object* v___f_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v_toApplicative_1978_; lean_object* v_toFunctor_1979_; lean_object* v_toSeq_1980_; lean_object* v_toSeqLeft_1981_; lean_object* v_toSeqRight_1982_; lean_object* v___f_1983_; lean_object* v___f_1984_; lean_object* v___x_1985_; lean_object* v___f_1986_; lean_object* v___f_1987_; lean_object* v___f_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v_toApplicative_1992_; lean_object* v___x_1994_; uint8_t v_isShared_1995_; uint8_t v_isSharedCheck_2023_; 
v___x_1959_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_1960_ = lean_ctor_get(v___x_1959_, 0);
v_toFunctor_1961_ = lean_ctor_get(v_toApplicative_1960_, 0);
v_toSeq_1962_ = lean_ctor_get(v_toApplicative_1960_, 2);
v_toSeqLeft_1963_ = lean_ctor_get(v_toApplicative_1960_, 3);
v_toSeqRight_1964_ = lean_ctor_get(v_toApplicative_1960_, 4);
v___f_1965_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_1966_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_1961_, 2);
v___f_1967_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1967_, 0, v_toFunctor_1961_);
v___f_1968_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1968_, 0, v_toFunctor_1961_);
v___x_1969_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1969_, 0, v___f_1967_);
lean_ctor_set(v___x_1969_, 1, v___f_1968_);
lean_inc(v_toSeqRight_1964_);
v___f_1970_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1970_, 0, v_toSeqRight_1964_);
lean_inc(v_toSeqLeft_1963_);
v___f_1971_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1971_, 0, v_toSeqLeft_1963_);
lean_inc(v_toSeq_1962_);
v___f_1972_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1972_, 0, v_toSeq_1962_);
v___x_1973_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1973_, 0, v___x_1969_);
lean_ctor_set(v___x_1973_, 1, v___f_1965_);
lean_ctor_set(v___x_1973_, 2, v___f_1972_);
lean_ctor_set(v___x_1973_, 3, v___f_1971_);
lean_ctor_set(v___x_1973_, 4, v___f_1970_);
v___x_1974_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1974_, 0, v___x_1973_);
lean_ctor_set(v___x_1974_, 1, v___f_1966_);
v___x_1975_ = l_StateRefT_x27_instMonad___redArg(v___x_1974_);
v___x_1976_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_1976_, 0, lean_box(0));
lean_closure_set(v___x_1976_, 1, lean_box(0));
lean_closure_set(v___x_1976_, 2, v___x_1975_);
v___x_1977_ = l_instMonadControlTOfPure___redArg(v___x_1976_);
v_toApplicative_1978_ = lean_ctor_get(v___x_1959_, 0);
v_toFunctor_1979_ = lean_ctor_get(v_toApplicative_1978_, 0);
v_toSeq_1980_ = lean_ctor_get(v_toApplicative_1978_, 2);
v_toSeqLeft_1981_ = lean_ctor_get(v_toApplicative_1978_, 3);
v_toSeqRight_1982_ = lean_ctor_get(v_toApplicative_1978_, 4);
lean_inc_ref_n(v_toFunctor_1979_, 2);
v___f_1983_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1983_, 0, v_toFunctor_1979_);
v___f_1984_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1984_, 0, v_toFunctor_1979_);
v___x_1985_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1985_, 0, v___f_1983_);
lean_ctor_set(v___x_1985_, 1, v___f_1984_);
lean_inc(v_toSeqRight_1982_);
v___f_1986_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1986_, 0, v_toSeqRight_1982_);
lean_inc(v_toSeqLeft_1981_);
v___f_1987_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1987_, 0, v_toSeqLeft_1981_);
lean_inc(v_toSeq_1980_);
v___f_1988_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1988_, 0, v_toSeq_1980_);
v___x_1989_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1989_, 0, v___x_1985_);
lean_ctor_set(v___x_1989_, 1, v___f_1965_);
lean_ctor_set(v___x_1989_, 2, v___f_1988_);
lean_ctor_set(v___x_1989_, 3, v___f_1987_);
lean_ctor_set(v___x_1989_, 4, v___f_1986_);
v___x_1990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1990_, 0, v___x_1989_);
lean_ctor_set(v___x_1990_, 1, v___f_1966_);
v___x_1991_ = l_StateRefT_x27_instMonad___redArg(v___x_1990_);
v_toApplicative_1992_ = lean_ctor_get(v___x_1991_, 0);
v_isSharedCheck_2023_ = !lean_is_exclusive(v___x_1991_);
if (v_isSharedCheck_2023_ == 0)
{
lean_object* v_unused_2024_; 
v_unused_2024_ = lean_ctor_get(v___x_1991_, 1);
lean_dec(v_unused_2024_);
v___x_1994_ = v___x_1991_;
v_isShared_1995_ = v_isSharedCheck_2023_;
goto v_resetjp_1993_;
}
else
{
lean_inc(v_toApplicative_1992_);
lean_dec(v___x_1991_);
v___x_1994_ = lean_box(0);
v_isShared_1995_ = v_isSharedCheck_2023_;
goto v_resetjp_1993_;
}
v_resetjp_1993_:
{
lean_object* v_toFunctor_1996_; lean_object* v_toSeq_1997_; lean_object* v_toSeqLeft_1998_; lean_object* v_toSeqRight_1999_; lean_object* v___x_2001_; uint8_t v_isShared_2002_; uint8_t v_isSharedCheck_2021_; 
v_toFunctor_1996_ = lean_ctor_get(v_toApplicative_1992_, 0);
v_toSeq_1997_ = lean_ctor_get(v_toApplicative_1992_, 2);
v_toSeqLeft_1998_ = lean_ctor_get(v_toApplicative_1992_, 3);
v_toSeqRight_1999_ = lean_ctor_get(v_toApplicative_1992_, 4);
v_isSharedCheck_2021_ = !lean_is_exclusive(v_toApplicative_1992_);
if (v_isSharedCheck_2021_ == 0)
{
lean_object* v_unused_2022_; 
v_unused_2022_ = lean_ctor_get(v_toApplicative_1992_, 1);
lean_dec(v_unused_2022_);
v___x_2001_ = v_toApplicative_1992_;
v_isShared_2002_ = v_isSharedCheck_2021_;
goto v_resetjp_2000_;
}
else
{
lean_inc(v_toSeqRight_1999_);
lean_inc(v_toSeqLeft_1998_);
lean_inc(v_toSeq_1997_);
lean_inc(v_toFunctor_1996_);
lean_dec(v_toApplicative_1992_);
v___x_2001_ = lean_box(0);
v_isShared_2002_ = v_isSharedCheck_2021_;
goto v_resetjp_2000_;
}
v_resetjp_2000_:
{
lean_object* v___f_2003_; lean_object* v___f_2004_; lean_object* v___f_2005_; lean_object* v___f_2006_; lean_object* v___f_2007_; lean_object* v___x_2008_; lean_object* v___f_2009_; lean_object* v___f_2010_; lean_object* v___f_2011_; lean_object* v___x_2013_; 
v___f_2003_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_2003_, 0, v_include_x3f_1953_);
v___f_2004_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0));
v___f_2005_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1));
lean_inc_ref(v_toFunctor_1996_);
v___f_2006_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2006_, 0, v_toFunctor_1996_);
v___f_2007_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2007_, 0, v_toFunctor_1996_);
v___x_2008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2008_, 0, v___f_2006_);
lean_ctor_set(v___x_2008_, 1, v___f_2007_);
v___f_2009_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2009_, 0, v_toSeqRight_1999_);
v___f_2010_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2010_, 0, v_toSeqLeft_1998_);
v___f_2011_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2011_, 0, v_toSeq_1997_);
if (v_isShared_2002_ == 0)
{
lean_ctor_set(v___x_2001_, 4, v___f_2009_);
lean_ctor_set(v___x_2001_, 3, v___f_2010_);
lean_ctor_set(v___x_2001_, 2, v___f_2011_);
lean_ctor_set(v___x_2001_, 1, v___f_2004_);
lean_ctor_set(v___x_2001_, 0, v___x_2008_);
v___x_2013_ = v___x_2001_;
goto v_reusejp_2012_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v___x_2008_);
lean_ctor_set(v_reuseFailAlloc_2020_, 1, v___f_2004_);
lean_ctor_set(v_reuseFailAlloc_2020_, 2, v___f_2011_);
lean_ctor_set(v_reuseFailAlloc_2020_, 3, v___f_2010_);
lean_ctor_set(v_reuseFailAlloc_2020_, 4, v___f_2009_);
v___x_2013_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2012_;
}
v_reusejp_2012_:
{
lean_object* v___x_2015_; 
if (v_isShared_1995_ == 0)
{
lean_ctor_set(v___x_1994_, 1, v___f_2005_);
lean_ctor_set(v___x_1994_, 0, v___x_2013_);
v___x_2015_ = v___x_1994_;
goto v_reusejp_2014_;
}
else
{
lean_object* v_reuseFailAlloc_2019_; 
v_reuseFailAlloc_2019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2019_, 0, v___x_2013_);
lean_ctor_set(v_reuseFailAlloc_2019_, 1, v___f_2005_);
v___x_2015_ = v_reuseFailAlloc_2019_;
goto v_reusejp_2014_;
}
v_reusejp_2014_:
{
lean_object* v___f_2016_; lean_object* v___x_834__overap_2017_; lean_object* v___x_2018_; 
lean_inc_ref(v___x_2015_);
lean_inc(v_goal_1952_);
v___f_2016_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_2016_, 0, v_goal_1952_);
lean_closure_set(v___f_2016_, 1, v_ri_1951_);
lean_closure_set(v___f_2016_, 2, v___x_2015_);
lean_closure_set(v___f_2016_, 3, v___f_2003_);
v___x_834__overap_2017_ = l_Lean_MVarId_withContext___redArg(v___x_1977_, v___x_2015_, v_goal_1952_, v___f_2016_);
lean_inc(v_a_1957_);
lean_inc_ref(v_a_1956_);
lean_inc(v_a_1955_);
lean_inc_ref(v_a_1954_);
v___x_2018_ = lean_apply_5(v___x_834__overap_2017_, v_a_1954_, v_a_1955_, v_a_1956_, v_a_1957_, lean_box(0));
return v___x_2018_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___boxed(lean_object* v_00_u03b1_2025_, lean_object* v_ri_2026_, lean_object* v_goal_2027_, lean_object* v_include_x3f_2028_, lean_object* v_a_2029_, lean_object* v_a_2030_, lean_object* v_a_2031_, lean_object* v_a_2032_, lean_object* v_a_2033_){
_start:
{
lean_object* v_res_2034_; 
v_res_2034_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules(v_00_u03b1_2025_, v_ri_2026_, v_goal_2027_, v_include_x3f_2028_, v_a_2029_, v_a_2030_, v_a_2031_, v_a_2032_);
lean_dec(v_a_2032_);
lean_dec_ref(v_a_2031_);
lean_dec(v_a_2030_);
lean_dec_ref(v_a_2029_);
return v_res_2034_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0(lean_object* v_include_x3f_2035_, lean_object* v_val_2036_, lean_object* v_a_2037_, lean_object* v_x_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_, lean_object* v___y_2042_, lean_object* v___y_2043_){
_start:
{
lean_object* v___x_2045_; uint8_t v___x_2046_; 
lean_inc_ref(v_a_2037_);
v___x_2045_ = lean_apply_1(v_include_x3f_2035_, v_a_2037_);
v___x_2046_ = lean_unbox(v___x_2045_);
if (v___x_2046_ == 0)
{
lean_object* v___x_2047_; lean_object* v___x_2048_; 
lean_dec_ref(v_a_2037_);
lean_dec_ref(v_val_2036_);
v___x_2047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2047_, 0, v___y_2039_);
v___x_2048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2048_, 0, v___x_2047_);
return v___x_2048_;
}
else
{
lean_object* v___x_2049_; lean_object* v___x_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; 
v___x_2049_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_2049_, 0, v_val_2036_);
v___x_2050_ = lean_unsigned_to_nat(1u);
v___x_2051_ = lean_mk_empty_array_with_capacity(v___x_2050_);
v___x_2052_ = lean_array_push(v___x_2051_, v___x_2049_);
v___x_2053_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2053_, 0, v_a_2037_);
lean_ctor_set(v___x_2053_, 1, v___x_2052_);
v___x_2054_ = lean_array_push(v___y_2039_, v___x_2053_);
v___x_2055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2055_, 0, v___x_2054_);
v___x_2056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2056_, 0, v___x_2055_);
return v___x_2056_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0___boxed(lean_object* v_include_x3f_2057_, lean_object* v_val_2058_, lean_object* v_a_2059_, lean_object* v_x_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
lean_object* v_res_2067_; 
v_res_2067_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0(v_include_x3f_2057_, v_val_2058_, v_a_2059_, v_x_2060_, v___y_2061_, v___y_2062_, v___y_2063_, v___y_2064_, v___y_2065_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
lean_dec(v___y_2063_);
lean_dec_ref(v___y_2062_);
return v_res_2067_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1(lean_object* v_ri_2068_, lean_object* v_include_x3f_2069_, lean_object* v___x_2070_, lean_object* v_d_x3f_2071_, lean_object* v_b_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_, lean_object* v___y_2076_){
_start:
{
if (lean_obj_tag(v_d_x3f_2071_) == 0)
{
lean_object* v___x_2078_; lean_object* v___x_2079_; 
lean_dec_ref(v___x_2070_);
lean_dec_ref(v_include_x3f_2069_);
lean_dec_ref(v_ri_2068_);
v___x_2078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2078_, 0, v_b_2072_);
v___x_2079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2079_, 0, v___x_2078_);
return v___x_2079_;
}
else
{
lean_object* v_val_2080_; lean_object* v___x_2082_; uint8_t v_isShared_2083_; uint8_t v_isSharedCheck_2125_; 
v_val_2080_ = lean_ctor_get(v_d_x3f_2071_, 0);
v_isSharedCheck_2125_ = !lean_is_exclusive(v_d_x3f_2071_);
if (v_isSharedCheck_2125_ == 0)
{
v___x_2082_ = v_d_x3f_2071_;
v_isShared_2083_ = v_isSharedCheck_2125_;
goto v_resetjp_2081_;
}
else
{
lean_inc(v_val_2080_);
lean_dec(v_d_x3f_2071_);
v___x_2082_ = lean_box(0);
v_isShared_2083_ = v_isSharedCheck_2125_;
goto v_resetjp_2081_;
}
v_resetjp_2081_:
{
uint8_t v___x_2084_; 
v___x_2084_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2080_);
if (v___x_2084_ == 0)
{
lean_object* v_byHyp_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; 
v_byHyp_2085_ = lean_ctor_get(v_ri_2068_, 1);
lean_inc_ref(v_byHyp_2085_);
lean_dec_ref(v_ri_2068_);
v___x_2086_ = l_Lean_LocalDecl_type(v_val_2080_);
v___x_2087_ = lp_aesop_Aesop_getUnify___redArg(v_byHyp_2085_, v___x_2086_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_);
if (lean_obj_tag(v___x_2087_) == 0)
{
lean_object* v_a_2088_; lean_object* v___f_2089_; size_t v_sz_2090_; size_t v___x_2091_; lean_object* v___x_1507__overap_2092_; lean_object* v___x_2093_; 
v_a_2088_ = lean_ctor_get(v___x_2087_, 0);
lean_inc(v_a_2088_);
lean_dec_ref_known(v___x_2087_, 1);
v___f_2089_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__0___boxed), 10, 2);
lean_closure_set(v___f_2089_, 0, v_include_x3f_2069_);
lean_closure_set(v___f_2089_, 1, v_val_2080_);
v_sz_2090_ = lean_array_size(v_a_2088_);
v___x_2091_ = ((size_t)0ULL);
v___x_1507__overap_2092_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_2070_, v_a_2088_, v___f_2089_, v_sz_2090_, v___x_2091_, v_b_2072_);
lean_inc(v___y_2076_);
lean_inc_ref(v___y_2075_);
lean_inc(v___y_2074_);
lean_inc_ref(v___y_2073_);
v___x_2093_ = lean_apply_5(v___x_1507__overap_2092_, v___y_2073_, v___y_2074_, v___y_2075_, v___y_2076_, lean_box(0));
if (lean_obj_tag(v___x_2093_) == 0)
{
lean_object* v_a_2094_; lean_object* v___x_2096_; uint8_t v_isShared_2097_; uint8_t v_isSharedCheck_2104_; 
v_a_2094_ = lean_ctor_get(v___x_2093_, 0);
v_isSharedCheck_2104_ = !lean_is_exclusive(v___x_2093_);
if (v_isSharedCheck_2104_ == 0)
{
v___x_2096_ = v___x_2093_;
v_isShared_2097_ = v_isSharedCheck_2104_;
goto v_resetjp_2095_;
}
else
{
lean_inc(v_a_2094_);
lean_dec(v___x_2093_);
v___x_2096_ = lean_box(0);
v_isShared_2097_ = v_isSharedCheck_2104_;
goto v_resetjp_2095_;
}
v_resetjp_2095_:
{
lean_object* v___x_2099_; 
if (v_isShared_2083_ == 0)
{
lean_ctor_set(v___x_2082_, 0, v_a_2094_);
v___x_2099_ = v___x_2082_;
goto v_reusejp_2098_;
}
else
{
lean_object* v_reuseFailAlloc_2103_; 
v_reuseFailAlloc_2103_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2103_, 0, v_a_2094_);
v___x_2099_ = v_reuseFailAlloc_2103_;
goto v_reusejp_2098_;
}
v_reusejp_2098_:
{
lean_object* v___x_2101_; 
if (v_isShared_2097_ == 0)
{
lean_ctor_set(v___x_2096_, 0, v___x_2099_);
v___x_2101_ = v___x_2096_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2102_; 
v_reuseFailAlloc_2102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2102_, 0, v___x_2099_);
v___x_2101_ = v_reuseFailAlloc_2102_;
goto v_reusejp_2100_;
}
v_reusejp_2100_:
{
return v___x_2101_;
}
}
}
}
else
{
lean_object* v_a_2105_; lean_object* v___x_2107_; uint8_t v_isShared_2108_; uint8_t v_isSharedCheck_2112_; 
lean_del_object(v___x_2082_);
v_a_2105_ = lean_ctor_get(v___x_2093_, 0);
v_isSharedCheck_2112_ = !lean_is_exclusive(v___x_2093_);
if (v_isSharedCheck_2112_ == 0)
{
v___x_2107_ = v___x_2093_;
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
else
{
lean_inc(v_a_2105_);
lean_dec(v___x_2093_);
v___x_2107_ = lean_box(0);
v_isShared_2108_ = v_isSharedCheck_2112_;
goto v_resetjp_2106_;
}
v_resetjp_2106_:
{
lean_object* v___x_2110_; 
if (v_isShared_2108_ == 0)
{
v___x_2110_ = v___x_2107_;
goto v_reusejp_2109_;
}
else
{
lean_object* v_reuseFailAlloc_2111_; 
v_reuseFailAlloc_2111_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2111_, 0, v_a_2105_);
v___x_2110_ = v_reuseFailAlloc_2111_;
goto v_reusejp_2109_;
}
v_reusejp_2109_:
{
return v___x_2110_;
}
}
}
}
else
{
lean_object* v_a_2113_; lean_object* v___x_2115_; uint8_t v_isShared_2116_; uint8_t v_isSharedCheck_2120_; 
lean_del_object(v___x_2082_);
lean_dec(v_val_2080_);
lean_dec_ref(v_b_2072_);
lean_dec_ref(v___x_2070_);
lean_dec_ref(v_include_x3f_2069_);
v_a_2113_ = lean_ctor_get(v___x_2087_, 0);
v_isSharedCheck_2120_ = !lean_is_exclusive(v___x_2087_);
if (v_isSharedCheck_2120_ == 0)
{
v___x_2115_ = v___x_2087_;
v_isShared_2116_ = v_isSharedCheck_2120_;
goto v_resetjp_2114_;
}
else
{
lean_inc(v_a_2113_);
lean_dec(v___x_2087_);
v___x_2115_ = lean_box(0);
v_isShared_2116_ = v_isSharedCheck_2120_;
goto v_resetjp_2114_;
}
v_resetjp_2114_:
{
lean_object* v___x_2118_; 
if (v_isShared_2116_ == 0)
{
v___x_2118_ = v___x_2115_;
goto v_reusejp_2117_;
}
else
{
lean_object* v_reuseFailAlloc_2119_; 
v_reuseFailAlloc_2119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2119_, 0, v_a_2113_);
v___x_2118_ = v_reuseFailAlloc_2119_;
goto v_reusejp_2117_;
}
v_reusejp_2117_:
{
return v___x_2118_;
}
}
}
}
else
{
lean_object* v___x_2122_; 
lean_dec(v_val_2080_);
lean_dec_ref(v___x_2070_);
lean_dec_ref(v_include_x3f_2069_);
lean_dec_ref(v_ri_2068_);
if (v_isShared_2083_ == 0)
{
lean_ctor_set(v___x_2082_, 0, v_b_2072_);
v___x_2122_ = v___x_2082_;
goto v_reusejp_2121_;
}
else
{
lean_object* v_reuseFailAlloc_2124_; 
v_reuseFailAlloc_2124_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2124_, 0, v_b_2072_);
v___x_2122_ = v_reuseFailAlloc_2124_;
goto v_reusejp_2121_;
}
v_reusejp_2121_:
{
lean_object* v___x_2123_; 
v___x_2123_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2123_, 0, v___x_2122_);
return v___x_2123_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed(lean_object* v_ri_2126_, lean_object* v_include_x3f_2127_, lean_object* v___x_2128_, lean_object* v_d_x3f_2129_, lean_object* v_b_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_){
_start:
{
lean_object* v_res_2136_; 
v_res_2136_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1(v_ri_2126_, v_include_x3f_2127_, v___x_2128_, v_d_x3f_2129_, v_b_2130_, v___y_2131_, v___y_2132_, v___y_2133_, v___y_2134_);
lean_dec(v___y_2134_);
lean_dec_ref(v___y_2133_);
lean_dec(v___y_2132_);
lean_dec_ref(v___y_2131_);
return v_res_2136_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2(lean_object* v___x_2137_, lean_object* v_rs_2138_, lean_object* v___f_2139_, lean_object* v___y_2140_, lean_object* v___y_2141_, lean_object* v___y_2142_, lean_object* v___y_2143_){
_start:
{
lean_object* v_lctx_2145_; lean_object* v_decls_2146_; lean_object* v___x_1527__overap_2147_; lean_object* v___x_2148_; 
v_lctx_2145_ = lean_ctor_get(v___y_2140_, 2);
v_decls_2146_ = lean_ctor_get(v_lctx_2145_, 1);
v___x_1527__overap_2147_ = l_Lean_PersistentArray_forIn___redArg(v___x_2137_, v_decls_2146_, v_rs_2138_, v___f_2139_);
v___x_2148_ = lean_apply_5(v___x_1527__overap_2147_, v___y_2140_, v___y_2141_, v___y_2142_, v___y_2143_, lean_box(0));
return v___x_2148_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed(lean_object* v___x_2149_, lean_object* v_rs_2150_, lean_object* v___f_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_){
_start:
{
lean_object* v_res_2157_; 
v_res_2157_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2(v___x_2149_, v_rs_2150_, v___f_2151_, v___y_2152_, v___y_2153_, v___y_2154_, v___y_2155_);
return v_res_2157_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg(lean_object* v_ri_2160_, lean_object* v_goal_2161_, lean_object* v_include_x3f_2162_, lean_object* v_a_2163_, lean_object* v_a_2164_, lean_object* v_a_2165_, lean_object* v_a_2166_){
_start:
{
lean_object* v___x_2168_; lean_object* v_toApplicative_2169_; lean_object* v_toFunctor_2170_; lean_object* v_toSeq_2171_; lean_object* v_toSeqLeft_2172_; lean_object* v_toSeqRight_2173_; lean_object* v___f_2174_; lean_object* v___f_2175_; lean_object* v___f_2176_; lean_object* v___f_2177_; lean_object* v___x_2178_; lean_object* v___f_2179_; lean_object* v___f_2180_; lean_object* v___f_2181_; lean_object* v___x_2182_; lean_object* v___x_2183_; lean_object* v___x_2184_; lean_object* v___x_2185_; lean_object* v___x_2186_; lean_object* v_toApplicative_2187_; lean_object* v_toFunctor_2188_; lean_object* v_toSeq_2189_; lean_object* v_toSeqLeft_2190_; lean_object* v_toSeqRight_2191_; lean_object* v___f_2192_; lean_object* v___f_2193_; lean_object* v___x_2194_; lean_object* v___f_2195_; lean_object* v___f_2196_; lean_object* v___f_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v_toApplicative_2201_; lean_object* v___x_2203_; uint8_t v_isShared_2204_; uint8_t v_isSharedCheck_2233_; 
v___x_2168_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_2169_ = lean_ctor_get(v___x_2168_, 0);
v_toFunctor_2170_ = lean_ctor_get(v_toApplicative_2169_, 0);
v_toSeq_2171_ = lean_ctor_get(v_toApplicative_2169_, 2);
v_toSeqLeft_2172_ = lean_ctor_get(v_toApplicative_2169_, 3);
v_toSeqRight_2173_ = lean_ctor_get(v_toApplicative_2169_, 4);
v___f_2174_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_2175_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_2170_, 2);
v___f_2176_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2176_, 0, v_toFunctor_2170_);
v___f_2177_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2177_, 0, v_toFunctor_2170_);
v___x_2178_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2178_, 0, v___f_2176_);
lean_ctor_set(v___x_2178_, 1, v___f_2177_);
lean_inc(v_toSeqRight_2173_);
v___f_2179_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2179_, 0, v_toSeqRight_2173_);
lean_inc(v_toSeqLeft_2172_);
v___f_2180_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2180_, 0, v_toSeqLeft_2172_);
lean_inc(v_toSeq_2171_);
v___f_2181_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2181_, 0, v_toSeq_2171_);
v___x_2182_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2182_, 0, v___x_2178_);
lean_ctor_set(v___x_2182_, 1, v___f_2174_);
lean_ctor_set(v___x_2182_, 2, v___f_2181_);
lean_ctor_set(v___x_2182_, 3, v___f_2180_);
lean_ctor_set(v___x_2182_, 4, v___f_2179_);
v___x_2183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___x_2182_);
lean_ctor_set(v___x_2183_, 1, v___f_2175_);
v___x_2184_ = l_StateRefT_x27_instMonad___redArg(v___x_2183_);
v___x_2185_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_2185_, 0, lean_box(0));
lean_closure_set(v___x_2185_, 1, lean_box(0));
lean_closure_set(v___x_2185_, 2, v___x_2184_);
v___x_2186_ = l_instMonadControlTOfPure___redArg(v___x_2185_);
v_toApplicative_2187_ = lean_ctor_get(v___x_2168_, 0);
v_toFunctor_2188_ = lean_ctor_get(v_toApplicative_2187_, 0);
v_toSeq_2189_ = lean_ctor_get(v_toApplicative_2187_, 2);
v_toSeqLeft_2190_ = lean_ctor_get(v_toApplicative_2187_, 3);
v_toSeqRight_2191_ = lean_ctor_get(v_toApplicative_2187_, 4);
lean_inc_ref_n(v_toFunctor_2188_, 2);
v___f_2192_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2192_, 0, v_toFunctor_2188_);
v___f_2193_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2193_, 0, v_toFunctor_2188_);
v___x_2194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___f_2192_);
lean_ctor_set(v___x_2194_, 1, v___f_2193_);
lean_inc(v_toSeqRight_2191_);
v___f_2195_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2195_, 0, v_toSeqRight_2191_);
lean_inc(v_toSeqLeft_2190_);
v___f_2196_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2196_, 0, v_toSeqLeft_2190_);
lean_inc(v_toSeq_2189_);
v___f_2197_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2197_, 0, v_toSeq_2189_);
v___x_2198_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2198_, 0, v___x_2194_);
lean_ctor_set(v___x_2198_, 1, v___f_2174_);
lean_ctor_set(v___x_2198_, 2, v___f_2197_);
lean_ctor_set(v___x_2198_, 3, v___f_2196_);
lean_ctor_set(v___x_2198_, 4, v___f_2195_);
v___x_2199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2199_, 0, v___x_2198_);
lean_ctor_set(v___x_2199_, 1, v___f_2175_);
v___x_2200_ = l_StateRefT_x27_instMonad___redArg(v___x_2199_);
v_toApplicative_2201_ = lean_ctor_get(v___x_2200_, 0);
v_isSharedCheck_2233_ = !lean_is_exclusive(v___x_2200_);
if (v_isSharedCheck_2233_ == 0)
{
lean_object* v_unused_2234_; 
v_unused_2234_ = lean_ctor_get(v___x_2200_, 1);
lean_dec(v_unused_2234_);
v___x_2203_ = v___x_2200_;
v_isShared_2204_ = v_isSharedCheck_2233_;
goto v_resetjp_2202_;
}
else
{
lean_inc(v_toApplicative_2201_);
lean_dec(v___x_2200_);
v___x_2203_ = lean_box(0);
v_isShared_2204_ = v_isSharedCheck_2233_;
goto v_resetjp_2202_;
}
v_resetjp_2202_:
{
lean_object* v_toFunctor_2205_; lean_object* v_toSeq_2206_; lean_object* v_toSeqLeft_2207_; lean_object* v_toSeqRight_2208_; lean_object* v___x_2210_; uint8_t v_isShared_2211_; uint8_t v_isSharedCheck_2231_; 
v_toFunctor_2205_ = lean_ctor_get(v_toApplicative_2201_, 0);
v_toSeq_2206_ = lean_ctor_get(v_toApplicative_2201_, 2);
v_toSeqLeft_2207_ = lean_ctor_get(v_toApplicative_2201_, 3);
v_toSeqRight_2208_ = lean_ctor_get(v_toApplicative_2201_, 4);
v_isSharedCheck_2231_ = !lean_is_exclusive(v_toApplicative_2201_);
if (v_isSharedCheck_2231_ == 0)
{
lean_object* v_unused_2232_; 
v_unused_2232_ = lean_ctor_get(v_toApplicative_2201_, 1);
lean_dec(v_unused_2232_);
v___x_2210_ = v_toApplicative_2201_;
v_isShared_2211_ = v_isSharedCheck_2231_;
goto v_resetjp_2209_;
}
else
{
lean_inc(v_toSeqRight_2208_);
lean_inc(v_toSeqLeft_2207_);
lean_inc(v_toSeq_2206_);
lean_inc(v_toFunctor_2205_);
lean_dec(v_toApplicative_2201_);
v___x_2210_ = lean_box(0);
v_isShared_2211_ = v_isSharedCheck_2231_;
goto v_resetjp_2209_;
}
v_resetjp_2209_:
{
lean_object* v___f_2212_; lean_object* v___f_2213_; lean_object* v___f_2214_; lean_object* v___f_2215_; lean_object* v___x_2216_; lean_object* v___f_2217_; lean_object* v___f_2218_; lean_object* v___f_2219_; lean_object* v___x_2221_; 
v___f_2212_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0));
v___f_2213_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1));
lean_inc_ref(v_toFunctor_2205_);
v___f_2214_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2214_, 0, v_toFunctor_2205_);
v___f_2215_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2215_, 0, v_toFunctor_2205_);
v___x_2216_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2216_, 0, v___f_2214_);
lean_ctor_set(v___x_2216_, 1, v___f_2215_);
v___f_2217_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2217_, 0, v_toSeqRight_2208_);
v___f_2218_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2218_, 0, v_toSeqLeft_2207_);
v___f_2219_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2219_, 0, v_toSeq_2206_);
if (v_isShared_2211_ == 0)
{
lean_ctor_set(v___x_2210_, 4, v___f_2217_);
lean_ctor_set(v___x_2210_, 3, v___f_2218_);
lean_ctor_set(v___x_2210_, 2, v___f_2219_);
lean_ctor_set(v___x_2210_, 1, v___f_2212_);
lean_ctor_set(v___x_2210_, 0, v___x_2216_);
v___x_2221_ = v___x_2210_;
goto v_reusejp_2220_;
}
else
{
lean_object* v_reuseFailAlloc_2230_; 
v_reuseFailAlloc_2230_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2230_, 0, v___x_2216_);
lean_ctor_set(v_reuseFailAlloc_2230_, 1, v___f_2212_);
lean_ctor_set(v_reuseFailAlloc_2230_, 2, v___f_2219_);
lean_ctor_set(v_reuseFailAlloc_2230_, 3, v___f_2218_);
lean_ctor_set(v_reuseFailAlloc_2230_, 4, v___f_2217_);
v___x_2221_ = v_reuseFailAlloc_2230_;
goto v_reusejp_2220_;
}
v_reusejp_2220_:
{
lean_object* v___x_2223_; 
if (v_isShared_2204_ == 0)
{
lean_ctor_set(v___x_2203_, 1, v___f_2213_);
lean_ctor_set(v___x_2203_, 0, v___x_2221_);
v___x_2223_ = v___x_2203_;
goto v_reusejp_2222_;
}
else
{
lean_object* v_reuseFailAlloc_2229_; 
v_reuseFailAlloc_2229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2229_, 0, v___x_2221_);
lean_ctor_set(v_reuseFailAlloc_2229_, 1, v___f_2213_);
v___x_2223_ = v_reuseFailAlloc_2229_;
goto v_reusejp_2222_;
}
v_reusejp_2222_:
{
lean_object* v___f_2224_; lean_object* v_rs_2225_; lean_object* v___f_2226_; lean_object* v___x_72__overap_2227_; lean_object* v___x_2228_; 
lean_inc_ref_n(v___x_2223_, 2);
v___f_2224_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed), 10, 3);
lean_closure_set(v___f_2224_, 0, v_ri_2160_);
lean_closure_set(v___f_2224_, 1, v_include_x3f_2162_);
lean_closure_set(v___f_2224_, 2, v___x_2223_);
v_rs_2225_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
v___f_2226_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_2226_, 0, v___x_2223_);
lean_closure_set(v___f_2226_, 1, v_rs_2225_);
lean_closure_set(v___f_2226_, 2, v___f_2224_);
v___x_72__overap_2227_ = l_Lean_MVarId_withContext___redArg(v___x_2186_, v___x_2223_, v_goal_2161_, v___f_2226_);
lean_inc(v_a_2166_);
lean_inc_ref(v_a_2165_);
lean_inc(v_a_2164_);
lean_inc_ref(v_a_2163_);
v___x_2228_ = lean_apply_5(v___x_72__overap_2227_, v_a_2163_, v_a_2164_, v_a_2165_, v_a_2166_, lean_box(0));
return v___x_2228_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___boxed(lean_object* v_ri_2235_, lean_object* v_goal_2236_, lean_object* v_include_x3f_2237_, lean_object* v_a_2238_, lean_object* v_a_2239_, lean_object* v_a_2240_, lean_object* v_a_2241_, lean_object* v_a_2242_){
_start:
{
lean_object* v_res_2243_; 
v_res_2243_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg(v_ri_2235_, v_goal_2236_, v_include_x3f_2237_, v_a_2238_, v_a_2239_, v_a_2240_, v_a_2241_);
lean_dec(v_a_2241_);
lean_dec_ref(v_a_2240_);
lean_dec(v_a_2239_);
lean_dec_ref(v_a_2238_);
return v_res_2243_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules(lean_object* v_00_u03b1_2244_, lean_object* v_ri_2245_, lean_object* v_goal_2246_, lean_object* v_include_x3f_2247_, lean_object* v_a_2248_, lean_object* v_a_2249_, lean_object* v_a_2250_, lean_object* v_a_2251_){
_start:
{
lean_object* v___x_2253_; lean_object* v_toApplicative_2254_; lean_object* v_toFunctor_2255_; lean_object* v_toSeq_2256_; lean_object* v_toSeqLeft_2257_; lean_object* v_toSeqRight_2258_; lean_object* v___f_2259_; lean_object* v___f_2260_; lean_object* v___f_2261_; lean_object* v___f_2262_; lean_object* v___x_2263_; lean_object* v___f_2264_; lean_object* v___f_2265_; lean_object* v___f_2266_; lean_object* v___x_2267_; lean_object* v___x_2268_; lean_object* v___x_2269_; lean_object* v___x_2270_; lean_object* v___x_2271_; lean_object* v_toApplicative_2272_; lean_object* v_toFunctor_2273_; lean_object* v_toSeq_2274_; lean_object* v_toSeqLeft_2275_; lean_object* v_toSeqRight_2276_; lean_object* v___f_2277_; lean_object* v___f_2278_; lean_object* v___x_2279_; lean_object* v___f_2280_; lean_object* v___f_2281_; lean_object* v___f_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v_toApplicative_2286_; lean_object* v___x_2288_; uint8_t v_isShared_2289_; uint8_t v_isSharedCheck_2318_; 
v___x_2253_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_2254_ = lean_ctor_get(v___x_2253_, 0);
v_toFunctor_2255_ = lean_ctor_get(v_toApplicative_2254_, 0);
v_toSeq_2256_ = lean_ctor_get(v_toApplicative_2254_, 2);
v_toSeqLeft_2257_ = lean_ctor_get(v_toApplicative_2254_, 3);
v_toSeqRight_2258_ = lean_ctor_get(v_toApplicative_2254_, 4);
v___f_2259_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_2260_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_2255_, 2);
v___f_2261_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2261_, 0, v_toFunctor_2255_);
v___f_2262_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2262_, 0, v_toFunctor_2255_);
v___x_2263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2263_, 0, v___f_2261_);
lean_ctor_set(v___x_2263_, 1, v___f_2262_);
lean_inc(v_toSeqRight_2258_);
v___f_2264_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2264_, 0, v_toSeqRight_2258_);
lean_inc(v_toSeqLeft_2257_);
v___f_2265_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2265_, 0, v_toSeqLeft_2257_);
lean_inc(v_toSeq_2256_);
v___f_2266_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2266_, 0, v_toSeq_2256_);
v___x_2267_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2267_, 0, v___x_2263_);
lean_ctor_set(v___x_2267_, 1, v___f_2259_);
lean_ctor_set(v___x_2267_, 2, v___f_2266_);
lean_ctor_set(v___x_2267_, 3, v___f_2265_);
lean_ctor_set(v___x_2267_, 4, v___f_2264_);
v___x_2268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2268_, 0, v___x_2267_);
lean_ctor_set(v___x_2268_, 1, v___f_2260_);
v___x_2269_ = l_StateRefT_x27_instMonad___redArg(v___x_2268_);
v___x_2270_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_2270_, 0, lean_box(0));
lean_closure_set(v___x_2270_, 1, lean_box(0));
lean_closure_set(v___x_2270_, 2, v___x_2269_);
v___x_2271_ = l_instMonadControlTOfPure___redArg(v___x_2270_);
v_toApplicative_2272_ = lean_ctor_get(v___x_2253_, 0);
v_toFunctor_2273_ = lean_ctor_get(v_toApplicative_2272_, 0);
v_toSeq_2274_ = lean_ctor_get(v_toApplicative_2272_, 2);
v_toSeqLeft_2275_ = lean_ctor_get(v_toApplicative_2272_, 3);
v_toSeqRight_2276_ = lean_ctor_get(v_toApplicative_2272_, 4);
lean_inc_ref_n(v_toFunctor_2273_, 2);
v___f_2277_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2277_, 0, v_toFunctor_2273_);
v___f_2278_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2278_, 0, v_toFunctor_2273_);
v___x_2279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2279_, 0, v___f_2277_);
lean_ctor_set(v___x_2279_, 1, v___f_2278_);
lean_inc(v_toSeqRight_2276_);
v___f_2280_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2280_, 0, v_toSeqRight_2276_);
lean_inc(v_toSeqLeft_2275_);
v___f_2281_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2281_, 0, v_toSeqLeft_2275_);
lean_inc(v_toSeq_2274_);
v___f_2282_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2282_, 0, v_toSeq_2274_);
v___x_2283_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2283_, 0, v___x_2279_);
lean_ctor_set(v___x_2283_, 1, v___f_2259_);
lean_ctor_set(v___x_2283_, 2, v___f_2282_);
lean_ctor_set(v___x_2283_, 3, v___f_2281_);
lean_ctor_set(v___x_2283_, 4, v___f_2280_);
v___x_2284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2284_, 0, v___x_2283_);
lean_ctor_set(v___x_2284_, 1, v___f_2260_);
v___x_2285_ = l_StateRefT_x27_instMonad___redArg(v___x_2284_);
v_toApplicative_2286_ = lean_ctor_get(v___x_2285_, 0);
v_isSharedCheck_2318_ = !lean_is_exclusive(v___x_2285_);
if (v_isSharedCheck_2318_ == 0)
{
lean_object* v_unused_2319_; 
v_unused_2319_ = lean_ctor_get(v___x_2285_, 1);
lean_dec(v_unused_2319_);
v___x_2288_ = v___x_2285_;
v_isShared_2289_ = v_isSharedCheck_2318_;
goto v_resetjp_2287_;
}
else
{
lean_inc(v_toApplicative_2286_);
lean_dec(v___x_2285_);
v___x_2288_ = lean_box(0);
v_isShared_2289_ = v_isSharedCheck_2318_;
goto v_resetjp_2287_;
}
v_resetjp_2287_:
{
lean_object* v_toFunctor_2290_; lean_object* v_toSeq_2291_; lean_object* v_toSeqLeft_2292_; lean_object* v_toSeqRight_2293_; lean_object* v___x_2295_; uint8_t v_isShared_2296_; uint8_t v_isSharedCheck_2316_; 
v_toFunctor_2290_ = lean_ctor_get(v_toApplicative_2286_, 0);
v_toSeq_2291_ = lean_ctor_get(v_toApplicative_2286_, 2);
v_toSeqLeft_2292_ = lean_ctor_get(v_toApplicative_2286_, 3);
v_toSeqRight_2293_ = lean_ctor_get(v_toApplicative_2286_, 4);
v_isSharedCheck_2316_ = !lean_is_exclusive(v_toApplicative_2286_);
if (v_isSharedCheck_2316_ == 0)
{
lean_object* v_unused_2317_; 
v_unused_2317_ = lean_ctor_get(v_toApplicative_2286_, 1);
lean_dec(v_unused_2317_);
v___x_2295_ = v_toApplicative_2286_;
v_isShared_2296_ = v_isSharedCheck_2316_;
goto v_resetjp_2294_;
}
else
{
lean_inc(v_toSeqRight_2293_);
lean_inc(v_toSeqLeft_2292_);
lean_inc(v_toSeq_2291_);
lean_inc(v_toFunctor_2290_);
lean_dec(v_toApplicative_2286_);
v___x_2295_ = lean_box(0);
v_isShared_2296_ = v_isSharedCheck_2316_;
goto v_resetjp_2294_;
}
v_resetjp_2294_:
{
lean_object* v___f_2297_; lean_object* v___f_2298_; lean_object* v___f_2299_; lean_object* v___f_2300_; lean_object* v___x_2301_; lean_object* v___f_2302_; lean_object* v___f_2303_; lean_object* v___f_2304_; lean_object* v___x_2306_; 
v___f_2297_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0));
v___f_2298_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1));
lean_inc_ref(v_toFunctor_2290_);
v___f_2299_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2299_, 0, v_toFunctor_2290_);
v___f_2300_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2300_, 0, v_toFunctor_2290_);
v___x_2301_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2301_, 0, v___f_2299_);
lean_ctor_set(v___x_2301_, 1, v___f_2300_);
v___f_2302_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2302_, 0, v_toSeqRight_2293_);
v___f_2303_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2303_, 0, v_toSeqLeft_2292_);
v___f_2304_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2304_, 0, v_toSeq_2291_);
if (v_isShared_2296_ == 0)
{
lean_ctor_set(v___x_2295_, 4, v___f_2302_);
lean_ctor_set(v___x_2295_, 3, v___f_2303_);
lean_ctor_set(v___x_2295_, 2, v___f_2304_);
lean_ctor_set(v___x_2295_, 1, v___f_2297_);
lean_ctor_set(v___x_2295_, 0, v___x_2301_);
v___x_2306_ = v___x_2295_;
goto v_reusejp_2305_;
}
else
{
lean_object* v_reuseFailAlloc_2315_; 
v_reuseFailAlloc_2315_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2315_, 0, v___x_2301_);
lean_ctor_set(v_reuseFailAlloc_2315_, 1, v___f_2297_);
lean_ctor_set(v_reuseFailAlloc_2315_, 2, v___f_2304_);
lean_ctor_set(v_reuseFailAlloc_2315_, 3, v___f_2303_);
lean_ctor_set(v_reuseFailAlloc_2315_, 4, v___f_2302_);
v___x_2306_ = v_reuseFailAlloc_2315_;
goto v_reusejp_2305_;
}
v_reusejp_2305_:
{
lean_object* v___x_2308_; 
if (v_isShared_2289_ == 0)
{
lean_ctor_set(v___x_2288_, 1, v___f_2298_);
lean_ctor_set(v___x_2288_, 0, v___x_2306_);
v___x_2308_ = v___x_2288_;
goto v_reusejp_2307_;
}
else
{
lean_object* v_reuseFailAlloc_2314_; 
v_reuseFailAlloc_2314_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2314_, 0, v___x_2306_);
lean_ctor_set(v_reuseFailAlloc_2314_, 1, v___f_2298_);
v___x_2308_ = v_reuseFailAlloc_2314_;
goto v_reusejp_2307_;
}
v_reusejp_2307_:
{
lean_object* v___f_2309_; lean_object* v_rs_2310_; lean_object* v___f_2311_; lean_object* v___x_1471__overap_2312_; lean_object* v___x_2313_; 
lean_inc_ref_n(v___x_2308_, 2);
v___f_2309_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed), 10, 3);
lean_closure_set(v___f_2309_, 0, v_ri_2245_);
lean_closure_set(v___f_2309_, 1, v_include_x3f_2247_);
lean_closure_set(v___f_2309_, 2, v___x_2308_);
v_rs_2310_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
v___f_2311_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_2311_, 0, v___x_2308_);
lean_closure_set(v___f_2311_, 1, v_rs_2310_);
lean_closure_set(v___f_2311_, 2, v___f_2309_);
v___x_1471__overap_2312_ = l_Lean_MVarId_withContext___redArg(v___x_2271_, v___x_2308_, v_goal_2246_, v___f_2311_);
lean_inc(v_a_2251_);
lean_inc_ref(v_a_2250_);
lean_inc(v_a_2249_);
lean_inc_ref(v_a_2248_);
v___x_2313_ = lean_apply_5(v___x_1471__overap_2312_, v_a_2248_, v_a_2249_, v_a_2250_, v_a_2251_, lean_box(0));
return v___x_2313_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___boxed(lean_object* v_00_u03b1_2320_, lean_object* v_ri_2321_, lean_object* v_goal_2322_, lean_object* v_include_x3f_2323_, lean_object* v_a_2324_, lean_object* v_a_2325_, lean_object* v_a_2326_, lean_object* v_a_2327_, lean_object* v_a_2328_){
_start:
{
lean_object* v_res_2329_; 
v_res_2329_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules(v_00_u03b1_2320_, v_ri_2321_, v_goal_2322_, v_include_x3f_2323_, v_a_2324_, v_a_2325_, v_a_2326_, v_a_2327_);
lean_dec(v_a_2327_);
lean_dec_ref(v_a_2326_);
lean_dec(v_a_2325_);
lean_dec_ref(v_a_2324_);
return v_res_2329_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0(lean_object* v_include_x3f_2334_, lean_object* v_d_2335_, lean_object* v_a_2336_, lean_object* v_x_2337_){
_start:
{
lean_object* v___x_2338_; uint8_t v___x_2339_; 
lean_inc_ref(v_a_2336_);
v___x_2338_ = lean_apply_1(v_include_x3f_2334_, v_a_2336_);
v___x_2339_ = lean_unbox(v___x_2338_);
if (v___x_2339_ == 0)
{
lean_dec_ref(v_a_2336_);
return v_d_2335_;
}
else
{
lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0___closed__0));
v___x_2341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2341_, 0, v_a_2336_);
lean_ctor_set(v___x_2341_, 1, v___x_2340_);
v___x_2342_ = lean_array_push(v_d_2335_, v___x_2341_);
return v___x_2342_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg(lean_object* v_ri_2343_, lean_object* v_include_x3f_2344_){
_start:
{
lean_object* v_unindexed_2345_; lean_object* v___f_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; 
v_unindexed_2345_ = lean_ctor_get(v_ri_2343_, 2);
lean_inc_ref(v_unindexed_2345_);
lean_dec_ref(v_ri_2343_);
v___f_2346_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2346_, 0, v_include_x3f_2344_);
v___x_2347_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
v___x_2348_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_2349_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_2348_, v___f_2346_, v_unindexed_2345_, v___x_2347_);
return v___x_2349_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules(lean_object* v_00_u03b1_2350_, lean_object* v_ri_2351_, lean_object* v_include_x3f_2352_){
_start:
{
lean_object* v_unindexed_2353_; lean_object* v___f_2354_; lean_object* v___x_2355_; lean_object* v___x_2356_; lean_object* v___x_2357_; 
v_unindexed_2353_ = lean_ctor_get(v_ri_2351_, 2);
lean_inc_ref(v_unindexed_2353_);
lean_dec_ref(v_ri_2351_);
v___f_2354_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0), 4, 1);
lean_closure_set(v___f_2354_, 0, v_include_x3f_2352_);
v___x_2355_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
v___x_2356_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_2357_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_2356_, v___f_2354_, v_unindexed_2353_, v___x_2355_);
return v___x_2357_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(uint8_t v___x_2358_, lean_object* v_x_2359_, lean_object* v_y_2360_){
_start:
{
switch(lean_obj_tag(v_x_2359_))
{
case 0:
{
if (lean_obj_tag(v_y_2360_) == 2)
{
return v___x_2358_;
}
else
{
uint8_t v___x_2361_; 
v___x_2361_ = 0;
return v___x_2361_;
}
}
case 1:
{
if (lean_obj_tag(v_y_2360_) == 1)
{
uint8_t v___x_2362_; 
v___x_2362_ = 0;
return v___x_2362_;
}
else
{
return v___x_2358_;
}
}
default: 
{
if (lean_obj_tag(v_y_2360_) == 2)
{
lean_object* v_ldecl_2363_; lean_object* v_ldecl_2364_; lean_object* v___x_2365_; lean_object* v___x_2366_; uint8_t v___x_2367_; 
v_ldecl_2363_ = lean_ctor_get(v_x_2359_, 0);
v_ldecl_2364_ = lean_ctor_get(v_y_2360_, 0);
v___x_2365_ = l_Lean_LocalDecl_index(v_ldecl_2363_);
v___x_2366_ = l_Lean_LocalDecl_index(v_ldecl_2364_);
v___x_2367_ = lean_nat_dec_lt(v___x_2365_, v___x_2366_);
lean_dec(v___x_2366_);
lean_dec(v___x_2365_);
return v___x_2367_;
}
else
{
uint8_t v___x_2368_; 
v___x_2368_ = 0;
return v___x_2368_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v___x_2369_, lean_object* v_x_2370_, lean_object* v_y_2371_){
_start:
{
uint8_t v___x_4305__boxed_2372_; uint8_t v_res_2373_; lean_object* v_r_2374_; 
v___x_4305__boxed_2372_ = lean_unbox(v___x_2369_);
v_res_2373_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(v___x_4305__boxed_2372_, v_x_2370_, v_y_2371_);
lean_dec(v_y_2371_);
lean_dec(v_x_2370_);
v_r_2374_ = lean_box(v_res_2373_);
return v_r_2374_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg(lean_object* v_hi_2375_, lean_object* v_pivot_2376_, lean_object* v_as_2377_, lean_object* v_i_2378_, lean_object* v_k_2379_){
_start:
{
uint8_t v___x_2390_; 
v___x_2390_ = lean_nat_dec_lt(v_k_2379_, v_hi_2375_);
if (v___x_2390_ == 0)
{
lean_object* v___x_2391_; lean_object* v___x_2392_; 
lean_dec(v_k_2379_);
v___x_2391_ = lean_array_fswap(v_as_2377_, v_i_2378_, v_hi_2375_);
v___x_2392_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2392_, 0, v_i_2378_);
lean_ctor_set(v___x_2392_, 1, v___x_2391_);
return v___x_2392_;
}
else
{
lean_object* v___x_2393_; 
v___x_2393_ = lean_array_fget_borrowed(v_as_2377_, v_k_2379_);
switch(lean_obj_tag(v___x_2393_))
{
case 0:
{
if (lean_obj_tag(v_pivot_2376_) == 2)
{
goto v___jp_2384_;
}
else
{
goto v___jp_2380_;
}
}
case 1:
{
if (lean_obj_tag(v_pivot_2376_) == 1)
{
goto v___jp_2380_;
}
else
{
goto v___jp_2384_;
}
}
default: 
{
if (lean_obj_tag(v_pivot_2376_) == 2)
{
lean_object* v_ldecl_2394_; lean_object* v_ldecl_2395_; lean_object* v___x_2396_; lean_object* v___x_2397_; uint8_t v___x_2398_; 
v_ldecl_2394_ = lean_ctor_get(v___x_2393_, 0);
v_ldecl_2395_ = lean_ctor_get(v_pivot_2376_, 0);
v___x_2396_ = l_Lean_LocalDecl_index(v_ldecl_2394_);
v___x_2397_ = l_Lean_LocalDecl_index(v_ldecl_2395_);
v___x_2398_ = lean_nat_dec_lt(v___x_2396_, v___x_2397_);
lean_dec(v___x_2397_);
lean_dec(v___x_2396_);
if (v___x_2398_ == 0)
{
goto v___jp_2380_;
}
else
{
goto v___jp_2384_;
}
}
else
{
goto v___jp_2380_;
}
}
}
}
v___jp_2380_:
{
lean_object* v___x_2381_; lean_object* v___x_2382_; 
v___x_2381_ = lean_unsigned_to_nat(1u);
v___x_2382_ = lean_nat_add(v_k_2379_, v___x_2381_);
lean_dec(v_k_2379_);
v_k_2379_ = v___x_2382_;
goto _start;
}
v___jp_2384_:
{
lean_object* v___x_2385_; lean_object* v___x_2386_; lean_object* v___x_2387_; lean_object* v___x_2388_; 
v___x_2385_ = lean_array_fswap(v_as_2377_, v_i_2378_, v_k_2379_);
v___x_2386_ = lean_unsigned_to_nat(1u);
v___x_2387_ = lean_nat_add(v_i_2378_, v___x_2386_);
lean_dec(v_i_2378_);
v___x_2388_ = lean_nat_add(v_k_2379_, v___x_2386_);
lean_dec(v_k_2379_);
v_as_2377_ = v___x_2385_;
v_i_2378_ = v___x_2387_;
v_k_2379_ = v___x_2388_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_hi_2399_, lean_object* v_pivot_2400_, lean_object* v_as_2401_, lean_object* v_i_2402_, lean_object* v_k_2403_){
_start:
{
lean_object* v_res_2404_; 
v_res_2404_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg(v_hi_2399_, v_pivot_2400_, v_as_2401_, v_i_2402_, v_k_2403_);
lean_dec(v_pivot_2400_);
lean_dec(v_hi_2399_);
return v_res_2404_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(lean_object* v_n_2405_, lean_object* v_as_2406_, lean_object* v_lo_2407_, lean_object* v_hi_2408_){
_start:
{
lean_object* v___y_2410_; uint8_t v___x_2420_; 
v___x_2420_ = lean_nat_dec_lt(v_lo_2407_, v_hi_2408_);
if (v___x_2420_ == 0)
{
lean_dec(v_lo_2407_);
return v_as_2406_;
}
else
{
lean_object* v___x_2421_; lean_object* v___x_2422_; lean_object* v_mid_2423_; lean_object* v___y_2425_; lean_object* v___y_2431_; lean_object* v___x_2436_; lean_object* v___x_2437_; uint8_t v___x_2438_; 
v___x_2421_ = lean_nat_add(v_lo_2407_, v_hi_2408_);
v___x_2422_ = lean_unsigned_to_nat(1u);
v_mid_2423_ = lean_nat_shiftr(v___x_2421_, v___x_2422_);
lean_dec(v___x_2421_);
v___x_2436_ = lean_array_fget_borrowed(v_as_2406_, v_mid_2423_);
v___x_2437_ = lean_array_fget_borrowed(v_as_2406_, v_lo_2407_);
v___x_2438_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(v___x_2420_, v___x_2436_, v___x_2437_);
if (v___x_2438_ == 0)
{
v___y_2431_ = v_as_2406_;
goto v___jp_2430_;
}
else
{
lean_object* v___x_2439_; 
v___x_2439_ = lean_array_fswap(v_as_2406_, v_lo_2407_, v_mid_2423_);
v___y_2431_ = v___x_2439_;
goto v___jp_2430_;
}
v___jp_2424_:
{
lean_object* v___x_2426_; lean_object* v___x_2427_; uint8_t v___x_2428_; 
v___x_2426_ = lean_array_fget_borrowed(v___y_2425_, v_mid_2423_);
v___x_2427_ = lean_array_fget_borrowed(v___y_2425_, v_hi_2408_);
v___x_2428_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(v___x_2420_, v___x_2426_, v___x_2427_);
if (v___x_2428_ == 0)
{
lean_dec(v_mid_2423_);
v___y_2410_ = v___y_2425_;
goto v___jp_2409_;
}
else
{
lean_object* v___x_2429_; 
v___x_2429_ = lean_array_fswap(v___y_2425_, v_mid_2423_, v_hi_2408_);
lean_dec(v_mid_2423_);
v___y_2410_ = v___x_2429_;
goto v___jp_2409_;
}
}
v___jp_2430_:
{
lean_object* v___x_2432_; lean_object* v___x_2433_; uint8_t v___x_2434_; 
v___x_2432_ = lean_array_fget_borrowed(v___y_2431_, v_hi_2408_);
v___x_2433_ = lean_array_fget_borrowed(v___y_2431_, v_lo_2407_);
v___x_2434_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___lam__0(v___x_2420_, v___x_2432_, v___x_2433_);
if (v___x_2434_ == 0)
{
v___y_2425_ = v___y_2431_;
goto v___jp_2424_;
}
else
{
lean_object* v___x_2435_; 
v___x_2435_ = lean_array_fswap(v___y_2431_, v_lo_2407_, v_hi_2408_);
v___y_2425_ = v___x_2435_;
goto v___jp_2424_;
}
}
}
v___jp_2409_:
{
lean_object* v_pivot_2411_; lean_object* v___x_2412_; lean_object* v_fst_2413_; lean_object* v_snd_2414_; uint8_t v___x_2415_; 
v_pivot_2411_ = lean_array_fget(v___y_2410_, v_hi_2408_);
lean_inc_n(v_lo_2407_, 2);
v___x_2412_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg(v_hi_2408_, v_pivot_2411_, v___y_2410_, v_lo_2407_, v_lo_2407_);
lean_dec(v_pivot_2411_);
v_fst_2413_ = lean_ctor_get(v___x_2412_, 0);
lean_inc(v_fst_2413_);
v_snd_2414_ = lean_ctor_get(v___x_2412_, 1);
lean_inc(v_snd_2414_);
lean_dec_ref(v___x_2412_);
v___x_2415_ = lean_nat_dec_le(v_hi_2408_, v_fst_2413_);
if (v___x_2415_ == 0)
{
lean_object* v___x_2416_; lean_object* v___x_2417_; lean_object* v___x_2418_; 
v___x_2416_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(v_n_2405_, v_snd_2414_, v_lo_2407_, v_fst_2413_);
v___x_2417_ = lean_unsigned_to_nat(1u);
v___x_2418_ = lean_nat_add(v_fst_2413_, v___x_2417_);
lean_dec(v_fst_2413_);
v_as_2406_ = v___x_2416_;
v_lo_2407_ = v___x_2418_;
goto _start;
}
else
{
lean_dec(v_fst_2413_);
lean_dec(v_lo_2407_);
return v_snd_2414_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg___boxed(lean_object* v_n_2440_, lean_object* v_as_2441_, lean_object* v_lo_2442_, lean_object* v_hi_2443_){
_start:
{
lean_object* v_res_2444_; 
v_res_2444_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(v_n_2440_, v_as_2441_, v_lo_2442_, v_hi_2443_);
lean_dec(v_hi_2443_);
lean_dec(v_n_2440_);
return v_res_2444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0(lean_object* v_xs_2445_){
_start:
{
lean_object* v___x_2446_; lean_object* v___x_2447_; uint8_t v___x_2448_; 
v___x_2446_ = lean_array_get_size(v_xs_2445_);
v___x_2447_ = lean_unsigned_to_nat(0u);
v___x_2448_ = lean_nat_dec_eq(v___x_2446_, v___x_2447_);
if (v___x_2448_ == 0)
{
lean_object* v___x_2449_; lean_object* v___x_2450_; lean_object* v___y_2452_; uint8_t v___x_2456_; 
v___x_2449_ = lean_unsigned_to_nat(1u);
v___x_2450_ = lean_nat_sub(v___x_2446_, v___x_2449_);
v___x_2456_ = lean_nat_dec_le(v___x_2447_, v___x_2450_);
if (v___x_2456_ == 0)
{
lean_inc(v___x_2450_);
v___y_2452_ = v___x_2450_;
goto v___jp_2451_;
}
else
{
v___y_2452_ = v___x_2447_;
goto v___jp_2451_;
}
v___jp_2451_:
{
uint8_t v___x_2453_; 
v___x_2453_ = lean_nat_dec_le(v___y_2452_, v___x_2450_);
if (v___x_2453_ == 0)
{
lean_object* v___x_2454_; 
lean_dec(v___x_2450_);
lean_inc(v___y_2452_);
v___x_2454_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(v___x_2446_, v_xs_2445_, v___y_2452_, v___y_2452_);
lean_dec(v___y_2452_);
return v___x_2454_;
}
else
{
lean_object* v___x_2455_; 
v___x_2455_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(v___x_2446_, v_xs_2445_, v___y_2452_, v___x_2450_);
lean_dec(v___x_2450_);
return v___x_2455_;
}
}
}
else
{
return v_xs_2445_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg(lean_object* v_a_2457_, lean_object* v_x_2458_){
_start:
{
if (lean_obj_tag(v_x_2458_) == 0)
{
lean_object* v___x_2459_; 
v___x_2459_ = lean_box(0);
return v___x_2459_;
}
else
{
lean_object* v_key_2460_; lean_object* v_value_2461_; lean_object* v_tail_2462_; uint8_t v___y_2464_; lean_object* v_name_2467_; uint8_t v_builder_2468_; uint8_t v_phase_2469_; uint8_t v_scope_2470_; uint64_t v_hash_2471_; lean_object* v_name_2472_; uint8_t v_builder_2473_; uint8_t v_phase_2474_; uint8_t v_scope_2475_; uint64_t v_hash_2476_; uint8_t v___y_2478_; uint8_t v___x_2483_; 
v_key_2460_ = lean_ctor_get(v_x_2458_, 0);
v_value_2461_ = lean_ctor_get(v_x_2458_, 1);
v_tail_2462_ = lean_ctor_get(v_x_2458_, 2);
v_name_2467_ = lean_ctor_get(v_key_2460_, 0);
v_builder_2468_ = lean_ctor_get_uint8(v_key_2460_, sizeof(void*)*1 + 8);
v_phase_2469_ = lean_ctor_get_uint8(v_key_2460_, sizeof(void*)*1 + 9);
v_scope_2470_ = lean_ctor_get_uint8(v_key_2460_, sizeof(void*)*1 + 10);
v_hash_2471_ = lean_ctor_get_uint64(v_key_2460_, sizeof(void*)*1);
v_name_2472_ = lean_ctor_get(v_a_2457_, 0);
v_builder_2473_ = lean_ctor_get_uint8(v_a_2457_, sizeof(void*)*1 + 8);
v_phase_2474_ = lean_ctor_get_uint8(v_a_2457_, sizeof(void*)*1 + 9);
v_scope_2475_ = lean_ctor_get_uint8(v_a_2457_, sizeof(void*)*1 + 10);
v_hash_2476_ = lean_ctor_get_uint64(v_a_2457_, sizeof(void*)*1);
v___x_2483_ = lean_uint64_dec_eq(v_hash_2471_, v_hash_2476_);
if (v___x_2483_ == 0)
{
v___y_2478_ = v___x_2483_;
goto v___jp_2477_;
}
else
{
uint8_t v___x_2484_; 
v___x_2484_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2468_, v_builder_2473_);
v___y_2478_ = v___x_2484_;
goto v___jp_2477_;
}
v___jp_2463_:
{
if (v___y_2464_ == 0)
{
v_x_2458_ = v_tail_2462_;
goto _start;
}
else
{
lean_object* v___x_2466_; 
lean_inc(v_value_2461_);
v___x_2466_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2466_, 0, v_value_2461_);
return v___x_2466_;
}
}
v___jp_2477_:
{
if (v___y_2478_ == 0)
{
v_x_2458_ = v_tail_2462_;
goto _start;
}
else
{
uint8_t v___x_2480_; 
v___x_2480_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_2469_, v_phase_2474_);
if (v___x_2480_ == 0)
{
v___y_2464_ = v___x_2480_;
goto v___jp_2463_;
}
else
{
uint8_t v___x_2481_; 
v___x_2481_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_2470_, v_scope_2475_);
if (v___x_2481_ == 0)
{
v___y_2464_ = v___x_2481_;
goto v___jp_2463_;
}
else
{
uint8_t v___x_2482_; 
v___x_2482_ = lean_name_eq(v_name_2467_, v_name_2472_);
v___y_2464_ = v___x_2482_;
goto v___jp_2463_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg___boxed(lean_object* v_a_2485_, lean_object* v_x_2486_){
_start:
{
lean_object* v_res_2487_; 
v_res_2487_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg(v_a_2485_, v_x_2486_);
lean_dec(v_x_2486_);
lean_dec_ref(v_a_2485_);
return v_res_2487_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg(lean_object* v_m_2488_, lean_object* v_a_2489_){
_start:
{
lean_object* v_buckets_2490_; uint64_t v_hash_2491_; lean_object* v___x_2492_; uint64_t v___x_2493_; uint64_t v___x_2494_; uint64_t v_fold_2495_; uint64_t v___x_2496_; uint64_t v___x_2497_; uint64_t v___x_2498_; size_t v___x_2499_; size_t v___x_2500_; size_t v___x_2501_; size_t v___x_2502_; size_t v___x_2503_; lean_object* v___x_2504_; lean_object* v___x_2505_; 
v_buckets_2490_ = lean_ctor_get(v_m_2488_, 1);
v_hash_2491_ = lean_ctor_get_uint64(v_a_2489_, sizeof(void*)*1);
v___x_2492_ = lean_array_get_size(v_buckets_2490_);
v___x_2493_ = 32ULL;
v___x_2494_ = lean_uint64_shift_right(v_hash_2491_, v___x_2493_);
v_fold_2495_ = lean_uint64_xor(v_hash_2491_, v___x_2494_);
v___x_2496_ = 16ULL;
v___x_2497_ = lean_uint64_shift_right(v_fold_2495_, v___x_2496_);
v___x_2498_ = lean_uint64_xor(v_fold_2495_, v___x_2497_);
v___x_2499_ = lean_uint64_to_usize(v___x_2498_);
v___x_2500_ = lean_usize_of_nat(v___x_2492_);
v___x_2501_ = ((size_t)1ULL);
v___x_2502_ = lean_usize_sub(v___x_2500_, v___x_2501_);
v___x_2503_ = lean_usize_land(v___x_2499_, v___x_2502_);
v___x_2504_ = lean_array_uget_borrowed(v_buckets_2490_, v___x_2503_);
v___x_2505_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg(v_a_2489_, v___x_2504_);
return v___x_2505_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg___boxed(lean_object* v_m_2506_, lean_object* v_a_2507_){
_start:
{
lean_object* v_res_2508_; 
v_res_2508_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg(v_m_2506_, v_a_2507_);
lean_dec_ref(v_a_2507_);
lean_dec_ref(v_m_2506_);
return v_res_2508_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0(lean_object* v_x_2509_, lean_object* v_x_2510_){
_start:
{
lean_inc(v_x_2509_);
return v_x_2509_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0___boxed(lean_object* v_x_2511_, lean_object* v_x_2512_){
_start:
{
lean_object* v_res_2513_; 
v_res_2513_ = lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___lam__0(v_x_2511_, v_x_2512_);
lean_dec(v_x_2512_);
lean_dec(v_x_2511_);
return v_res_2513_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4(lean_object* v_f_2514_, lean_object* v_xs_2515_, lean_object* v_acc_2516_, lean_object* v_i_2517_, lean_object* v_hd_2518_){
_start:
{
lean_object* v___x_2519_; uint8_t v___x_2520_; 
v___x_2519_ = lean_array_get_size(v_xs_2515_);
v___x_2520_ = lean_nat_dec_lt(v_i_2517_, v___x_2519_);
if (v___x_2520_ == 0)
{
lean_object* v___x_2521_; 
lean_dec(v_i_2517_);
lean_dec_ref(v_f_2514_);
v___x_2521_ = lean_array_push(v_acc_2516_, v_hd_2518_);
return v___x_2521_;
}
else
{
lean_object* v_x_2522_; 
v_x_2522_ = lean_array_fget_borrowed(v_xs_2515_, v_i_2517_);
switch(lean_obj_tag(v_x_2522_))
{
case 0:
{
if (lean_obj_tag(v_hd_2518_) == 0)
{
goto v___jp_2528_;
}
else
{
goto v___jp_2523_;
}
}
case 1:
{
if (lean_obj_tag(v_hd_2518_) == 1)
{
goto v___jp_2528_;
}
else
{
goto v___jp_2523_;
}
}
default: 
{
if (lean_obj_tag(v_hd_2518_) == 2)
{
lean_object* v_ldecl_2533_; lean_object* v_ldecl_2534_; lean_object* v___x_2535_; lean_object* v___x_2536_; uint8_t v___x_2537_; 
v_ldecl_2533_ = lean_ctor_get(v_x_2522_, 0);
v_ldecl_2534_ = lean_ctor_get(v_hd_2518_, 0);
v___x_2535_ = l_Lean_LocalDecl_index(v_ldecl_2533_);
v___x_2536_ = l_Lean_LocalDecl_index(v_ldecl_2534_);
v___x_2537_ = lean_nat_dec_eq(v___x_2535_, v___x_2536_);
lean_dec(v___x_2536_);
lean_dec(v___x_2535_);
if (v___x_2537_ == 0)
{
goto v___jp_2523_;
}
else
{
goto v___jp_2528_;
}
}
else
{
goto v___jp_2523_;
}
}
}
v___jp_2523_:
{
lean_object* v___x_2524_; lean_object* v___x_2525_; lean_object* v___x_2526_; 
v___x_2524_ = lean_array_push(v_acc_2516_, v_hd_2518_);
v___x_2525_ = lean_unsigned_to_nat(1u);
v___x_2526_ = lean_nat_add(v_i_2517_, v___x_2525_);
lean_dec(v_i_2517_);
lean_inc(v_x_2522_);
v_acc_2516_ = v___x_2524_;
v_i_2517_ = v___x_2526_;
v_hd_2518_ = v_x_2522_;
goto _start;
}
v___jp_2528_:
{
lean_object* v___x_2529_; lean_object* v___x_2530_; lean_object* v___x_2531_; 
v___x_2529_ = lean_unsigned_to_nat(1u);
v___x_2530_ = lean_nat_add(v_i_2517_, v___x_2529_);
lean_dec(v_i_2517_);
lean_inc_ref(v_f_2514_);
lean_inc(v_x_2522_);
v___x_2531_ = lean_apply_2(v_f_2514_, v_hd_2518_, v_x_2522_);
v_i_2517_ = v___x_2530_;
v_hd_2518_ = v___x_2531_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4___boxed(lean_object* v_f_2538_, lean_object* v_xs_2539_, lean_object* v_acc_2540_, lean_object* v_i_2541_, lean_object* v_hd_2542_){
_start:
{
lean_object* v_res_2543_; 
v_res_2543_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4(v_f_2538_, v_xs_2539_, v_acc_2540_, v_i_2541_, v_hd_2542_);
lean_dec_ref(v_xs_2539_);
return v_res_2543_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2(lean_object* v_f_2544_, lean_object* v_xs_2545_){
_start:
{
lean_object* v___x_2546_; lean_object* v___x_2547_; uint8_t v___x_2548_; 
v___x_2546_ = lean_unsigned_to_nat(0u);
v___x_2547_ = lean_array_get_size(v_xs_2545_);
v___x_2548_ = lean_nat_dec_lt(v___x_2546_, v___x_2547_);
if (v___x_2548_ == 0)
{
lean_dec_ref(v_f_2544_);
lean_inc_ref(v_xs_2545_);
return v_xs_2545_;
}
else
{
lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2549_ = lean_mk_empty_array_with_capacity(v___x_2547_);
v___x_2550_ = lean_unsigned_to_nat(1u);
v___x_2551_ = lean_array_fget_borrowed(v_xs_2545_, v___x_2546_);
lean_inc(v___x_2551_);
v___x_2552_ = lp_aesop_Array_mergeAdjacentDups_go___at___00Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2_spec__4(v_f_2544_, v_xs_2545_, v___x_2549_, v___x_2550_, v___x_2551_);
return v___x_2552_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2___boxed(lean_object* v_f_2553_, lean_object* v_xs_2554_){
_start:
{
lean_object* v_res_2555_; 
v_res_2555_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2(v_f_2553_, v_xs_2554_);
lean_dec_ref(v_xs_2554_);
return v_res_2555_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1(lean_object* v_xs_2557_){
_start:
{
lean_object* v___f_2558_; lean_object* v___x_2559_; 
v___f_2558_ = ((lean_object*)(lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___closed__0));
v___x_2559_ = lp_aesop_Array_mergeAdjacentDups___at___00Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1_spec__2(v___f_2558_, v_xs_2557_);
return v___x_2559_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1___boxed(lean_object* v_xs_2560_){
_start:
{
lean_object* v_res_2561_; 
v_res_2561_ = lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1(v_xs_2560_);
lean_dec_ref(v_xs_2560_);
return v_res_2561_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg(lean_object* v_patSubstMap_2562_, lean_object* v_a_2563_, lean_object* v_a_2564_){
_start:
{
if (lean_obj_tag(v_a_2563_) == 0)
{
lean_object* v___x_2566_; lean_object* v___x_2567_; 
v___x_2566_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2566_, 0, v_a_2564_);
v___x_2567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2567_, 0, v___x_2566_);
return v___x_2567_;
}
else
{
lean_object* v_key_2568_; lean_object* v_pattern_x3f_2569_; 
v_key_2568_ = lean_ctor_get(v_a_2563_, 0);
lean_inc(v_key_2568_);
v_pattern_x3f_2569_ = lean_ctor_get(v_key_2568_, 2);
if (lean_obj_tag(v_pattern_x3f_2569_) == 0)
{
lean_object* v_value_2570_; lean_object* v_tail_2571_; lean_object* v___x_2573_; uint8_t v_isShared_2574_; uint8_t v_isSharedCheck_2583_; 
v_value_2570_ = lean_ctor_get(v_a_2563_, 1);
v_tail_2571_ = lean_ctor_get(v_a_2563_, 2);
v_isSharedCheck_2583_ = !lean_is_exclusive(v_a_2563_);
if (v_isSharedCheck_2583_ == 0)
{
lean_object* v_unused_2584_; 
v_unused_2584_ = lean_ctor_get(v_a_2563_, 0);
lean_dec(v_unused_2584_);
v___x_2573_ = v_a_2563_;
v_isShared_2574_ = v_isSharedCheck_2583_;
goto v_resetjp_2572_;
}
else
{
lean_inc(v_tail_2571_);
lean_inc(v_value_2570_);
lean_dec(v_a_2563_);
v___x_2573_ = lean_box(0);
v_isShared_2574_ = v_isSharedCheck_2583_;
goto v_resetjp_2572_;
}
v_resetjp_2572_:
{
lean_object* v___x_2575_; lean_object* v___x_2576_; lean_object* v___x_2577_; lean_object* v___x_2579_; 
v___x_2575_ = lp_aesop_Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0(v_value_2570_);
v___x_2576_ = lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1(v___x_2575_);
lean_dec_ref(v___x_2575_);
v___x_2577_ = lean_box(0);
if (v_isShared_2574_ == 0)
{
lean_ctor_set_tag(v___x_2573_, 0);
lean_ctor_set(v___x_2573_, 2, v___x_2577_);
lean_ctor_set(v___x_2573_, 1, v___x_2576_);
v___x_2579_ = v___x_2573_;
goto v_reusejp_2578_;
}
else
{
lean_object* v_reuseFailAlloc_2582_; 
v_reuseFailAlloc_2582_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2582_, 0, v_key_2568_);
lean_ctor_set(v_reuseFailAlloc_2582_, 1, v___x_2576_);
lean_ctor_set(v_reuseFailAlloc_2582_, 2, v___x_2577_);
v___x_2579_ = v_reuseFailAlloc_2582_;
goto v_reusejp_2578_;
}
v_reusejp_2578_:
{
lean_object* v___x_2580_; 
v___x_2580_ = lean_array_push(v_a_2564_, v___x_2579_);
v_a_2563_ = v_tail_2571_;
v_a_2564_ = v___x_2580_;
goto _start;
}
}
}
else
{
lean_object* v_value_2585_; lean_object* v_tail_2586_; lean_object* v___x_2588_; uint8_t v_isShared_2589_; uint8_t v_isSharedCheck_2609_; 
v_value_2585_ = lean_ctor_get(v_a_2563_, 1);
v_tail_2586_ = lean_ctor_get(v_a_2563_, 2);
v_isSharedCheck_2609_ = !lean_is_exclusive(v_a_2563_);
if (v_isSharedCheck_2609_ == 0)
{
lean_object* v_unused_2610_; 
v_unused_2610_ = lean_ctor_get(v_a_2563_, 0);
lean_dec(v_unused_2610_);
v___x_2588_ = v_a_2563_;
v_isShared_2589_ = v_isSharedCheck_2609_;
goto v_resetjp_2587_;
}
else
{
lean_inc(v_tail_2586_);
lean_inc(v_value_2585_);
lean_dec(v_a_2563_);
v___x_2588_ = lean_box(0);
v_isShared_2589_ = v_isSharedCheck_2609_;
goto v_resetjp_2587_;
}
v_resetjp_2587_:
{
lean_object* v_name_2590_; lean_object* v___x_2591_; 
v_name_2590_ = lean_ctor_get(v_key_2568_, 0);
v___x_2591_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg(v_patSubstMap_2562_, v_name_2590_);
if (lean_obj_tag(v___x_2591_) == 1)
{
lean_object* v_val_2592_; lean_object* v___x_2594_; uint8_t v_isShared_2595_; uint8_t v_isSharedCheck_2607_; 
v_val_2592_ = lean_ctor_get(v___x_2591_, 0);
v_isSharedCheck_2607_ = !lean_is_exclusive(v___x_2591_);
if (v_isSharedCheck_2607_ == 0)
{
v___x_2594_ = v___x_2591_;
v_isShared_2595_ = v_isSharedCheck_2607_;
goto v_resetjp_2593_;
}
else
{
lean_inc(v_val_2592_);
lean_dec(v___x_2591_);
v___x_2594_ = lean_box(0);
v_isShared_2595_ = v_isSharedCheck_2607_;
goto v_resetjp_2593_;
}
v_resetjp_2593_:
{
lean_object* v_toArray_2596_; lean_object* v___x_2597_; lean_object* v___x_2598_; lean_object* v___x_2600_; 
v_toArray_2596_ = lean_ctor_get(v_val_2592_, 0);
lean_inc_ref(v_toArray_2596_);
lean_dec(v_val_2592_);
v___x_2597_ = lp_aesop_Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0(v_value_2585_);
v___x_2598_ = lp_aesop_Array_dedupSorted___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__1(v___x_2597_);
lean_dec_ref(v___x_2597_);
if (v_isShared_2595_ == 0)
{
lean_ctor_set(v___x_2594_, 0, v_toArray_2596_);
v___x_2600_ = v___x_2594_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2606_; 
v_reuseFailAlloc_2606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2606_, 0, v_toArray_2596_);
v___x_2600_ = v_reuseFailAlloc_2606_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
lean_object* v___x_2602_; 
if (v_isShared_2589_ == 0)
{
lean_ctor_set_tag(v___x_2588_, 0);
lean_ctor_set(v___x_2588_, 2, v___x_2600_);
lean_ctor_set(v___x_2588_, 1, v___x_2598_);
v___x_2602_ = v___x_2588_;
goto v_reusejp_2601_;
}
else
{
lean_object* v_reuseFailAlloc_2605_; 
v_reuseFailAlloc_2605_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2605_, 0, v_key_2568_);
lean_ctor_set(v_reuseFailAlloc_2605_, 1, v___x_2598_);
lean_ctor_set(v_reuseFailAlloc_2605_, 2, v___x_2600_);
v___x_2602_ = v_reuseFailAlloc_2605_;
goto v_reusejp_2601_;
}
v_reusejp_2601_:
{
lean_object* v___x_2603_; 
v___x_2603_ = lean_array_push(v_a_2564_, v___x_2602_);
v_a_2563_ = v_tail_2586_;
v_a_2564_ = v___x_2603_;
goto _start;
}
}
}
}
else
{
lean_dec(v___x_2591_);
lean_del_object(v___x_2588_);
lean_dec(v_value_2585_);
lean_dec(v_key_2568_);
v_a_2563_ = v_tail_2586_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg___boxed(lean_object* v_patSubstMap_2611_, lean_object* v_a_2612_, lean_object* v_a_2613_, lean_object* v___y_2614_){
_start:
{
lean_object* v_res_2615_; 
v_res_2615_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg(v_patSubstMap_2611_, v_a_2612_, v_a_2613_);
lean_dec_ref(v_patSubstMap_2611_);
return v_res_2615_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg(lean_object* v_patSubstMap_2616_, lean_object* v_as_2617_, size_t v_sz_2618_, size_t v_i_2619_, lean_object* v_b_2620_, lean_object* v___y_2621_, lean_object* v___y_2622_, lean_object* v___y_2623_, lean_object* v___y_2624_){
_start:
{
uint8_t v___x_2626_; 
v___x_2626_ = lean_usize_dec_lt(v_i_2619_, v_sz_2618_);
if (v___x_2626_ == 0)
{
lean_object* v___x_2627_; 
v___x_2627_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2627_, 0, v_b_2620_);
return v___x_2627_;
}
else
{
lean_object* v_a_2628_; lean_object* v___x_2629_; 
v_a_2628_ = lean_array_uget_borrowed(v_as_2617_, v_i_2619_);
lean_inc(v_a_2628_);
v___x_2629_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg(v_patSubstMap_2616_, v_a_2628_, v_b_2620_);
if (lean_obj_tag(v___x_2629_) == 0)
{
lean_object* v_a_2630_; lean_object* v___x_2632_; uint8_t v_isShared_2633_; uint8_t v_isSharedCheck_2642_; 
v_a_2630_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2642_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2632_ = v___x_2629_;
v_isShared_2633_ = v_isSharedCheck_2642_;
goto v_resetjp_2631_;
}
else
{
lean_inc(v_a_2630_);
lean_dec(v___x_2629_);
v___x_2632_ = lean_box(0);
v_isShared_2633_ = v_isSharedCheck_2642_;
goto v_resetjp_2631_;
}
v_resetjp_2631_:
{
if (lean_obj_tag(v_a_2630_) == 0)
{
lean_object* v_a_2634_; lean_object* v___x_2636_; 
v_a_2634_ = lean_ctor_get(v_a_2630_, 0);
lean_inc(v_a_2634_);
lean_dec_ref_known(v_a_2630_, 1);
if (v_isShared_2633_ == 0)
{
lean_ctor_set(v___x_2632_, 0, v_a_2634_);
v___x_2636_ = v___x_2632_;
goto v_reusejp_2635_;
}
else
{
lean_object* v_reuseFailAlloc_2637_; 
v_reuseFailAlloc_2637_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2637_, 0, v_a_2634_);
v___x_2636_ = v_reuseFailAlloc_2637_;
goto v_reusejp_2635_;
}
v_reusejp_2635_:
{
return v___x_2636_;
}
}
else
{
lean_object* v_a_2638_; size_t v___x_2639_; size_t v___x_2640_; 
lean_del_object(v___x_2632_);
v_a_2638_ = lean_ctor_get(v_a_2630_, 0);
lean_inc(v_a_2638_);
lean_dec_ref_known(v_a_2630_, 1);
v___x_2639_ = ((size_t)1ULL);
v___x_2640_ = lean_usize_add(v_i_2619_, v___x_2639_);
v_i_2619_ = v___x_2640_;
v_b_2620_ = v_a_2638_;
goto _start;
}
}
}
else
{
lean_object* v_a_2643_; lean_object* v___x_2645_; uint8_t v_isShared_2646_; uint8_t v_isSharedCheck_2650_; 
v_a_2643_ = lean_ctor_get(v___x_2629_, 0);
v_isSharedCheck_2650_ = !lean_is_exclusive(v___x_2629_);
if (v_isSharedCheck_2650_ == 0)
{
v___x_2645_ = v___x_2629_;
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
else
{
lean_inc(v_a_2643_);
lean_dec(v___x_2629_);
v___x_2645_ = lean_box(0);
v_isShared_2646_ = v_isSharedCheck_2650_;
goto v_resetjp_2644_;
}
v_resetjp_2644_:
{
lean_object* v___x_2648_; 
if (v_isShared_2646_ == 0)
{
v___x_2648_ = v___x_2645_;
goto v_reusejp_2647_;
}
else
{
lean_object* v_reuseFailAlloc_2649_; 
v_reuseFailAlloc_2649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2649_, 0, v_a_2643_);
v___x_2648_ = v_reuseFailAlloc_2649_;
goto v_reusejp_2647_;
}
v_reusejp_2647_:
{
return v___x_2648_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg___boxed(lean_object* v_patSubstMap_2651_, lean_object* v_as_2652_, lean_object* v_sz_2653_, lean_object* v_i_2654_, lean_object* v_b_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_, lean_object* v___y_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_){
_start:
{
size_t v_sz_boxed_2661_; size_t v_i_boxed_2662_; lean_object* v_res_2663_; 
v_sz_boxed_2661_ = lean_unbox_usize(v_sz_2653_);
lean_dec(v_sz_2653_);
v_i_boxed_2662_ = lean_unbox_usize(v_i_2654_);
lean_dec(v_i_2654_);
v_res_2663_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg(v_patSubstMap_2651_, v_as_2652_, v_sz_boxed_2661_, v_i_boxed_2662_, v_b_2655_, v___y_2656_, v___y_2657_, v___y_2658_, v___y_2659_);
lean_dec(v___y_2659_);
lean_dec_ref(v___y_2658_);
lean_dec(v___y_2657_);
lean_dec_ref(v___y_2656_);
lean_dec_ref(v_as_2652_);
lean_dec_ref(v_patSubstMap_2651_);
return v_res_2663_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg(lean_object* v_a_2664_, lean_object* v_x_2665_){
_start:
{
if (lean_obj_tag(v_x_2665_) == 0)
{
lean_object* v___x_2666_; 
v___x_2666_ = lean_box(0);
return v___x_2666_;
}
else
{
lean_object* v_key_2667_; lean_object* v_value_2668_; lean_object* v_tail_2669_; uint8_t v___y_2671_; lean_object* v_name_2674_; lean_object* v_name_2675_; lean_object* v_name_2676_; uint8_t v_builder_2677_; uint8_t v_phase_2678_; uint8_t v_scope_2679_; uint64_t v_hash_2680_; lean_object* v_name_2681_; uint8_t v_builder_2682_; uint8_t v_phase_2683_; uint8_t v_scope_2684_; uint64_t v_hash_2685_; uint8_t v___y_2687_; uint8_t v___x_2692_; 
v_key_2667_ = lean_ctor_get(v_x_2665_, 0);
v_value_2668_ = lean_ctor_get(v_x_2665_, 1);
v_tail_2669_ = lean_ctor_get(v_x_2665_, 2);
v_name_2674_ = lean_ctor_get(v_key_2667_, 0);
v_name_2675_ = lean_ctor_get(v_a_2664_, 0);
v_name_2676_ = lean_ctor_get(v_name_2674_, 0);
v_builder_2677_ = lean_ctor_get_uint8(v_name_2674_, sizeof(void*)*1 + 8);
v_phase_2678_ = lean_ctor_get_uint8(v_name_2674_, sizeof(void*)*1 + 9);
v_scope_2679_ = lean_ctor_get_uint8(v_name_2674_, sizeof(void*)*1 + 10);
v_hash_2680_ = lean_ctor_get_uint64(v_name_2674_, sizeof(void*)*1);
v_name_2681_ = lean_ctor_get(v_name_2675_, 0);
v_builder_2682_ = lean_ctor_get_uint8(v_name_2675_, sizeof(void*)*1 + 8);
v_phase_2683_ = lean_ctor_get_uint8(v_name_2675_, sizeof(void*)*1 + 9);
v_scope_2684_ = lean_ctor_get_uint8(v_name_2675_, sizeof(void*)*1 + 10);
v_hash_2685_ = lean_ctor_get_uint64(v_name_2675_, sizeof(void*)*1);
v___x_2692_ = lean_uint64_dec_eq(v_hash_2680_, v_hash_2685_);
if (v___x_2692_ == 0)
{
v___y_2687_ = v___x_2692_;
goto v___jp_2686_;
}
else
{
uint8_t v___x_2693_; 
v___x_2693_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2677_, v_builder_2682_);
v___y_2687_ = v___x_2693_;
goto v___jp_2686_;
}
v___jp_2670_:
{
if (v___y_2671_ == 0)
{
v_x_2665_ = v_tail_2669_;
goto _start;
}
else
{
lean_object* v___x_2673_; 
lean_inc(v_value_2668_);
v___x_2673_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2673_, 0, v_value_2668_);
return v___x_2673_;
}
}
v___jp_2686_:
{
if (v___y_2687_ == 0)
{
v_x_2665_ = v_tail_2669_;
goto _start;
}
else
{
uint8_t v___x_2689_; 
v___x_2689_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_2678_, v_phase_2683_);
if (v___x_2689_ == 0)
{
v___y_2671_ = v___x_2689_;
goto v___jp_2670_;
}
else
{
uint8_t v___x_2690_; 
v___x_2690_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_2679_, v_scope_2684_);
if (v___x_2690_ == 0)
{
v___y_2671_ = v___x_2690_;
goto v___jp_2670_;
}
else
{
uint8_t v___x_2691_; 
v___x_2691_ = lean_name_eq(v_name_2676_, v_name_2681_);
v___y_2671_ = v___x_2691_;
goto v___jp_2670_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg___boxed(lean_object* v_a_2694_, lean_object* v_x_2695_){
_start:
{
lean_object* v_res_2696_; 
v_res_2696_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg(v_a_2694_, v_x_2695_);
lean_dec(v_x_2695_);
lean_dec_ref(v_a_2694_);
return v_res_2696_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg(lean_object* v_m_2697_, lean_object* v_a_2698_){
_start:
{
lean_object* v_name_2699_; lean_object* v_buckets_2700_; uint64_t v_hash_2701_; lean_object* v___x_2702_; uint64_t v___x_2703_; uint64_t v___x_2704_; uint64_t v_fold_2705_; uint64_t v___x_2706_; uint64_t v___x_2707_; uint64_t v___x_2708_; size_t v___x_2709_; size_t v___x_2710_; size_t v___x_2711_; size_t v___x_2712_; size_t v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; 
v_name_2699_ = lean_ctor_get(v_a_2698_, 0);
v_buckets_2700_ = lean_ctor_get(v_m_2697_, 1);
v_hash_2701_ = lean_ctor_get_uint64(v_name_2699_, sizeof(void*)*1);
v___x_2702_ = lean_array_get_size(v_buckets_2700_);
v___x_2703_ = 32ULL;
v___x_2704_ = lean_uint64_shift_right(v_hash_2701_, v___x_2703_);
v_fold_2705_ = lean_uint64_xor(v_hash_2701_, v___x_2704_);
v___x_2706_ = 16ULL;
v___x_2707_ = lean_uint64_shift_right(v_fold_2705_, v___x_2706_);
v___x_2708_ = lean_uint64_xor(v_fold_2705_, v___x_2707_);
v___x_2709_ = lean_uint64_to_usize(v___x_2708_);
v___x_2710_ = lean_usize_of_nat(v___x_2702_);
v___x_2711_ = ((size_t)1ULL);
v___x_2712_ = lean_usize_sub(v___x_2710_, v___x_2711_);
v___x_2713_ = lean_usize_land(v___x_2709_, v___x_2712_);
v___x_2714_ = lean_array_uget_borrowed(v_buckets_2700_, v___x_2713_);
v___x_2715_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg(v_a_2698_, v___x_2714_);
return v___x_2715_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg___boxed(lean_object* v_m_2716_, lean_object* v_a_2717_){
_start:
{
lean_object* v_res_2718_; 
v_res_2718_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg(v_m_2716_, v_a_2717_);
lean_dec_ref(v_a_2717_);
lean_dec_ref(v_m_2716_);
return v_res_2718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17___redArg(lean_object* v_x_2719_, lean_object* v_x_2720_){
_start:
{
if (lean_obj_tag(v_x_2720_) == 0)
{
return v_x_2719_;
}
else
{
lean_object* v_key_2721_; lean_object* v_name_2722_; lean_object* v_value_2723_; lean_object* v_tail_2724_; lean_object* v___x_2726_; uint8_t v_isShared_2727_; uint8_t v_isSharedCheck_2747_; 
v_key_2721_ = lean_ctor_get(v_x_2720_, 0);
lean_inc(v_key_2721_);
v_name_2722_ = lean_ctor_get(v_key_2721_, 0);
v_value_2723_ = lean_ctor_get(v_x_2720_, 1);
v_tail_2724_ = lean_ctor_get(v_x_2720_, 2);
v_isSharedCheck_2747_ = !lean_is_exclusive(v_x_2720_);
if (v_isSharedCheck_2747_ == 0)
{
lean_object* v_unused_2748_; 
v_unused_2748_ = lean_ctor_get(v_x_2720_, 0);
lean_dec(v_unused_2748_);
v___x_2726_ = v_x_2720_;
v_isShared_2727_ = v_isSharedCheck_2747_;
goto v_resetjp_2725_;
}
else
{
lean_inc(v_tail_2724_);
lean_inc(v_value_2723_);
lean_dec(v_x_2720_);
v___x_2726_ = lean_box(0);
v_isShared_2727_ = v_isSharedCheck_2747_;
goto v_resetjp_2725_;
}
v_resetjp_2725_:
{
uint64_t v_hash_2728_; lean_object* v___x_2729_; uint64_t v___x_2730_; uint64_t v___x_2731_; uint64_t v_fold_2732_; uint64_t v___x_2733_; uint64_t v___x_2734_; uint64_t v___x_2735_; size_t v___x_2736_; size_t v___x_2737_; size_t v___x_2738_; size_t v___x_2739_; size_t v___x_2740_; lean_object* v___x_2741_; lean_object* v___x_2743_; 
v_hash_2728_ = lean_ctor_get_uint64(v_name_2722_, sizeof(void*)*1);
v___x_2729_ = lean_array_get_size(v_x_2719_);
v___x_2730_ = 32ULL;
v___x_2731_ = lean_uint64_shift_right(v_hash_2728_, v___x_2730_);
v_fold_2732_ = lean_uint64_xor(v_hash_2728_, v___x_2731_);
v___x_2733_ = 16ULL;
v___x_2734_ = lean_uint64_shift_right(v_fold_2732_, v___x_2733_);
v___x_2735_ = lean_uint64_xor(v_fold_2732_, v___x_2734_);
v___x_2736_ = lean_uint64_to_usize(v___x_2735_);
v___x_2737_ = lean_usize_of_nat(v___x_2729_);
v___x_2738_ = ((size_t)1ULL);
v___x_2739_ = lean_usize_sub(v___x_2737_, v___x_2738_);
v___x_2740_ = lean_usize_land(v___x_2736_, v___x_2739_);
v___x_2741_ = lean_array_uget_borrowed(v_x_2719_, v___x_2740_);
lean_inc(v___x_2741_);
if (v_isShared_2727_ == 0)
{
lean_ctor_set(v___x_2726_, 2, v___x_2741_);
v___x_2743_ = v___x_2726_;
goto v_reusejp_2742_;
}
else
{
lean_object* v_reuseFailAlloc_2746_; 
v_reuseFailAlloc_2746_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2746_, 0, v_key_2721_);
lean_ctor_set(v_reuseFailAlloc_2746_, 1, v_value_2723_);
lean_ctor_set(v_reuseFailAlloc_2746_, 2, v___x_2741_);
v___x_2743_ = v_reuseFailAlloc_2746_;
goto v_reusejp_2742_;
}
v_reusejp_2742_:
{
lean_object* v___x_2744_; 
v___x_2744_ = lean_array_uset(v_x_2719_, v___x_2740_, v___x_2743_);
v_x_2719_ = v___x_2744_;
v_x_2720_ = v_tail_2724_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13___redArg(lean_object* v_i_2749_, lean_object* v_source_2750_, lean_object* v_target_2751_){
_start:
{
lean_object* v___x_2752_; uint8_t v___x_2753_; 
v___x_2752_ = lean_array_get_size(v_source_2750_);
v___x_2753_ = lean_nat_dec_lt(v_i_2749_, v___x_2752_);
if (v___x_2753_ == 0)
{
lean_dec_ref(v_source_2750_);
lean_dec(v_i_2749_);
return v_target_2751_;
}
else
{
lean_object* v_es_2754_; lean_object* v___x_2755_; lean_object* v_source_2756_; lean_object* v_target_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; 
v_es_2754_ = lean_array_fget(v_source_2750_, v_i_2749_);
v___x_2755_ = lean_box(0);
v_source_2756_ = lean_array_fset(v_source_2750_, v_i_2749_, v___x_2755_);
v_target_2757_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17___redArg(v_target_2751_, v_es_2754_);
v___x_2758_ = lean_unsigned_to_nat(1u);
v___x_2759_ = lean_nat_add(v_i_2749_, v___x_2758_);
lean_dec(v_i_2749_);
v_i_2749_ = v___x_2759_;
v_source_2750_ = v_source_2756_;
v_target_2751_ = v_target_2757_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10___redArg(lean_object* v_data_2761_){
_start:
{
lean_object* v___x_2762_; lean_object* v___x_2763_; lean_object* v_nbuckets_2764_; lean_object* v___x_2765_; lean_object* v___x_2766_; lean_object* v___x_2767_; lean_object* v___x_2768_; 
v___x_2762_ = lean_array_get_size(v_data_2761_);
v___x_2763_ = lean_unsigned_to_nat(2u);
v_nbuckets_2764_ = lean_nat_mul(v___x_2762_, v___x_2763_);
v___x_2765_ = lean_unsigned_to_nat(0u);
v___x_2766_ = lean_box(0);
v___x_2767_ = lean_mk_array(v_nbuckets_2764_, v___x_2766_);
v___x_2768_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13___redArg(v___x_2765_, v_data_2761_, v___x_2767_);
return v___x_2768_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11___redArg(lean_object* v_a_2769_, lean_object* v_b_2770_, lean_object* v_x_2771_){
_start:
{
if (lean_obj_tag(v_x_2771_) == 0)
{
lean_dec(v_b_2770_);
lean_dec_ref(v_a_2769_);
return v_x_2771_;
}
else
{
lean_object* v_key_2772_; lean_object* v_value_2773_; lean_object* v_tail_2774_; lean_object* v___x_2776_; uint8_t v_isShared_2777_; uint8_t v_isSharedCheck_2805_; 
v_key_2772_ = lean_ctor_get(v_x_2771_, 0);
v_value_2773_ = lean_ctor_get(v_x_2771_, 1);
v_tail_2774_ = lean_ctor_get(v_x_2771_, 2);
v_isSharedCheck_2805_ = !lean_is_exclusive(v_x_2771_);
if (v_isSharedCheck_2805_ == 0)
{
v___x_2776_ = v_x_2771_;
v_isShared_2777_ = v_isSharedCheck_2805_;
goto v_resetjp_2775_;
}
else
{
lean_inc(v_tail_2774_);
lean_inc(v_value_2773_);
lean_inc(v_key_2772_);
lean_dec(v_x_2771_);
v___x_2776_ = lean_box(0);
v_isShared_2777_ = v_isSharedCheck_2805_;
goto v_resetjp_2775_;
}
v_resetjp_2775_:
{
uint8_t v___y_2784_; lean_object* v_name_2786_; lean_object* v_name_2787_; lean_object* v_name_2788_; uint8_t v_builder_2789_; uint8_t v_phase_2790_; uint8_t v_scope_2791_; uint64_t v_hash_2792_; lean_object* v_name_2793_; uint8_t v_builder_2794_; uint8_t v_phase_2795_; uint8_t v_scope_2796_; uint64_t v_hash_2797_; uint8_t v___y_2799_; uint8_t v___x_2803_; 
v_name_2786_ = lean_ctor_get(v_key_2772_, 0);
v_name_2787_ = lean_ctor_get(v_a_2769_, 0);
v_name_2788_ = lean_ctor_get(v_name_2786_, 0);
v_builder_2789_ = lean_ctor_get_uint8(v_name_2786_, sizeof(void*)*1 + 8);
v_phase_2790_ = lean_ctor_get_uint8(v_name_2786_, sizeof(void*)*1 + 9);
v_scope_2791_ = lean_ctor_get_uint8(v_name_2786_, sizeof(void*)*1 + 10);
v_hash_2792_ = lean_ctor_get_uint64(v_name_2786_, sizeof(void*)*1);
v_name_2793_ = lean_ctor_get(v_name_2787_, 0);
v_builder_2794_ = lean_ctor_get_uint8(v_name_2787_, sizeof(void*)*1 + 8);
v_phase_2795_ = lean_ctor_get_uint8(v_name_2787_, sizeof(void*)*1 + 9);
v_scope_2796_ = lean_ctor_get_uint8(v_name_2787_, sizeof(void*)*1 + 10);
v_hash_2797_ = lean_ctor_get_uint64(v_name_2787_, sizeof(void*)*1);
v___x_2803_ = lean_uint64_dec_eq(v_hash_2792_, v_hash_2797_);
if (v___x_2803_ == 0)
{
v___y_2799_ = v___x_2803_;
goto v___jp_2798_;
}
else
{
uint8_t v___x_2804_; 
v___x_2804_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2789_, v_builder_2794_);
v___y_2799_ = v___x_2804_;
goto v___jp_2798_;
}
v___jp_2778_:
{
lean_object* v___x_2779_; lean_object* v___x_2781_; 
v___x_2779_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11___redArg(v_a_2769_, v_b_2770_, v_tail_2774_);
if (v_isShared_2777_ == 0)
{
lean_ctor_set(v___x_2776_, 2, v___x_2779_);
v___x_2781_ = v___x_2776_;
goto v_reusejp_2780_;
}
else
{
lean_object* v_reuseFailAlloc_2782_; 
v_reuseFailAlloc_2782_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2782_, 0, v_key_2772_);
lean_ctor_set(v_reuseFailAlloc_2782_, 1, v_value_2773_);
lean_ctor_set(v_reuseFailAlloc_2782_, 2, v___x_2779_);
v___x_2781_ = v_reuseFailAlloc_2782_;
goto v_reusejp_2780_;
}
v_reusejp_2780_:
{
return v___x_2781_;
}
}
v___jp_2783_:
{
if (v___y_2784_ == 0)
{
goto v___jp_2778_;
}
else
{
lean_object* v___x_2785_; 
lean_del_object(v___x_2776_);
lean_dec(v_value_2773_);
lean_dec(v_key_2772_);
v___x_2785_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2785_, 0, v_a_2769_);
lean_ctor_set(v___x_2785_, 1, v_b_2770_);
lean_ctor_set(v___x_2785_, 2, v_tail_2774_);
return v___x_2785_;
}
}
v___jp_2798_:
{
if (v___y_2799_ == 0)
{
goto v___jp_2778_;
}
else
{
uint8_t v___x_2800_; 
v___x_2800_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_2790_, v_phase_2795_);
if (v___x_2800_ == 0)
{
v___y_2784_ = v___x_2800_;
goto v___jp_2783_;
}
else
{
uint8_t v___x_2801_; 
v___x_2801_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_2791_, v_scope_2796_);
if (v___x_2801_ == 0)
{
v___y_2784_ = v___x_2801_;
goto v___jp_2783_;
}
else
{
uint8_t v___x_2802_; 
v___x_2802_ = lean_name_eq(v_name_2788_, v_name_2793_);
v___y_2784_ = v___x_2802_;
goto v___jp_2783_;
}
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg(lean_object* v_a_2806_, lean_object* v_x_2807_){
_start:
{
if (lean_obj_tag(v_x_2807_) == 0)
{
uint8_t v___x_2808_; 
v___x_2808_ = 0;
return v___x_2808_;
}
else
{
lean_object* v_key_2809_; lean_object* v_tail_2810_; uint8_t v___y_2812_; lean_object* v_name_2814_; lean_object* v_name_2815_; lean_object* v_name_2816_; uint8_t v_builder_2817_; uint8_t v_phase_2818_; uint8_t v_scope_2819_; uint64_t v_hash_2820_; lean_object* v_name_2821_; uint8_t v_builder_2822_; uint8_t v_phase_2823_; uint8_t v_scope_2824_; uint64_t v_hash_2825_; uint8_t v___y_2827_; uint8_t v___x_2832_; 
v_key_2809_ = lean_ctor_get(v_x_2807_, 0);
v_tail_2810_ = lean_ctor_get(v_x_2807_, 2);
v_name_2814_ = lean_ctor_get(v_key_2809_, 0);
v_name_2815_ = lean_ctor_get(v_a_2806_, 0);
v_name_2816_ = lean_ctor_get(v_name_2814_, 0);
v_builder_2817_ = lean_ctor_get_uint8(v_name_2814_, sizeof(void*)*1 + 8);
v_phase_2818_ = lean_ctor_get_uint8(v_name_2814_, sizeof(void*)*1 + 9);
v_scope_2819_ = lean_ctor_get_uint8(v_name_2814_, sizeof(void*)*1 + 10);
v_hash_2820_ = lean_ctor_get_uint64(v_name_2814_, sizeof(void*)*1);
v_name_2821_ = lean_ctor_get(v_name_2815_, 0);
v_builder_2822_ = lean_ctor_get_uint8(v_name_2815_, sizeof(void*)*1 + 8);
v_phase_2823_ = lean_ctor_get_uint8(v_name_2815_, sizeof(void*)*1 + 9);
v_scope_2824_ = lean_ctor_get_uint8(v_name_2815_, sizeof(void*)*1 + 10);
v_hash_2825_ = lean_ctor_get_uint64(v_name_2815_, sizeof(void*)*1);
v___x_2832_ = lean_uint64_dec_eq(v_hash_2820_, v_hash_2825_);
if (v___x_2832_ == 0)
{
v___y_2827_ = v___x_2832_;
goto v___jp_2826_;
}
else
{
uint8_t v___x_2833_; 
v___x_2833_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_2817_, v_builder_2822_);
v___y_2827_ = v___x_2833_;
goto v___jp_2826_;
}
v___jp_2811_:
{
if (v___y_2812_ == 0)
{
v_x_2807_ = v_tail_2810_;
goto _start;
}
else
{
return v___y_2812_;
}
}
v___jp_2826_:
{
if (v___y_2827_ == 0)
{
v_x_2807_ = v_tail_2810_;
goto _start;
}
else
{
uint8_t v___x_2829_; 
v___x_2829_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_2818_, v_phase_2823_);
if (v___x_2829_ == 0)
{
v___y_2812_ = v___x_2829_;
goto v___jp_2811_;
}
else
{
uint8_t v___x_2830_; 
v___x_2830_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_2819_, v_scope_2824_);
if (v___x_2830_ == 0)
{
v___y_2812_ = v___x_2830_;
goto v___jp_2811_;
}
else
{
uint8_t v___x_2831_; 
v___x_2831_ = lean_name_eq(v_name_2816_, v_name_2821_);
v___y_2812_ = v___x_2831_;
goto v___jp_2811_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg___boxed(lean_object* v_a_2834_, lean_object* v_x_2835_){
_start:
{
uint8_t v_res_2836_; lean_object* v_r_2837_; 
v_res_2836_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg(v_a_2834_, v_x_2835_);
lean_dec(v_x_2835_);
lean_dec_ref(v_a_2834_);
v_r_2837_ = lean_box(v_res_2836_);
return v_r_2837_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5___redArg(lean_object* v_m_2838_, lean_object* v_a_2839_, lean_object* v_b_2840_){
_start:
{
lean_object* v_name_2841_; lean_object* v_size_2842_; lean_object* v_buckets_2843_; lean_object* v___x_2845_; uint8_t v_isShared_2846_; uint8_t v_isSharedCheck_2886_; 
v_name_2841_ = lean_ctor_get(v_a_2839_, 0);
v_size_2842_ = lean_ctor_get(v_m_2838_, 0);
v_buckets_2843_ = lean_ctor_get(v_m_2838_, 1);
v_isSharedCheck_2886_ = !lean_is_exclusive(v_m_2838_);
if (v_isSharedCheck_2886_ == 0)
{
v___x_2845_ = v_m_2838_;
v_isShared_2846_ = v_isSharedCheck_2886_;
goto v_resetjp_2844_;
}
else
{
lean_inc(v_buckets_2843_);
lean_inc(v_size_2842_);
lean_dec(v_m_2838_);
v___x_2845_ = lean_box(0);
v_isShared_2846_ = v_isSharedCheck_2886_;
goto v_resetjp_2844_;
}
v_resetjp_2844_:
{
uint64_t v_hash_2847_; lean_object* v___x_2848_; uint64_t v___x_2849_; uint64_t v___x_2850_; uint64_t v_fold_2851_; uint64_t v___x_2852_; uint64_t v___x_2853_; uint64_t v___x_2854_; size_t v___x_2855_; size_t v___x_2856_; size_t v___x_2857_; size_t v___x_2858_; size_t v___x_2859_; lean_object* v_bkt_2860_; uint8_t v___x_2861_; 
v_hash_2847_ = lean_ctor_get_uint64(v_name_2841_, sizeof(void*)*1);
v___x_2848_ = lean_array_get_size(v_buckets_2843_);
v___x_2849_ = 32ULL;
v___x_2850_ = lean_uint64_shift_right(v_hash_2847_, v___x_2849_);
v_fold_2851_ = lean_uint64_xor(v_hash_2847_, v___x_2850_);
v___x_2852_ = 16ULL;
v___x_2853_ = lean_uint64_shift_right(v_fold_2851_, v___x_2852_);
v___x_2854_ = lean_uint64_xor(v_fold_2851_, v___x_2853_);
v___x_2855_ = lean_uint64_to_usize(v___x_2854_);
v___x_2856_ = lean_usize_of_nat(v___x_2848_);
v___x_2857_ = ((size_t)1ULL);
v___x_2858_ = lean_usize_sub(v___x_2856_, v___x_2857_);
v___x_2859_ = lean_usize_land(v___x_2855_, v___x_2858_);
v_bkt_2860_ = lean_array_uget_borrowed(v_buckets_2843_, v___x_2859_);
v___x_2861_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg(v_a_2839_, v_bkt_2860_);
if (v___x_2861_ == 0)
{
lean_object* v___x_2862_; lean_object* v_size_x27_2863_; lean_object* v___x_2864_; lean_object* v_buckets_x27_2865_; lean_object* v___x_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; uint8_t v___x_2871_; 
v___x_2862_ = lean_unsigned_to_nat(1u);
v_size_x27_2863_ = lean_nat_add(v_size_2842_, v___x_2862_);
lean_dec(v_size_2842_);
lean_inc(v_bkt_2860_);
v___x_2864_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_2864_, 0, v_a_2839_);
lean_ctor_set(v___x_2864_, 1, v_b_2840_);
lean_ctor_set(v___x_2864_, 2, v_bkt_2860_);
v_buckets_x27_2865_ = lean_array_uset(v_buckets_2843_, v___x_2859_, v___x_2864_);
v___x_2866_ = lean_unsigned_to_nat(4u);
v___x_2867_ = lean_nat_mul(v_size_x27_2863_, v___x_2866_);
v___x_2868_ = lean_unsigned_to_nat(3u);
v___x_2869_ = lean_nat_div(v___x_2867_, v___x_2868_);
lean_dec(v___x_2867_);
v___x_2870_ = lean_array_get_size(v_buckets_x27_2865_);
v___x_2871_ = lean_nat_dec_le(v___x_2869_, v___x_2870_);
lean_dec(v___x_2869_);
if (v___x_2871_ == 0)
{
lean_object* v_val_2872_; lean_object* v___x_2874_; 
v_val_2872_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10___redArg(v_buckets_x27_2865_);
if (v_isShared_2846_ == 0)
{
lean_ctor_set(v___x_2845_, 1, v_val_2872_);
lean_ctor_set(v___x_2845_, 0, v_size_x27_2863_);
v___x_2874_ = v___x_2845_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2875_; 
v_reuseFailAlloc_2875_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2875_, 0, v_size_x27_2863_);
lean_ctor_set(v_reuseFailAlloc_2875_, 1, v_val_2872_);
v___x_2874_ = v_reuseFailAlloc_2875_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
return v___x_2874_;
}
}
else
{
lean_object* v___x_2877_; 
if (v_isShared_2846_ == 0)
{
lean_ctor_set(v___x_2845_, 1, v_buckets_x27_2865_);
lean_ctor_set(v___x_2845_, 0, v_size_x27_2863_);
v___x_2877_ = v___x_2845_;
goto v_reusejp_2876_;
}
else
{
lean_object* v_reuseFailAlloc_2878_; 
v_reuseFailAlloc_2878_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2878_, 0, v_size_x27_2863_);
lean_ctor_set(v_reuseFailAlloc_2878_, 1, v_buckets_x27_2865_);
v___x_2877_ = v_reuseFailAlloc_2878_;
goto v_reusejp_2876_;
}
v_reusejp_2876_:
{
return v___x_2877_;
}
}
}
else
{
lean_object* v___x_2879_; lean_object* v_buckets_x27_2880_; lean_object* v___x_2881_; lean_object* v___x_2882_; lean_object* v___x_2884_; 
lean_inc(v_bkt_2860_);
v___x_2879_ = lean_box(0);
v_buckets_x27_2880_ = lean_array_uset(v_buckets_2843_, v___x_2859_, v___x_2879_);
v___x_2881_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11___redArg(v_a_2839_, v_b_2840_, v_bkt_2860_);
v___x_2882_ = lean_array_uset(v_buckets_x27_2880_, v___x_2859_, v___x_2881_);
if (v_isShared_2846_ == 0)
{
lean_ctor_set(v___x_2845_, 1, v___x_2882_);
v___x_2884_ = v___x_2845_;
goto v_reusejp_2883_;
}
else
{
lean_object* v_reuseFailAlloc_2885_; 
v_reuseFailAlloc_2885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2885_, 0, v_size_2842_);
lean_ctor_set(v_reuseFailAlloc_2885_, 1, v___x_2882_);
v___x_2884_ = v_reuseFailAlloc_2885_;
goto v_reusejp_2883_;
}
v_reusejp_2883_:
{
return v___x_2884_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg(lean_object* v_as_2887_, size_t v_sz_2888_, size_t v_i_2889_, lean_object* v_b_2890_){
_start:
{
lean_object* v_a_2893_; uint8_t v___x_2897_; 
v___x_2897_ = lean_usize_dec_lt(v_i_2889_, v_sz_2888_);
if (v___x_2897_ == 0)
{
lean_object* v___x_2898_; 
v___x_2898_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2898_, 0, v_b_2890_);
return v___x_2898_;
}
else
{
lean_object* v_a_2899_; lean_object* v_fst_2900_; lean_object* v_snd_2901_; lean_object* v___x_2902_; 
v_a_2899_ = lean_array_uget_borrowed(v_as_2887_, v_i_2889_);
v_fst_2900_ = lean_ctor_get(v_a_2899_, 0);
v_snd_2901_ = lean_ctor_get(v_a_2899_, 1);
v___x_2902_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg(v_b_2890_, v_fst_2900_);
if (lean_obj_tag(v___x_2902_) == 1)
{
lean_object* v_val_2903_; lean_object* v___x_2904_; lean_object* v___x_2905_; 
v_val_2903_ = lean_ctor_get(v___x_2902_, 0);
lean_inc(v_val_2903_);
lean_dec_ref_known(v___x_2902_, 1);
v___x_2904_ = l_Array_append___redArg(v_val_2903_, v_snd_2901_);
lean_inc(v_fst_2900_);
v___x_2905_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5___redArg(v_b_2890_, v_fst_2900_, v___x_2904_);
v_a_2893_ = v___x_2905_;
goto v___jp_2892_;
}
else
{
lean_object* v___x_2906_; 
lean_dec(v___x_2902_);
lean_inc(v_snd_2901_);
lean_inc(v_fst_2900_);
v___x_2906_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5___redArg(v_b_2890_, v_fst_2900_, v_snd_2901_);
v_a_2893_ = v___x_2906_;
goto v___jp_2892_;
}
}
v___jp_2892_:
{
size_t v___x_2894_; size_t v___x_2895_; 
v___x_2894_ = ((size_t)1ULL);
v___x_2895_ = lean_usize_add(v_i_2889_, v___x_2894_);
v_i_2889_ = v___x_2895_;
v_b_2890_ = v_a_2893_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg___boxed(lean_object* v_as_2907_, lean_object* v_sz_2908_, lean_object* v_i_2909_, lean_object* v_b_2910_, lean_object* v___y_2911_){
_start:
{
size_t v_sz_boxed_2912_; size_t v_i_boxed_2913_; lean_object* v_res_2914_; 
v_sz_boxed_2912_ = lean_unbox_usize(v_sz_2908_);
lean_dec(v_sz_2908_);
v_i_boxed_2913_ = lean_unbox_usize(v_i_2909_);
lean_dec(v_i_2909_);
v_res_2914_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg(v_as_2907_, v_sz_boxed_2912_, v_i_boxed_2913_, v_b_2910_);
lean_dec_ref(v_as_2907_);
return v_res_2914_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg(lean_object* v_as_2915_, size_t v_sz_2916_, size_t v_i_2917_, lean_object* v_b_2918_, lean_object* v___y_2919_, lean_object* v___y_2920_, lean_object* v___y_2921_, lean_object* v___y_2922_){
_start:
{
uint8_t v___x_2924_; 
v___x_2924_ = lean_usize_dec_lt(v_i_2917_, v_sz_2916_);
if (v___x_2924_ == 0)
{
lean_object* v___x_2925_; 
v___x_2925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2925_, 0, v_b_2918_);
return v___x_2925_;
}
else
{
lean_object* v_a_2926_; size_t v_sz_2927_; size_t v___x_2928_; lean_object* v___x_2929_; 
v_a_2926_ = lean_array_uget_borrowed(v_as_2915_, v_i_2917_);
v_sz_2927_ = lean_array_size(v_a_2926_);
v___x_2928_ = ((size_t)0ULL);
v___x_2929_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg(v_a_2926_, v_sz_2927_, v___x_2928_, v_b_2918_);
if (lean_obj_tag(v___x_2929_) == 0)
{
lean_object* v_a_2930_; size_t v___x_2931_; size_t v___x_2932_; 
v_a_2930_ = lean_ctor_get(v___x_2929_, 0);
lean_inc(v_a_2930_);
lean_dec_ref_known(v___x_2929_, 1);
v___x_2931_ = ((size_t)1ULL);
v___x_2932_ = lean_usize_add(v_i_2917_, v___x_2931_);
v_i_2917_ = v___x_2932_;
v_b_2918_ = v_a_2930_;
goto _start;
}
else
{
return v___x_2929_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg___boxed(lean_object* v_as_2934_, lean_object* v_sz_2935_, lean_object* v_i_2936_, lean_object* v_b_2937_, lean_object* v___y_2938_, lean_object* v___y_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_){
_start:
{
size_t v_sz_boxed_2943_; size_t v_i_boxed_2944_; lean_object* v_res_2945_; 
v_sz_boxed_2943_ = lean_unbox_usize(v_sz_2935_);
lean_dec(v_sz_2935_);
v_i_boxed_2944_ = lean_unbox_usize(v_i_2936_);
lean_dec(v_i_2936_);
v_res_2945_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg(v_as_2934_, v_sz_boxed_2943_, v_i_boxed_2944_, v_b_2937_, v___y_2938_, v___y_2939_, v___y_2940_, v___y_2941_);
lean_dec(v___y_2941_);
lean_dec_ref(v___y_2940_);
lean_dec(v___y_2939_);
lean_dec_ref(v___y_2938_);
lean_dec_ref(v_as_2934_);
return v_res_2945_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0(void){
_start:
{
lean_object* v___x_2946_; lean_object* v___x_2947_; lean_object* v___x_2948_; 
v___x_2946_ = lean_box(0);
v___x_2947_ = lean_unsigned_to_nat(16u);
v___x_2948_ = lean_mk_array(v___x_2947_, v___x_2946_);
return v___x_2948_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1(void){
_start:
{
lean_object* v___x_2949_; lean_object* v___x_2950_; lean_object* v_locMap_2951_; 
v___x_2949_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0, &lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__0);
v___x_2950_ = lean_unsigned_to_nat(0u);
v_locMap_2951_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_locMap_2951_, 0, v___x_2950_);
lean_ctor_set(v_locMap_2951_, 1, v___x_2949_);
return v_locMap_2951_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(lean_object* v_patSubstMap_2952_, lean_object* v_acc_2953_, lean_object* v_ruless_2954_, lean_object* v_a_2955_, lean_object* v_a_2956_, lean_object* v_a_2957_, lean_object* v_a_2958_){
_start:
{
lean_object* v_locMap_2960_; size_t v_sz_2961_; size_t v___x_2962_; lean_object* v___x_2963_; 
v_locMap_2960_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___closed__1);
v_sz_2961_ = lean_array_size(v_ruless_2954_);
v___x_2962_ = ((size_t)0ULL);
v___x_2963_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg(v_ruless_2954_, v_sz_2961_, v___x_2962_, v_locMap_2960_, v_a_2955_, v_a_2956_, v_a_2957_, v_a_2958_);
if (lean_obj_tag(v___x_2963_) == 0)
{
lean_object* v_a_2964_; lean_object* v_buckets_2965_; size_t v_sz_2966_; lean_object* v___x_2967_; 
v_a_2964_ = lean_ctor_get(v___x_2963_, 0);
lean_inc(v_a_2964_);
lean_dec_ref_known(v___x_2963_, 1);
v_buckets_2965_ = lean_ctor_get(v_a_2964_, 1);
lean_inc_ref(v_buckets_2965_);
lean_dec(v_a_2964_);
v_sz_2966_ = lean_array_size(v_buckets_2965_);
v___x_2967_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg(v_patSubstMap_2952_, v_buckets_2965_, v_sz_2966_, v___x_2962_, v_acc_2953_, v_a_2955_, v_a_2956_, v_a_2957_, v_a_2958_);
lean_dec_ref(v_buckets_2965_);
return v___x_2967_;
}
else
{
lean_object* v_a_2968_; lean_object* v___x_2970_; uint8_t v_isShared_2971_; uint8_t v_isSharedCheck_2975_; 
lean_dec_ref(v_acc_2953_);
v_a_2968_ = lean_ctor_get(v___x_2963_, 0);
v_isSharedCheck_2975_ = !lean_is_exclusive(v___x_2963_);
if (v_isSharedCheck_2975_ == 0)
{
v___x_2970_ = v___x_2963_;
v_isShared_2971_ = v_isSharedCheck_2975_;
goto v_resetjp_2969_;
}
else
{
lean_inc(v_a_2968_);
lean_dec(v___x_2963_);
v___x_2970_ = lean_box(0);
v_isShared_2971_ = v_isSharedCheck_2975_;
goto v_resetjp_2969_;
}
v_resetjp_2969_:
{
lean_object* v___x_2973_; 
if (v_isShared_2971_ == 0)
{
v___x_2973_ = v___x_2970_;
goto v_reusejp_2972_;
}
else
{
lean_object* v_reuseFailAlloc_2974_; 
v_reuseFailAlloc_2974_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2974_, 0, v_a_2968_);
v___x_2973_ = v_reuseFailAlloc_2974_;
goto v_reusejp_2972_;
}
v_reusejp_2972_:
{
return v___x_2973_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg___boxed(lean_object* v_patSubstMap_2976_, lean_object* v_acc_2977_, lean_object* v_ruless_2978_, lean_object* v_a_2979_, lean_object* v_a_2980_, lean_object* v_a_2981_, lean_object* v_a_2982_, lean_object* v_a_2983_){
_start:
{
lean_object* v_res_2984_; 
v_res_2984_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_2976_, v_acc_2977_, v_ruless_2978_, v_a_2979_, v_a_2980_, v_a_2981_, v_a_2982_);
lean_dec(v_a_2982_);
lean_dec_ref(v_a_2981_);
lean_dec(v_a_2980_);
lean_dec_ref(v_a_2979_);
lean_dec_ref(v_ruless_2978_);
lean_dec_ref(v_patSubstMap_2976_);
return v_res_2984_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules(lean_object* v_00_u03b1_2985_, lean_object* v_patSubstMap_2986_, lean_object* v_acc_2987_, lean_object* v_ruless_2988_, lean_object* v_a_2989_, lean_object* v_a_2990_, lean_object* v_a_2991_, lean_object* v_a_2992_){
_start:
{
lean_object* v___x_2994_; 
v___x_2994_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_2986_, v_acc_2987_, v_ruless_2988_, v_a_2989_, v_a_2990_, v_a_2991_, v_a_2992_);
return v___x_2994_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___boxed(lean_object* v_00_u03b1_2995_, lean_object* v_patSubstMap_2996_, lean_object* v_acc_2997_, lean_object* v_ruless_2998_, lean_object* v_a_2999_, lean_object* v_a_3000_, lean_object* v_a_3001_, lean_object* v_a_3002_, lean_object* v_a_3003_){
_start:
{
lean_object* v_res_3004_; 
v_res_3004_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules(v_00_u03b1_2995_, v_patSubstMap_2996_, v_acc_2997_, v_ruless_2998_, v_a_2999_, v_a_3000_, v_a_3001_, v_a_3002_);
lean_dec(v_a_3002_);
lean_dec_ref(v_a_3001_);
lean_dec(v_a_3000_);
lean_dec_ref(v_a_2999_);
lean_dec_ref(v_ruless_2998_);
lean_dec_ref(v_patSubstMap_2996_);
return v_res_3004_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2(lean_object* v_00_u03b2_3005_, lean_object* v_m_3006_, lean_object* v_a_3007_){
_start:
{
lean_object* v___x_3008_; 
v___x_3008_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___redArg(v_m_3006_, v_a_3007_);
return v___x_3008_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2___boxed(lean_object* v_00_u03b2_3009_, lean_object* v_m_3010_, lean_object* v_a_3011_){
_start:
{
lean_object* v_res_3012_; 
v_res_3012_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2(v_00_u03b2_3009_, v_m_3010_, v_a_3011_);
lean_dec_ref(v_a_3011_);
lean_dec_ref(v_m_3010_);
return v_res_3012_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3(lean_object* v_00_u03b1_3013_, lean_object* v_patSubstMap_3014_, lean_object* v_a_3015_, lean_object* v_a_3016_, lean_object* v___y_3017_, lean_object* v___y_3018_, lean_object* v___y_3019_, lean_object* v___y_3020_){
_start:
{
lean_object* v___x_3022_; 
v___x_3022_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___redArg(v_patSubstMap_3014_, v_a_3015_, v_a_3016_);
return v___x_3022_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3___boxed(lean_object* v_00_u03b1_3023_, lean_object* v_patSubstMap_3024_, lean_object* v_a_3025_, lean_object* v_a_3026_, lean_object* v___y_3027_, lean_object* v___y_3028_, lean_object* v___y_3029_, lean_object* v___y_3030_, lean_object* v___y_3031_){
_start:
{
lean_object* v_res_3032_; 
v_res_3032_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__3(v_00_u03b1_3023_, v_patSubstMap_3024_, v_a_3025_, v_a_3026_, v___y_3027_, v___y_3028_, v___y_3029_, v___y_3030_);
lean_dec(v___y_3030_);
lean_dec_ref(v___y_3029_);
lean_dec(v___y_3028_);
lean_dec_ref(v___y_3027_);
lean_dec_ref(v_patSubstMap_3024_);
return v_res_3032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4(lean_object* v_00_u03b1_3033_, lean_object* v_00_u03b2_3034_, lean_object* v_m_3035_, lean_object* v_a_3036_){
_start:
{
lean_object* v___x_3037_; 
v___x_3037_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___redArg(v_m_3035_, v_a_3036_);
return v___x_3037_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4___boxed(lean_object* v_00_u03b1_3038_, lean_object* v_00_u03b2_3039_, lean_object* v_m_3040_, lean_object* v_a_3041_){
_start:
{
lean_object* v_res_3042_; 
v_res_3042_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4(v_00_u03b1_3038_, v_00_u03b2_3039_, v_m_3040_, v_a_3041_);
lean_dec_ref(v_a_3041_);
lean_dec_ref(v_m_3040_);
return v_res_3042_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5(lean_object* v_00_u03b1_3043_, lean_object* v_00_u03b2_3044_, lean_object* v_m_3045_, lean_object* v_a_3046_, lean_object* v_b_3047_){
_start:
{
lean_object* v___x_3048_; 
v___x_3048_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5___redArg(v_m_3045_, v_a_3046_, v_b_3047_);
return v___x_3048_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6(lean_object* v_00_u03b1_3049_, lean_object* v_as_3050_, size_t v_sz_3051_, size_t v_i_3052_, lean_object* v_b_3053_, lean_object* v___y_3054_, lean_object* v___y_3055_, lean_object* v___y_3056_, lean_object* v___y_3057_){
_start:
{
lean_object* v___x_3059_; 
v___x_3059_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___redArg(v_as_3050_, v_sz_3051_, v_i_3052_, v_b_3053_);
return v___x_3059_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6___boxed(lean_object* v_00_u03b1_3060_, lean_object* v_as_3061_, lean_object* v_sz_3062_, lean_object* v_i_3063_, lean_object* v_b_3064_, lean_object* v___y_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_){
_start:
{
size_t v_sz_boxed_3070_; size_t v_i_boxed_3071_; lean_object* v_res_3072_; 
v_sz_boxed_3070_ = lean_unbox_usize(v_sz_3062_);
lean_dec(v_sz_3062_);
v_i_boxed_3071_ = lean_unbox_usize(v_i_3063_);
lean_dec(v_i_3063_);
v_res_3072_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__6(v_00_u03b1_3060_, v_as_3061_, v_sz_boxed_3070_, v_i_boxed_3071_, v_b_3064_, v___y_3065_, v___y_3066_, v___y_3067_, v___y_3068_);
lean_dec(v___y_3068_);
lean_dec_ref(v___y_3067_);
lean_dec(v___y_3066_);
lean_dec_ref(v___y_3065_);
lean_dec_ref(v_as_3061_);
return v_res_3072_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7(lean_object* v_00_u03b1_3073_, lean_object* v_as_3074_, size_t v_sz_3075_, size_t v_i_3076_, lean_object* v_b_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_, lean_object* v___y_3080_, lean_object* v___y_3081_){
_start:
{
lean_object* v___x_3083_; 
v___x_3083_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___redArg(v_as_3074_, v_sz_3075_, v_i_3076_, v_b_3077_, v___y_3078_, v___y_3079_, v___y_3080_, v___y_3081_);
return v___x_3083_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7___boxed(lean_object* v_00_u03b1_3084_, lean_object* v_as_3085_, lean_object* v_sz_3086_, lean_object* v_i_3087_, lean_object* v_b_3088_, lean_object* v___y_3089_, lean_object* v___y_3090_, lean_object* v___y_3091_, lean_object* v___y_3092_, lean_object* v___y_3093_){
_start:
{
size_t v_sz_boxed_3094_; size_t v_i_boxed_3095_; lean_object* v_res_3096_; 
v_sz_boxed_3094_ = lean_unbox_usize(v_sz_3086_);
lean_dec(v_sz_3086_);
v_i_boxed_3095_ = lean_unbox_usize(v_i_3087_);
lean_dec(v_i_3087_);
v_res_3096_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__7(v_00_u03b1_3084_, v_as_3085_, v_sz_boxed_3094_, v_i_boxed_3095_, v_b_3088_, v___y_3089_, v___y_3090_, v___y_3091_, v___y_3092_);
lean_dec(v___y_3092_);
lean_dec_ref(v___y_3091_);
lean_dec(v___y_3090_);
lean_dec_ref(v___y_3089_);
lean_dec_ref(v_as_3085_);
return v_res_3096_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8(lean_object* v_00_u03b1_3097_, lean_object* v_patSubstMap_3098_, lean_object* v_as_3099_, size_t v_sz_3100_, size_t v_i_3101_, lean_object* v_b_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_){
_start:
{
lean_object* v___x_3108_; 
v___x_3108_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___redArg(v_patSubstMap_3098_, v_as_3099_, v_sz_3100_, v_i_3101_, v_b_3102_, v___y_3103_, v___y_3104_, v___y_3105_, v___y_3106_);
return v___x_3108_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8___boxed(lean_object* v_00_u03b1_3109_, lean_object* v_patSubstMap_3110_, lean_object* v_as_3111_, lean_object* v_sz_3112_, lean_object* v_i_3113_, lean_object* v_b_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_){
_start:
{
size_t v_sz_boxed_3120_; size_t v_i_boxed_3121_; lean_object* v_res_3122_; 
v_sz_boxed_3120_ = lean_unbox_usize(v_sz_3112_);
lean_dec(v_sz_3112_);
v_i_boxed_3121_ = lean_unbox_usize(v_i_3113_);
lean_dec(v_i_3113_);
v_res_3122_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__8(v_00_u03b1_3109_, v_patSubstMap_3110_, v_as_3111_, v_sz_boxed_3120_, v_i_boxed_3121_, v_b_3114_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
lean_dec(v___y_3116_);
lean_dec_ref(v___y_3115_);
lean_dec_ref(v_as_3111_);
lean_dec_ref(v_patSubstMap_3110_);
return v_res_3122_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0(lean_object* v_n_3123_, lean_object* v_as_3124_, lean_object* v_lo_3125_, lean_object* v_hi_3126_, lean_object* v_w_3127_, lean_object* v_hlo_3128_, lean_object* v_hhi_3129_){
_start:
{
lean_object* v___x_3130_; 
v___x_3130_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___redArg(v_n_3123_, v_as_3124_, v_lo_3125_, v_hi_3126_);
return v___x_3130_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0___boxed(lean_object* v_n_3131_, lean_object* v_as_3132_, lean_object* v_lo_3133_, lean_object* v_hi_3134_, lean_object* v_w_3135_, lean_object* v_hlo_3136_, lean_object* v_hhi_3137_){
_start:
{
lean_object* v_res_3138_; 
v_res_3138_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0(v_n_3131_, v_as_3132_, v_lo_3133_, v_hi_3134_, v_w_3135_, v_hlo_3136_, v_hhi_3137_);
lean_dec(v_hi_3134_);
lean_dec(v_n_3131_);
return v_res_3138_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4(lean_object* v_00_u03b2_3139_, lean_object* v_a_3140_, lean_object* v_x_3141_){
_start:
{
lean_object* v___x_3142_; 
v___x_3142_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___redArg(v_a_3140_, v_x_3141_);
return v___x_3142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4___boxed(lean_object* v_00_u03b2_3143_, lean_object* v_a_3144_, lean_object* v_x_3145_){
_start:
{
lean_object* v_res_3146_; 
v_res_3146_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__2_spec__4(v_00_u03b2_3143_, v_a_3144_, v_x_3145_);
lean_dec(v_x_3145_);
lean_dec_ref(v_a_3144_);
return v_res_3146_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7(lean_object* v_00_u03b1_3147_, lean_object* v_00_u03b2_3148_, lean_object* v_a_3149_, lean_object* v_x_3150_){
_start:
{
lean_object* v___x_3151_; 
v___x_3151_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___redArg(v_a_3149_, v_x_3150_);
return v___x_3151_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7___boxed(lean_object* v_00_u03b1_3152_, lean_object* v_00_u03b2_3153_, lean_object* v_a_3154_, lean_object* v_x_3155_){
_start:
{
lean_object* v_res_3156_; 
v_res_3156_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__4_spec__7(v_00_u03b1_3152_, v_00_u03b2_3153_, v_a_3154_, v_x_3155_);
lean_dec(v_x_3155_);
lean_dec_ref(v_a_3154_);
return v_res_3156_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9(lean_object* v_00_u03b1_3157_, lean_object* v_00_u03b2_3158_, lean_object* v_a_3159_, lean_object* v_x_3160_){
_start:
{
uint8_t v___x_3161_; 
v___x_3161_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___redArg(v_a_3159_, v_x_3160_);
return v___x_3161_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9___boxed(lean_object* v_00_u03b1_3162_, lean_object* v_00_u03b2_3163_, lean_object* v_a_3164_, lean_object* v_x_3165_){
_start:
{
uint8_t v_res_3166_; lean_object* v_r_3167_; 
v_res_3166_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__9(v_00_u03b1_3162_, v_00_u03b2_3163_, v_a_3164_, v_x_3165_);
lean_dec(v_x_3165_);
lean_dec_ref(v_a_3164_);
v_r_3167_ = lean_box(v_res_3166_);
return v_r_3167_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10(lean_object* v_00_u03b1_3168_, lean_object* v_00_u03b2_3169_, lean_object* v_data_3170_){
_start:
{
lean_object* v___x_3171_; 
v___x_3171_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10___redArg(v_data_3170_);
return v___x_3171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11(lean_object* v_00_u03b1_3172_, lean_object* v_00_u03b2_3173_, lean_object* v_a_3174_, lean_object* v_b_3175_, lean_object* v_x_3176_){
_start:
{
lean_object* v___x_3177_; 
v___x_3177_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__11___redArg(v_a_3174_, v_b_3175_, v_x_3176_);
return v___x_3177_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1(lean_object* v_n_3178_, lean_object* v_lo_3179_, lean_object* v_hi_3180_, lean_object* v_hhi_3181_, lean_object* v_pivot_3182_, lean_object* v_as_3183_, lean_object* v_i_3184_, lean_object* v_k_3185_, lean_object* v_ilo_3186_, lean_object* v_ik_3187_, lean_object* v_w_3188_){
_start:
{
lean_object* v___x_3189_; 
v___x_3189_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___redArg(v_hi_3180_, v_pivot_3182_, v_as_3183_, v_i_3184_, v_k_3185_);
return v___x_3189_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1___boxed(lean_object* v_n_3190_, lean_object* v_lo_3191_, lean_object* v_hi_3192_, lean_object* v_hhi_3193_, lean_object* v_pivot_3194_, lean_object* v_as_3195_, lean_object* v_i_3196_, lean_object* v_k_3197_, lean_object* v_ilo_3198_, lean_object* v_ik_3199_, lean_object* v_w_3200_){
_start:
{
lean_object* v_res_3201_; 
v_res_3201_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__0_spec__0_spec__1(v_n_3190_, v_lo_3191_, v_hi_3192_, v_hhi_3193_, v_pivot_3194_, v_as_3195_, v_i_3196_, v_k_3197_, v_ilo_3198_, v_ik_3199_, v_w_3200_);
lean_dec(v_pivot_3194_);
lean_dec(v_hi_3192_);
lean_dec(v_lo_3191_);
lean_dec(v_n_3190_);
return v_res_3201_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13(lean_object* v_00_u03b1_3202_, lean_object* v_00_u03b2_3203_, lean_object* v_i_3204_, lean_object* v_source_3205_, lean_object* v_target_3206_){
_start:
{
lean_object* v___x_3207_; 
v___x_3207_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13___redArg(v_i_3204_, v_source_3205_, v_target_3206_);
return v___x_3207_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17(lean_object* v_00_u03b1_3208_, lean_object* v_00_u03b2_3209_, lean_object* v_x_3210_, lean_object* v_x_3211_){
_start:
{
lean_object* v___x_3212_; 
v___x_3212_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Index_0__Aesop_Index_applicableRules_addRules_spec__5_spec__10_spec__13_spec__17___redArg(v_x_3210_, v_x_3211_);
return v___x_3212_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6(uint8_t v_a_3227_, lean_object* v_x_3228_){
_start:
{
lean_object* v_rule_3229_; lean_object* v_name_3230_; lean_object* v_name_3231_; uint8_t v_builder_3232_; uint8_t v_phase_3233_; uint8_t v_scope_3234_; lean_object* v___y_3236_; lean_object* v___y_3237_; lean_object* v___y_3238_; lean_object* v___y_3246_; lean_object* v___y_3247_; lean_object* v___y_3248_; lean_object* v___y_3254_; 
v_rule_3229_ = lean_ctor_get(v_x_3228_, 0);
lean_inc(v_rule_3229_);
lean_dec_ref(v_x_3228_);
v_name_3230_ = lean_ctor_get(v_rule_3229_, 0);
lean_inc_ref(v_name_3230_);
lean_dec(v_rule_3229_);
v_name_3231_ = lean_ctor_get(v_name_3230_, 0);
lean_inc(v_name_3231_);
v_builder_3232_ = lean_ctor_get_uint8(v_name_3230_, sizeof(void*)*1 + 8);
v_phase_3233_ = lean_ctor_get_uint8(v_name_3230_, sizeof(void*)*1 + 9);
v_scope_3234_ = lean_ctor_get_uint8(v_name_3230_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_3230_);
switch(v_phase_3233_)
{
case 0:
{
lean_object* v___x_3265_; 
v___x_3265_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11));
v___y_3254_ = v___x_3265_;
goto v___jp_3253_;
}
case 1:
{
lean_object* v___x_3266_; 
v___x_3266_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12));
v___y_3254_ = v___x_3266_;
goto v___jp_3253_;
}
default: 
{
lean_object* v___x_3267_; 
v___x_3267_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13));
v___y_3254_ = v___x_3267_;
goto v___jp_3253_;
}
}
v___jp_3235_:
{
lean_object* v___x_3239_; lean_object* v___x_3240_; lean_object* v___x_3241_; lean_object* v___x_3242_; lean_object* v___x_3243_; lean_object* v___x_3244_; 
v___x_3239_ = lean_string_append(v___y_3236_, v___y_3238_);
v___x_3240_ = lean_string_append(v___x_3239_, v___y_3237_);
v___x_3241_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3231_, v_a_3227_);
v___x_3242_ = lean_string_append(v___x_3240_, v___x_3241_);
lean_dec_ref(v___x_3241_);
v___x_3243_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3243_, 0, v___x_3242_);
v___x_3244_ = l_Lean_MessageData_ofFormat(v___x_3243_);
return v___x_3244_;
}
v___jp_3245_:
{
lean_object* v___x_3249_; lean_object* v___x_3250_; 
v___x_3249_ = lean_string_append(v___y_3246_, v___y_3248_);
v___x_3250_ = lean_string_append(v___x_3249_, v___y_3247_);
if (v_scope_3234_ == 0)
{
lean_object* v___x_3251_; 
v___x_3251_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0));
v___y_3236_ = v___x_3250_;
v___y_3237_ = v___y_3247_;
v___y_3238_ = v___x_3251_;
goto v___jp_3235_;
}
else
{
lean_object* v___x_3252_; 
v___x_3252_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1));
v___y_3236_ = v___x_3250_;
v___y_3237_ = v___y_3247_;
v___y_3238_ = v___x_3252_;
goto v___jp_3235_;
}
}
v___jp_3253_:
{
lean_object* v___x_3255_; lean_object* v___x_3256_; 
v___x_3255_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2));
lean_inc_ref(v___y_3254_);
v___x_3256_ = lean_string_append(v___y_3254_, v___x_3255_);
switch(v_builder_3232_)
{
case 0:
{
lean_object* v___x_3257_; 
v___x_3257_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3257_;
goto v___jp_3245_;
}
case 1:
{
lean_object* v___x_3258_; 
v___x_3258_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3258_;
goto v___jp_3245_;
}
case 2:
{
lean_object* v___x_3259_; 
v___x_3259_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3259_;
goto v___jp_3245_;
}
case 3:
{
lean_object* v___x_3260_; 
v___x_3260_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3260_;
goto v___jp_3245_;
}
case 4:
{
lean_object* v___x_3261_; 
v___x_3261_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3261_;
goto v___jp_3245_;
}
case 5:
{
lean_object* v___x_3262_; 
v___x_3262_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3262_;
goto v___jp_3245_;
}
case 6:
{
lean_object* v___x_3263_; 
v___x_3263_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3263_;
goto v___jp_3245_;
}
default: 
{
lean_object* v___x_3264_; 
v___x_3264_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10));
v___y_3246_ = v___x_3256_;
v___y_3247_ = v___x_3255_;
v___y_3248_ = v___x_3264_;
goto v___jp_3245_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___boxed(lean_object* v_a_3268_, lean_object* v_x_3269_){
_start:
{
uint8_t v_a_17846__boxed_3270_; lean_object* v_res_3271_; 
v_a_17846__boxed_3270_ = lean_unbox(v_a_3268_);
v_res_3271_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__6(v_a_17846__boxed_3270_, v_x_3269_);
return v_res_3271_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__0(lean_object* v_inst_3272_, uint8_t v_hasTrace_3273_, lean_object* v_x_3274_, lean_object* v_y_3275_){
_start:
{
lean_object* v_rule_3276_; lean_object* v_rule_3277_; uint8_t v___x_3278_; 
v_rule_3276_ = lean_ctor_get(v_x_3274_, 0);
lean_inc(v_rule_3276_);
lean_dec_ref(v_x_3274_);
v_rule_3277_ = lean_ctor_get(v_y_3275_, 0);
lean_inc(v_rule_3277_);
lean_dec_ref(v_y_3275_);
v___x_3278_ = lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(v_inst_3272_, v_rule_3276_, v_rule_3277_);
if (v___x_3278_ == 0)
{
uint8_t v___x_3279_; 
v___x_3279_ = 1;
return v___x_3279_;
}
else
{
return v_hasTrace_3273_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__0___boxed(lean_object* v_inst_3280_, lean_object* v_hasTrace_3281_, lean_object* v_x_3282_, lean_object* v_y_3283_){
_start:
{
uint8_t v_hasTrace_boxed_3284_; uint8_t v_res_3285_; lean_object* v_r_3286_; 
v_hasTrace_boxed_3284_ = lean_unbox(v_hasTrace_3281_);
v_res_3285_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__0(v_inst_3280_, v_hasTrace_boxed_3284_, v_x_3282_, v_y_3283_);
v_r_3286_ = lean_box(v_res_3285_);
return v_r_3286_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__12(uint8_t v_hasTrace_3287_, lean_object* v_x_3288_){
_start:
{
lean_object* v_rule_3289_; lean_object* v_name_3290_; lean_object* v_name_3291_; uint8_t v_builder_3292_; uint8_t v_phase_3293_; uint8_t v_scope_3294_; lean_object* v___y_3296_; lean_object* v___y_3297_; lean_object* v___y_3298_; lean_object* v___y_3306_; lean_object* v___y_3307_; lean_object* v___y_3308_; lean_object* v___y_3314_; 
v_rule_3289_ = lean_ctor_get(v_x_3288_, 0);
lean_inc(v_rule_3289_);
lean_dec_ref(v_x_3288_);
v_name_3290_ = lean_ctor_get(v_rule_3289_, 0);
lean_inc_ref(v_name_3290_);
lean_dec(v_rule_3289_);
v_name_3291_ = lean_ctor_get(v_name_3290_, 0);
lean_inc(v_name_3291_);
v_builder_3292_ = lean_ctor_get_uint8(v_name_3290_, sizeof(void*)*1 + 8);
v_phase_3293_ = lean_ctor_get_uint8(v_name_3290_, sizeof(void*)*1 + 9);
v_scope_3294_ = lean_ctor_get_uint8(v_name_3290_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_3290_);
switch(v_phase_3293_)
{
case 0:
{
lean_object* v___x_3325_; 
v___x_3325_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11));
v___y_3314_ = v___x_3325_;
goto v___jp_3313_;
}
case 1:
{
lean_object* v___x_3326_; 
v___x_3326_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12));
v___y_3314_ = v___x_3326_;
goto v___jp_3313_;
}
default: 
{
lean_object* v___x_3327_; 
v___x_3327_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13));
v___y_3314_ = v___x_3327_;
goto v___jp_3313_;
}
}
v___jp_3295_:
{
lean_object* v___x_3299_; lean_object* v___x_3300_; lean_object* v___x_3301_; lean_object* v___x_3302_; lean_object* v___x_3303_; lean_object* v___x_3304_; 
v___x_3299_ = lean_string_append(v___y_3297_, v___y_3298_);
v___x_3300_ = lean_string_append(v___x_3299_, v___y_3296_);
v___x_3301_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3291_, v_hasTrace_3287_);
v___x_3302_ = lean_string_append(v___x_3300_, v___x_3301_);
lean_dec_ref(v___x_3301_);
v___x_3303_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3303_, 0, v___x_3302_);
v___x_3304_ = l_Lean_MessageData_ofFormat(v___x_3303_);
return v___x_3304_;
}
v___jp_3305_:
{
lean_object* v___x_3309_; lean_object* v___x_3310_; 
v___x_3309_ = lean_string_append(v___y_3306_, v___y_3308_);
v___x_3310_ = lean_string_append(v___x_3309_, v___y_3307_);
if (v_scope_3294_ == 0)
{
lean_object* v___x_3311_; 
v___x_3311_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0));
v___y_3296_ = v___y_3307_;
v___y_3297_ = v___x_3310_;
v___y_3298_ = v___x_3311_;
goto v___jp_3295_;
}
else
{
lean_object* v___x_3312_; 
v___x_3312_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1));
v___y_3296_ = v___y_3307_;
v___y_3297_ = v___x_3310_;
v___y_3298_ = v___x_3312_;
goto v___jp_3295_;
}
}
v___jp_3313_:
{
lean_object* v___x_3315_; lean_object* v___x_3316_; 
v___x_3315_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2));
lean_inc_ref(v___y_3314_);
v___x_3316_ = lean_string_append(v___y_3314_, v___x_3315_);
switch(v_builder_3292_)
{
case 0:
{
lean_object* v___x_3317_; 
v___x_3317_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3317_;
goto v___jp_3305_;
}
case 1:
{
lean_object* v___x_3318_; 
v___x_3318_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3318_;
goto v___jp_3305_;
}
case 2:
{
lean_object* v___x_3319_; 
v___x_3319_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3319_;
goto v___jp_3305_;
}
case 3:
{
lean_object* v___x_3320_; 
v___x_3320_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3320_;
goto v___jp_3305_;
}
case 4:
{
lean_object* v___x_3321_; 
v___x_3321_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3321_;
goto v___jp_3305_;
}
case 5:
{
lean_object* v___x_3322_; 
v___x_3322_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3322_;
goto v___jp_3305_;
}
case 6:
{
lean_object* v___x_3323_; 
v___x_3323_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3323_;
goto v___jp_3305_;
}
default: 
{
lean_object* v___x_3324_; 
v___x_3324_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10));
v___y_3306_ = v___x_3316_;
v___y_3307_ = v___x_3315_;
v___y_3308_ = v___x_3324_;
goto v___jp_3305_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__12___boxed(lean_object* v_hasTrace_3328_, lean_object* v_x_3329_){
_start:
{
uint8_t v_hasTrace_boxed_3330_; lean_object* v_res_3331_; 
v_hasTrace_boxed_3330_ = lean_unbox(v_hasTrace_3328_);
v_res_3331_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__12(v_hasTrace_boxed_3330_, v_x_3329_);
return v_res_3331_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__1(lean_object* v___x_3332_, lean_object* v_x_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_, lean_object* v___y_3336_, lean_object* v___y_3337_){
_start:
{
lean_object* v___x_3339_; 
v___x_3339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3339_, 0, v___x_3332_);
return v___x_3339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__1___boxed(lean_object* v___x_3340_, lean_object* v_x_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_, lean_object* v___y_3345_, lean_object* v___y_3346_){
_start:
{
lean_object* v_res_3347_; 
v_res_3347_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__1(v___x_3340_, v_x_3341_, v___y_3342_, v___y_3343_, v___y_3344_, v___y_3345_);
lean_dec(v___y_3345_);
lean_dec_ref(v___y_3344_);
lean_dec(v___y_3343_);
lean_dec_ref(v___y_3342_);
lean_dec_ref(v_x_3341_);
return v_res_3347_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__3(lean_object* v_inst_3348_, uint8_t v_hasTrace_3349_, uint8_t v___x_3350_, lean_object* v_x_3351_, lean_object* v_y_3352_){
_start:
{
lean_object* v_rule_3353_; lean_object* v_rule_3354_; uint8_t v___x_3355_; 
v_rule_3353_ = lean_ctor_get(v_x_3351_, 0);
lean_inc(v_rule_3353_);
lean_dec_ref(v_x_3351_);
v_rule_3354_ = lean_ctor_get(v_y_3352_, 0);
lean_inc(v_rule_3354_);
lean_dec_ref(v_y_3352_);
v___x_3355_ = lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(v_inst_3348_, v_rule_3353_, v_rule_3354_);
if (v___x_3355_ == 0)
{
return v_hasTrace_3349_;
}
else
{
return v___x_3350_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__3___boxed(lean_object* v_inst_3356_, lean_object* v_hasTrace_3357_, lean_object* v___x_3358_, lean_object* v_x_3359_, lean_object* v_y_3360_){
_start:
{
uint8_t v_hasTrace_boxed_3361_; uint8_t v___x_18046__boxed_3362_; uint8_t v_res_3363_; lean_object* v_r_3364_; 
v_hasTrace_boxed_3361_ = lean_unbox(v_hasTrace_3357_);
v___x_18046__boxed_3362_ = lean_unbox(v___x_3358_);
v_res_3363_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__3(v_inst_3356_, v_hasTrace_boxed_3361_, v___x_18046__boxed_3362_, v_x_3359_, v_y_3360_);
v_r_3364_ = lean_box(v_res_3363_);
return v_r_3364_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__4(uint8_t v___x_3365_, lean_object* v_x_3366_){
_start:
{
lean_object* v_rule_3367_; lean_object* v_name_3368_; lean_object* v_name_3369_; uint8_t v_builder_3370_; uint8_t v_phase_3371_; uint8_t v_scope_3372_; lean_object* v___y_3374_; lean_object* v___y_3375_; lean_object* v___y_3376_; lean_object* v___y_3384_; lean_object* v___y_3385_; lean_object* v___y_3386_; lean_object* v___y_3392_; 
v_rule_3367_ = lean_ctor_get(v_x_3366_, 0);
lean_inc(v_rule_3367_);
lean_dec_ref(v_x_3366_);
v_name_3368_ = lean_ctor_get(v_rule_3367_, 0);
lean_inc_ref(v_name_3368_);
lean_dec(v_rule_3367_);
v_name_3369_ = lean_ctor_get(v_name_3368_, 0);
lean_inc(v_name_3369_);
v_builder_3370_ = lean_ctor_get_uint8(v_name_3368_, sizeof(void*)*1 + 8);
v_phase_3371_ = lean_ctor_get_uint8(v_name_3368_, sizeof(void*)*1 + 9);
v_scope_3372_ = lean_ctor_get_uint8(v_name_3368_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_3368_);
switch(v_phase_3371_)
{
case 0:
{
lean_object* v___x_3403_; 
v___x_3403_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11));
v___y_3392_ = v___x_3403_;
goto v___jp_3391_;
}
case 1:
{
lean_object* v___x_3404_; 
v___x_3404_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12));
v___y_3392_ = v___x_3404_;
goto v___jp_3391_;
}
default: 
{
lean_object* v___x_3405_; 
v___x_3405_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13));
v___y_3392_ = v___x_3405_;
goto v___jp_3391_;
}
}
v___jp_3373_:
{
lean_object* v___x_3377_; lean_object* v___x_3378_; lean_object* v___x_3379_; lean_object* v___x_3380_; lean_object* v___x_3381_; lean_object* v___x_3382_; 
v___x_3377_ = lean_string_append(v___y_3375_, v___y_3376_);
v___x_3378_ = lean_string_append(v___x_3377_, v___y_3374_);
v___x_3379_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3369_, v___x_3365_);
v___x_3380_ = lean_string_append(v___x_3378_, v___x_3379_);
lean_dec_ref(v___x_3379_);
v___x_3381_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3381_, 0, v___x_3380_);
v___x_3382_ = l_Lean_MessageData_ofFormat(v___x_3381_);
return v___x_3382_;
}
v___jp_3383_:
{
lean_object* v___x_3387_; lean_object* v___x_3388_; 
v___x_3387_ = lean_string_append(v___y_3385_, v___y_3386_);
v___x_3388_ = lean_string_append(v___x_3387_, v___y_3384_);
if (v_scope_3372_ == 0)
{
lean_object* v___x_3389_; 
v___x_3389_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0));
v___y_3374_ = v___y_3384_;
v___y_3375_ = v___x_3388_;
v___y_3376_ = v___x_3389_;
goto v___jp_3373_;
}
else
{
lean_object* v___x_3390_; 
v___x_3390_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1));
v___y_3374_ = v___y_3384_;
v___y_3375_ = v___x_3388_;
v___y_3376_ = v___x_3390_;
goto v___jp_3373_;
}
}
v___jp_3391_:
{
lean_object* v___x_3393_; lean_object* v___x_3394_; 
v___x_3393_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2));
lean_inc_ref(v___y_3392_);
v___x_3394_ = lean_string_append(v___y_3392_, v___x_3393_);
switch(v_builder_3370_)
{
case 0:
{
lean_object* v___x_3395_; 
v___x_3395_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3395_;
goto v___jp_3383_;
}
case 1:
{
lean_object* v___x_3396_; 
v___x_3396_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3396_;
goto v___jp_3383_;
}
case 2:
{
lean_object* v___x_3397_; 
v___x_3397_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3397_;
goto v___jp_3383_;
}
case 3:
{
lean_object* v___x_3398_; 
v___x_3398_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3398_;
goto v___jp_3383_;
}
case 4:
{
lean_object* v___x_3399_; 
v___x_3399_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3399_;
goto v___jp_3383_;
}
case 5:
{
lean_object* v___x_3400_; 
v___x_3400_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3400_;
goto v___jp_3383_;
}
case 6:
{
lean_object* v___x_3401_; 
v___x_3401_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3401_;
goto v___jp_3383_;
}
default: 
{
lean_object* v___x_3402_; 
v___x_3402_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10));
v___y_3384_ = v___x_3393_;
v___y_3385_ = v___x_3394_;
v___y_3386_ = v___x_3402_;
goto v___jp_3383_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__4___boxed(lean_object* v___x_3406_, lean_object* v_x_3407_){
_start:
{
uint8_t v___x_18067__boxed_3408_; lean_object* v_res_3409_; 
v___x_18067__boxed_3408_ = lean_unbox(v___x_3406_);
v_res_3409_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__4(v___x_18067__boxed_3408_, v_x_3407_);
return v_res_3409_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Index_applicableRules___redArg___lam__2(lean_object* v_inst_3410_, uint8_t v___x_3411_, uint8_t v___x_3412_, lean_object* v_x_3413_, lean_object* v_y_3414_){
_start:
{
lean_object* v_rule_3415_; lean_object* v_rule_3416_; uint8_t v___x_3417_; 
v_rule_3415_ = lean_ctor_get(v_x_3413_, 0);
lean_inc(v_rule_3415_);
lean_dec_ref(v_x_3413_);
v_rule_3416_ = lean_ctor_get(v_y_3414_, 0);
lean_inc(v_rule_3416_);
lean_dec_ref(v_y_3414_);
v___x_3417_ = lp_aesop_Aesop_Rule_compareByPriorityThenName___redArg(v_inst_3410_, v_rule_3415_, v_rule_3416_);
if (v___x_3417_ == 0)
{
return v___x_3411_;
}
else
{
return v___x_3412_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__2___boxed(lean_object* v_inst_3418_, lean_object* v___x_3419_, lean_object* v___x_3420_, lean_object* v_x_3421_, lean_object* v_y_3422_){
_start:
{
uint8_t v___x_18140__boxed_3423_; uint8_t v___x_18141__boxed_3424_; uint8_t v_res_3425_; lean_object* v_r_3426_; 
v___x_18140__boxed_3423_ = lean_unbox(v___x_3419_);
v___x_18141__boxed_3424_ = lean_unbox(v___x_3420_);
v_res_3425_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__2(v_inst_3418_, v___x_18140__boxed_3423_, v___x_18141__boxed_3424_, v_x_3421_, v_y_3422_);
v_r_3426_ = lean_box(v_res_3425_);
return v_r_3426_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__13(uint8_t v_hasTrace_3427_, lean_object* v_x_3428_){
_start:
{
lean_object* v_rule_3429_; lean_object* v_name_3430_; lean_object* v_name_3431_; uint8_t v_builder_3432_; uint8_t v_phase_3433_; uint8_t v_scope_3434_; lean_object* v___y_3436_; lean_object* v___y_3437_; lean_object* v___y_3438_; lean_object* v___y_3446_; lean_object* v___y_3447_; lean_object* v___y_3448_; lean_object* v___y_3454_; 
v_rule_3429_ = lean_ctor_get(v_x_3428_, 0);
lean_inc(v_rule_3429_);
lean_dec_ref(v_x_3428_);
v_name_3430_ = lean_ctor_get(v_rule_3429_, 0);
lean_inc_ref(v_name_3430_);
lean_dec(v_rule_3429_);
v_name_3431_ = lean_ctor_get(v_name_3430_, 0);
lean_inc(v_name_3431_);
v_builder_3432_ = lean_ctor_get_uint8(v_name_3430_, sizeof(void*)*1 + 8);
v_phase_3433_ = lean_ctor_get_uint8(v_name_3430_, sizeof(void*)*1 + 9);
v_scope_3434_ = lean_ctor_get_uint8(v_name_3430_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_3430_);
switch(v_phase_3433_)
{
case 0:
{
lean_object* v___x_3465_; 
v___x_3465_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__11));
v___y_3454_ = v___x_3465_;
goto v___jp_3453_;
}
case 1:
{
lean_object* v___x_3466_; 
v___x_3466_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__12));
v___y_3454_ = v___x_3466_;
goto v___jp_3453_;
}
default: 
{
lean_object* v___x_3467_; 
v___x_3467_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__13));
v___y_3454_ = v___x_3467_;
goto v___jp_3453_;
}
}
v___jp_3435_:
{
lean_object* v___x_3439_; lean_object* v___x_3440_; lean_object* v___x_3441_; lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3444_; 
v___x_3439_ = lean_string_append(v___y_3437_, v___y_3438_);
v___x_3440_ = lean_string_append(v___x_3439_, v___y_3436_);
v___x_3441_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_3431_, v_hasTrace_3427_);
v___x_3442_ = lean_string_append(v___x_3440_, v___x_3441_);
lean_dec_ref(v___x_3441_);
v___x_3443_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3443_, 0, v___x_3442_);
v___x_3444_ = l_Lean_MessageData_ofFormat(v___x_3443_);
return v___x_3444_;
}
v___jp_3445_:
{
lean_object* v___x_3449_; lean_object* v___x_3450_; 
v___x_3449_ = lean_string_append(v___y_3447_, v___y_3448_);
v___x_3450_ = lean_string_append(v___x_3449_, v___y_3446_);
if (v_scope_3434_ == 0)
{
lean_object* v___x_3451_; 
v___x_3451_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__0));
v___y_3436_ = v___y_3446_;
v___y_3437_ = v___x_3450_;
v___y_3438_ = v___x_3451_;
goto v___jp_3435_;
}
else
{
lean_object* v___x_3452_; 
v___x_3452_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__1));
v___y_3436_ = v___y_3446_;
v___y_3437_ = v___x_3450_;
v___y_3438_ = v___x_3452_;
goto v___jp_3435_;
}
}
v___jp_3453_:
{
lean_object* v___x_3455_; lean_object* v___x_3456_; 
v___x_3455_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__2));
lean_inc_ref(v___y_3454_);
v___x_3456_ = lean_string_append(v___y_3454_, v___x_3455_);
switch(v_builder_3432_)
{
case 0:
{
lean_object* v___x_3457_; 
v___x_3457_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__3));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3457_;
goto v___jp_3445_;
}
case 1:
{
lean_object* v___x_3458_; 
v___x_3458_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__4));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3458_;
goto v___jp_3445_;
}
case 2:
{
lean_object* v___x_3459_; 
v___x_3459_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__5));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3459_;
goto v___jp_3445_;
}
case 3:
{
lean_object* v___x_3460_; 
v___x_3460_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__6));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3460_;
goto v___jp_3445_;
}
case 4:
{
lean_object* v___x_3461_; 
v___x_3461_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__7));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3461_;
goto v___jp_3445_;
}
case 5:
{
lean_object* v___x_3462_; 
v___x_3462_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__8));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3462_;
goto v___jp_3445_;
}
case 6:
{
lean_object* v___x_3463_; 
v___x_3463_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__9));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3463_;
goto v___jp_3445_;
}
default: 
{
lean_object* v___x_3464_; 
v___x_3464_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___closed__10));
v___y_3446_ = v___x_3455_;
v___y_3447_ = v___x_3456_;
v___y_3448_ = v___x_3464_;
goto v___jp_3445_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___lam__13___boxed(lean_object* v_hasTrace_3468_, lean_object* v_x_3469_){
_start:
{
uint8_t v_hasTrace_boxed_3470_; lean_object* v_res_3471_; 
v_hasTrace_boxed_3470_ = lean_unbox(v_hasTrace_3468_);
v_res_3471_ = lp_aesop_Aesop_Index_applicableRules___redArg___lam__13(v_hasTrace_boxed_3470_, v_x_3469_);
return v_res_3471_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__2(void){
_start:
{
lean_object* v___x_3474_; lean_object* v___x_3475_; lean_object* v___x_3476_; 
v___x_3474_ = l_Lean_Core_instMonadTraceCoreM;
v___x_3475_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__1));
v___x_3476_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_3475_, v___x_3474_);
return v___x_3476_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__3(void){
_start:
{
lean_object* v___x_3477_; lean_object* v___f_3478_; lean_object* v___x_3479_; 
v___x_3477_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__2, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__2_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__2);
v___f_3478_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__0));
v___x_3479_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_3478_, v___x_3477_);
return v___x_3479_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__6(void){
_start:
{
lean_object* v___x_3482_; lean_object* v___x_3483_; lean_object* v___x_3484_; lean_object* v___x_3485_; 
v___x_3482_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_3483_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__1));
v___x_3484_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__5));
v___x_3485_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_3484_, v___x_3483_, v___x_3482_);
return v___x_3485_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__7(void){
_start:
{
lean_object* v___x_3486_; lean_object* v___f_3487_; lean_object* v___f_3488_; lean_object* v___x_3489_; 
v___x_3486_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__6, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__6_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__6);
v___f_3487_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__0));
v___f_3488_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__4));
v___x_3489_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_3488_, v___f_3487_, v___x_3486_);
return v___x_3489_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__8(void){
_start:
{
lean_object* v___x_3490_; lean_object* v___x_3491_; 
v___x_3490_ = lean_obj_once(&lp_aesop_Aesop_Index_trace___redArg___closed__3, &lp_aesop_Aesop_Index_trace___redArg___closed__3_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__3);
v___x_3491_ = l_Lean_instMonadAlwaysExceptStateRefT_x27___redArg(v___x_3490_);
return v___x_3491_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__9(void){
_start:
{
lean_object* v___x_3492_; lean_object* v___x_3493_; 
v___x_3492_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__8, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__8_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__8);
v___x_3493_ = l_Lean_instMonadAlwaysExceptReaderT___redArg(v___x_3492_);
return v___x_3493_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__10(void){
_start:
{
lean_object* v___x_3494_; lean_object* v___f_3495_; 
v___x_3494_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_3495_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_3495_, 0, v___x_3494_);
return v___f_3495_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__11(void){
_start:
{
lean_object* v___x_3496_; lean_object* v___f_3497_; 
v___x_3496_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_3497_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_3497_, 0, v___x_3496_);
return v___f_3497_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__12(void){
_start:
{
lean_object* v___f_3498_; lean_object* v___f_3499_; lean_object* v___x_3500_; 
v___f_3498_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__11, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__11_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__11);
v___f_3499_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__10, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__10_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__10);
v___x_3500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3500_, 0, v___f_3499_);
lean_ctor_set(v___x_3500_, 1, v___f_3498_);
return v___x_3500_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__13(void){
_start:
{
lean_object* v___x_3501_; lean_object* v___f_3502_; 
v___x_3501_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__12, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__12_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__12);
v___f_3502_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_3502_, 0, v___x_3501_);
return v___f_3502_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__14(void){
_start:
{
lean_object* v___x_3503_; lean_object* v___f_3504_; 
v___x_3503_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__12, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__12_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__12);
v___f_3504_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_3504_, 0, v___x_3503_);
return v___f_3504_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__15(void){
_start:
{
lean_object* v___f_3505_; lean_object* v___f_3506_; lean_object* v___x_3507_; 
v___f_3505_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__14, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__14_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__14);
v___f_3506_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__13, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__13_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__13);
v___x_3507_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3507_, 0, v___f_3506_);
lean_ctor_set(v___x_3507_, 1, v___f_3505_);
return v___x_3507_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__19(void){
_start:
{
lean_object* v___x_3513_; lean_object* v___x_3514_; 
v___x_3513_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__18));
v___x_3514_ = l_Lean_stringToMessageData(v___x_3513_);
return v___x_3514_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__22(void){
_start:
{
lean_object* v___x_3518_; lean_object* v___x_3519_; 
v___x_3518_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__21));
v___x_3519_ = l_Lean_MessageData_ofFormat(v___x_3518_);
return v___x_3519_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__25(void){
_start:
{
lean_object* v___x_3523_; lean_object* v___x_3524_; 
v___x_3523_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__24));
v___x_3524_ = l_Lean_MessageData_ofFormat(v___x_3523_);
return v___x_3524_;
}
}
static lean_object* _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__26(void){
_start:
{
lean_object* v___x_3525_; lean_object* v___f_3526_; 
v___x_3525_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__25, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__25_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__25);
v___f_3526_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__1___boxed), 7, 1);
lean_closure_set(v___f_3526_, 0, v___x_3525_);
return v___f_3526_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg(lean_object* v_inst_3527_, lean_object* v_ri_3528_, lean_object* v_goal_3529_, lean_object* v_patSubstMap_3530_, lean_object* v_additionalRules_3531_, lean_object* v_include_x3f_3532_, lean_object* v_a_3533_, lean_object* v_a_3534_, lean_object* v_a_3535_, lean_object* v_a_3536_){
_start:
{
lean_object* v___x_3538_; lean_object* v_toApplicative_3539_; lean_object* v_toFunctor_3540_; lean_object* v_toSeq_3541_; lean_object* v_toSeqLeft_3542_; lean_object* v_toSeqRight_3543_; lean_object* v___f_3544_; lean_object* v___f_3545_; lean_object* v___f_3546_; lean_object* v___f_3547_; lean_object* v___x_3548_; lean_object* v___f_3549_; lean_object* v___f_3550_; lean_object* v___f_3551_; lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3554_; lean_object* v_toApplicative_3555_; lean_object* v___x_3557_; uint8_t v_isShared_3558_; uint8_t v_isSharedCheck_4116_; 
v___x_3538_ = lean_obj_once(&lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1, &lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__1);
v_toApplicative_3539_ = lean_ctor_get(v___x_3538_, 0);
v_toFunctor_3540_ = lean_ctor_get(v_toApplicative_3539_, 0);
v_toSeq_3541_ = lean_ctor_get(v_toApplicative_3539_, 2);
v_toSeqLeft_3542_ = lean_ctor_get(v_toApplicative_3539_, 3);
v_toSeqRight_3543_ = lean_ctor_get(v_toApplicative_3539_, 4);
v___f_3544_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__2));
v___f_3545_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__3));
lean_inc_ref_n(v_toFunctor_3540_, 2);
v___f_3546_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3546_, 0, v_toFunctor_3540_);
v___f_3547_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3547_, 0, v_toFunctor_3540_);
v___x_3548_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3548_, 0, v___f_3546_);
lean_ctor_set(v___x_3548_, 1, v___f_3547_);
lean_inc(v_toSeqRight_3543_);
v___f_3549_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3549_, 0, v_toSeqRight_3543_);
lean_inc(v_toSeqLeft_3542_);
v___f_3550_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3550_, 0, v_toSeqLeft_3542_);
lean_inc(v_toSeq_3541_);
v___f_3551_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3551_, 0, v_toSeq_3541_);
v___x_3552_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3552_, 0, v___x_3548_);
lean_ctor_set(v___x_3552_, 1, v___f_3544_);
lean_ctor_set(v___x_3552_, 2, v___f_3551_);
lean_ctor_set(v___x_3552_, 3, v___f_3550_);
lean_ctor_set(v___x_3552_, 4, v___f_3549_);
v___x_3553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3553_, 0, v___x_3552_);
lean_ctor_set(v___x_3553_, 1, v___f_3545_);
v___x_3554_ = l_StateRefT_x27_instMonad___redArg(v___x_3553_);
v_toApplicative_3555_ = lean_ctor_get(v___x_3554_, 0);
v_isSharedCheck_4116_ = !lean_is_exclusive(v___x_3554_);
if (v_isSharedCheck_4116_ == 0)
{
lean_object* v_unused_4117_; 
v_unused_4117_ = lean_ctor_get(v___x_3554_, 1);
lean_dec(v_unused_4117_);
v___x_3557_ = v___x_3554_;
v_isShared_3558_ = v_isSharedCheck_4116_;
goto v_resetjp_3556_;
}
else
{
lean_inc(v_toApplicative_3555_);
lean_dec(v___x_3554_);
v___x_3557_ = lean_box(0);
v_isShared_3558_ = v_isSharedCheck_4116_;
goto v_resetjp_3556_;
}
v_resetjp_3556_:
{
lean_object* v_toFunctor_3559_; lean_object* v_toSeq_3560_; lean_object* v_toSeqLeft_3561_; lean_object* v_toSeqRight_3562_; lean_object* v___x_3564_; uint8_t v_isShared_3565_; uint8_t v_isSharedCheck_4114_; 
v_toFunctor_3559_ = lean_ctor_get(v_toApplicative_3555_, 0);
v_toSeq_3560_ = lean_ctor_get(v_toApplicative_3555_, 2);
v_toSeqLeft_3561_ = lean_ctor_get(v_toApplicative_3555_, 3);
v_toSeqRight_3562_ = lean_ctor_get(v_toApplicative_3555_, 4);
v_isSharedCheck_4114_ = !lean_is_exclusive(v_toApplicative_3555_);
if (v_isSharedCheck_4114_ == 0)
{
lean_object* v_unused_4115_; 
v_unused_4115_ = lean_ctor_get(v_toApplicative_3555_, 1);
lean_dec(v_unused_4115_);
v___x_3564_ = v_toApplicative_3555_;
v_isShared_3565_ = v_isSharedCheck_4114_;
goto v_resetjp_3563_;
}
else
{
lean_inc(v_toSeqRight_3562_);
lean_inc(v_toSeqLeft_3561_);
lean_inc(v_toSeq_3560_);
lean_inc(v_toFunctor_3559_);
lean_dec(v_toApplicative_3555_);
v___x_3564_ = lean_box(0);
v_isShared_3565_ = v_isSharedCheck_4114_;
goto v_resetjp_3563_;
}
v_resetjp_3563_:
{
lean_object* v___f_3566_; lean_object* v___f_3567_; lean_object* v___f_3568_; lean_object* v___f_3569_; lean_object* v___x_3570_; lean_object* v___f_3571_; lean_object* v___f_3572_; lean_object* v___f_3573_; lean_object* v___x_3575_; 
v___f_3566_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__0));
v___f_3567_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___closed__1));
lean_inc_ref(v_toFunctor_3559_);
v___f_3568_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3568_, 0, v_toFunctor_3559_);
v___f_3569_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3569_, 0, v_toFunctor_3559_);
v___x_3570_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3570_, 0, v___f_3568_);
lean_ctor_set(v___x_3570_, 1, v___f_3569_);
v___f_3571_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3571_, 0, v_toSeqRight_3562_);
v___f_3572_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3572_, 0, v_toSeqLeft_3561_);
v___f_3573_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3573_, 0, v_toSeq_3560_);
if (v_isShared_3565_ == 0)
{
lean_ctor_set(v___x_3564_, 4, v___f_3571_);
lean_ctor_set(v___x_3564_, 3, v___f_3572_);
lean_ctor_set(v___x_3564_, 2, v___f_3573_);
lean_ctor_set(v___x_3564_, 1, v___f_3566_);
lean_ctor_set(v___x_3564_, 0, v___x_3570_);
v___x_3575_ = v___x_3564_;
goto v_reusejp_3574_;
}
else
{
lean_object* v_reuseFailAlloc_4113_; 
v_reuseFailAlloc_4113_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_4113_, 0, v___x_3570_);
lean_ctor_set(v_reuseFailAlloc_4113_, 1, v___f_3566_);
lean_ctor_set(v_reuseFailAlloc_4113_, 2, v___f_3573_);
lean_ctor_set(v_reuseFailAlloc_4113_, 3, v___f_3572_);
lean_ctor_set(v_reuseFailAlloc_4113_, 4, v___f_3571_);
v___x_3575_ = v_reuseFailAlloc_4113_;
goto v_reusejp_3574_;
}
v_reusejp_3574_:
{
lean_object* v___x_3577_; 
if (v_isShared_3558_ == 0)
{
lean_ctor_set(v___x_3557_, 1, v___f_3567_);
lean_ctor_set(v___x_3557_, 0, v___x_3575_);
v___x_3577_ = v___x_3557_;
goto v_reusejp_3576_;
}
else
{
lean_object* v_reuseFailAlloc_4112_; 
v_reuseFailAlloc_4112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4112_, 0, v___x_3575_);
lean_ctor_set(v_reuseFailAlloc_4112_, 1, v___f_3567_);
v___x_3577_ = v_reuseFailAlloc_4112_;
goto v_reusejp_3576_;
}
v_reusejp_3576_:
{
lean_object* v___x_3578_; lean_object* v___x_3579_; lean_object* v_toMonadRef_3580_; lean_object* v___x_3581_; lean_object* v___x_3582_; lean_object* v___x_3583_; lean_object* v___x_3584_; lean_object* v___x_3585_; lean_object* v___x_3586_; lean_object* v_options_3587_; lean_object* v_inheritedTraceOptions_3588_; uint8_t v_hasTrace_3589_; lean_object* v___x_3590_; lean_object* v___x_3591_; 
v___x_3578_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__3, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__3_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__3);
v___x_3579_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__7, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__7_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__7);
v_toMonadRef_3580_ = lean_ctor_get(v___x_3579_, 0);
v___x_3581_ = l_Lean_Meta_instAddMessageContextMetaM;
v___x_3582_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__9, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__9_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__9);
v___x_3583_ = l_Lean_Meta_instMonadMCtxMetaM;
v___x_3584_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__15, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__15_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__15);
lean_inc_ref(v___x_3577_);
v___x_3585_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_3581_, v___x_3577_);
lean_inc_ref(v_toMonadRef_3580_);
v___x_3586_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_3586_, 0, v___x_3584_);
lean_ctor_set(v___x_3586_, 1, v_toMonadRef_3580_);
lean_ctor_set(v___x_3586_, 2, v___x_3585_);
v_options_3587_ = lean_ctor_get(v_a_3535_, 2);
v_inheritedTraceOptions_3588_ = lean_ctor_get(v_a_3535_, 13);
v_hasTrace_3589_ = lean_ctor_get_uint8(v_options_3587_, sizeof(void*)*1);
v___x_3590_ = ((lean_object*)(lp_aesop_Aesop_Index_applicableRules___redArg___closed__17));
v___x_3591_ = lp_aesop_Aesop_TraceOption_debug;
if (v_hasTrace_3589_ == 0)
{
lean_object* v___x_14804__overap_3592_; lean_object* v___x_3593_; 
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
v___x_14804__overap_3592_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg(v___x_3577_, v___x_3583_, v___x_3586_, v_goal_3529_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3593_ = lean_apply_5(v___x_14804__overap_3592_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3593_) == 0)
{
lean_object* v_toApplicative_3594_; lean_object* v_toFunctor_3595_; lean_object* v_toSeq_3596_; lean_object* v_toSeqLeft_3597_; lean_object* v_toSeqRight_3598_; lean_object* v___f_3599_; lean_object* v___f_3600_; lean_object* v___f_3601_; lean_object* v___f_3602_; lean_object* v___x_3603_; lean_object* v___f_3604_; lean_object* v___f_3605_; lean_object* v___f_3606_; lean_object* v___x_3607_; lean_object* v___x_3608_; lean_object* v___x_3609_; lean_object* v___x_3610_; lean_object* v___x_3611_; lean_object* v___x_14855__overap_3612_; lean_object* v___x_3613_; 
lean_dec_ref_known(v___x_3593_, 1);
v_toApplicative_3594_ = lean_ctor_get(v___x_3538_, 0);
v_toFunctor_3595_ = lean_ctor_get(v_toApplicative_3594_, 0);
v_toSeq_3596_ = lean_ctor_get(v_toApplicative_3594_, 2);
v_toSeqLeft_3597_ = lean_ctor_get(v_toApplicative_3594_, 3);
v_toSeqRight_3598_ = lean_ctor_get(v_toApplicative_3594_, 4);
lean_inc_ref(v_include_x3f_3532_);
v___f_3599_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_3599_, 0, v_include_x3f_3532_);
lean_inc_ref_n(v___x_3577_, 2);
lean_inc_ref(v_ri_3528_);
lean_inc_n(v_goal_3529_, 2);
v___f_3600_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_3600_, 0, v_goal_3529_);
lean_closure_set(v___f_3600_, 1, v_ri_3528_);
lean_closure_set(v___f_3600_, 2, v___x_3577_);
lean_closure_set(v___f_3600_, 3, v___f_3599_);
lean_inc_ref_n(v_toFunctor_3595_, 2);
v___f_3601_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3601_, 0, v_toFunctor_3595_);
v___f_3602_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3602_, 0, v_toFunctor_3595_);
v___x_3603_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3603_, 0, v___f_3601_);
lean_ctor_set(v___x_3603_, 1, v___f_3602_);
lean_inc(v_toSeqRight_3598_);
v___f_3604_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3604_, 0, v_toSeqRight_3598_);
lean_inc(v_toSeqLeft_3597_);
v___f_3605_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3605_, 0, v_toSeqLeft_3597_);
lean_inc(v_toSeq_3596_);
v___f_3606_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3606_, 0, v_toSeq_3596_);
v___x_3607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3607_, 0, v___x_3603_);
lean_ctor_set(v___x_3607_, 1, v___f_3544_);
lean_ctor_set(v___x_3607_, 2, v___f_3606_);
lean_ctor_set(v___x_3607_, 3, v___f_3605_);
lean_ctor_set(v___x_3607_, 4, v___f_3604_);
v___x_3608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3608_, 0, v___x_3607_);
lean_ctor_set(v___x_3608_, 1, v___f_3545_);
v___x_3609_ = l_StateRefT_x27_instMonad___redArg(v___x_3608_);
v___x_3610_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_3610_, 0, lean_box(0));
lean_closure_set(v___x_3610_, 1, lean_box(0));
lean_closure_set(v___x_3610_, 2, v___x_3609_);
v___x_3611_ = l_instMonadControlTOfPure___redArg(v___x_3610_);
lean_inc_ref(v___x_3611_);
v___x_14855__overap_3612_ = l_Lean_MVarId_withContext___redArg(v___x_3611_, v___x_3577_, v_goal_3529_, v___f_3600_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3613_ = lean_apply_5(v___x_14855__overap_3612_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3613_) == 0)
{
lean_object* v_a_3614_; lean_object* v___f_3615_; lean_object* v___x_3616_; lean_object* v_rs_3617_; lean_object* v___f_3618_; lean_object* v___x_14922__overap_3619_; lean_object* v___x_3620_; 
v_a_3614_ = lean_ctor_get(v___x_3613_, 0);
lean_inc(v_a_3614_);
lean_dec_ref_known(v___x_3613_, 1);
lean_inc_ref_n(v___x_3577_, 3);
lean_inc_ref(v_include_x3f_3532_);
lean_inc_ref(v_ri_3528_);
v___f_3615_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed), 10, 3);
lean_closure_set(v___f_3615_, 0, v_ri_3528_);
lean_closure_set(v___f_3615_, 1, v_include_x3f_3532_);
lean_closure_set(v___f_3615_, 2, v___x_3577_);
v___x_3616_ = lean_unsigned_to_nat(0u);
v_rs_3617_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
v___f_3618_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_3618_, 0, v___x_3577_);
lean_closure_set(v___f_3618_, 1, v_rs_3617_);
lean_closure_set(v___f_3618_, 2, v___f_3615_);
v___x_14922__overap_3619_ = l_Lean_MVarId_withContext___redArg(v___x_3611_, v___x_3577_, v_goal_3529_, v___f_3618_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3620_ = lean_apply_5(v___x_14922__overap_3619_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3620_) == 0)
{
lean_object* v_a_3621_; lean_object* v_unindexed_3622_; lean_object* v___f_3623_; lean_object* v___x_3624_; lean_object* v___x_3625_; lean_object* v___x_3626_; lean_object* v___x_3627_; lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3631_; 
v_a_3621_ = lean_ctor_get(v___x_3620_, 0);
lean_inc(v_a_3621_);
lean_dec_ref_known(v___x_3620_, 1);
v_unindexed_3622_ = lean_ctor_get(v_ri_3528_, 2);
lean_inc_ref(v_unindexed_3622_);
lean_dec_ref(v_ri_3528_);
v___f_3623_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0), 4, 1);
lean_closure_set(v___f_3623_, 0, v_include_x3f_3532_);
v___x_3624_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_3625_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_3624_, v___f_3623_, v_unindexed_3622_, v_rs_3617_);
v___x_3626_ = lean_unsigned_to_nat(3u);
v___x_3627_ = lean_mk_empty_array_with_capacity(v___x_3626_);
v___x_3628_ = lean_array_push(v___x_3627_, v_a_3614_);
v___x_3629_ = lean_array_push(v___x_3628_, v_a_3621_);
v___x_3630_ = lean_array_push(v___x_3629_, v___x_3625_);
v___x_3631_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_3530_, v_additionalRules_3531_, v___x_3630_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_);
lean_dec_ref(v___x_3630_);
if (lean_obj_tag(v___x_3631_) == 0)
{
lean_object* v_a_3632_; lean_object* v___y_3634_; lean_object* v___x_3682_; uint8_t v___x_3683_; 
v_a_3632_ = lean_ctor_get(v___x_3631_, 0);
lean_inc(v_a_3632_);
lean_dec_ref_known(v___x_3631_, 1);
v___x_3682_ = lean_array_get_size(v_a_3632_);
v___x_3683_ = lean_nat_dec_eq(v___x_3682_, v___x_3616_);
if (v___x_3683_ == 0)
{
lean_object* v___x_3684_; lean_object* v___f_3685_; lean_object* v___y_3687_; lean_object* v___y_3688_; lean_object* v___x_3690_; lean_object* v___x_3691_; lean_object* v___y_3693_; uint8_t v___x_3695_; 
v___x_3684_ = lean_box(v_hasTrace_3589_);
v___f_3685_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_3685_, 0, v_inst_3527_);
lean_closure_set(v___f_3685_, 1, v___x_3684_);
v___x_3690_ = lean_unsigned_to_nat(1u);
v___x_3691_ = lean_nat_sub(v___x_3682_, v___x_3690_);
v___x_3695_ = lean_nat_dec_le(v___x_3616_, v___x_3691_);
if (v___x_3695_ == 0)
{
lean_inc(v___x_3691_);
v___y_3693_ = v___x_3691_;
goto v___jp_3692_;
}
else
{
v___y_3693_ = v___x_3616_;
goto v___jp_3692_;
}
v___jp_3686_:
{
lean_object* v___x_3689_; 
v___x_3689_ = l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_box(0), v___f_3685_, v___x_3682_, v_a_3632_, v___y_3687_, v___y_3688_, lean_box(0), lean_box(0), lean_box(0));
lean_dec(v___y_3688_);
v___y_3634_ = v___x_3689_;
goto v___jp_3633_;
}
v___jp_3692_:
{
uint8_t v___x_3694_; 
v___x_3694_ = lean_nat_dec_le(v___y_3693_, v___x_3691_);
if (v___x_3694_ == 0)
{
lean_dec(v___x_3691_);
lean_inc(v___y_3693_);
v___y_3687_ = v___y_3693_;
v___y_3688_ = v___y_3693_;
goto v___jp_3686_;
}
else
{
v___y_3687_ = v___y_3693_;
v___y_3688_ = v___x_3691_;
goto v___jp_3686_;
}
}
}
else
{
lean_dec_ref(v_inst_3527_);
v___y_3634_ = v_a_3632_;
goto v___jp_3633_;
}
v___jp_3633_:
{
lean_object* v___x_14961__overap_3635_; lean_object* v___x_3636_; 
lean_inc_ref(v___x_3577_);
v___x_14961__overap_3635_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_3577_, v___x_3590_, v___x_3591_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3636_ = lean_apply_5(v___x_14961__overap_3635_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3636_) == 0)
{
lean_object* v_a_3637_; lean_object* v___x_3639_; uint8_t v_isShared_3640_; uint8_t v_isSharedCheck_3673_; 
v_a_3637_ = lean_ctor_get(v___x_3636_, 0);
v_isSharedCheck_3673_ = !lean_is_exclusive(v___x_3636_);
if (v_isSharedCheck_3673_ == 0)
{
v___x_3639_ = v___x_3636_;
v_isShared_3640_ = v_isSharedCheck_3673_;
goto v_resetjp_3638_;
}
else
{
lean_inc(v_a_3637_);
lean_dec(v___x_3636_);
v___x_3639_ = lean_box(0);
v_isShared_3640_ = v_isSharedCheck_3673_;
goto v_resetjp_3638_;
}
v_resetjp_3638_:
{
uint8_t v___x_3641_; 
v___x_3641_ = lean_unbox(v_a_3637_);
if (v___x_3641_ == 0)
{
lean_object* v___x_3643_; 
lean_dec(v_a_3637_);
lean_dec_ref(v___x_3577_);
if (v_isShared_3640_ == 0)
{
lean_ctor_set(v___x_3639_, 0, v___y_3634_);
v___x_3643_ = v___x_3639_;
goto v_reusejp_3642_;
}
else
{
lean_object* v_reuseFailAlloc_3644_; 
v_reuseFailAlloc_3644_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3644_, 0, v___y_3634_);
v___x_3643_ = v_reuseFailAlloc_3644_;
goto v_reusejp_3642_;
}
v_reusejp_3642_:
{
return v___x_3643_;
}
}
else
{
lean_object* v_traceClass_3645_; lean_object* v___f_3646_; lean_object* v___x_3647_; size_t v_sz_3648_; size_t v___x_3649_; lean_object* v___x_3650_; lean_object* v___x_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_3654_; lean_object* v___x_16157__overap_3655_; lean_object* v___x_3656_; 
lean_del_object(v___x_3639_);
v_traceClass_3645_ = lean_ctor_get(v___x_3591_, 0);
v___f_3646_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__6___boxed), 2, 1);
lean_closure_set(v___f_3646_, 0, v_a_3637_);
v___x_3647_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__19, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__19_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__19);
v_sz_3648_ = lean_array_size(v___y_3634_);
v___x_3649_ = ((size_t)0ULL);
lean_inc_ref(v___y_3634_);
v___x_3650_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_3624_, v___f_3646_, v_sz_3648_, v___x_3649_, v___y_3634_);
v___x_3651_ = lean_array_to_list(v___x_3650_);
v___x_3652_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__22, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__22_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__22);
v___x_3653_ = l_Lean_MessageData_joinSep(v___x_3651_, v___x_3652_);
v___x_3654_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3654_, 0, v___x_3647_);
lean_ctor_set(v___x_3654_, 1, v___x_3653_);
lean_inc(v_traceClass_3645_);
lean_inc_ref(v_toMonadRef_3580_);
v___x_16157__overap_3655_ = l_Lean_addTrace___redArg(v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, v_traceClass_3645_, v___x_3654_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3656_ = lean_apply_5(v___x_16157__overap_3655_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3656_) == 0)
{
lean_object* v___x_3658_; uint8_t v_isShared_3659_; uint8_t v_isSharedCheck_3663_; 
v_isSharedCheck_3663_ = !lean_is_exclusive(v___x_3656_);
if (v_isSharedCheck_3663_ == 0)
{
lean_object* v_unused_3664_; 
v_unused_3664_ = lean_ctor_get(v___x_3656_, 0);
lean_dec(v_unused_3664_);
v___x_3658_ = v___x_3656_;
v_isShared_3659_ = v_isSharedCheck_3663_;
goto v_resetjp_3657_;
}
else
{
lean_dec(v___x_3656_);
v___x_3658_ = lean_box(0);
v_isShared_3659_ = v_isSharedCheck_3663_;
goto v_resetjp_3657_;
}
v_resetjp_3657_:
{
lean_object* v___x_3661_; 
if (v_isShared_3659_ == 0)
{
lean_ctor_set(v___x_3658_, 0, v___y_3634_);
v___x_3661_ = v___x_3658_;
goto v_reusejp_3660_;
}
else
{
lean_object* v_reuseFailAlloc_3662_; 
v_reuseFailAlloc_3662_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3662_, 0, v___y_3634_);
v___x_3661_ = v_reuseFailAlloc_3662_;
goto v_reusejp_3660_;
}
v_reusejp_3660_:
{
return v___x_3661_;
}
}
}
else
{
lean_object* v_a_3665_; lean_object* v___x_3667_; uint8_t v_isShared_3668_; uint8_t v_isSharedCheck_3672_; 
lean_dec_ref(v___y_3634_);
v_a_3665_ = lean_ctor_get(v___x_3656_, 0);
v_isSharedCheck_3672_ = !lean_is_exclusive(v___x_3656_);
if (v_isSharedCheck_3672_ == 0)
{
v___x_3667_ = v___x_3656_;
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
else
{
lean_inc(v_a_3665_);
lean_dec(v___x_3656_);
v___x_3667_ = lean_box(0);
v_isShared_3668_ = v_isSharedCheck_3672_;
goto v_resetjp_3666_;
}
v_resetjp_3666_:
{
lean_object* v___x_3670_; 
if (v_isShared_3668_ == 0)
{
v___x_3670_ = v___x_3667_;
goto v_reusejp_3669_;
}
else
{
lean_object* v_reuseFailAlloc_3671_; 
v_reuseFailAlloc_3671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3671_, 0, v_a_3665_);
v___x_3670_ = v_reuseFailAlloc_3671_;
goto v_reusejp_3669_;
}
v_reusejp_3669_:
{
return v___x_3670_;
}
}
}
}
}
}
else
{
lean_object* v_a_3674_; lean_object* v___x_3676_; uint8_t v_isShared_3677_; uint8_t v_isSharedCheck_3681_; 
lean_dec_ref(v___y_3634_);
lean_dec_ref(v___x_3577_);
v_a_3674_ = lean_ctor_get(v___x_3636_, 0);
v_isSharedCheck_3681_ = !lean_is_exclusive(v___x_3636_);
if (v_isSharedCheck_3681_ == 0)
{
v___x_3676_ = v___x_3636_;
v_isShared_3677_ = v_isSharedCheck_3681_;
goto v_resetjp_3675_;
}
else
{
lean_inc(v_a_3674_);
lean_dec(v___x_3636_);
v___x_3676_ = lean_box(0);
v_isShared_3677_ = v_isSharedCheck_3681_;
goto v_resetjp_3675_;
}
v_resetjp_3675_:
{
lean_object* v___x_3679_; 
if (v_isShared_3677_ == 0)
{
v___x_3679_ = v___x_3676_;
goto v_reusejp_3678_;
}
else
{
lean_object* v_reuseFailAlloc_3680_; 
v_reuseFailAlloc_3680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3680_, 0, v_a_3674_);
v___x_3679_ = v_reuseFailAlloc_3680_;
goto v_reusejp_3678_;
}
v_reusejp_3678_:
{
return v___x_3679_;
}
}
}
}
}
else
{
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_inst_3527_);
return v___x_3631_;
}
}
else
{
lean_object* v_a_3696_; lean_object* v___x_3698_; uint8_t v_isShared_3699_; uint8_t v_isSharedCheck_3703_; 
lean_dec(v_a_3614_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_include_x3f_3532_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3696_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3703_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3703_ == 0)
{
v___x_3698_ = v___x_3620_;
v_isShared_3699_ = v_isSharedCheck_3703_;
goto v_resetjp_3697_;
}
else
{
lean_inc(v_a_3696_);
lean_dec(v___x_3620_);
v___x_3698_ = lean_box(0);
v_isShared_3699_ = v_isSharedCheck_3703_;
goto v_resetjp_3697_;
}
v_resetjp_3697_:
{
lean_object* v___x_3701_; 
if (v_isShared_3699_ == 0)
{
v___x_3701_ = v___x_3698_;
goto v_reusejp_3700_;
}
else
{
lean_object* v_reuseFailAlloc_3702_; 
v_reuseFailAlloc_3702_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3702_, 0, v_a_3696_);
v___x_3701_ = v_reuseFailAlloc_3702_;
goto v_reusejp_3700_;
}
v_reusejp_3700_:
{
return v___x_3701_;
}
}
}
}
else
{
lean_object* v_a_3704_; lean_object* v___x_3706_; uint8_t v_isShared_3707_; uint8_t v_isSharedCheck_3711_; 
lean_dec_ref(v___x_3611_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_include_x3f_3532_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3704_ = lean_ctor_get(v___x_3613_, 0);
v_isSharedCheck_3711_ = !lean_is_exclusive(v___x_3613_);
if (v_isSharedCheck_3711_ == 0)
{
v___x_3706_ = v___x_3613_;
v_isShared_3707_ = v_isSharedCheck_3711_;
goto v_resetjp_3705_;
}
else
{
lean_inc(v_a_3704_);
lean_dec(v___x_3613_);
v___x_3706_ = lean_box(0);
v_isShared_3707_ = v_isSharedCheck_3711_;
goto v_resetjp_3705_;
}
v_resetjp_3705_:
{
lean_object* v___x_3709_; 
if (v_isShared_3707_ == 0)
{
v___x_3709_ = v___x_3706_;
goto v_reusejp_3708_;
}
else
{
lean_object* v_reuseFailAlloc_3710_; 
v_reuseFailAlloc_3710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3710_, 0, v_a_3704_);
v___x_3709_ = v_reuseFailAlloc_3710_;
goto v_reusejp_3708_;
}
v_reusejp_3708_:
{
return v___x_3709_;
}
}
}
}
else
{
lean_object* v_a_3712_; lean_object* v___x_3714_; uint8_t v_isShared_3715_; uint8_t v_isSharedCheck_3719_; 
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_include_x3f_3532_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3712_ = lean_ctor_get(v___x_3593_, 0);
v_isSharedCheck_3719_ = !lean_is_exclusive(v___x_3593_);
if (v_isSharedCheck_3719_ == 0)
{
v___x_3714_ = v___x_3593_;
v_isShared_3715_ = v_isSharedCheck_3719_;
goto v_resetjp_3713_;
}
else
{
lean_inc(v_a_3712_);
lean_dec(v___x_3593_);
v___x_3714_ = lean_box(0);
v_isShared_3715_ = v_isSharedCheck_3719_;
goto v_resetjp_3713_;
}
v_resetjp_3713_:
{
lean_object* v___x_3717_; 
if (v_isShared_3715_ == 0)
{
v___x_3717_ = v___x_3714_;
goto v_reusejp_3716_;
}
else
{
lean_object* v_reuseFailAlloc_3718_; 
v_reuseFailAlloc_3718_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3718_, 0, v_a_3712_);
v___x_3717_ = v_reuseFailAlloc_3718_;
goto v_reusejp_3716_;
}
v_reusejp_3716_:
{
return v___x_3717_;
}
}
}
}
else
{
lean_object* v_traceClass_3720_; lean_object* v___f_3721_; lean_object* v___f_3722_; lean_object* v___f_3723_; lean_object* v___f_3724_; lean_object* v___f_3725_; lean_object* v___x_3726_; lean_object* v___f_3727_; lean_object* v___f_3728_; lean_object* v___x_3729_; lean_object* v___x_3730_; lean_object* v___x_3731_; uint8_t v___x_3732_; lean_object* v___y_3734_; lean_object* v___y_3735_; lean_object* v_a_3736_; lean_object* v___y_3750_; lean_object* v___y_3751_; lean_object* v_a_3752_; lean_object* v___y_3755_; lean_object* v___y_3756_; lean_object* v_a_3757_; lean_object* v___y_3760_; lean_object* v___y_3761_; lean_object* v___y_3762_; lean_object* v___y_3763_; lean_object* v___y_3781_; lean_object* v___y_3782_; lean_object* v___y_3783_; lean_object* v___y_3784_; lean_object* v___y_3785_; lean_object* v___y_3786_; lean_object* v___y_3787_; lean_object* v___y_3788_; lean_object* v___y_3791_; lean_object* v___y_3792_; lean_object* v___y_3793_; lean_object* v___y_3794_; lean_object* v___y_3795_; lean_object* v___y_3796_; lean_object* v___y_3797_; lean_object* v___y_3798_; lean_object* v___y_3801_; lean_object* v___y_3802_; lean_object* v_a_3803_; lean_object* v___y_3814_; lean_object* v___y_3815_; lean_object* v_a_3816_; lean_object* v___y_3819_; lean_object* v___y_3820_; lean_object* v_a_3821_; lean_object* v___y_3824_; lean_object* v___y_3825_; lean_object* v___y_3826_; lean_object* v___y_3827_; lean_object* v___y_3828_; lean_object* v___y_3846_; lean_object* v___y_3847_; lean_object* v___y_3848_; lean_object* v___y_3849_; lean_object* v___y_3850_; lean_object* v___y_3851_; lean_object* v___y_3852_; lean_object* v___y_3853_; lean_object* v___y_3854_; lean_object* v___y_3857_; lean_object* v___y_3858_; lean_object* v___y_3859_; lean_object* v___y_3860_; lean_object* v___y_3861_; lean_object* v___y_3862_; lean_object* v___y_3863_; lean_object* v___y_3864_; lean_object* v___y_3865_; 
v_traceClass_3720_ = lean_ctor_get(v___x_3591_, 0);
lean_inc_ref_n(v_include_x3f_3532_, 2);
v___f_3721_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableUnindexedRules___redArg___lam__0), 4, 1);
lean_closure_set(v___f_3721_, 0, v_include_x3f_3532_);
v___f_3722_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_3722_, 0, v_include_x3f_3532_);
lean_inc_ref_n(v___x_3577_, 2);
lean_inc_ref_n(v_ri_3528_, 2);
v___f_3723_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__1___boxed), 10, 3);
lean_closure_set(v___f_3723_, 0, v_ri_3528_);
lean_closure_set(v___f_3723_, 1, v_include_x3f_3532_);
lean_closure_set(v___f_3723_, 2, v___x_3577_);
lean_inc(v_goal_3529_);
v___f_3724_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByTargetRules___redArg___lam__1___boxed), 9, 4);
lean_closure_set(v___f_3724_, 0, v_goal_3529_);
lean_closure_set(v___f_3724_, 1, v_ri_3528_);
lean_closure_set(v___f_3724_, 2, v___x_3577_);
lean_closure_set(v___f_3724_, 3, v___f_3722_);
v___f_3725_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__6));
v___x_3726_ = lean_box(v_hasTrace_3589_);
v___f_3727_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__12___boxed), 2, 1);
lean_closure_set(v___f_3727_, 0, v___x_3726_);
v___f_3728_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__26, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__26_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__26);
v___x_3729_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__13));
v___x_3730_ = ((lean_object*)(lp_aesop_Aesop_Index_trace___redArg___closed__15));
lean_inc(v_traceClass_3720_);
v___x_3731_ = l_Lean_Name_append(v___x_3730_, v_traceClass_3720_);
v___x_3732_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3588_, v_options_3587_, v___x_3731_);
lean_dec(v___x_3731_);
if (v___x_3732_ == 0)
{
lean_object* v___x_3984_; lean_object* v___x_3985_; lean_object* v___x_3986_; uint8_t v___x_3987_; 
v___x_3984_ = l_Lean_KVMap_instValueBool;
v___x_3985_ = l_Lean_trace_profiler;
v___x_3986_ = l_Lean_Option_get___redArg(v___x_3984_, v_options_3587_, v___x_3985_);
v___x_3987_ = lean_unbox(v___x_3986_);
if (v___x_3987_ == 0)
{
lean_object* v___x_16976__overap_3988_; lean_object* v___x_3989_; 
lean_dec_ref(v___f_3727_);
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
v___x_16976__overap_3988_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg(v___x_3577_, v___x_3583_, v___x_3586_, v_goal_3529_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3989_ = lean_apply_5(v___x_16976__overap_3988_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3989_) == 0)
{
lean_object* v_toApplicative_3990_; lean_object* v_toFunctor_3991_; lean_object* v_toSeq_3992_; lean_object* v_toSeqLeft_3993_; lean_object* v_toSeqRight_3994_; lean_object* v___f_3995_; lean_object* v___f_3996_; lean_object* v___x_3997_; lean_object* v___f_3998_; lean_object* v___f_3999_; lean_object* v___f_4000_; lean_object* v___x_4001_; lean_object* v___x_4002_; lean_object* v___x_4003_; lean_object* v___x_4004_; lean_object* v___x_4005_; lean_object* v___x_17024__overap_4006_; lean_object* v___x_4007_; 
lean_dec_ref_known(v___x_3989_, 1);
v_toApplicative_3990_ = lean_ctor_get(v___x_3538_, 0);
v_toFunctor_3991_ = lean_ctor_get(v_toApplicative_3990_, 0);
v_toSeq_3992_ = lean_ctor_get(v_toApplicative_3990_, 2);
v_toSeqLeft_3993_ = lean_ctor_get(v_toApplicative_3990_, 3);
v_toSeqRight_3994_ = lean_ctor_get(v_toApplicative_3990_, 4);
lean_inc_ref_n(v_toFunctor_3991_, 2);
v___f_3995_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3995_, 0, v_toFunctor_3991_);
v___f_3996_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3996_, 0, v_toFunctor_3991_);
v___x_3997_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3997_, 0, v___f_3995_);
lean_ctor_set(v___x_3997_, 1, v___f_3996_);
lean_inc(v_toSeqRight_3994_);
v___f_3998_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3998_, 0, v_toSeqRight_3994_);
lean_inc(v_toSeqLeft_3993_);
v___f_3999_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3999_, 0, v_toSeqLeft_3993_);
lean_inc(v_toSeq_3992_);
v___f_4000_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_4000_, 0, v_toSeq_3992_);
v___x_4001_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4001_, 0, v___x_3997_);
lean_ctor_set(v___x_4001_, 1, v___f_3544_);
lean_ctor_set(v___x_4001_, 2, v___f_4000_);
lean_ctor_set(v___x_4001_, 3, v___f_3999_);
lean_ctor_set(v___x_4001_, 4, v___f_3998_);
v___x_4002_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4002_, 0, v___x_4001_);
lean_ctor_set(v___x_4002_, 1, v___f_3545_);
v___x_4003_ = l_StateRefT_x27_instMonad___redArg(v___x_4002_);
v___x_4004_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_4004_, 0, lean_box(0));
lean_closure_set(v___x_4004_, 1, lean_box(0));
lean_closure_set(v___x_4004_, 2, v___x_4003_);
v___x_4005_ = l_instMonadControlTOfPure___redArg(v___x_4004_);
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
lean_inc_ref(v___x_4005_);
v___x_17024__overap_4006_ = l_Lean_MVarId_withContext___redArg(v___x_4005_, v___x_3577_, v_goal_3529_, v___f_3724_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_4007_ = lean_apply_5(v___x_17024__overap_4006_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_4007_) == 0)
{
lean_object* v_a_4008_; lean_object* v___x_4009_; lean_object* v_rs_4010_; lean_object* v___f_4011_; lean_object* v___x_17085__overap_4012_; lean_object* v___x_4013_; 
v_a_4008_ = lean_ctor_get(v___x_4007_, 0);
lean_inc(v_a_4008_);
lean_dec_ref_known(v___x_4007_, 1);
v___x_4009_ = lean_unsigned_to_nat(0u);
v_rs_4010_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
lean_inc_ref_n(v___x_3577_, 2);
v___f_4011_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_4011_, 0, v___x_3577_);
lean_closure_set(v___f_4011_, 1, v_rs_4010_);
lean_closure_set(v___f_4011_, 2, v___f_3723_);
v___x_17085__overap_4012_ = l_Lean_MVarId_withContext___redArg(v___x_4005_, v___x_3577_, v_goal_3529_, v___f_4011_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_4013_ = lean_apply_5(v___x_17085__overap_4012_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_4013_) == 0)
{
lean_object* v_a_4014_; lean_object* v_unindexed_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; lean_object* v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v___x_4022_; lean_object* v___x_4023_; 
v_a_4014_ = lean_ctor_get(v___x_4013_, 0);
lean_inc(v_a_4014_);
lean_dec_ref_known(v___x_4013_, 1);
v_unindexed_4015_ = lean_ctor_get(v_ri_3528_, 2);
lean_inc_ref(v_unindexed_4015_);
lean_dec_ref(v_ri_3528_);
v___x_4016_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_4017_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_4016_, v___f_3721_, v_unindexed_4015_, v_rs_4010_);
v___x_4018_ = lean_unsigned_to_nat(3u);
v___x_4019_ = lean_mk_empty_array_with_capacity(v___x_4018_);
v___x_4020_ = lean_array_push(v___x_4019_, v_a_4008_);
v___x_4021_ = lean_array_push(v___x_4020_, v_a_4014_);
v___x_4022_ = lean_array_push(v___x_4021_, v___x_4017_);
v___x_4023_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_3530_, v_additionalRules_3531_, v___x_4022_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_);
lean_dec_ref(v___x_4022_);
if (lean_obj_tag(v___x_4023_) == 0)
{
lean_object* v_a_4024_; lean_object* v___x_4025_; lean_object* v___f_4026_; lean_object* v___y_4028_; lean_object* v___x_4074_; uint8_t v___x_4075_; 
v_a_4024_ = lean_ctor_get(v___x_4023_, 0);
lean_inc(v_a_4024_);
lean_dec_ref_known(v___x_4023_, 1);
v___x_4025_ = lean_box(v_hasTrace_3589_);
v___f_4026_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__13___boxed), 2, 1);
lean_closure_set(v___f_4026_, 0, v___x_4025_);
v___x_4074_ = lean_array_get_size(v_a_4024_);
v___x_4075_ = lean_nat_dec_eq(v___x_4074_, v___x_4009_);
if (v___x_4075_ == 0)
{
lean_object* v___x_4076_; lean_object* v___f_4077_; lean_object* v___y_4079_; lean_object* v___y_4080_; lean_object* v___x_4082_; lean_object* v___x_4083_; lean_object* v___y_4085_; uint8_t v___x_4087_; 
v___x_4076_ = lean_box(v_hasTrace_3589_);
v___f_4077_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__3___boxed), 5, 3);
lean_closure_set(v___f_4077_, 0, v_inst_3527_);
lean_closure_set(v___f_4077_, 1, v___x_4076_);
lean_closure_set(v___f_4077_, 2, v___x_3986_);
v___x_4082_ = lean_unsigned_to_nat(1u);
v___x_4083_ = lean_nat_sub(v___x_4074_, v___x_4082_);
v___x_4087_ = lean_nat_dec_le(v___x_4009_, v___x_4083_);
if (v___x_4087_ == 0)
{
lean_inc(v___x_4083_);
v___y_4085_ = v___x_4083_;
goto v___jp_4084_;
}
else
{
v___y_4085_ = v___x_4009_;
goto v___jp_4084_;
}
v___jp_4078_:
{
lean_object* v___x_4081_; 
v___x_4081_ = l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_box(0), v___f_4077_, v___x_4074_, v_a_4024_, v___y_4079_, v___y_4080_, lean_box(0), lean_box(0), lean_box(0));
lean_dec(v___y_4080_);
v___y_4028_ = v___x_4081_;
goto v___jp_4027_;
}
v___jp_4084_:
{
uint8_t v___x_4086_; 
v___x_4086_ = lean_nat_dec_le(v___y_4085_, v___x_4083_);
if (v___x_4086_ == 0)
{
lean_dec(v___x_4083_);
lean_inc(v___y_4085_);
v___y_4079_ = v___y_4085_;
v___y_4080_ = v___y_4085_;
goto v___jp_4078_;
}
else
{
v___y_4079_ = v___y_4085_;
v___y_4080_ = v___x_4083_;
goto v___jp_4078_;
}
}
}
else
{
lean_dec(v___x_3986_);
lean_dec_ref(v_inst_3527_);
v___y_4028_ = v_a_4024_;
goto v___jp_4027_;
}
v___jp_4027_:
{
lean_object* v___x_17112__overap_4029_; lean_object* v___x_4030_; 
lean_inc_ref(v___x_3577_);
v___x_17112__overap_4029_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_3577_, v___x_3590_, v___x_3591_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_4030_ = lean_apply_5(v___x_17112__overap_4029_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_4030_) == 0)
{
lean_object* v_a_4031_; lean_object* v___x_4033_; uint8_t v_isShared_4034_; uint8_t v_isSharedCheck_4065_; 
v_a_4031_ = lean_ctor_get(v___x_4030_, 0);
v_isSharedCheck_4065_ = !lean_is_exclusive(v___x_4030_);
if (v_isSharedCheck_4065_ == 0)
{
v___x_4033_ = v___x_4030_;
v_isShared_4034_ = v_isSharedCheck_4065_;
goto v_resetjp_4032_;
}
else
{
lean_inc(v_a_4031_);
lean_dec(v___x_4030_);
v___x_4033_ = lean_box(0);
v_isShared_4034_ = v_isSharedCheck_4065_;
goto v_resetjp_4032_;
}
v_resetjp_4032_:
{
uint8_t v___x_4035_; 
v___x_4035_ = lean_unbox(v_a_4031_);
lean_dec(v_a_4031_);
if (v___x_4035_ == 0)
{
lean_object* v___x_4037_; 
lean_dec_ref(v___f_4026_);
lean_dec_ref(v___x_3577_);
if (v_isShared_4034_ == 0)
{
lean_ctor_set(v___x_4033_, 0, v___y_4028_);
v___x_4037_ = v___x_4033_;
goto v_reusejp_4036_;
}
else
{
lean_object* v_reuseFailAlloc_4038_; 
v_reuseFailAlloc_4038_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4038_, 0, v___y_4028_);
v___x_4037_ = v_reuseFailAlloc_4038_;
goto v_reusejp_4036_;
}
v_reusejp_4036_:
{
return v___x_4037_;
}
}
else
{
lean_object* v___x_4039_; size_t v_sz_4040_; size_t v___x_4041_; lean_object* v___x_4042_; lean_object* v___x_4043_; lean_object* v___x_4044_; lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_17126__overap_4047_; lean_object* v___x_4048_; 
lean_del_object(v___x_4033_);
v___x_4039_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__19, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__19_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__19);
v_sz_4040_ = lean_array_size(v___y_4028_);
v___x_4041_ = ((size_t)0ULL);
lean_inc_ref(v___y_4028_);
v___x_4042_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___x_4016_, v___f_4026_, v_sz_4040_, v___x_4041_, v___y_4028_);
v___x_4043_ = lean_array_to_list(v___x_4042_);
v___x_4044_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__22, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__22_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__22);
v___x_4045_ = l_Lean_MessageData_joinSep(v___x_4043_, v___x_4044_);
v___x_4046_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4046_, 0, v___x_4039_);
lean_ctor_set(v___x_4046_, 1, v___x_4045_);
lean_inc(v_traceClass_3720_);
lean_inc_ref(v_toMonadRef_3580_);
v___x_17126__overap_4047_ = l_Lean_addTrace___redArg(v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, v_traceClass_3720_, v___x_4046_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_4048_ = lean_apply_5(v___x_17126__overap_4047_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_4048_) == 0)
{
lean_object* v___x_4050_; uint8_t v_isShared_4051_; uint8_t v_isSharedCheck_4055_; 
v_isSharedCheck_4055_ = !lean_is_exclusive(v___x_4048_);
if (v_isSharedCheck_4055_ == 0)
{
lean_object* v_unused_4056_; 
v_unused_4056_ = lean_ctor_get(v___x_4048_, 0);
lean_dec(v_unused_4056_);
v___x_4050_ = v___x_4048_;
v_isShared_4051_ = v_isSharedCheck_4055_;
goto v_resetjp_4049_;
}
else
{
lean_dec(v___x_4048_);
v___x_4050_ = lean_box(0);
v_isShared_4051_ = v_isSharedCheck_4055_;
goto v_resetjp_4049_;
}
v_resetjp_4049_:
{
lean_object* v___x_4053_; 
if (v_isShared_4051_ == 0)
{
lean_ctor_set(v___x_4050_, 0, v___y_4028_);
v___x_4053_ = v___x_4050_;
goto v_reusejp_4052_;
}
else
{
lean_object* v_reuseFailAlloc_4054_; 
v_reuseFailAlloc_4054_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4054_, 0, v___y_4028_);
v___x_4053_ = v_reuseFailAlloc_4054_;
goto v_reusejp_4052_;
}
v_reusejp_4052_:
{
return v___x_4053_;
}
}
}
else
{
lean_object* v_a_4057_; lean_object* v___x_4059_; uint8_t v_isShared_4060_; uint8_t v_isSharedCheck_4064_; 
lean_dec_ref(v___y_4028_);
v_a_4057_ = lean_ctor_get(v___x_4048_, 0);
v_isSharedCheck_4064_ = !lean_is_exclusive(v___x_4048_);
if (v_isSharedCheck_4064_ == 0)
{
v___x_4059_ = v___x_4048_;
v_isShared_4060_ = v_isSharedCheck_4064_;
goto v_resetjp_4058_;
}
else
{
lean_inc(v_a_4057_);
lean_dec(v___x_4048_);
v___x_4059_ = lean_box(0);
v_isShared_4060_ = v_isSharedCheck_4064_;
goto v_resetjp_4058_;
}
v_resetjp_4058_:
{
lean_object* v___x_4062_; 
if (v_isShared_4060_ == 0)
{
v___x_4062_ = v___x_4059_;
goto v_reusejp_4061_;
}
else
{
lean_object* v_reuseFailAlloc_4063_; 
v_reuseFailAlloc_4063_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4063_, 0, v_a_4057_);
v___x_4062_ = v_reuseFailAlloc_4063_;
goto v_reusejp_4061_;
}
v_reusejp_4061_:
{
return v___x_4062_;
}
}
}
}
}
}
else
{
lean_object* v_a_4066_; lean_object* v___x_4068_; uint8_t v_isShared_4069_; uint8_t v_isSharedCheck_4073_; 
lean_dec_ref(v___y_4028_);
lean_dec_ref(v___f_4026_);
lean_dec_ref(v___x_3577_);
v_a_4066_ = lean_ctor_get(v___x_4030_, 0);
v_isSharedCheck_4073_ = !lean_is_exclusive(v___x_4030_);
if (v_isSharedCheck_4073_ == 0)
{
v___x_4068_ = v___x_4030_;
v_isShared_4069_ = v_isSharedCheck_4073_;
goto v_resetjp_4067_;
}
else
{
lean_inc(v_a_4066_);
lean_dec(v___x_4030_);
v___x_4068_ = lean_box(0);
v_isShared_4069_ = v_isSharedCheck_4073_;
goto v_resetjp_4067_;
}
v_resetjp_4067_:
{
lean_object* v___x_4071_; 
if (v_isShared_4069_ == 0)
{
v___x_4071_ = v___x_4068_;
goto v_reusejp_4070_;
}
else
{
lean_object* v_reuseFailAlloc_4072_; 
v_reuseFailAlloc_4072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4072_, 0, v_a_4066_);
v___x_4071_ = v_reuseFailAlloc_4072_;
goto v_reusejp_4070_;
}
v_reusejp_4070_:
{
return v___x_4071_;
}
}
}
}
}
else
{
lean_dec(v___x_3986_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_inst_3527_);
return v___x_4023_;
}
}
else
{
lean_object* v_a_4088_; lean_object* v___x_4090_; uint8_t v_isShared_4091_; uint8_t v_isSharedCheck_4095_; 
lean_dec(v_a_4008_);
lean_dec(v___x_3986_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_4088_ = lean_ctor_get(v___x_4013_, 0);
v_isSharedCheck_4095_ = !lean_is_exclusive(v___x_4013_);
if (v_isSharedCheck_4095_ == 0)
{
v___x_4090_ = v___x_4013_;
v_isShared_4091_ = v_isSharedCheck_4095_;
goto v_resetjp_4089_;
}
else
{
lean_inc(v_a_4088_);
lean_dec(v___x_4013_);
v___x_4090_ = lean_box(0);
v_isShared_4091_ = v_isSharedCheck_4095_;
goto v_resetjp_4089_;
}
v_resetjp_4089_:
{
lean_object* v___x_4093_; 
if (v_isShared_4091_ == 0)
{
v___x_4093_ = v___x_4090_;
goto v_reusejp_4092_;
}
else
{
lean_object* v_reuseFailAlloc_4094_; 
v_reuseFailAlloc_4094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4094_, 0, v_a_4088_);
v___x_4093_ = v_reuseFailAlloc_4094_;
goto v_reusejp_4092_;
}
v_reusejp_4092_:
{
return v___x_4093_;
}
}
}
}
else
{
lean_object* v_a_4096_; lean_object* v___x_4098_; uint8_t v_isShared_4099_; uint8_t v_isSharedCheck_4103_; 
lean_dec_ref(v___x_4005_);
lean_dec(v___x_3986_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_4096_ = lean_ctor_get(v___x_4007_, 0);
v_isSharedCheck_4103_ = !lean_is_exclusive(v___x_4007_);
if (v_isSharedCheck_4103_ == 0)
{
v___x_4098_ = v___x_4007_;
v_isShared_4099_ = v_isSharedCheck_4103_;
goto v_resetjp_4097_;
}
else
{
lean_inc(v_a_4096_);
lean_dec(v___x_4007_);
v___x_4098_ = lean_box(0);
v_isShared_4099_ = v_isSharedCheck_4103_;
goto v_resetjp_4097_;
}
v_resetjp_4097_:
{
lean_object* v___x_4101_; 
if (v_isShared_4099_ == 0)
{
v___x_4101_ = v___x_4098_;
goto v_reusejp_4100_;
}
else
{
lean_object* v_reuseFailAlloc_4102_; 
v_reuseFailAlloc_4102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4102_, 0, v_a_4096_);
v___x_4101_ = v_reuseFailAlloc_4102_;
goto v_reusejp_4100_;
}
v_reusejp_4100_:
{
return v___x_4101_;
}
}
}
}
else
{
lean_object* v_a_4104_; lean_object* v___x_4106_; uint8_t v_isShared_4107_; uint8_t v_isSharedCheck_4111_; 
lean_dec(v___x_3986_);
lean_dec_ref(v___f_3724_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_4104_ = lean_ctor_get(v___x_3989_, 0);
v_isSharedCheck_4111_ = !lean_is_exclusive(v___x_3989_);
if (v_isSharedCheck_4111_ == 0)
{
v___x_4106_ = v___x_3989_;
v_isShared_4107_ = v_isSharedCheck_4111_;
goto v_resetjp_4105_;
}
else
{
lean_inc(v_a_4104_);
lean_dec(v___x_3989_);
v___x_4106_ = lean_box(0);
v_isShared_4107_ = v_isSharedCheck_4111_;
goto v_resetjp_4105_;
}
v_resetjp_4105_:
{
lean_object* v___x_4109_; 
if (v_isShared_4107_ == 0)
{
v___x_4109_ = v___x_4106_;
goto v_reusejp_4108_;
}
else
{
lean_object* v_reuseFailAlloc_4110_; 
v_reuseFailAlloc_4110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4110_, 0, v_a_4104_);
v___x_4109_ = v_reuseFailAlloc_4110_;
goto v_reusejp_4108_;
}
v_reusejp_4108_:
{
return v___x_4109_;
}
}
}
}
else
{
lean_dec(v___x_3986_);
goto v___jp_3867_;
}
}
else
{
goto v___jp_3867_;
}
v___jp_3733_:
{
lean_object* v___x_3737_; double v___x_3738_; double v___x_3739_; double v___x_3740_; double v___x_3741_; double v___x_3742_; lean_object* v___x_3743_; lean_object* v___x_3744_; lean_object* v___x_3745_; lean_object* v___x_3746_; lean_object* v___x_16573__overap_3747_; lean_object* v___x_3748_; 
v___x_3737_ = lean_io_mono_nanos_now();
v___x_3738_ = lean_float_of_nat(v___y_3734_);
v___x_3739_ = lean_float_once(&lp_aesop_Aesop_Index_trace___redArg___closed__7, &lp_aesop_Aesop_Index_trace___redArg___closed__7_once, _init_lp_aesop_Aesop_Index_trace___redArg___closed__7);
v___x_3740_ = lean_float_div(v___x_3738_, v___x_3739_);
v___x_3741_ = lean_float_of_nat(v___x_3737_);
v___x_3742_ = lean_float_div(v___x_3741_, v___x_3739_);
v___x_3743_ = lean_box_float(v___x_3740_);
v___x_3744_ = lean_box_float(v___x_3742_);
v___x_3745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3745_, 0, v___x_3743_);
lean_ctor_set(v___x_3745_, 1, v___x_3744_);
v___x_3746_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3746_, 0, v_a_3736_);
lean_ctor_set(v___x_3746_, 1, v___x_3745_);
lean_inc(v_traceClass_3720_);
lean_inc_ref(v_toMonadRef_3580_);
v___x_16573__overap_3747_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, lean_box(0), v___x_3582_, v___f_3725_, v_traceClass_3720_, v_hasTrace_3589_, v___x_3729_, v_options_3587_, v___x_3732_, v___y_3735_, v___f_3728_, v___x_3746_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3748_ = lean_apply_5(v___x_16573__overap_3747_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
return v___x_3748_;
}
v___jp_3749_:
{
lean_object* v___x_3753_; 
v___x_3753_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3753_, 0, v_a_3752_);
v___y_3734_ = v___y_3750_;
v___y_3735_ = v___y_3751_;
v_a_3736_ = v___x_3753_;
goto v___jp_3733_;
}
v___jp_3754_:
{
lean_object* v___x_3758_; 
v___x_3758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3758_, 0, v_a_3757_);
v___y_3734_ = v___y_3755_;
v___y_3735_ = v___y_3756_;
v_a_3736_ = v___x_3758_;
goto v___jp_3733_;
}
v___jp_3759_:
{
lean_object* v___x_16718__overap_3764_; lean_object* v___x_3765_; 
lean_inc_ref(v___x_3577_);
v___x_16718__overap_3764_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_3577_, v___x_3590_, v___x_3591_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3765_ = lean_apply_5(v___x_16718__overap_3764_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3765_) == 0)
{
lean_object* v_a_3766_; uint8_t v___x_3767_; 
v_a_3766_ = lean_ctor_get(v___x_3765_, 0);
lean_inc(v_a_3766_);
lean_dec_ref_known(v___x_3765_, 1);
v___x_3767_ = lean_unbox(v_a_3766_);
lean_dec(v_a_3766_);
if (v___x_3767_ == 0)
{
lean_dec_ref(v___y_3762_);
lean_dec_ref(v___f_3727_);
v___y_3755_ = v___y_3760_;
v___y_3756_ = v___y_3761_;
v_a_3757_ = v___y_3763_;
goto v___jp_3754_;
}
else
{
lean_object* v___x_3768_; size_t v_sz_3769_; size_t v___x_3770_; lean_object* v___x_3771_; lean_object* v___x_3772_; lean_object* v___x_3773_; lean_object* v___x_3774_; lean_object* v___x_3775_; lean_object* v___x_16731__overap_3776_; lean_object* v___x_3777_; 
v___x_3768_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__19, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__19_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__19);
v_sz_3769_ = lean_array_size(v___y_3763_);
v___x_3770_ = ((size_t)0ULL);
lean_inc_ref(v___y_3763_);
v___x_3771_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___y_3762_, v___f_3727_, v_sz_3769_, v___x_3770_, v___y_3763_);
v___x_3772_ = lean_array_to_list(v___x_3771_);
v___x_3773_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__22, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__22_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__22);
v___x_3774_ = l_Lean_MessageData_joinSep(v___x_3772_, v___x_3773_);
v___x_3775_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3775_, 0, v___x_3768_);
lean_ctor_set(v___x_3775_, 1, v___x_3774_);
lean_inc(v_traceClass_3720_);
lean_inc_ref(v_toMonadRef_3580_);
lean_inc_ref(v___x_3577_);
v___x_16731__overap_3776_ = l_Lean_addTrace___redArg(v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, v_traceClass_3720_, v___x_3775_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3777_ = lean_apply_5(v___x_16731__overap_3776_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3777_) == 0)
{
lean_dec_ref_known(v___x_3777_, 1);
v___y_3755_ = v___y_3760_;
v___y_3756_ = v___y_3761_;
v_a_3757_ = v___y_3763_;
goto v___jp_3754_;
}
else
{
lean_object* v_a_3778_; 
lean_dec_ref(v___y_3763_);
v_a_3778_ = lean_ctor_get(v___x_3777_, 0);
lean_inc(v_a_3778_);
lean_dec_ref_known(v___x_3777_, 1);
v___y_3750_ = v___y_3760_;
v___y_3751_ = v___y_3761_;
v_a_3752_ = v_a_3778_;
goto v___jp_3749_;
}
}
}
else
{
lean_object* v_a_3779_; 
lean_dec_ref(v___y_3763_);
lean_dec_ref(v___y_3762_);
lean_dec_ref(v___f_3727_);
v_a_3779_ = lean_ctor_get(v___x_3765_, 0);
lean_inc(v_a_3779_);
lean_dec_ref_known(v___x_3765_, 1);
v___y_3750_ = v___y_3760_;
v___y_3751_ = v___y_3761_;
v_a_3752_ = v_a_3779_;
goto v___jp_3749_;
}
}
v___jp_3780_:
{
lean_object* v___x_3789_; 
v___x_3789_ = l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_box(0), v___y_3786_, v___y_3785_, v___y_3787_, v___y_3784_, v___y_3788_, lean_box(0), lean_box(0), lean_box(0));
lean_dec(v___y_3788_);
lean_dec(v___y_3785_);
v___y_3760_ = v___y_3781_;
v___y_3761_ = v___y_3782_;
v___y_3762_ = v___y_3783_;
v___y_3763_ = v___x_3789_;
goto v___jp_3759_;
}
v___jp_3790_:
{
uint8_t v___x_3799_; 
v___x_3799_ = lean_nat_dec_le(v___y_3798_, v___y_3794_);
if (v___x_3799_ == 0)
{
lean_dec(v___y_3794_);
lean_inc(v___y_3798_);
v___y_3781_ = v___y_3791_;
v___y_3782_ = v___y_3792_;
v___y_3783_ = v___y_3793_;
v___y_3784_ = v___y_3798_;
v___y_3785_ = v___y_3796_;
v___y_3786_ = v___y_3795_;
v___y_3787_ = v___y_3797_;
v___y_3788_ = v___y_3798_;
goto v___jp_3780_;
}
else
{
v___y_3781_ = v___y_3791_;
v___y_3782_ = v___y_3792_;
v___y_3783_ = v___y_3793_;
v___y_3784_ = v___y_3798_;
v___y_3785_ = v___y_3796_;
v___y_3786_ = v___y_3795_;
v___y_3787_ = v___y_3797_;
v___y_3788_ = v___y_3794_;
goto v___jp_3780_;
}
}
v___jp_3800_:
{
lean_object* v___x_3804_; double v___x_3805_; double v___x_3806_; lean_object* v___x_3807_; lean_object* v___x_3808_; lean_object* v___x_3809_; lean_object* v___x_3810_; lean_object* v___x_16776__overap_3811_; lean_object* v___x_3812_; 
v___x_3804_ = lean_io_get_num_heartbeats();
v___x_3805_ = lean_float_of_nat(v___y_3802_);
v___x_3806_ = lean_float_of_nat(v___x_3804_);
v___x_3807_ = lean_box_float(v___x_3805_);
v___x_3808_ = lean_box_float(v___x_3806_);
v___x_3809_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3809_, 0, v___x_3807_);
lean_ctor_set(v___x_3809_, 1, v___x_3808_);
v___x_3810_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3810_, 0, v_a_3803_);
lean_ctor_set(v___x_3810_, 1, v___x_3809_);
lean_inc(v_traceClass_3720_);
lean_inc_ref(v_toMonadRef_3580_);
v___x_16776__overap_3811_ = l___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback(lean_box(0), lean_box(0), v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, lean_box(0), v___x_3582_, v___f_3725_, v_traceClass_3720_, v_hasTrace_3589_, v___x_3729_, v_options_3587_, v___x_3732_, v___y_3801_, v___f_3728_, v___x_3810_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3812_ = lean_apply_5(v___x_16776__overap_3811_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
return v___x_3812_;
}
v___jp_3813_:
{
lean_object* v___x_3817_; 
v___x_3817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3817_, 0, v_a_3816_);
v___y_3801_ = v___y_3814_;
v___y_3802_ = v___y_3815_;
v_a_3803_ = v___x_3817_;
goto v___jp_3800_;
}
v___jp_3818_:
{
lean_object* v___x_3822_; 
v___x_3822_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3822_, 0, v_a_3821_);
v___y_3801_ = v___y_3819_;
v___y_3802_ = v___y_3820_;
v_a_3803_ = v___x_3822_;
goto v___jp_3800_;
}
v___jp_3823_:
{
lean_object* v___x_16921__overap_3829_; lean_object* v___x_3830_; 
lean_inc_ref(v___x_3577_);
v___x_16921__overap_3829_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_3577_, v___x_3590_, v___x_3591_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3830_ = lean_apply_5(v___x_16921__overap_3829_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3830_) == 0)
{
lean_object* v_a_3831_; uint8_t v___x_3832_; 
v_a_3831_ = lean_ctor_get(v___x_3830_, 0);
lean_inc(v_a_3831_);
lean_dec_ref_known(v___x_3830_, 1);
v___x_3832_ = lean_unbox(v_a_3831_);
lean_dec(v_a_3831_);
if (v___x_3832_ == 0)
{
lean_dec_ref(v___y_3826_);
lean_dec_ref(v___y_3825_);
v___y_3814_ = v___y_3824_;
v___y_3815_ = v___y_3827_;
v_a_3816_ = v___y_3828_;
goto v___jp_3813_;
}
else
{
lean_object* v___x_3833_; size_t v_sz_3834_; size_t v___x_3835_; lean_object* v___x_3836_; lean_object* v___x_3837_; lean_object* v___x_3838_; lean_object* v___x_3839_; lean_object* v___x_3840_; lean_object* v___x_16934__overap_3841_; lean_object* v___x_3842_; 
v___x_3833_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__19, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__19_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__19);
v_sz_3834_ = lean_array_size(v___y_3828_);
v___x_3835_ = ((size_t)0ULL);
lean_inc_ref(v___y_3828_);
v___x_3836_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v___y_3826_, v___y_3825_, v_sz_3834_, v___x_3835_, v___y_3828_);
v___x_3837_ = lean_array_to_list(v___x_3836_);
v___x_3838_ = lean_obj_once(&lp_aesop_Aesop_Index_applicableRules___redArg___closed__22, &lp_aesop_Aesop_Index_applicableRules___redArg___closed__22_once, _init_lp_aesop_Aesop_Index_applicableRules___redArg___closed__22);
v___x_3839_ = l_Lean_MessageData_joinSep(v___x_3837_, v___x_3838_);
v___x_3840_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3840_, 0, v___x_3833_);
lean_ctor_set(v___x_3840_, 1, v___x_3839_);
lean_inc(v_traceClass_3720_);
lean_inc_ref(v_toMonadRef_3580_);
lean_inc_ref(v___x_3577_);
v___x_16934__overap_3841_ = l_Lean_addTrace___redArg(v___x_3577_, v___x_3578_, v_toMonadRef_3580_, v___x_3581_, v_traceClass_3720_, v___x_3840_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3842_ = lean_apply_5(v___x_16934__overap_3841_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3842_) == 0)
{
lean_dec_ref_known(v___x_3842_, 1);
v___y_3814_ = v___y_3824_;
v___y_3815_ = v___y_3827_;
v_a_3816_ = v___y_3828_;
goto v___jp_3813_;
}
else
{
lean_object* v_a_3843_; 
lean_dec_ref(v___y_3828_);
v_a_3843_ = lean_ctor_get(v___x_3842_, 0);
lean_inc(v_a_3843_);
lean_dec_ref_known(v___x_3842_, 1);
v___y_3819_ = v___y_3824_;
v___y_3820_ = v___y_3827_;
v_a_3821_ = v_a_3843_;
goto v___jp_3818_;
}
}
}
else
{
lean_object* v_a_3844_; 
lean_dec_ref(v___y_3828_);
lean_dec_ref(v___y_3826_);
lean_dec_ref(v___y_3825_);
v_a_3844_ = lean_ctor_get(v___x_3830_, 0);
lean_inc(v_a_3844_);
lean_dec_ref_known(v___x_3830_, 1);
v___y_3819_ = v___y_3824_;
v___y_3820_ = v___y_3827_;
v_a_3821_ = v_a_3844_;
goto v___jp_3818_;
}
}
v___jp_3845_:
{
lean_object* v___x_3855_; 
v___x_3855_ = l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_box(0), v___y_3853_, v___y_3852_, v___y_3848_, v___y_3851_, v___y_3854_, lean_box(0), lean_box(0), lean_box(0));
lean_dec(v___y_3854_);
lean_dec(v___y_3852_);
v___y_3824_ = v___y_3846_;
v___y_3825_ = v___y_3847_;
v___y_3826_ = v___y_3849_;
v___y_3827_ = v___y_3850_;
v___y_3828_ = v___x_3855_;
goto v___jp_3823_;
}
v___jp_3856_:
{
uint8_t v___x_3866_; 
v___x_3866_ = lean_nat_dec_le(v___y_3865_, v___y_3860_);
if (v___x_3866_ == 0)
{
lean_dec(v___y_3860_);
lean_inc(v___y_3865_);
v___y_3846_ = v___y_3857_;
v___y_3847_ = v___y_3859_;
v___y_3848_ = v___y_3858_;
v___y_3849_ = v___y_3861_;
v___y_3850_ = v___y_3862_;
v___y_3851_ = v___y_3865_;
v___y_3852_ = v___y_3864_;
v___y_3853_ = v___y_3863_;
v___y_3854_ = v___y_3865_;
goto v___jp_3845_;
}
else
{
v___y_3846_ = v___y_3857_;
v___y_3847_ = v___y_3859_;
v___y_3848_ = v___y_3858_;
v___y_3849_ = v___y_3861_;
v___y_3850_ = v___y_3862_;
v___y_3851_ = v___y_3865_;
v___y_3852_ = v___y_3864_;
v___y_3853_ = v___y_3863_;
v___y_3854_ = v___y_3860_;
goto v___jp_3845_;
}
}
v___jp_3867_:
{
lean_object* v___x_16504__overap_3868_; lean_object* v___x_3869_; 
lean_inc_ref(v___x_3577_);
v___x_16504__overap_3868_ = l___private_Lean_Util_Trace_0__Lean_getResetTraces(lean_box(0), v___x_3577_, v___x_3578_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3869_ = lean_apply_5(v___x_16504__overap_3868_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3869_) == 0)
{
lean_object* v_a_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; uint8_t v___x_3874_; 
v_a_3870_ = lean_ctor_get(v___x_3869_, 0);
lean_inc(v_a_3870_);
lean_dec_ref_known(v___x_3869_, 1);
v___x_3871_ = l_Lean_KVMap_instValueBool;
v___x_3872_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3873_ = l_Lean_Option_get___redArg(v___x_3871_, v_options_3587_, v___x_3872_);
v___x_3874_ = lean_unbox(v___x_3873_);
if (v___x_3874_ == 0)
{
lean_object* v___x_3875_; lean_object* v___x_16582__overap_3876_; lean_object* v___x_3877_; 
v___x_3875_ = lean_io_mono_nanos_now();
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
v___x_16582__overap_3876_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg(v___x_3577_, v___x_3583_, v___x_3586_, v_goal_3529_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3877_ = lean_apply_5(v___x_16582__overap_3876_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3877_) == 0)
{
lean_object* v_toApplicative_3878_; lean_object* v_toFunctor_3879_; lean_object* v_toSeq_3880_; lean_object* v_toSeqLeft_3881_; lean_object* v_toSeqRight_3882_; lean_object* v___f_3883_; lean_object* v___f_3884_; lean_object* v___x_3885_; lean_object* v___f_3886_; lean_object* v___f_3887_; lean_object* v___f_3888_; lean_object* v___x_3889_; lean_object* v___x_3890_; lean_object* v___x_3891_; lean_object* v___x_3892_; lean_object* v___x_3893_; lean_object* v___x_16630__overap_3894_; lean_object* v___x_3895_; 
lean_dec_ref_known(v___x_3877_, 1);
v_toApplicative_3878_ = lean_ctor_get(v___x_3538_, 0);
v_toFunctor_3879_ = lean_ctor_get(v_toApplicative_3878_, 0);
v_toSeq_3880_ = lean_ctor_get(v_toApplicative_3878_, 2);
v_toSeqLeft_3881_ = lean_ctor_get(v_toApplicative_3878_, 3);
v_toSeqRight_3882_ = lean_ctor_get(v_toApplicative_3878_, 4);
lean_inc_ref_n(v_toFunctor_3879_, 2);
v___f_3883_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3883_, 0, v_toFunctor_3879_);
v___f_3884_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3884_, 0, v_toFunctor_3879_);
v___x_3885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3885_, 0, v___f_3883_);
lean_ctor_set(v___x_3885_, 1, v___f_3884_);
lean_inc(v_toSeqRight_3882_);
v___f_3886_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3886_, 0, v_toSeqRight_3882_);
lean_inc(v_toSeqLeft_3881_);
v___f_3887_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3887_, 0, v_toSeqLeft_3881_);
lean_inc(v_toSeq_3880_);
v___f_3888_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3888_, 0, v_toSeq_3880_);
v___x_3889_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3889_, 0, v___x_3885_);
lean_ctor_set(v___x_3889_, 1, v___f_3544_);
lean_ctor_set(v___x_3889_, 2, v___f_3888_);
lean_ctor_set(v___x_3889_, 3, v___f_3887_);
lean_ctor_set(v___x_3889_, 4, v___f_3886_);
v___x_3890_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3890_, 0, v___x_3889_);
lean_ctor_set(v___x_3890_, 1, v___f_3545_);
v___x_3891_ = l_StateRefT_x27_instMonad___redArg(v___x_3890_);
v___x_3892_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_3892_, 0, lean_box(0));
lean_closure_set(v___x_3892_, 1, lean_box(0));
lean_closure_set(v___x_3892_, 2, v___x_3891_);
v___x_3893_ = l_instMonadControlTOfPure___redArg(v___x_3892_);
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
lean_inc_ref(v___x_3893_);
v___x_16630__overap_3894_ = l_Lean_MVarId_withContext___redArg(v___x_3893_, v___x_3577_, v_goal_3529_, v___f_3724_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3895_ = lean_apply_5(v___x_16630__overap_3894_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3895_) == 0)
{
lean_object* v_a_3896_; lean_object* v___x_3897_; lean_object* v_rs_3898_; lean_object* v___f_3899_; lean_object* v___x_16691__overap_3900_; lean_object* v___x_3901_; 
v_a_3896_ = lean_ctor_get(v___x_3895_, 0);
lean_inc(v_a_3896_);
lean_dec_ref_known(v___x_3895_, 1);
v___x_3897_ = lean_unsigned_to_nat(0u);
v_rs_3898_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
lean_inc_ref_n(v___x_3577_, 2);
v___f_3899_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_3899_, 0, v___x_3577_);
lean_closure_set(v___f_3899_, 1, v_rs_3898_);
lean_closure_set(v___f_3899_, 2, v___f_3723_);
v___x_16691__overap_3900_ = l_Lean_MVarId_withContext___redArg(v___x_3893_, v___x_3577_, v_goal_3529_, v___f_3899_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3901_ = lean_apply_5(v___x_16691__overap_3900_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3901_) == 0)
{
lean_object* v_a_3902_; lean_object* v_unindexed_3903_; lean_object* v___x_3904_; lean_object* v___x_3905_; lean_object* v___x_3906_; lean_object* v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; lean_object* v___x_3911_; 
v_a_3902_ = lean_ctor_get(v___x_3901_, 0);
lean_inc(v_a_3902_);
lean_dec_ref_known(v___x_3901_, 1);
v_unindexed_3903_ = lean_ctor_get(v_ri_3528_, 2);
lean_inc_ref(v_unindexed_3903_);
lean_dec_ref(v_ri_3528_);
v___x_3904_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_3905_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_3904_, v___f_3721_, v_unindexed_3903_, v_rs_3898_);
v___x_3906_ = lean_unsigned_to_nat(3u);
v___x_3907_ = lean_mk_empty_array_with_capacity(v___x_3906_);
v___x_3908_ = lean_array_push(v___x_3907_, v_a_3896_);
v___x_3909_ = lean_array_push(v___x_3908_, v_a_3902_);
v___x_3910_ = lean_array_push(v___x_3909_, v___x_3905_);
v___x_3911_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_3530_, v_additionalRules_3531_, v___x_3910_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_);
lean_dec_ref(v___x_3910_);
if (lean_obj_tag(v___x_3911_) == 0)
{
lean_object* v_a_3912_; lean_object* v___x_3913_; uint8_t v___x_3914_; 
v_a_3912_ = lean_ctor_get(v___x_3911_, 0);
lean_inc(v_a_3912_);
lean_dec_ref_known(v___x_3911_, 1);
v___x_3913_ = lean_array_get_size(v_a_3912_);
v___x_3914_ = lean_nat_dec_eq(v___x_3913_, v___x_3897_);
if (v___x_3914_ == 0)
{
lean_object* v___x_3915_; lean_object* v___f_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; uint8_t v___x_3919_; 
v___x_3915_ = lean_box(v_hasTrace_3589_);
v___f_3916_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__3___boxed), 5, 3);
lean_closure_set(v___f_3916_, 0, v_inst_3527_);
lean_closure_set(v___f_3916_, 1, v___x_3915_);
lean_closure_set(v___f_3916_, 2, v___x_3873_);
v___x_3917_ = lean_unsigned_to_nat(1u);
v___x_3918_ = lean_nat_sub(v___x_3913_, v___x_3917_);
v___x_3919_ = lean_nat_dec_le(v___x_3897_, v___x_3918_);
if (v___x_3919_ == 0)
{
lean_inc(v___x_3918_);
v___y_3791_ = v___x_3875_;
v___y_3792_ = v_a_3870_;
v___y_3793_ = v___x_3904_;
v___y_3794_ = v___x_3918_;
v___y_3795_ = v___f_3916_;
v___y_3796_ = v___x_3913_;
v___y_3797_ = v_a_3912_;
v___y_3798_ = v___x_3918_;
goto v___jp_3790_;
}
else
{
v___y_3791_ = v___x_3875_;
v___y_3792_ = v_a_3870_;
v___y_3793_ = v___x_3904_;
v___y_3794_ = v___x_3918_;
v___y_3795_ = v___f_3916_;
v___y_3796_ = v___x_3913_;
v___y_3797_ = v_a_3912_;
v___y_3798_ = v___x_3897_;
goto v___jp_3790_;
}
}
else
{
lean_dec(v___x_3873_);
lean_dec_ref(v_inst_3527_);
v___y_3760_ = v___x_3875_;
v___y_3761_ = v_a_3870_;
v___y_3762_ = v___x_3904_;
v___y_3763_ = v_a_3912_;
goto v___jp_3759_;
}
}
else
{
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3727_);
lean_dec_ref(v_inst_3527_);
if (lean_obj_tag(v___x_3911_) == 0)
{
lean_object* v_a_3920_; 
v_a_3920_ = lean_ctor_get(v___x_3911_, 0);
lean_inc(v_a_3920_);
lean_dec_ref_known(v___x_3911_, 1);
v___y_3755_ = v___x_3875_;
v___y_3756_ = v_a_3870_;
v_a_3757_ = v_a_3920_;
goto v___jp_3754_;
}
else
{
lean_object* v_a_3921_; 
v_a_3921_ = lean_ctor_get(v___x_3911_, 0);
lean_inc(v_a_3921_);
lean_dec_ref_known(v___x_3911_, 1);
v___y_3750_ = v___x_3875_;
v___y_3751_ = v_a_3870_;
v_a_3752_ = v_a_3921_;
goto v___jp_3749_;
}
}
}
else
{
lean_object* v_a_3922_; 
lean_dec(v_a_3896_);
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3727_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3922_ = lean_ctor_get(v___x_3901_, 0);
lean_inc(v_a_3922_);
lean_dec_ref_known(v___x_3901_, 1);
v___y_3750_ = v___x_3875_;
v___y_3751_ = v_a_3870_;
v_a_3752_ = v_a_3922_;
goto v___jp_3749_;
}
}
else
{
lean_object* v_a_3923_; 
lean_dec_ref(v___x_3893_);
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3727_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3923_ = lean_ctor_get(v___x_3895_, 0);
lean_inc(v_a_3923_);
lean_dec_ref_known(v___x_3895_, 1);
v___y_3750_ = v___x_3875_;
v___y_3751_ = v_a_3870_;
v_a_3752_ = v_a_3923_;
goto v___jp_3749_;
}
}
else
{
lean_object* v_a_3924_; 
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3727_);
lean_dec_ref(v___f_3724_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3924_ = lean_ctor_get(v___x_3877_, 0);
lean_inc(v_a_3924_);
lean_dec_ref_known(v___x_3877_, 1);
v___y_3750_ = v___x_3875_;
v___y_3751_ = v_a_3870_;
v_a_3752_ = v_a_3924_;
goto v___jp_3749_;
}
}
else
{
lean_object* v___x_3925_; lean_object* v___x_16785__overap_3926_; lean_object* v___x_3927_; 
lean_dec_ref(v___f_3727_);
v___x_3925_ = lean_io_get_num_heartbeats();
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
v___x_16785__overap_3926_ = lp_batteries_Lean_MVarId_instantiateMVars___redArg(v___x_3577_, v___x_3583_, v___x_3586_, v_goal_3529_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3927_ = lean_apply_5(v___x_16785__overap_3926_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3927_) == 0)
{
lean_object* v_toApplicative_3928_; lean_object* v_toFunctor_3929_; lean_object* v_toSeq_3930_; lean_object* v_toSeqLeft_3931_; lean_object* v_toSeqRight_3932_; lean_object* v___f_3933_; lean_object* v___f_3934_; lean_object* v___x_3935_; lean_object* v___f_3936_; lean_object* v___f_3937_; lean_object* v___f_3938_; lean_object* v___x_3939_; lean_object* v___x_3940_; lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_16833__overap_3944_; lean_object* v___x_3945_; 
lean_dec_ref_known(v___x_3927_, 1);
v_toApplicative_3928_ = lean_ctor_get(v___x_3538_, 0);
v_toFunctor_3929_ = lean_ctor_get(v_toApplicative_3928_, 0);
v_toSeq_3930_ = lean_ctor_get(v_toApplicative_3928_, 2);
v_toSeqLeft_3931_ = lean_ctor_get(v_toApplicative_3928_, 3);
v_toSeqRight_3932_ = lean_ctor_get(v_toApplicative_3928_, 4);
lean_inc_ref_n(v_toFunctor_3929_, 2);
v___f_3933_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3933_, 0, v_toFunctor_3929_);
v___f_3934_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3934_, 0, v_toFunctor_3929_);
v___x_3935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3935_, 0, v___f_3933_);
lean_ctor_set(v___x_3935_, 1, v___f_3934_);
lean_inc(v_toSeqRight_3932_);
v___f_3936_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3936_, 0, v_toSeqRight_3932_);
lean_inc(v_toSeqLeft_3931_);
v___f_3937_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3937_, 0, v_toSeqLeft_3931_);
lean_inc(v_toSeq_3930_);
v___f_3938_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3938_, 0, v_toSeq_3930_);
v___x_3939_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3939_, 0, v___x_3935_);
lean_ctor_set(v___x_3939_, 1, v___f_3544_);
lean_ctor_set(v___x_3939_, 2, v___f_3938_);
lean_ctor_set(v___x_3939_, 3, v___f_3937_);
lean_ctor_set(v___x_3939_, 4, v___f_3936_);
v___x_3940_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3940_, 0, v___x_3939_);
lean_ctor_set(v___x_3940_, 1, v___f_3545_);
v___x_3941_ = l_StateRefT_x27_instMonad___redArg(v___x_3940_);
v___x_3942_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_3942_, 0, lean_box(0));
lean_closure_set(v___x_3942_, 1, lean_box(0));
lean_closure_set(v___x_3942_, 2, v___x_3941_);
v___x_3943_ = l_instMonadControlTOfPure___redArg(v___x_3942_);
lean_inc(v_goal_3529_);
lean_inc_ref(v___x_3577_);
lean_inc_ref(v___x_3943_);
v___x_16833__overap_3944_ = l_Lean_MVarId_withContext___redArg(v___x_3943_, v___x_3577_, v_goal_3529_, v___f_3724_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3945_ = lean_apply_5(v___x_16833__overap_3944_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3945_) == 0)
{
lean_object* v_a_3946_; lean_object* v___x_3947_; lean_object* v_rs_3948_; lean_object* v___f_3949_; lean_object* v___x_16894__overap_3950_; lean_object* v___x_3951_; 
v_a_3946_ = lean_ctor_get(v___x_3945_, 0);
lean_inc(v_a_3946_);
lean_dec_ref_known(v___x_3945_, 1);
v___x_3947_ = lean_unsigned_to_nat(0u);
v_rs_3948_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___closed__0));
lean_inc_ref_n(v___x_3577_, 2);
v___f_3949_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableByHypRules___redArg___lam__2___boxed), 8, 3);
lean_closure_set(v___f_3949_, 0, v___x_3577_);
lean_closure_set(v___f_3949_, 1, v_rs_3948_);
lean_closure_set(v___f_3949_, 2, v___f_3723_);
v___x_16894__overap_3950_ = l_Lean_MVarId_withContext___redArg(v___x_3943_, v___x_3577_, v_goal_3529_, v___f_3949_);
lean_inc(v_a_3536_);
lean_inc_ref(v_a_3535_);
lean_inc(v_a_3534_);
lean_inc_ref(v_a_3533_);
v___x_3951_ = lean_apply_5(v___x_16894__overap_3950_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_, lean_box(0));
if (lean_obj_tag(v___x_3951_) == 0)
{
lean_object* v_a_3952_; lean_object* v_unindexed_3953_; lean_object* v___x_3954_; lean_object* v___x_3955_; lean_object* v___x_3956_; lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3960_; lean_object* v___x_3961_; 
v_a_3952_ = lean_ctor_get(v___x_3951_, 0);
lean_inc(v_a_3952_);
lean_dec_ref_known(v___x_3951_, 1);
v_unindexed_3953_ = lean_ctor_get(v_ri_3528_, 2);
lean_inc_ref(v_unindexed_3953_);
lean_dec_ref(v_ri_3528_);
v___x_3954_ = ((lean_object*)(lp_aesop___private_Aesop_Index_0__Aesop_Index_trace_traceArray___redArg___closed__14));
v___x_3955_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_3954_, v___f_3721_, v_unindexed_3953_, v_rs_3948_);
v___x_3956_ = lean_unsigned_to_nat(3u);
v___x_3957_ = lean_mk_empty_array_with_capacity(v___x_3956_);
v___x_3958_ = lean_array_push(v___x_3957_, v_a_3946_);
v___x_3959_ = lean_array_push(v___x_3958_, v_a_3952_);
v___x_3960_ = lean_array_push(v___x_3959_, v___x_3955_);
v___x_3961_ = lp_aesop___private_Aesop_Index_0__Aesop_Index_applicableRules_addRules___redArg(v_patSubstMap_3530_, v_additionalRules_3531_, v___x_3960_, v_a_3533_, v_a_3534_, v_a_3535_, v_a_3536_);
lean_dec_ref(v___x_3960_);
if (lean_obj_tag(v___x_3961_) == 0)
{
lean_object* v_a_3962_; lean_object* v___f_3963_; lean_object* v___x_3964_; uint8_t v___x_3965_; 
v_a_3962_ = lean_ctor_get(v___x_3961_, 0);
lean_inc(v_a_3962_);
lean_dec_ref_known(v___x_3961_, 1);
lean_inc(v___x_3873_);
v___f_3963_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__4___boxed), 2, 1);
lean_closure_set(v___f_3963_, 0, v___x_3873_);
v___x_3964_ = lean_array_get_size(v_a_3962_);
v___x_3965_ = lean_nat_dec_eq(v___x_3964_, v___x_3947_);
if (v___x_3965_ == 0)
{
lean_object* v___x_3966_; lean_object* v___f_3967_; lean_object* v___x_3968_; lean_object* v___x_3969_; uint8_t v___x_3970_; 
v___x_3966_ = lean_box(v___x_3965_);
v___f_3967_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Index_applicableRules___redArg___lam__2___boxed), 5, 3);
lean_closure_set(v___f_3967_, 0, v_inst_3527_);
lean_closure_set(v___f_3967_, 1, v___x_3873_);
lean_closure_set(v___f_3967_, 2, v___x_3966_);
v___x_3968_ = lean_unsigned_to_nat(1u);
v___x_3969_ = lean_nat_sub(v___x_3964_, v___x_3968_);
v___x_3970_ = lean_nat_dec_le(v___x_3947_, v___x_3969_);
if (v___x_3970_ == 0)
{
lean_inc(v___x_3969_);
v___y_3857_ = v_a_3870_;
v___y_3858_ = v_a_3962_;
v___y_3859_ = v___f_3963_;
v___y_3860_ = v___x_3969_;
v___y_3861_ = v___x_3954_;
v___y_3862_ = v___x_3925_;
v___y_3863_ = v___f_3967_;
v___y_3864_ = v___x_3964_;
v___y_3865_ = v___x_3969_;
goto v___jp_3856_;
}
else
{
v___y_3857_ = v_a_3870_;
v___y_3858_ = v_a_3962_;
v___y_3859_ = v___f_3963_;
v___y_3860_ = v___x_3969_;
v___y_3861_ = v___x_3954_;
v___y_3862_ = v___x_3925_;
v___y_3863_ = v___f_3967_;
v___y_3864_ = v___x_3964_;
v___y_3865_ = v___x_3947_;
goto v___jp_3856_;
}
}
else
{
lean_dec(v___x_3873_);
lean_dec_ref(v_inst_3527_);
v___y_3824_ = v_a_3870_;
v___y_3825_ = v___f_3963_;
v___y_3826_ = v___x_3954_;
v___y_3827_ = v___x_3925_;
v___y_3828_ = v_a_3962_;
goto v___jp_3823_;
}
}
else
{
lean_dec(v___x_3873_);
lean_dec_ref(v_inst_3527_);
if (lean_obj_tag(v___x_3961_) == 0)
{
lean_object* v_a_3971_; 
v_a_3971_ = lean_ctor_get(v___x_3961_, 0);
lean_inc(v_a_3971_);
lean_dec_ref_known(v___x_3961_, 1);
v___y_3814_ = v_a_3870_;
v___y_3815_ = v___x_3925_;
v_a_3816_ = v_a_3971_;
goto v___jp_3813_;
}
else
{
lean_object* v_a_3972_; 
v_a_3972_ = lean_ctor_get(v___x_3961_, 0);
lean_inc(v_a_3972_);
lean_dec_ref_known(v___x_3961_, 1);
v___y_3819_ = v_a_3870_;
v___y_3820_ = v___x_3925_;
v_a_3821_ = v_a_3972_;
goto v___jp_3818_;
}
}
}
else
{
lean_object* v_a_3973_; 
lean_dec(v_a_3946_);
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3973_ = lean_ctor_get(v___x_3951_, 0);
lean_inc(v_a_3973_);
lean_dec_ref_known(v___x_3951_, 1);
v___y_3819_ = v_a_3870_;
v___y_3820_ = v___x_3925_;
v_a_3821_ = v_a_3973_;
goto v___jp_3818_;
}
}
else
{
lean_object* v_a_3974_; 
lean_dec_ref(v___x_3943_);
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3974_ = lean_ctor_get(v___x_3945_, 0);
lean_inc(v_a_3974_);
lean_dec_ref_known(v___x_3945_, 1);
v___y_3819_ = v_a_3870_;
v___y_3820_ = v___x_3925_;
v_a_3821_ = v_a_3974_;
goto v___jp_3818_;
}
}
else
{
lean_object* v_a_3975_; 
lean_dec(v___x_3873_);
lean_dec_ref(v___f_3724_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3975_ = lean_ctor_get(v___x_3927_, 0);
lean_inc(v_a_3975_);
lean_dec_ref_known(v___x_3927_, 1);
v___y_3819_ = v_a_3870_;
v___y_3820_ = v___x_3925_;
v_a_3821_ = v_a_3975_;
goto v___jp_3818_;
}
}
}
else
{
lean_object* v_a_3976_; lean_object* v___x_3978_; uint8_t v_isShared_3979_; uint8_t v_isSharedCheck_3983_; 
lean_dec_ref(v___f_3727_);
lean_dec_ref(v___f_3724_);
lean_dec_ref(v___f_3723_);
lean_dec_ref(v___f_3721_);
lean_dec_ref_known(v___x_3586_, 3);
lean_dec_ref(v___x_3577_);
lean_dec_ref(v_additionalRules_3531_);
lean_dec(v_goal_3529_);
lean_dec_ref(v_ri_3528_);
lean_dec_ref(v_inst_3527_);
v_a_3976_ = lean_ctor_get(v___x_3869_, 0);
v_isSharedCheck_3983_ = !lean_is_exclusive(v___x_3869_);
if (v_isSharedCheck_3983_ == 0)
{
v___x_3978_ = v___x_3869_;
v_isShared_3979_ = v_isSharedCheck_3983_;
goto v_resetjp_3977_;
}
else
{
lean_inc(v_a_3976_);
lean_dec(v___x_3869_);
v___x_3978_ = lean_box(0);
v_isShared_3979_ = v_isSharedCheck_3983_;
goto v_resetjp_3977_;
}
v_resetjp_3977_:
{
lean_object* v___x_3981_; 
if (v_isShared_3979_ == 0)
{
v___x_3981_ = v___x_3978_;
goto v_reusejp_3980_;
}
else
{
lean_object* v_reuseFailAlloc_3982_; 
v_reuseFailAlloc_3982_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3982_, 0, v_a_3976_);
v___x_3981_ = v_reuseFailAlloc_3982_;
goto v_reusejp_3980_;
}
v_reusejp_3980_:
{
return v___x_3981_;
}
}
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___redArg___boxed(lean_object* v_inst_4118_, lean_object* v_ri_4119_, lean_object* v_goal_4120_, lean_object* v_patSubstMap_4121_, lean_object* v_additionalRules_4122_, lean_object* v_include_x3f_4123_, lean_object* v_a_4124_, lean_object* v_a_4125_, lean_object* v_a_4126_, lean_object* v_a_4127_, lean_object* v_a_4128_){
_start:
{
lean_object* v_res_4129_; 
v_res_4129_ = lp_aesop_Aesop_Index_applicableRules___redArg(v_inst_4118_, v_ri_4119_, v_goal_4120_, v_patSubstMap_4121_, v_additionalRules_4122_, v_include_x3f_4123_, v_a_4124_, v_a_4125_, v_a_4126_, v_a_4127_);
lean_dec(v_a_4127_);
lean_dec_ref(v_a_4126_);
lean_dec(v_a_4125_);
lean_dec_ref(v_a_4124_);
lean_dec_ref(v_patSubstMap_4121_);
return v_res_4129_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules(lean_object* v_00_u03b1_4130_, lean_object* v_inst_4131_, lean_object* v_ri_4132_, lean_object* v_goal_4133_, lean_object* v_patSubstMap_4134_, lean_object* v_additionalRules_4135_, lean_object* v_include_x3f_4136_, lean_object* v_a_4137_, lean_object* v_a_4138_, lean_object* v_a_4139_, lean_object* v_a_4140_){
_start:
{
lean_object* v___x_4142_; 
v___x_4142_ = lp_aesop_Aesop_Index_applicableRules___redArg(v_inst_4131_, v_ri_4132_, v_goal_4133_, v_patSubstMap_4134_, v_additionalRules_4135_, v_include_x3f_4136_, v_a_4137_, v_a_4138_, v_a_4139_, v_a_4140_);
return v___x_4142_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Index_applicableRules___boxed(lean_object* v_00_u03b1_4143_, lean_object* v_inst_4144_, lean_object* v_ri_4145_, lean_object* v_goal_4146_, lean_object* v_patSubstMap_4147_, lean_object* v_additionalRules_4148_, lean_object* v_include_x3f_4149_, lean_object* v_a_4150_, lean_object* v_a_4151_, lean_object* v_a_4152_, lean_object* v_a_4153_, lean_object* v_a_4154_){
_start:
{
lean_object* v_res_4155_; 
v_res_4155_ = lp_aesop_Aesop_Index_applicableRules(v_00_u03b1_4143_, v_inst_4144_, v_ri_4145_, v_goal_4146_, v_patSubstMap_4147_, v_additionalRules_4148_, v_include_x3f_4149_, v_a_4150_, v_a_4151_, v_a_4152_, v_a_4153_);
lean_dec(v_a_4153_);
lean_dec_ref(v_a_4152_);
lean_dec(v_a_4151_);
lean_dec_ref(v_a_4150_);
lean_dec_ref(v_patSubstMap_4147_);
return v_res_4155_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Index_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Index_RulePattern(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_PersistentHashSet(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Index(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index_RulePattern(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_PersistentHashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Index(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Index_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Index_DiscrTreeConfig(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Index_RulePattern(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Rule_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_PersistentHashSet(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Index(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Index_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Index_DiscrTreeConfig(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Index_RulePattern(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_InstantiateMVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_PersistentHashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Index(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Index(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Index(builtin);
}
#ifdef __cplusplus
}
#endif
