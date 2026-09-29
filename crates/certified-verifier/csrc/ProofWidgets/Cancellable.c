// Lean compiler output
// Module: ProofWidgets.Cancellable
// Imports: public import Init public meta import Init public meta import Lean.Server.Rpc.RequestHandling public import ProofWidgets.Compat public import ProofWidgets.Util
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
lean_object* lean_io_cancel(lean_object*);
lean_object* l_Lean_Json_getTag_x3f(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Json_parseCtorFields(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_invalidParams(lean_object*);
uint8_t lean_io_get_task_state(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_task_get_own(lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* l_Lean_Json_getNat_x3f(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Server_ServerTask_mapCheap___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_instDecidableEqNat___boxed(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addAndCompile(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Server_registerRpcProcedure(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerBuiltinAttribute(lean_object*);
lean_object* l_UInt64_ofNat___boxed(lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_runningRequests;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0;
static const lean_closure_object lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_UInt64_ofNat___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "cancelRequest"};
static const lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(242, 193, 45, 57, 217, 154, 107, 26)}};
static const lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_cancelRequest___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__3_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped;
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_running_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_running_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_done_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_done_elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorIdx(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_running_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_running_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_done_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_done_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "no inductive tag found"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "running"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "done"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "no inductive constructor matched"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__4_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "result"};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__7_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value),LEAN_SCALAR_PTR_LITERAL(180, 131, 177, 30, 113, 24, 63, 83)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__7_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__7_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_array_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__8_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__7_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__8_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__8_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__9_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__8_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__9_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__9_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__10_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__10_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__10_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14__value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34____boxed(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34__value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___lam__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse = (const lean_object*)&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Request '"};
static const lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__0_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "' has already finished, or the ID is invalid."};
static const lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "checkRequest"};
static const lean_object* lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(104, 84, 75, 161, 240, 21, 175, 224)}};
static const lean_object* lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_checkRequest___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__2_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped;
static const lean_string_object lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "_cancellable"};
static const lean_object* lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__0_value),LEAN_SCALAR_PTR_LITERAL(206, 168, 255, 116, 103, 27, 7, 58)}};
static const lean_object* lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__1_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_cancellableSuffix = (const lean_object*)&lp_proofwidgets_ProofWidgets_cancellableSuffix___closed__1_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__1_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "unsolvedGoals"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__2 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__2_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "synthPlaceholder"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__3 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__3_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "lean"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__4 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__4_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "inductionWithNoAlts"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__5 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__5_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "_namedError"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__6 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__6_value;
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__7 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__7_value;
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 24, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 1, 1, 0),LEAN_SCALAR_PTR_LITERAL(1, 1, 0, 1, 1, 1, 2, 1),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 372, .m_capacity = 372, .m_length = 371, .m_data = "This attribute is deprecated since ProofWidgets v0.0.93.\n\nTo migrate, replace `@[server_rpc_method_cancellable]` with `@[server_rpc_method]`,\nand replace calls to `IO.checkCanceled` with `RequestM.checkCancelled`.\n\nIf the RPC method spawns `CoreM` computations,\nit is also encouraged to pass `RequestContext.cancelTk.cancelledByCancelRequest`\n        into `Core.Context`."};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "mkCancellable"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0;
static lean_once_cell_t lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1;
static lean_once_cell_t lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2;
static lean_once_cell_t lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3;
static lean_once_cell_t lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Attribute `["};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "]` cannot be erased"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static lean_once_cell_t lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 75, 30, 82, 27, 168, 252, 212)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Cancellable"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 67, 199, 11, 16, 125, 85, 16)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_closure_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed, .m_arity = 9, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value)} };
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(4, 148, 253, 139, 180, 121, 43, 80)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 51, 82, 253, 94, 154, 27, 13)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__8_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(91, 71, 160, 233, 27, 32, 138, 159)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(190, 4, 160, 67, 177, 32, 157, 66)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__12_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(116, 71, 210, 96, 172, 90, 15, 243)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__12_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__12_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__13_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__12_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(72, 214, 193, 8, 74, 139, 201, 15)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__13_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__13_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__14_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__13_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)(((size_t)(680169540) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(62, 52, 204, 247, 134, 245, 172, 4)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__14_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__14_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__15_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__15_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__15_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__16_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__14_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__15_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(105, 188, 204, 63, 227, 0, 127, 93)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__16_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__16_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__17_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__17_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__17_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__18_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__16_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__17_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(129, 41, 30, 254, 74, 123, 3, 78)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__18_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__18_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__19_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__18_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)(((size_t)(2) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(28, 45, 176, 0, 116, 218, 117, 7)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__19_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__19_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__20_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "server_rpc_method_cancellable"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__20_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__20_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__21_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__20_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(28, 119, 190, 100, 2, 156, 158, 165)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__21_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__21_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_closure_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__22_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__21_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value)} };
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__22_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__22_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__23_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 236, .m_capacity = 236, .m_length = 235, .m_data = "Like `server_rpc_method`, but requests for this method can be cancelled. The method should check for that using `IO.checkCanceled`. Cancellable methods are invoked differently from JavaScript: see `callCancellable` in `cancellable.ts`."};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__23_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__23_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__24_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__19_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__21_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__23_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__24_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__24_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__25_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__24_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__22_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__25_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_ = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__25_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = lean_unsigned_to_nat(16u);
v___x_3_ = lean_mk_array(v___x_2_, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__0_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_);
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
lean_ctor_set(v___x_6_, 1, v___x_4_);
return v___x_6_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_7_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__1_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_);
v___x_8_ = lean_unsigned_to_nat(0u);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__2_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_);
v___x_12_ = lean_st_mk_ref(v___x_11_);
v___x_13_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2____boxed(lean_object* v_a_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_();
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__0(lean_object* v_inst_16_, lean_object* v_x_17_){
_start:
{
if (lean_obj_tag(v_x_17_) == 0)
{
lean_object* v_a_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_25_; 
lean_dec_ref(v_inst_16_);
v_a_18_ = lean_ctor_get(v_x_17_, 0);
v_isSharedCheck_25_ = !lean_is_exclusive(v_x_17_);
if (v_isSharedCheck_25_ == 0)
{
v___x_20_ = v_x_17_;
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_a_18_);
lean_dec(v_x_17_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_25_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_23_; 
if (v_isShared_21_ == 0)
{
v___x_23_ = v___x_20_;
goto v_reusejp_22_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_a_18_);
v___x_23_ = v_reuseFailAlloc_24_;
goto v_reusejp_22_;
}
v_reusejp_22_:
{
return v___x_23_;
}
}
}
else
{
lean_object* v_rpcEncode_26_; lean_object* v_a_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_35_; 
v_rpcEncode_26_ = lean_ctor_get(v_inst_16_, 0);
lean_inc_ref(v_rpcEncode_26_);
lean_dec_ref(v_inst_16_);
v_a_27_ = lean_ctor_get(v_x_17_, 0);
v_isSharedCheck_35_ = !lean_is_exclusive(v_x_17_);
if (v_isSharedCheck_35_ == 0)
{
v___x_29_ = v_x_17_;
v_isShared_30_ = v_isSharedCheck_35_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_a_27_);
lean_dec(v_x_17_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_35_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v___x_31_; lean_object* v___x_33_; 
v___x_31_ = lean_apply_1(v_rpcEncode_26_, v_a_27_);
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 0, v___x_31_);
v___x_33_ = v___x_29_;
goto v_reusejp_32_;
}
else
{
lean_object* v_reuseFailAlloc_34_; 
v_reuseFailAlloc_34_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_34_, 0, v___x_31_);
v___x_33_ = v_reuseFailAlloc_34_;
goto v_reusejp_32_;
}
v_reusejp_32_:
{
return v___x_33_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1(lean_object* v_t_36_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lean_io_cancel(v_t_36_);
v___x_39_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1___boxed(lean_object* v_t_40_, lean_object* v___y_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1(v_t_40_);
lean_dec_ref(v_t_40_);
return v_res_42_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0(void){
_start:
{
lean_object* v___x_43_; lean_object* v___f_44_; 
v___x_43_ = lean_alloc_closure((void*)(l_instDecidableEqNat___boxed), 2, 0);
v___f_44_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_44_, 0, v___x_43_);
return v___f_44_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2(lean_object* v___f_45_, lean_object* v___f_46_, lean_object* v_t_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v_fst_52_; lean_object* v_snd_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_69_; 
v___x_50_ = lp_proofwidgets_ProofWidgets_runningRequests;
v___x_51_ = lean_st_ref_take(v___x_50_);
v_fst_52_ = lean_ctor_get(v___x_51_, 0);
v_snd_53_ = lean_ctor_get(v___x_51_, 1);
v_isSharedCheck_69_ = !lean_is_exclusive(v___x_51_);
if (v_isSharedCheck_69_ == 0)
{
v___x_55_ = v___x_51_;
v_isShared_56_ = v_isSharedCheck_69_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_snd_53_);
lean_inc(v_fst_52_);
lean_dec(v___x_51_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_69_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___f_57_; lean_object* v_t_x27_58_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___f_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_65_; 
lean_inc_ref(v_t_47_);
v___f_57_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_57_, 0, v_t_47_);
v_t_x27_58_ = l_Lean_Server_ServerTask_mapCheap___redArg(v___f_45_, v_t_47_);
v___x_59_ = lean_unsigned_to_nat(1u);
v___x_60_ = lean_nat_add(v_fst_52_, v___x_59_);
v___f_61_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0, &lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0_once, _init_lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___closed__0);
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v_t_x27_58_);
lean_ctor_set(v___x_62_, 1, v___f_57_);
lean_inc(v_fst_52_);
v___x_63_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___f_61_, v___f_46_, v_snd_53_, v_fst_52_, v___x_62_);
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 1, v___x_63_);
lean_ctor_set(v___x_55_, 0, v___x_60_);
v___x_65_ = v___x_55_;
goto v_reusejp_64_;
}
else
{
lean_object* v_reuseFailAlloc_68_; 
v_reuseFailAlloc_68_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_68_, 0, v___x_60_);
lean_ctor_set(v_reuseFailAlloc_68_, 1, v___x_63_);
v___x_65_ = v_reuseFailAlloc_68_;
goto v_reusejp_64_;
}
v_reusejp_64_:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = lean_st_ref_set(v___x_50_, v___x_65_);
v___x_67_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_67_, 0, v_fst_52_);
return v___x_67_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___boxed(lean_object* v___f_70_, lean_object* v___f_71_, lean_object* v_t_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2(v___f_70_, v___f_71_, v_t_72_, v___y_73_);
lean_dec_ref(v___y_73_);
return v_res_75_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0(void){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = l_instMonadEIO(lean_box(0));
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg(lean_object* v_inst_78_, lean_object* v_handler_79_, lean_object* v_a_80_, lean_object* v_a_81_){
_start:
{
lean_object* v___f_83_; lean_object* v___x_84_; lean_object* v___f_85_; lean_object* v___f_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; 
v___f_83_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__0), 2, 1);
lean_closure_set(v___f_83_, 0, v_inst_78_);
v___x_84_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0, &lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0_once, _init_lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__0);
v___f_85_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_mkCancellable___redArg___closed__1));
v___f_86_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkCancellable___redArg___lam__2___boxed), 5, 2);
lean_closure_set(v___f_86_, 0, v___f_83_);
lean_closure_set(v___f_86_, 1, v___f_85_);
v___x_87_ = lean_apply_1(v_handler_79_, v_a_80_);
v___x_88_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_88_, 0, lean_box(0));
lean_closure_set(v___x_88_, 1, lean_box(0));
lean_closure_set(v___x_88_, 2, v___x_84_);
lean_closure_set(v___x_88_, 3, lean_box(0));
lean_closure_set(v___x_88_, 4, lean_box(0));
lean_closure_set(v___x_88_, 5, v___x_87_);
lean_closure_set(v___x_88_, 6, v___f_86_);
v___x_89_ = l_Lean_Server_RequestM_asTask___redArg(v___x_88_, v_a_81_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___redArg___boxed(lean_object* v_inst_90_, lean_object* v_handler_91_, lean_object* v_a_92_, lean_object* v_a_93_, lean_object* v_a_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_proofwidgets_ProofWidgets_mkCancellable___redArg(v_inst_90_, v_handler_91_, v_a_92_, v_a_93_);
lean_dec_ref(v_a_93_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable(lean_object* v_00_u03b2_96_, lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_handler_99_, lean_object* v_a_100_, lean_object* v_a_101_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lp_proofwidgets_ProofWidgets_mkCancellable___redArg(v_inst_98_, v_handler_99_, v_a_100_, v_a_101_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkCancellable___boxed(lean_object* v_00_u03b2_104_, lean_object* v_00_u03b1_105_, lean_object* v_inst_106_, lean_object* v_handler_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_proofwidgets_ProofWidgets_mkCancellable(v_00_u03b2_104_, v_00_u03b1_105_, v_inst_106_, v_handler_107_, v_a_108_, v_a_109_);
lean_dec_ref(v_a_109_);
return v_res_111_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg(lean_object* v_a_112_, lean_object* v_x_113_){
_start:
{
if (lean_obj_tag(v_x_113_) == 0)
{
uint8_t v___x_114_; 
v___x_114_ = 0;
return v___x_114_;
}
else
{
lean_object* v_key_115_; lean_object* v_tail_116_; uint8_t v___x_117_; 
v_key_115_ = lean_ctor_get(v_x_113_, 0);
v_tail_116_ = lean_ctor_get(v_x_113_, 2);
v___x_117_ = lean_nat_dec_eq(v_key_115_, v_a_112_);
if (v___x_117_ == 0)
{
v_x_113_ = v_tail_116_;
goto _start;
}
else
{
return v___x_117_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg___boxed(lean_object* v_a_119_, lean_object* v_x_120_){
_start:
{
uint8_t v_res_121_; lean_object* v_r_122_; 
v_res_121_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg(v_a_119_, v_x_120_);
lean_dec(v_x_120_);
lean_dec(v_a_119_);
v_r_122_ = lean_box(v_res_121_);
return v_r_122_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(lean_object* v_a_123_, lean_object* v_x_124_){
_start:
{
if (lean_obj_tag(v_x_124_) == 0)
{
return v_x_124_;
}
else
{
lean_object* v_key_125_; lean_object* v_value_126_; lean_object* v_tail_127_; lean_object* v___x_129_; uint8_t v_isShared_130_; uint8_t v_isSharedCheck_136_; 
v_key_125_ = lean_ctor_get(v_x_124_, 0);
v_value_126_ = lean_ctor_get(v_x_124_, 1);
v_tail_127_ = lean_ctor_get(v_x_124_, 2);
v_isSharedCheck_136_ = !lean_is_exclusive(v_x_124_);
if (v_isSharedCheck_136_ == 0)
{
v___x_129_ = v_x_124_;
v_isShared_130_ = v_isSharedCheck_136_;
goto v_resetjp_128_;
}
else
{
lean_inc(v_tail_127_);
lean_inc(v_value_126_);
lean_inc(v_key_125_);
lean_dec(v_x_124_);
v___x_129_ = lean_box(0);
v_isShared_130_ = v_isSharedCheck_136_;
goto v_resetjp_128_;
}
v_resetjp_128_:
{
uint8_t v___x_131_; 
v___x_131_ = lean_nat_dec_eq(v_key_125_, v_a_123_);
if (v___x_131_ == 0)
{
lean_object* v___x_132_; lean_object* v___x_134_; 
v___x_132_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(v_a_123_, v_tail_127_);
if (v_isShared_130_ == 0)
{
lean_ctor_set(v___x_129_, 2, v___x_132_);
v___x_134_ = v___x_129_;
goto v_reusejp_133_;
}
else
{
lean_object* v_reuseFailAlloc_135_; 
v_reuseFailAlloc_135_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_135_, 0, v_key_125_);
lean_ctor_set(v_reuseFailAlloc_135_, 1, v_value_126_);
lean_ctor_set(v_reuseFailAlloc_135_, 2, v___x_132_);
v___x_134_ = v_reuseFailAlloc_135_;
goto v_reusejp_133_;
}
v_reusejp_133_:
{
return v___x_134_;
}
}
else
{
lean_del_object(v___x_129_);
lean_dec(v_value_126_);
lean_dec(v_key_125_);
return v_tail_127_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg___boxed(lean_object* v_a_137_, lean_object* v_x_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(v_a_137_, v_x_138_);
lean_dec(v_a_137_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(lean_object* v_m_140_, lean_object* v_a_141_){
_start:
{
lean_object* v_size_142_; lean_object* v_buckets_143_; lean_object* v___x_144_; uint64_t v___x_145_; uint64_t v___x_146_; uint64_t v___x_147_; uint64_t v_fold_148_; uint64_t v___x_149_; uint64_t v___x_150_; uint64_t v___x_151_; size_t v___x_152_; size_t v___x_153_; size_t v___x_154_; size_t v___x_155_; size_t v___x_156_; lean_object* v_bkt_157_; uint8_t v___x_158_; 
v_size_142_ = lean_ctor_get(v_m_140_, 0);
v_buckets_143_ = lean_ctor_get(v_m_140_, 1);
v___x_144_ = lean_array_get_size(v_buckets_143_);
v___x_145_ = lean_uint64_of_nat(v_a_141_);
v___x_146_ = 32ULL;
v___x_147_ = lean_uint64_shift_right(v___x_145_, v___x_146_);
v_fold_148_ = lean_uint64_xor(v___x_145_, v___x_147_);
v___x_149_ = 16ULL;
v___x_150_ = lean_uint64_shift_right(v_fold_148_, v___x_149_);
v___x_151_ = lean_uint64_xor(v_fold_148_, v___x_150_);
v___x_152_ = lean_uint64_to_usize(v___x_151_);
v___x_153_ = lean_usize_of_nat(v___x_144_);
v___x_154_ = ((size_t)1ULL);
v___x_155_ = lean_usize_sub(v___x_153_, v___x_154_);
v___x_156_ = lean_usize_land(v___x_152_, v___x_155_);
v_bkt_157_ = lean_array_uget_borrowed(v_buckets_143_, v___x_156_);
v___x_158_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg(v_a_141_, v_bkt_157_);
if (v___x_158_ == 0)
{
return v_m_140_;
}
else
{
lean_object* v___x_160_; uint8_t v_isShared_161_; uint8_t v_isSharedCheck_171_; 
lean_inc(v_bkt_157_);
lean_inc_ref(v_buckets_143_);
lean_inc(v_size_142_);
v_isSharedCheck_171_ = !lean_is_exclusive(v_m_140_);
if (v_isSharedCheck_171_ == 0)
{
lean_object* v_unused_172_; lean_object* v_unused_173_; 
v_unused_172_ = lean_ctor_get(v_m_140_, 1);
lean_dec(v_unused_172_);
v_unused_173_ = lean_ctor_get(v_m_140_, 0);
lean_dec(v_unused_173_);
v___x_160_ = v_m_140_;
v_isShared_161_ = v_isSharedCheck_171_;
goto v_resetjp_159_;
}
else
{
lean_dec(v_m_140_);
v___x_160_ = lean_box(0);
v_isShared_161_ = v_isSharedCheck_171_;
goto v_resetjp_159_;
}
v_resetjp_159_:
{
lean_object* v___x_162_; lean_object* v_buckets_x27_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_169_; 
v___x_162_ = lean_box(0);
v_buckets_x27_163_ = lean_array_uset(v_buckets_143_, v___x_156_, v___x_162_);
v___x_164_ = lean_unsigned_to_nat(1u);
v___x_165_ = lean_nat_sub(v_size_142_, v___x_164_);
lean_dec(v_size_142_);
v___x_166_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(v_a_141_, v_bkt_157_);
v___x_167_ = lean_array_uset(v_buckets_x27_163_, v___x_156_, v___x_166_);
if (v_isShared_161_ == 0)
{
lean_ctor_set(v___x_160_, 1, v___x_167_);
lean_ctor_set(v___x_160_, 0, v___x_165_);
v___x_169_ = v___x_160_;
goto v_reusejp_168_;
}
else
{
lean_object* v_reuseFailAlloc_170_; 
v_reuseFailAlloc_170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_170_, 0, v___x_165_);
lean_ctor_set(v_reuseFailAlloc_170_, 1, v___x_167_);
v___x_169_ = v_reuseFailAlloc_170_;
goto v_reusejp_168_;
}
v_reusejp_168_:
{
return v___x_169_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg___boxed(lean_object* v_m_174_, lean_object* v_a_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(v_m_174_, v_a_175_);
lean_dec(v_a_175_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg(lean_object* v_a_177_, lean_object* v_x_178_){
_start:
{
if (lean_obj_tag(v_x_178_) == 0)
{
lean_object* v___x_179_; 
v___x_179_ = lean_box(0);
return v___x_179_;
}
else
{
lean_object* v_key_180_; lean_object* v_value_181_; lean_object* v_tail_182_; uint8_t v___x_183_; 
v_key_180_ = lean_ctor_get(v_x_178_, 0);
v_value_181_ = lean_ctor_get(v_x_178_, 1);
v_tail_182_ = lean_ctor_get(v_x_178_, 2);
v___x_183_ = lean_nat_dec_eq(v_key_180_, v_a_177_);
if (v___x_183_ == 0)
{
v_x_178_ = v_tail_182_;
goto _start;
}
else
{
lean_object* v___x_185_; 
lean_inc(v_value_181_);
v___x_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_185_, 0, v_value_181_);
return v___x_185_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg___boxed(lean_object* v_a_186_, lean_object* v_x_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg(v_a_186_, v_x_187_);
lean_dec(v_x_187_);
lean_dec(v_a_186_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(lean_object* v_m_189_, lean_object* v_a_190_){
_start:
{
lean_object* v_buckets_191_; lean_object* v___x_192_; uint64_t v___x_193_; uint64_t v___x_194_; uint64_t v___x_195_; uint64_t v_fold_196_; uint64_t v___x_197_; uint64_t v___x_198_; uint64_t v___x_199_; size_t v___x_200_; size_t v___x_201_; size_t v___x_202_; size_t v___x_203_; size_t v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v_buckets_191_ = lean_ctor_get(v_m_189_, 1);
v___x_192_ = lean_array_get_size(v_buckets_191_);
v___x_193_ = lean_uint64_of_nat(v_a_190_);
v___x_194_ = 32ULL;
v___x_195_ = lean_uint64_shift_right(v___x_193_, v___x_194_);
v_fold_196_ = lean_uint64_xor(v___x_193_, v___x_195_);
v___x_197_ = 16ULL;
v___x_198_ = lean_uint64_shift_right(v_fold_196_, v___x_197_);
v___x_199_ = lean_uint64_xor(v_fold_196_, v___x_198_);
v___x_200_ = lean_uint64_to_usize(v___x_199_);
v___x_201_ = lean_usize_of_nat(v___x_192_);
v___x_202_ = ((size_t)1ULL);
v___x_203_ = lean_usize_sub(v___x_201_, v___x_202_);
v___x_204_ = lean_usize_land(v___x_200_, v___x_203_);
v___x_205_ = lean_array_uget_borrowed(v_buckets_191_, v___x_204_);
v___x_206_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg(v_a_190_, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg___boxed(lean_object* v_m_207_, lean_object* v_a_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(v_m_207_, v_a_208_);
lean_dec(v_a_208_);
lean_dec_ref(v_m_207_);
return v_res_209_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___lam__0(lean_object* v___x_210_, lean_object* v_rid_211_, lean_object* v___y_212_){
_start:
{
lean_object* v___x_214_; lean_object* v_fst_215_; lean_object* v_snd_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_248_; 
v___x_214_ = lean_st_ref_take(v___x_210_);
v_fst_215_ = lean_ctor_get(v___x_214_, 0);
v_snd_216_ = lean_ctor_get(v___x_214_, 1);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_214_);
if (v_isSharedCheck_248_ == 0)
{
v___x_218_ = v___x_214_;
v_isShared_219_ = v_isSharedCheck_248_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_snd_216_);
lean_inc(v_fst_215_);
lean_dec(v___x_214_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_248_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_223_; 
v___x_220_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(v_snd_216_, v_rid_211_);
v___x_221_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(v_snd_216_, v_rid_211_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 1, v___x_221_);
v___x_223_ = v___x_218_;
goto v_reusejp_222_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_fst_215_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v___x_221_);
v___x_223_ = v_reuseFailAlloc_247_;
goto v_reusejp_222_;
}
v_reusejp_222_:
{
lean_object* v___x_224_; 
v___x_224_ = lean_st_ref_set(v___x_210_, v___x_223_);
if (lean_obj_tag(v___x_220_) == 1)
{
lean_object* v_val_225_; lean_object* v_cancel_226_; lean_object* v___x_227_; 
v_val_225_ = lean_ctor_get(v___x_220_, 0);
lean_inc(v_val_225_);
lean_dec_ref_known(v___x_220_, 1);
v_cancel_226_ = lean_ctor_get(v_val_225_, 1);
lean_inc_ref(v_cancel_226_);
lean_dec(v_val_225_);
v___x_227_ = lean_apply_1(v_cancel_226_, lean_box(0));
if (lean_obj_tag(v___x_227_) == 0)
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_235_; 
v_a_228_ = lean_ctor_get(v___x_227_, 0);
v_isSharedCheck_235_ = !lean_is_exclusive(v___x_227_);
if (v_isSharedCheck_235_ == 0)
{
v___x_230_ = v___x_227_;
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_227_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_235_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_233_; 
if (v_isShared_231_ == 0)
{
v___x_233_ = v___x_230_;
goto v_reusejp_232_;
}
else
{
lean_object* v_reuseFailAlloc_234_; 
v_reuseFailAlloc_234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_234_, 0, v_a_228_);
v___x_233_ = v_reuseFailAlloc_234_;
goto v_reusejp_232_;
}
v_reusejp_232_:
{
return v___x_233_;
}
}
}
else
{
lean_object* v_a_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_244_; 
v_a_236_ = lean_ctor_get(v___x_227_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_227_);
if (v_isSharedCheck_244_ == 0)
{
v___x_238_ = v___x_227_;
v_isShared_239_ = v_isSharedCheck_244_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_a_236_);
lean_dec(v___x_227_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_244_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_240_; lean_object* v___x_242_; 
v___x_240_ = l_Lean_Server_RequestError_ofIoError(v_a_236_);
if (v_isShared_239_ == 0)
{
lean_ctor_set(v___x_238_, 0, v___x_240_);
v___x_242_ = v___x_238_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v___x_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
else
{
lean_object* v___x_245_; lean_object* v___x_246_; 
lean_dec(v___x_220_);
v___x_245_ = lean_box(0);
v___x_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_246_, 0, v___x_245_);
return v___x_246_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___lam__0___boxed(lean_object* v___x_249_, lean_object* v_rid_250_, lean_object* v___y_251_, lean_object* v___y_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_proofwidgets_ProofWidgets_cancelRequest___lam__0(v___x_249_, v_rid_250_, v___y_251_);
lean_dec_ref(v___y_251_);
lean_dec(v_rid_250_);
lean_dec(v___x_249_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest(lean_object* v_rid_254_, lean_object* v_a_255_){
_start:
{
lean_object* v___x_257_; lean_object* v___f_258_; lean_object* v___x_259_; 
v___x_257_ = lp_proofwidgets_ProofWidgets_runningRequests;
v___f_258_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_cancelRequest___lam__0___boxed), 4, 2);
lean_closure_set(v___f_258_, 0, v___x_257_);
lean_closure_set(v___f_258_, 1, v_rid_254_);
v___x_259_ = l_Lean_Server_RequestM_asTask___redArg(v___f_258_, v_a_255_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_cancelRequest___boxed(lean_object* v_rid_260_, lean_object* v_a_261_, lean_object* v_a_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_proofwidgets_ProofWidgets_cancelRequest(v_rid_260_, v_a_261_);
lean_dec_ref(v_a_261_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0(lean_object* v_00_u03b2_264_, lean_object* v_m_265_, lean_object* v_a_266_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(v_m_265_, v_a_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___boxed(lean_object* v_00_u03b2_268_, lean_object* v_m_269_, lean_object* v_a_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0(v_00_u03b2_268_, v_m_269_, v_a_270_);
lean_dec(v_a_270_);
lean_dec_ref(v_m_269_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1(lean_object* v_00_u03b2_272_, lean_object* v_m_273_, lean_object* v_a_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(v_m_273_, v_a_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___boxed(lean_object* v_00_u03b2_276_, lean_object* v_m_277_, lean_object* v_a_278_){
_start:
{
lean_object* v_res_279_; 
v_res_279_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1(v_00_u03b2_276_, v_m_277_, v_a_278_);
lean_dec(v_a_278_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0(lean_object* v_00_u03b2_280_, lean_object* v_a_281_, lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___redArg(v_a_281_, v_x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0___boxed(lean_object* v_00_u03b2_284_, lean_object* v_a_285_, lean_object* v_x_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0_spec__0(v_00_u03b2_284_, v_a_285_, v_x_286_);
lean_dec(v_x_286_);
lean_dec(v_a_285_);
return v_res_287_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2(lean_object* v_00_u03b2_288_, lean_object* v_a_289_, lean_object* v_x_290_){
_start:
{
uint8_t v___x_291_; 
v___x_291_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___redArg(v_a_289_, v_x_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2___boxed(lean_object* v_00_u03b2_292_, lean_object* v_a_293_, lean_object* v_x_294_){
_start:
{
uint8_t v_res_295_; lean_object* v_r_296_; 
v_res_295_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__2(v_00_u03b2_292_, v_a_293_, v_x_294_);
lean_dec(v_x_294_);
lean_dec(v_a_293_);
v_r_296_ = lean_box(v_res_295_);
return v_r_296_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3(lean_object* v_00_u03b2_297_, lean_object* v_a_298_, lean_object* v_x_299_){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___redArg(v_a_298_, v_x_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3___boxed(lean_object* v_00_u03b2_301_, lean_object* v_a_302_, lean_object* v_x_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_proofwidgets_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1_spec__3(v_00_u03b2_301_, v_a_302_, v_x_303_);
lean_dec(v_a_302_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_305_, lean_object* v_x_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_307_, 0, v_x_306_);
lean_ctor_set(v___x_307_, 1, v_expireTime_305_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2(lean_object* v_val_308_, lean_object* v___f_309_, lean_object* v_x_310_, lean_object* v___y_311_){
_start:
{
if (lean_obj_tag(v_x_310_) == 0)
{
lean_object* v_a_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_320_; 
lean_dec_ref(v___f_309_);
v_a_313_ = lean_ctor_get(v_x_310_, 0);
v_isSharedCheck_320_ = !lean_is_exclusive(v_x_310_);
if (v_isSharedCheck_320_ == 0)
{
v___x_315_ = v_x_310_;
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_a_313_);
lean_dec(v_x_310_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_320_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
lean_object* v___x_318_; 
if (v_isShared_316_ == 0)
{
lean_ctor_set_tag(v___x_315_, 1);
v___x_318_ = v___x_315_;
goto v_reusejp_317_;
}
else
{
lean_object* v_reuseFailAlloc_319_; 
v_reuseFailAlloc_319_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_319_, 0, v_a_313_);
v___x_318_ = v_reuseFailAlloc_319_;
goto v_reusejp_317_;
}
v_reusejp_317_:
{
return v___x_318_;
}
}
}
else
{
lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_343_; 
v_isSharedCheck_343_ = !lean_is_exclusive(v_x_310_);
if (v_isSharedCheck_343_ == 0)
{
lean_object* v_unused_344_; 
v_unused_344_ = lean_ctor_get(v_x_310_, 0);
lean_dec(v_unused_344_);
v___x_322_ = v_x_310_;
v_isShared_323_ = v_isSharedCheck_343_;
goto v_resetjp_321_;
}
else
{
lean_dec(v_x_310_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_343_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___x_324_; lean_object* v_objects_325_; lean_object* v_expireTime_326_; lean_object* v___x_328_; uint8_t v_isShared_329_; uint8_t v_isSharedCheck_342_; 
v___x_324_ = lean_st_ref_take(v_val_308_);
v_objects_325_ = lean_ctor_get(v___x_324_, 0);
v_expireTime_326_ = lean_ctor_get(v___x_324_, 1);
v_isSharedCheck_342_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_342_ == 0)
{
v___x_328_ = v___x_324_;
v_isShared_329_ = v_isSharedCheck_342_;
goto v_resetjp_327_;
}
else
{
lean_inc(v_expireTime_326_);
lean_inc(v_objects_325_);
lean_dec(v___x_324_);
v___x_328_ = lean_box(0);
v_isShared_329_ = v_isSharedCheck_342_;
goto v_resetjp_327_;
}
v_resetjp_327_:
{
lean_object* v___f_330_; lean_object* v___x_331_; lean_object* v___x_333_; 
v___f_330_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_330_, 0, v_expireTime_326_);
v___x_331_ = lean_box(0);
if (v_isShared_329_ == 0)
{
lean_ctor_set(v___x_328_, 1, v_objects_325_);
lean_ctor_set(v___x_328_, 0, v___x_331_);
v___x_333_ = v___x_328_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_341_; 
v_reuseFailAlloc_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_341_, 0, v___x_331_);
lean_ctor_set(v_reuseFailAlloc_341_, 1, v_objects_325_);
v___x_333_ = v_reuseFailAlloc_341_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
lean_object* v___x_334_; lean_object* v_fst_335_; lean_object* v_snd_336_; lean_object* v___x_337_; lean_object* v___x_339_; 
v___x_334_ = l_Prod_map___redArg(v___f_309_, v___f_330_, v___x_333_);
v_fst_335_ = lean_ctor_get(v___x_334_, 0);
lean_inc(v_fst_335_);
v_snd_336_ = lean_ctor_get(v___x_334_, 1);
lean_inc(v_snd_336_);
lean_dec_ref(v___x_334_);
v___x_337_ = lean_st_ref_set(v_val_308_, v_snd_336_);
if (v_isShared_323_ == 0)
{
lean_ctor_set_tag(v___x_322_, 0);
lean_ctor_set(v___x_322_, 0, v_fst_335_);
v___x_339_ = v___x_322_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_fst_335_);
v___x_339_ = v_reuseFailAlloc_340_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
return v___x_339_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_345_, lean_object* v___f_346_, lean_object* v_x_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2(v_val_345_, v___f_346_, v_x_347_, v___y_348_);
lean_dec_ref(v___y_348_);
lean_dec(v_val_345_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_351_, uint64_t v_k_352_){
_start:
{
if (lean_obj_tag(v_t_351_) == 0)
{
lean_object* v_k_353_; lean_object* v_v_354_; lean_object* v_l_355_; lean_object* v_r_356_; uint64_t v___x_357_; uint8_t v___x_358_; 
v_k_353_ = lean_ctor_get(v_t_351_, 1);
v_v_354_ = lean_ctor_get(v_t_351_, 2);
v_l_355_ = lean_ctor_get(v_t_351_, 3);
v_r_356_ = lean_ctor_get(v_t_351_, 4);
v___x_357_ = lean_unbox_uint64(v_k_353_);
v___x_358_ = lean_uint64_dec_lt(v_k_352_, v___x_357_);
if (v___x_358_ == 0)
{
uint64_t v___x_359_; uint8_t v___x_360_; 
v___x_359_ = lean_unbox_uint64(v_k_353_);
v___x_360_ = lean_uint64_dec_eq(v_k_352_, v___x_359_);
if (v___x_360_ == 0)
{
v_t_351_ = v_r_356_;
goto _start;
}
else
{
lean_object* v___x_362_; 
lean_inc(v_v_354_);
v___x_362_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_362_, 0, v_v_354_);
return v___x_362_;
}
}
else
{
v_t_351_ = v_l_355_;
goto _start;
}
}
else
{
lean_object* v___x_364_; 
v___x_364_ = lean_box(0);
return v___x_364_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_365_, lean_object* v_k_366_){
_start:
{
uint64_t v_k_boxed_367_; lean_object* v_res_368_; 
v_k_boxed_367_ = lean_unbox_uint64(v_k_366_);
lean_dec_ref(v_k_366_);
v_res_368_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(v_t_365_, v_k_boxed_367_);
lean_dec(v_t_365_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3(lean_object* v_method_376_, lean_object* v_handler_377_, lean_object* v___f_378_, uint64_t v_seshId_379_, lean_object* v_j_380_, lean_object* v___y_381_){
_start:
{
lean_object* v_rpcSessions_383_; lean_object* v___x_384_; 
v_rpcSessions_383_ = lean_ctor_get(v___y_381_, 0);
v___x_384_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_383_, v_seshId_379_);
if (lean_obj_tag(v___x_384_) == 1)
{
lean_object* v_val_385_; lean_object* v___x_386_; lean_object* v___x_387_; 
v_val_385_ = lean_ctor_get(v___x_384_, 0);
lean_inc(v_val_385_);
lean_dec_ref_known(v___x_384_, 1);
v___x_386_ = lean_st_ref_get(v_val_385_);
lean_dec(v___x_386_);
lean_inc(v_j_380_);
v___x_387_ = l_Lean_Json_getNat_x3f(v_j_380_);
if (lean_obj_tag(v___x_387_) == 0)
{
lean_object* v_a_388_; lean_object* v___x_390_; uint8_t v_isShared_391_; uint8_t v_isSharedCheck_408_; 
lean_dec(v_val_385_);
lean_dec_ref(v___f_378_);
lean_dec_ref(v_handler_377_);
v_a_388_ = lean_ctor_get(v___x_387_, 0);
v_isSharedCheck_408_ = !lean_is_exclusive(v___x_387_);
if (v_isSharedCheck_408_ == 0)
{
v___x_390_ = v___x_387_;
v_isShared_391_ = v_isSharedCheck_408_;
goto v_resetjp_389_;
}
else
{
lean_inc(v_a_388_);
lean_dec(v___x_387_);
v___x_390_ = lean_box(0);
v_isShared_391_ = v_isSharedCheck_408_;
goto v_resetjp_389_;
}
v_resetjp_389_:
{
uint8_t v___x_392_; lean_object* v___x_393_; uint8_t v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_406_; 
v___x_392_ = 3;
v___x_393_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_394_ = 1;
v___x_395_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_376_, v___x_394_);
v___x_396_ = lean_string_append(v___x_393_, v___x_395_);
lean_dec_ref(v___x_395_);
v___x_397_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_398_ = lean_string_append(v___x_396_, v___x_397_);
v___x_399_ = l_Lean_Json_compress(v_j_380_);
v___x_400_ = lean_string_append(v___x_398_, v___x_399_);
lean_dec_ref(v___x_399_);
v___x_401_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_402_ = lean_string_append(v___x_400_, v___x_401_);
v___x_403_ = lean_string_append(v___x_402_, v_a_388_);
lean_dec(v_a_388_);
v___x_404_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_404_, 0, v___x_403_);
lean_ctor_set_uint8(v___x_404_, sizeof(void*)*1, v___x_392_);
if (v_isShared_391_ == 0)
{
lean_ctor_set_tag(v___x_390_, 1);
lean_ctor_set(v___x_390_, 0, v___x_404_);
v___x_406_ = v___x_390_;
goto v_reusejp_405_;
}
else
{
lean_object* v_reuseFailAlloc_407_; 
v_reuseFailAlloc_407_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_407_, 0, v___x_404_);
v___x_406_ = v_reuseFailAlloc_407_;
goto v_reusejp_405_;
}
v_reusejp_405_:
{
return v___x_406_;
}
}
}
else
{
lean_object* v_a_409_; lean_object* v___x_410_; 
lean_dec(v_j_380_);
lean_dec(v_method_376_);
v_a_409_ = lean_ctor_get(v___x_387_, 0);
lean_inc(v_a_409_);
lean_dec_ref_known(v___x_387_, 1);
lean_inc_ref(v___y_381_);
v___x_410_ = lean_apply_3(v_handler_377_, v_a_409_, v___y_381_, lean_box(0));
if (lean_obj_tag(v___x_410_) == 0)
{
lean_object* v_a_411_; lean_object* v___f_412_; lean_object* v___x_413_; 
v_a_411_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_a_411_);
lean_dec_ref_known(v___x_410_, 1);
v___f_412_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_412_, 0, v_val_385_);
lean_closure_set(v___f_412_, 1, v___f_378_);
v___x_413_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_411_, v___f_412_, v___y_381_);
return v___x_413_;
}
else
{
lean_object* v_a_414_; lean_object* v___x_416_; uint8_t v_isShared_417_; uint8_t v_isSharedCheck_421_; 
lean_dec(v_val_385_);
lean_dec_ref(v___f_378_);
v_a_414_ = lean_ctor_get(v___x_410_, 0);
v_isSharedCheck_421_ = !lean_is_exclusive(v___x_410_);
if (v_isSharedCheck_421_ == 0)
{
v___x_416_ = v___x_410_;
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
else
{
lean_inc(v_a_414_);
lean_dec(v___x_410_);
v___x_416_ = lean_box(0);
v_isShared_417_ = v_isSharedCheck_421_;
goto v_resetjp_415_;
}
v_resetjp_415_:
{
lean_object* v___x_419_; 
if (v_isShared_417_ == 0)
{
v___x_419_ = v___x_416_;
goto v_reusejp_418_;
}
else
{
lean_object* v_reuseFailAlloc_420_; 
v_reuseFailAlloc_420_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_420_, 0, v_a_414_);
v___x_419_ = v_reuseFailAlloc_420_;
goto v_reusejp_418_;
}
v_reusejp_418_:
{
return v___x_419_;
}
}
}
}
}
else
{
lean_object* v___x_422_; lean_object* v___x_423_; 
lean_dec(v___x_384_);
lean_dec(v_j_380_);
lean_dec_ref(v___f_378_);
lean_dec_ref(v_handler_377_);
lean_dec(v_method_376_);
v___x_422_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
return v___x_423_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_424_, lean_object* v_handler_425_, lean_object* v___f_426_, lean_object* v_seshId_427_, lean_object* v_j_428_, lean_object* v___y_429_, lean_object* v___y_430_){
_start:
{
uint64_t v_seshId_boxed_431_; lean_object* v_res_432_; 
v_seshId_boxed_431_ = lean_unbox_uint64(v_seshId_427_);
lean_dec_ref(v_seshId_427_);
v_res_432_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3(v_method_424_, v_handler_425_, v___f_426_, v_seshId_boxed_431_, v_j_428_, v___y_429_);
lean_dec_ref(v___y_429_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0(lean_object* v___y_433_){
_start:
{
lean_inc(v___y_433_);
return v___y_433_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__0(v___y_434_);
lean_dec(v___y_434_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0(lean_object* v_method_437_, lean_object* v_handler_438_){
_start:
{
lean_object* v___f_439_; lean_object* v___f_440_; 
v___f_439_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___closed__0));
v___f_440_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_440_, 0, v_method_437_);
lean_closure_set(v___f_440_, 1, v_handler_438_);
lean_closure_set(v___f_440_, 2, v___f_439_);
return v___f_440_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4(void){
_start:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_447_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__3));
v___x_448_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__2));
v___x_449_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0(v___x_448_, v___x_447_);
return v___x_449_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped(void){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4, &lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4_once, _init_lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped___closed__4);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___redArg(lean_object* v_x_451_){
_start:
{
lean_inc_ref(v_x_451_);
return v_x_451_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___redArg___boxed(lean_object* v_x_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___redArg(v_x_452_);
lean_dec_ref(v_x_452_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1(lean_object* v_00_u03b1_454_, lean_object* v_x_455_, lean_object* v___y_456_){
_start:
{
lean_inc_ref(v_x_455_);
return v_x_455_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1___boxed(lean_object* v_00_u03b1_457_, lean_object* v_x_458_, lean_object* v___y_459_){
_start:
{
lean_object* v_res_460_; 
v_res_460_ = lp_proofwidgets_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__1(v_00_u03b1_457_, v_x_458_, v___y_459_);
lean_dec_ref(v___y_459_);
lean_dec_ref(v_x_458_);
return v_res_460_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_461_, lean_object* v_t_462_, uint64_t v_k_463_){
_start:
{
lean_object* v___x_464_; 
v___x_464_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(v_t_462_, v_k_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_465_, lean_object* v_t_466_, lean_object* v_k_467_){
_start:
{
uint64_t v_k_boxed_468_; lean_object* v_res_469_; 
v_k_boxed_468_ = lean_unbox_uint64(v_k_467_);
lean_dec_ref(v_k_467_);
v_res_469_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0(v_00_u03b4_465_, v_t_466_, v_k_boxed_468_);
lean_dec(v_t_466_);
return v_res_469_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorIdx(lean_object* v_x_470_){
_start:
{
if (lean_obj_tag(v_x_470_) == 0)
{
lean_object* v___x_471_; 
v___x_471_ = lean_unsigned_to_nat(0u);
return v___x_471_;
}
else
{
lean_object* v___x_472_; 
v___x_472_ = lean_unsigned_to_nat(1u);
return v___x_472_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorIdx___boxed(lean_object* v_x_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorIdx(v_x_473_);
lean_dec(v_x_473_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(lean_object* v_t_475_, lean_object* v_k_476_){
_start:
{
if (lean_obj_tag(v_t_475_) == 0)
{
return v_k_476_;
}
else
{
lean_object* v_result_477_; lean_object* v___x_478_; 
v_result_477_ = lean_ctor_get(v_t_475_, 0);
lean_inc_ref(v_result_477_);
lean_dec_ref_known(v_t_475_, 1);
v___x_478_ = lean_apply_1(v_k_476_, v_result_477_);
return v___x_478_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim(lean_object* v_motive_479_, lean_object* v_ctorIdx_480_, lean_object* v_t_481_, lean_object* v_h_482_, lean_object* v_k_483_){
_start:
{
lean_object* v___x_484_; 
v___x_484_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(v_t_481_, v_k_483_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___boxed(lean_object* v_motive_485_, lean_object* v_ctorIdx_486_, lean_object* v_t_487_, lean_object* v_h_488_, lean_object* v_k_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim(v_motive_485_, v_ctorIdx_486_, v_t_487_, v_h_488_, v_k_489_);
lean_dec(v_ctorIdx_486_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_running_elim___redArg(lean_object* v_t_491_, lean_object* v_running_492_){
_start:
{
lean_object* v___x_493_; 
v___x_493_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(v_t_491_, v_running_492_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_running_elim(lean_object* v_motive_494_, lean_object* v_t_495_, lean_object* v_h_496_, lean_object* v_running_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(v_t_495_, v_running_497_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_done_elim___redArg(lean_object* v_t_499_, lean_object* v_done_500_){
_start:
{
lean_object* v___x_501_; 
v___x_501_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(v_t_499_, v_done_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_CheckRequestResponse_done_elim(lean_object* v_motive_502_, lean_object* v_t_503_, lean_object* v_h_504_, lean_object* v_done_505_){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_proofwidgets_ProofWidgets_CheckRequestResponse_ctorElim___redArg(v_t_503_, v_done_505_);
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorIdx(lean_object* v_x_507_){
_start:
{
if (lean_obj_tag(v_x_507_) == 0)
{
lean_object* v___x_508_; 
v___x_508_ = lean_unsigned_to_nat(0u);
return v___x_508_;
}
else
{
lean_object* v___x_509_; 
v___x_509_ = lean_unsigned_to_nat(1u);
return v___x_509_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorIdx___boxed(lean_object* v_x_510_){
_start:
{
lean_object* v_res_511_; 
v_res_511_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorIdx(v_x_510_);
lean_dec(v_x_510_);
return v_res_511_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(lean_object* v_t_512_, lean_object* v_k_513_){
_start:
{
if (lean_obj_tag(v_t_512_) == 0)
{
return v_k_513_;
}
else
{
lean_object* v_result_514_; lean_object* v___x_515_; 
v_result_514_ = lean_ctor_get(v_t_512_, 0);
lean_inc(v_result_514_);
lean_dec_ref_known(v_t_512_, 1);
v___x_515_ = lean_apply_1(v_k_513_, v_result_514_);
return v___x_515_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim(lean_object* v_motive_516_, lean_object* v_ctorIdx_517_, lean_object* v_t_518_, lean_object* v_h_519_, lean_object* v_k_520_){
_start:
{
lean_object* v___x_521_; 
v___x_521_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(v_t_518_, v_k_520_);
return v___x_521_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___boxed(lean_object* v_motive_522_, lean_object* v_ctorIdx_523_, lean_object* v_t_524_, lean_object* v_h_525_, lean_object* v_k_526_){
_start:
{
lean_object* v_res_527_; 
v_res_527_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim(v_motive_522_, v_ctorIdx_523_, v_t_524_, v_h_525_, v_k_526_);
lean_dec(v_ctorIdx_523_);
return v_res_527_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_running_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim___redArg(lean_object* v_t_528_, lean_object* v_ProofWidgets_RpcEncodablePacket_running_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(v_t_528_, v_ProofWidgets_RpcEncodablePacket_running_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_running_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim(lean_object* v_motive_531_, lean_object* v_t_532_, lean_object* v_h_533_, lean_object* v_ProofWidgets_RpcEncodablePacket_running_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(v_t_532_, v_ProofWidgets_RpcEncodablePacket_running_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_done_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim___redArg(lean_object* v_t_536_, lean_object* v_ProofWidgets_RpcEncodablePacket_done_537_){
_start:
{
lean_object* v___x_538_; 
v___x_538_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(v_t_536_, v_ProofWidgets_RpcEncodablePacket_done_537_);
return v___x_538_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RpcEncodablePacket_done_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__elim(lean_object* v_motive_539_, lean_object* v_t_540_, lean_object* v_h_541_, lean_object* v_ProofWidgets_RpcEncodablePacket_done_542_){
_start:
{
lean_object* v___x_543_; 
v___x_543_ = lp_proofwidgets_ProofWidgets_RpcEncodablePacket_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__ctorElim___redArg(v_t_540_, v_ProofWidgets_RpcEncodablePacket_done_542_);
return v___x_543_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_(lean_object* v_json_563_){
_start:
{
lean_object* v___x_564_; 
lean_inc(v_json_563_);
v___x_564_ = l_Lean_Json_getTag_x3f(v_json_563_);
if (lean_obj_tag(v___x_564_) == 0)
{
lean_object* v___x_565_; 
lean_dec(v_json_563_);
v___x_565_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
return v___x_565_;
}
else
{
lean_object* v_val_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_601_; 
v_val_566_ = lean_ctor_get(v___x_564_, 0);
v_isSharedCheck_601_ = !lean_is_exclusive(v___x_564_);
if (v_isSharedCheck_601_ == 0)
{
v___x_568_ = v___x_564_;
v_isShared_569_ = v_isSharedCheck_601_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_val_566_);
lean_dec(v___x_564_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_601_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_570_; uint8_t v___x_571_; 
v___x_570_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
v___x_571_ = lean_string_dec_eq(v_val_566_, v___x_570_);
if (v___x_571_ == 0)
{
lean_object* v___x_572_; uint8_t v___x_573_; 
v___x_572_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
v___x_573_ = lean_string_dec_eq(v_val_566_, v___x_572_);
lean_dec(v_val_566_);
if (v___x_573_ == 0)
{
lean_object* v___x_574_; 
lean_del_object(v___x_568_);
lean_dec(v_json_563_);
v___x_574_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__5_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
return v___x_574_;
}
else
{
lean_object* v___x_575_; lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_575_ = lean_unsigned_to_nat(1u);
v___x_576_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__9_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
v___x_577_ = l_Lean_Json_parseCtorFields(v_json_563_, v___x_572_, v___x_575_, v___x_576_);
if (lean_obj_tag(v___x_577_) == 0)
{
lean_object* v_a_578_; lean_object* v___x_580_; uint8_t v_isShared_581_; uint8_t v_isSharedCheck_585_; 
lean_del_object(v___x_568_);
v_a_578_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_585_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_585_ == 0)
{
v___x_580_ = v___x_577_;
v_isShared_581_ = v_isSharedCheck_585_;
goto v_resetjp_579_;
}
else
{
lean_inc(v_a_578_);
lean_dec(v___x_577_);
v___x_580_ = lean_box(0);
v_isShared_581_ = v_isSharedCheck_585_;
goto v_resetjp_579_;
}
v_resetjp_579_:
{
lean_object* v___x_583_; 
if (v_isShared_581_ == 0)
{
v___x_583_ = v___x_580_;
goto v_reusejp_582_;
}
else
{
lean_object* v_reuseFailAlloc_584_; 
v_reuseFailAlloc_584_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_584_, 0, v_a_578_);
v___x_583_ = v_reuseFailAlloc_584_;
goto v_reusejp_582_;
}
v_reusejp_582_:
{
return v___x_583_;
}
}
}
else
{
lean_object* v_a_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_599_; 
v_a_586_ = lean_ctor_get(v___x_577_, 0);
v_isSharedCheck_599_ = !lean_is_exclusive(v___x_577_);
if (v_isSharedCheck_599_ == 0)
{
v___x_588_ = v___x_577_;
v_isShared_589_ = v_isSharedCheck_599_;
goto v_resetjp_587_;
}
else
{
lean_inc(v_a_586_);
lean_dec(v___x_577_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_599_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_594_; 
v___x_590_ = lean_box(0);
v___x_591_ = lean_unsigned_to_nat(0u);
v___x_592_ = lean_array_get(v___x_590_, v_a_586_, v___x_591_);
lean_dec(v_a_586_);
if (v_isShared_569_ == 0)
{
lean_ctor_set(v___x_568_, 0, v___x_592_);
v___x_594_ = v___x_568_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_598_; 
v_reuseFailAlloc_598_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_598_, 0, v___x_592_);
v___x_594_ = v_reuseFailAlloc_598_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
lean_object* v___x_596_; 
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v___x_594_);
v___x_596_ = v___x_588_;
goto v_reusejp_595_;
}
else
{
lean_object* v_reuseFailAlloc_597_; 
v_reuseFailAlloc_597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_597_, 0, v___x_594_);
v___x_596_ = v_reuseFailAlloc_597_;
goto v_reusejp_595_;
}
v_reusejp_595_:
{
return v___x_596_;
}
}
}
}
}
}
else
{
lean_object* v___x_600_; 
lean_del_object(v___x_568_);
lean_dec(v_val_566_);
lean_dec(v_json_563_);
v___x_600_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__10_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
return v___x_600_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_(lean_object* v_x_606_){
_start:
{
if (lean_obj_tag(v_x_606_) == 0)
{
lean_object* v___x_607_; 
v___x_607_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_));
return v___x_607_;
}
else
{
lean_object* v_result_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v_result_608_ = lean_ctor_get(v_x_606_, 0);
v___x_609_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
v___x_610_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson___closed__6_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_));
lean_inc(v_result_608_);
v___x_611_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_611_, 0, v___x_610_);
lean_ctor_set(v___x_611_, 1, v_result_608_);
v___x_612_ = lean_box(0);
v___x_613_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_613_, 0, v___x_611_);
lean_ctor_set(v___x_613_, 1, v___x_612_);
v___x_614_ = l_Lean_Json_mkObj(v___x_613_);
lean_dec_ref_known(v___x_613_, 2);
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_609_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
v___x_616_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_615_);
lean_ctor_set(v___x_616_, 1, v___x_612_);
v___x_617_ = l_Lean_Json_mkObj(v___x_616_);
lean_dec_ref_known(v___x_616_, 2);
return v___x_617_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34____boxed(lean_object* v_x_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_(v_x_618_);
lean_dec(v_x_618_);
return v_res_619_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(void){
_start:
{
lean_object* v___x_622_; lean_object* v___x_623_; 
v___x_622_ = lean_box(0);
v___x_623_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_(v___x_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object* v_x_624_, lean_object* v_a_625_){
_start:
{
if (lean_obj_tag(v_x_624_) == 0)
{
lean_object* v___x_626_; lean_object* v___x_627_; 
v___x_626_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_, &lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1__once, _init_lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_);
v___x_627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_627_, 0, v___x_626_);
lean_ctor_set(v___x_627_, 1, v_a_625_);
return v___x_627_;
}
else
{
lean_object* v_result_628_; lean_object* v___x_630_; uint8_t v_isShared_631_; uint8_t v_isSharedCheck_646_; 
v_result_628_ = lean_ctor_get(v_x_624_, 0);
v_isSharedCheck_646_ = !lean_is_exclusive(v_x_624_);
if (v_isSharedCheck_646_ == 0)
{
v___x_630_ = v_x_624_;
v_isShared_631_ = v_isSharedCheck_646_;
goto v_resetjp_629_;
}
else
{
lean_inc(v_result_628_);
lean_dec(v_x_624_);
v___x_630_ = lean_box(0);
v_isShared_631_ = v_isSharedCheck_646_;
goto v_resetjp_629_;
}
v_resetjp_629_:
{
lean_object* v___x_632_; lean_object* v_fst_633_; lean_object* v_snd_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_645_; 
v___x_632_ = lean_apply_1(v_result_628_, v_a_625_);
v_fst_633_ = lean_ctor_get(v___x_632_, 0);
v_snd_634_ = lean_ctor_get(v___x_632_, 1);
v_isSharedCheck_645_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_645_ == 0)
{
v___x_636_ = v___x_632_;
v_isShared_637_ = v_isSharedCheck_645_;
goto v_resetjp_635_;
}
else
{
lean_inc(v_snd_634_);
lean_inc(v_fst_633_);
lean_dec(v___x_632_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_645_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v___x_639_; 
if (v_isShared_631_ == 0)
{
lean_ctor_set(v___x_630_, 0, v_fst_633_);
v___x_639_ = v___x_630_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v_fst_633_);
v___x_639_ = v_reuseFailAlloc_644_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
lean_object* v___x_640_; lean_object* v___x_642_; 
v___x_640_ = lp_proofwidgets_ProofWidgets_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_34_(v___x_639_);
lean_dec_ref(v___x_639_);
if (v_isShared_637_ == 0)
{
lean_ctor_set(v___x_636_, 0, v___x_640_);
v___x_642_ = v___x_636_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v___x_640_);
lean_ctor_set(v_reuseFailAlloc_643_, 1, v_snd_634_);
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
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___lam__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object* v_result_647_, lean_object* v___y_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_649_, 0, v_result_647_);
lean_ctor_set(v___x_649_, 1, v___y_648_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object* v_j_652_){
_start:
{
lean_object* v___x_653_; 
v___x_653_ = lp_proofwidgets_ProofWidgets_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Cancellable_2911678472____hygCtx___hyg_14_(v_j_652_);
if (lean_obj_tag(v___x_653_) == 0)
{
lean_object* v_a_654_; lean_object* v___x_656_; uint8_t v_isShared_657_; uint8_t v_isSharedCheck_661_; 
v_a_654_ = lean_ctor_get(v___x_653_, 0);
v_isSharedCheck_661_ = !lean_is_exclusive(v___x_653_);
if (v_isSharedCheck_661_ == 0)
{
v___x_656_ = v___x_653_;
v_isShared_657_ = v_isSharedCheck_661_;
goto v_resetjp_655_;
}
else
{
lean_inc(v_a_654_);
lean_dec(v___x_653_);
v___x_656_ = lean_box(0);
v_isShared_657_ = v_isSharedCheck_661_;
goto v_resetjp_655_;
}
v_resetjp_655_:
{
lean_object* v___x_659_; 
if (v_isShared_657_ == 0)
{
v___x_659_ = v___x_656_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v_a_654_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
}
else
{
lean_object* v_a_662_; lean_object* v___x_664_; uint8_t v_isShared_665_; uint8_t v_isSharedCheck_679_; 
v_a_662_ = lean_ctor_get(v___x_653_, 0);
v_isSharedCheck_679_ = !lean_is_exclusive(v___x_653_);
if (v_isSharedCheck_679_ == 0)
{
v___x_664_ = v___x_653_;
v_isShared_665_ = v_isSharedCheck_679_;
goto v_resetjp_663_;
}
else
{
lean_inc(v_a_662_);
lean_dec(v___x_653_);
v___x_664_ = lean_box(0);
v_isShared_665_ = v_isSharedCheck_679_;
goto v_resetjp_663_;
}
v_resetjp_663_:
{
if (lean_obj_tag(v_a_662_) == 0)
{
lean_object* v___x_666_; 
lean_del_object(v___x_664_);
v___x_666_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___closed__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_));
return v___x_666_;
}
else
{
lean_object* v_result_667_; lean_object* v___x_669_; uint8_t v_isShared_670_; uint8_t v_isSharedCheck_678_; 
v_result_667_ = lean_ctor_get(v_a_662_, 0);
v_isSharedCheck_678_ = !lean_is_exclusive(v_a_662_);
if (v_isSharedCheck_678_ == 0)
{
v___x_669_ = v_a_662_;
v_isShared_670_ = v_isSharedCheck_678_;
goto v_resetjp_668_;
}
else
{
lean_inc(v_result_667_);
lean_dec(v_a_662_);
v___x_669_ = lean_box(0);
v_isShared_670_ = v_isSharedCheck_678_;
goto v_resetjp_668_;
}
v_resetjp_668_:
{
lean_object* v___f_671_; lean_object* v___x_673_; 
v___f_671_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg___lam__0_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_), 2, 1);
lean_closure_set(v___f_671_, 0, v_result_667_);
if (v_isShared_670_ == 0)
{
lean_ctor_set(v___x_669_, 0, v___f_671_);
v___x_673_ = v___x_669_;
goto v_reusejp_672_;
}
else
{
lean_object* v_reuseFailAlloc_677_; 
v_reuseFailAlloc_677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_677_, 0, v___f_671_);
v___x_673_ = v_reuseFailAlloc_677_;
goto v_reusejp_672_;
}
v_reusejp_672_:
{
lean_object* v___x_675_; 
if (v_isShared_665_ == 0)
{
lean_ctor_set(v___x_664_, 0, v___x_673_);
v___x_675_ = v___x_664_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_676_; 
v_reuseFailAlloc_676_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_676_, 0, v___x_673_);
v___x_675_ = v_reuseFailAlloc_676_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
return v___x_675_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(lean_object* v_j_680_, lean_object* v_a_681_){
_start:
{
lean_object* v___x_682_; 
v___x_682_ = lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec___redArg_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(v_j_680_);
return v___x_682_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1____boxed(lean_object* v_j_683_, lean_object* v_a_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_dec_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(v_j_683_, v_a_684_);
lean_dec_ref(v_a_684_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0(lean_object* v___x_694_, lean_object* v_rid_695_, lean_object* v___y_696_){
_start:
{
lean_object* v___x_698_; lean_object* v_snd_699_; lean_object* v___x_700_; 
v___x_698_ = lean_st_ref_get(v___x_694_);
v_snd_699_ = lean_ctor_get(v___x_698_, 1);
lean_inc(v_snd_699_);
lean_dec(v___x_698_);
v___x_700_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00ProofWidgets_cancelRequest_spec__0___redArg(v_snd_699_, v_rid_695_);
lean_dec(v_snd_699_);
if (lean_obj_tag(v___x_700_) == 0)
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; 
v___x_701_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__0));
v___x_702_ = l_Nat_reprFast(v_rid_695_);
v___x_703_ = lean_string_append(v___x_701_, v___x_702_);
lean_dec_ref(v___x_702_);
v___x_704_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_checkRequest___lam__0___closed__1));
v___x_705_ = lean_string_append(v___x_703_, v___x_704_);
v___x_706_ = l_Lean_Server_RequestError_invalidParams(v___x_705_);
v___x_707_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_707_, 0, v___x_706_);
return v___x_707_;
}
else
{
lean_object* v_val_708_; lean_object* v___x_710_; uint8_t v_isShared_711_; uint8_t v_isSharedCheck_750_; 
v_val_708_ = lean_ctor_get(v___x_700_, 0);
v_isSharedCheck_750_ = !lean_is_exclusive(v___x_700_);
if (v_isSharedCheck_750_ == 0)
{
v___x_710_ = v___x_700_;
v_isShared_711_ = v_isSharedCheck_750_;
goto v_resetjp_709_;
}
else
{
lean_inc(v_val_708_);
lean_dec(v___x_700_);
v___x_710_ = lean_box(0);
v_isShared_711_ = v_isSharedCheck_750_;
goto v_resetjp_709_;
}
v_resetjp_709_:
{
lean_object* v_task_712_; uint8_t v___x_713_; 
v_task_712_ = lean_ctor_get(v_val_708_, 0);
lean_inc_ref(v_task_712_);
lean_dec(v_val_708_);
v___x_713_ = lean_io_get_task_state(v_task_712_);
if (v___x_713_ == 2)
{
lean_object* v___x_714_; lean_object* v_fst_715_; lean_object* v_snd_716_; lean_object* v___x_718_; uint8_t v_isShared_719_; uint8_t v_isSharedCheck_745_; 
v___x_714_ = lean_st_ref_take(v___x_694_);
v_fst_715_ = lean_ctor_get(v___x_714_, 0);
v_snd_716_ = lean_ctor_get(v___x_714_, 1);
v_isSharedCheck_745_ = !lean_is_exclusive(v___x_714_);
if (v_isSharedCheck_745_ == 0)
{
v___x_718_ = v___x_714_;
v_isShared_719_ = v_isSharedCheck_745_;
goto v_resetjp_717_;
}
else
{
lean_inc(v_snd_716_);
lean_inc(v_fst_715_);
lean_dec(v___x_714_);
v___x_718_ = lean_box(0);
v_isShared_719_ = v_isSharedCheck_745_;
goto v_resetjp_717_;
}
v_resetjp_717_:
{
lean_object* v___x_720_; lean_object* v___x_722_; 
v___x_720_ = lp_proofwidgets_Std_DHashMap_Internal_Raw_u2080_erase___at___00ProofWidgets_cancelRequest_spec__1___redArg(v_snd_716_, v_rid_695_);
lean_dec(v_rid_695_);
if (v_isShared_719_ == 0)
{
lean_ctor_set(v___x_718_, 1, v___x_720_);
v___x_722_ = v___x_718_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_744_; 
v_reuseFailAlloc_744_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_744_, 0, v_fst_715_);
lean_ctor_set(v_reuseFailAlloc_744_, 1, v___x_720_);
v___x_722_ = v_reuseFailAlloc_744_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
lean_object* v___x_723_; lean_object* v___x_724_; 
v___x_723_ = lean_st_ref_set(v___x_694_, v___x_722_);
v___x_724_ = lean_task_get_own(v_task_712_);
if (lean_obj_tag(v___x_724_) == 0)
{
lean_object* v_a_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_732_; 
lean_del_object(v___x_710_);
v_a_725_ = lean_ctor_get(v___x_724_, 0);
v_isSharedCheck_732_ = !lean_is_exclusive(v___x_724_);
if (v_isSharedCheck_732_ == 0)
{
v___x_727_ = v___x_724_;
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_a_725_);
lean_dec(v___x_724_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_732_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_730_; 
if (v_isShared_728_ == 0)
{
lean_ctor_set_tag(v___x_727_, 1);
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
else
{
lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_743_; 
v_a_733_ = lean_ctor_get(v___x_724_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_724_);
if (v_isSharedCheck_743_ == 0)
{
v___x_735_ = v___x_724_;
v_isShared_736_ = v_isSharedCheck_743_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_724_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_743_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_738_; 
if (v_isShared_736_ == 0)
{
v___x_738_ = v___x_735_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_733_);
v___x_738_ = v_reuseFailAlloc_742_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
lean_object* v___x_740_; 
if (v_isShared_711_ == 0)
{
lean_ctor_set_tag(v___x_710_, 0);
lean_ctor_set(v___x_710_, 0, v___x_738_);
v___x_740_ = v___x_710_;
goto v_reusejp_739_;
}
else
{
lean_object* v_reuseFailAlloc_741_; 
v_reuseFailAlloc_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_741_, 0, v___x_738_);
v___x_740_ = v_reuseFailAlloc_741_;
goto v_reusejp_739_;
}
v_reusejp_739_:
{
return v___x_740_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_746_; lean_object* v___x_748_; 
lean_dec_ref(v_task_712_);
lean_dec(v_rid_695_);
v___x_746_ = lean_box(0);
if (v_isShared_711_ == 0)
{
lean_ctor_set_tag(v___x_710_, 0);
lean_ctor_set(v___x_710_, 0, v___x_746_);
v___x_748_ = v___x_710_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_749_; 
v_reuseFailAlloc_749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_749_, 0, v___x_746_);
v___x_748_ = v_reuseFailAlloc_749_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
return v___x_748_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___lam__0___boxed(lean_object* v___x_751_, lean_object* v_rid_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_proofwidgets_ProofWidgets_checkRequest___lam__0(v___x_751_, v_rid_752_, v___y_753_);
lean_dec_ref(v___y_753_);
lean_dec(v___x_751_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest(lean_object* v_rid_756_, lean_object* v_a_757_){
_start:
{
lean_object* v___x_759_; lean_object* v___f_760_; lean_object* v___x_761_; 
v___x_759_ = lp_proofwidgets_ProofWidgets_runningRequests;
v___f_760_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_checkRequest___lam__0___boxed), 4, 2);
lean_closure_set(v___f_760_, 0, v___x_759_);
lean_closure_set(v___f_760_, 1, v_rid_756_);
v___x_761_ = l_Lean_Server_RequestM_asTask___redArg(v___f_760_, v_a_757_);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_checkRequest___boxed(lean_object* v_rid_762_, lean_object* v_a_763_, lean_object* v_a_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_proofwidgets_ProofWidgets_checkRequest(v_rid_762_, v_a_763_);
lean_dec_ref(v_a_763_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2(lean_object* v_val_766_, lean_object* v___f_767_, lean_object* v_x_768_, lean_object* v___y_769_){
_start:
{
if (lean_obj_tag(v_x_768_) == 0)
{
lean_object* v_a_771_; lean_object* v___x_773_; uint8_t v_isShared_774_; uint8_t v_isSharedCheck_778_; 
lean_dec_ref(v___f_767_);
v_a_771_ = lean_ctor_get(v_x_768_, 0);
v_isSharedCheck_778_ = !lean_is_exclusive(v_x_768_);
if (v_isSharedCheck_778_ == 0)
{
v___x_773_ = v_x_768_;
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
else
{
lean_inc(v_a_771_);
lean_dec(v_x_768_);
v___x_773_ = lean_box(0);
v_isShared_774_ = v_isSharedCheck_778_;
goto v_resetjp_772_;
}
v_resetjp_772_:
{
lean_object* v___x_776_; 
if (v_isShared_774_ == 0)
{
lean_ctor_set_tag(v___x_773_, 1);
v___x_776_ = v___x_773_;
goto v_reusejp_775_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v_a_771_);
v___x_776_ = v_reuseFailAlloc_777_;
goto v_reusejp_775_;
}
v_reusejp_775_:
{
return v___x_776_;
}
}
}
else
{
lean_object* v_a_779_; lean_object* v___x_781_; uint8_t v_isShared_782_; uint8_t v_isSharedCheck_795_; 
v_a_779_ = lean_ctor_get(v_x_768_, 0);
v_isSharedCheck_795_ = !lean_is_exclusive(v_x_768_);
if (v_isSharedCheck_795_ == 0)
{
v___x_781_ = v_x_768_;
v_isShared_782_ = v_isSharedCheck_795_;
goto v_resetjp_780_;
}
else
{
lean_inc(v_a_779_);
lean_dec(v_x_768_);
v___x_781_ = lean_box(0);
v_isShared_782_ = v_isSharedCheck_795_;
goto v_resetjp_780_;
}
v_resetjp_780_:
{
lean_object* v___x_783_; lean_object* v_objects_784_; lean_object* v_expireTime_785_; lean_object* v___f_786_; lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v_fst_789_; lean_object* v_snd_790_; lean_object* v___x_791_; lean_object* v___x_793_; 
v___x_783_ = lean_st_ref_take(v_val_766_);
v_objects_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc_ref(v_objects_784_);
v_expireTime_785_ = lean_ctor_get(v___x_783_, 1);
lean_inc(v_expireTime_785_);
lean_dec(v___x_783_);
v___f_786_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_786_, 0, v_expireTime_785_);
v___x_787_ = lp_proofwidgets_ProofWidgets_instRpcEncodableCheckRequestResponse_enc_00___x40_ProofWidgets_Cancellable_1202226121____hygCtx___hyg_1_(v_a_779_, v_objects_784_);
v___x_788_ = l_Prod_map___redArg(v___f_767_, v___f_786_, v___x_787_);
v_fst_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_fst_789_);
v_snd_790_ = lean_ctor_get(v___x_788_, 1);
lean_inc(v_snd_790_);
lean_dec_ref(v___x_788_);
v___x_791_ = lean_st_ref_set(v_val_766_, v_snd_790_);
if (v_isShared_782_ == 0)
{
lean_ctor_set_tag(v___x_781_, 0);
lean_ctor_set(v___x_781_, 0, v_fst_789_);
v___x_793_ = v___x_781_;
goto v_reusejp_792_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v_fst_789_);
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
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_796_, lean_object* v___f_797_, lean_object* v_x_798_, lean_object* v___y_799_, lean_object* v___y_800_){
_start:
{
lean_object* v_res_801_; 
v_res_801_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2(v_val_796_, v___f_797_, v_x_798_, v___y_799_);
lean_dec_ref(v___y_799_);
lean_dec(v_val_796_);
return v_res_801_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0(lean_object* v_method_802_, lean_object* v_handler_803_, lean_object* v___f_804_, uint64_t v_seshId_805_, lean_object* v_j_806_, lean_object* v___y_807_){
_start:
{
lean_object* v_rpcSessions_809_; lean_object* v___x_810_; 
v_rpcSessions_809_ = lean_ctor_get(v___y_807_, 0);
v___x_810_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_809_, v_seshId_805_);
if (lean_obj_tag(v___x_810_) == 1)
{
lean_object* v_val_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v_val_811_ = lean_ctor_get(v___x_810_, 0);
lean_inc(v_val_811_);
lean_dec_ref_known(v___x_810_, 1);
v___x_812_ = lean_st_ref_get(v_val_811_);
lean_dec(v___x_812_);
lean_inc(v_j_806_);
v___x_813_ = l_Lean_Json_getNat_x3f(v_j_806_);
if (lean_obj_tag(v___x_813_) == 0)
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_834_; 
lean_dec(v_val_811_);
lean_dec_ref(v___f_804_);
lean_dec_ref(v_handler_803_);
v_a_814_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_834_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_834_ == 0)
{
v___x_816_ = v___x_813_;
v_isShared_817_ = v_isSharedCheck_834_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_813_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_834_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
uint8_t v___x_818_; lean_object* v___x_819_; uint8_t v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; lean_object* v___x_828_; lean_object* v___x_829_; lean_object* v___x_830_; lean_object* v___x_832_; 
v___x_818_ = 3;
v___x_819_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_820_ = 1;
v___x_821_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_802_, v___x_820_);
v___x_822_ = lean_string_append(v___x_819_, v___x_821_);
lean_dec_ref(v___x_821_);
v___x_823_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_824_ = lean_string_append(v___x_822_, v___x_823_);
v___x_825_ = l_Lean_Json_compress(v_j_806_);
v___x_826_ = lean_string_append(v___x_824_, v___x_825_);
lean_dec_ref(v___x_825_);
v___x_827_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_828_ = lean_string_append(v___x_826_, v___x_827_);
v___x_829_ = lean_string_append(v___x_828_, v_a_814_);
lean_dec(v_a_814_);
v___x_830_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_830_, 0, v___x_829_);
lean_ctor_set_uint8(v___x_830_, sizeof(void*)*1, v___x_818_);
if (v_isShared_817_ == 0)
{
lean_ctor_set_tag(v___x_816_, 1);
lean_ctor_set(v___x_816_, 0, v___x_830_);
v___x_832_ = v___x_816_;
goto v_reusejp_831_;
}
else
{
lean_object* v_reuseFailAlloc_833_; 
v_reuseFailAlloc_833_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_833_, 0, v___x_830_);
v___x_832_ = v_reuseFailAlloc_833_;
goto v_reusejp_831_;
}
v_reusejp_831_:
{
return v___x_832_;
}
}
}
else
{
lean_object* v_a_835_; lean_object* v___x_836_; 
lean_dec(v_j_806_);
lean_dec(v_method_802_);
v_a_835_ = lean_ctor_get(v___x_813_, 0);
lean_inc(v_a_835_);
lean_dec_ref_known(v___x_813_, 1);
lean_inc_ref(v___y_807_);
v___x_836_ = lean_apply_3(v_handler_803_, v_a_835_, v___y_807_, lean_box(0));
if (lean_obj_tag(v___x_836_) == 0)
{
lean_object* v_a_837_; lean_object* v___f_838_; lean_object* v___x_839_; 
v_a_837_ = lean_ctor_get(v___x_836_, 0);
lean_inc(v_a_837_);
lean_dec_ref_known(v___x_836_, 1);
v___f_838_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_838_, 0, v_val_811_);
lean_closure_set(v___f_838_, 1, v___f_804_);
v___x_839_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_837_, v___f_838_, v___y_807_);
return v___x_839_;
}
else
{
lean_object* v_a_840_; lean_object* v___x_842_; uint8_t v_isShared_843_; uint8_t v_isSharedCheck_847_; 
lean_dec(v_val_811_);
lean_dec_ref(v___f_804_);
v_a_840_ = lean_ctor_get(v___x_836_, 0);
v_isSharedCheck_847_ = !lean_is_exclusive(v___x_836_);
if (v_isSharedCheck_847_ == 0)
{
v___x_842_ = v___x_836_;
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
else
{
lean_inc(v_a_840_);
lean_dec(v___x_836_);
v___x_842_ = lean_box(0);
v_isShared_843_ = v_isSharedCheck_847_;
goto v_resetjp_841_;
}
v_resetjp_841_:
{
lean_object* v___x_845_; 
if (v_isShared_843_ == 0)
{
v___x_845_ = v___x_842_;
goto v_reusejp_844_;
}
else
{
lean_object* v_reuseFailAlloc_846_; 
v_reuseFailAlloc_846_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_846_, 0, v_a_840_);
v___x_845_ = v_reuseFailAlloc_846_;
goto v_reusejp_844_;
}
v_reusejp_844_:
{
return v___x_845_;
}
}
}
}
}
else
{
lean_object* v___x_848_; lean_object* v___x_849_; 
lean_dec(v___x_810_);
lean_dec(v_j_806_);
lean_dec_ref(v___f_804_);
lean_dec_ref(v_handler_803_);
lean_dec(v_method_802_);
v___x_848_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_849_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v_method_850_, lean_object* v_handler_851_, lean_object* v___f_852_, lean_object* v_seshId_853_, lean_object* v_j_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
uint64_t v_seshId_boxed_857_; lean_object* v_res_858_; 
v_seshId_boxed_857_ = lean_unbox_uint64(v_seshId_853_);
lean_dec_ref(v_seshId_853_);
v_res_858_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0(v_method_850_, v_handler_851_, v___f_852_, v_seshId_boxed_857_, v_j_854_, v___y_855_);
lean_dec_ref(v___y_855_);
return v_res_858_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0(lean_object* v_method_859_, lean_object* v_handler_860_){
_start:
{
lean_object* v___f_861_; lean_object* v___f_862_; 
v___f_861_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_cancelRequest___rpc__wrapped_spec__0___closed__0));
v___f_862_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0___lam__0___boxed), 7, 3);
lean_closure_set(v___f_862_, 0, v_method_859_);
lean_closure_set(v___f_862_, 1, v_handler_860_);
lean_closure_set(v___f_862_, 2, v___f_861_);
return v___f_862_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3(void){
_start:
{
lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_868_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__2));
v___x_869_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__1));
v___x_870_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_checkRequest___rpc__wrapped_spec__0(v___x_869_, v___x_868_);
return v___x_870_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped(void){
_start:
{
lean_object* v___x_871_; 
v___x_871_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3, &lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3_once, _init_lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped___closed__3);
return v___x_871_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0(uint8_t v___y_884_, uint8_t v_suppressElabErrors_885_, lean_object* v_x_886_){
_start:
{
if (lean_obj_tag(v_x_886_) == 1)
{
lean_object* v_pre_887_; 
v_pre_887_ = lean_ctor_get(v_x_886_, 0);
switch(lean_obj_tag(v_pre_887_))
{
case 1:
{
lean_object* v_pre_888_; 
v_pre_888_ = lean_ctor_get(v_pre_887_, 0);
switch(lean_obj_tag(v_pre_888_))
{
case 0:
{
lean_object* v_str_889_; lean_object* v_str_890_; lean_object* v___x_891_; uint8_t v___x_892_; 
v_str_889_ = lean_ctor_get(v_x_886_, 1);
v_str_890_ = lean_ctor_get(v_pre_887_, 1);
v___x_891_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__0));
v___x_892_ = lean_string_dec_eq(v_str_890_, v___x_891_);
if (v___x_892_ == 0)
{
lean_object* v___x_893_; uint8_t v___x_894_; 
v___x_893_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__1));
v___x_894_ = lean_string_dec_eq(v_str_890_, v___x_893_);
if (v___x_894_ == 0)
{
return v___y_884_;
}
else
{
lean_object* v___x_895_; uint8_t v___x_896_; 
v___x_895_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__2));
v___x_896_ = lean_string_dec_eq(v_str_889_, v___x_895_);
if (v___x_896_ == 0)
{
return v___y_884_;
}
else
{
return v_suppressElabErrors_885_;
}
}
}
else
{
lean_object* v___x_897_; uint8_t v___x_898_; 
v___x_897_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__3));
v___x_898_ = lean_string_dec_eq(v_str_889_, v___x_897_);
if (v___x_898_ == 0)
{
return v___y_884_;
}
else
{
return v_suppressElabErrors_885_;
}
}
}
case 1:
{
lean_object* v_pre_899_; 
v_pre_899_ = lean_ctor_get(v_pre_888_, 0);
if (lean_obj_tag(v_pre_899_) == 0)
{
lean_object* v_str_900_; lean_object* v_str_901_; lean_object* v_str_902_; lean_object* v___x_903_; uint8_t v___x_904_; 
v_str_900_ = lean_ctor_get(v_x_886_, 1);
v_str_901_ = lean_ctor_get(v_pre_887_, 1);
v_str_902_ = lean_ctor_get(v_pre_888_, 1);
v___x_903_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__4));
v___x_904_ = lean_string_dec_eq(v_str_902_, v___x_903_);
if (v___x_904_ == 0)
{
return v___y_884_;
}
else
{
lean_object* v___x_905_; uint8_t v___x_906_; 
v___x_905_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__5));
v___x_906_ = lean_string_dec_eq(v_str_901_, v___x_905_);
if (v___x_906_ == 0)
{
return v___y_884_;
}
else
{
lean_object* v___x_907_; uint8_t v___x_908_; 
v___x_907_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__6));
v___x_908_ = lean_string_dec_eq(v_str_900_, v___x_907_);
if (v___x_908_ == 0)
{
return v___y_884_;
}
else
{
return v_suppressElabErrors_885_;
}
}
}
}
else
{
return v___y_884_;
}
}
default: 
{
return v___y_884_;
}
}
}
case 0:
{
lean_object* v_str_909_; lean_object* v___x_910_; uint8_t v___x_911_; 
v_str_909_ = lean_ctor_get(v_x_886_, 1);
v___x_910_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___closed__7));
v___x_911_ = lean_string_dec_eq(v_str_909_, v___x_910_);
if (v___x_911_ == 0)
{
return v___y_884_;
}
else
{
return v_suppressElabErrors_885_;
}
}
default: 
{
return v___y_884_;
}
}
}
else
{
return v___y_884_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v___y_912_, lean_object* v_suppressElabErrors_913_, lean_object* v_x_914_){
_start:
{
uint8_t v___y_4995__boxed_915_; uint8_t v_suppressElabErrors_boxed_916_; uint8_t v_res_917_; lean_object* v_r_918_; 
v___y_4995__boxed_915_ = lean_unbox(v___y_912_);
v_suppressElabErrors_boxed_916_ = lean_unbox(v_suppressElabErrors_913_);
v_res_917_ = lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0(v___y_4995__boxed_915_, v_suppressElabErrors_boxed_916_, v_x_914_);
lean_dec(v_x_914_);
v_r_918_ = lean_box(v_res_917_);
return v_r_918_;
}
}
LEAN_EXPORT uint8_t lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(lean_object* v_opts_919_, lean_object* v_opt_920_){
_start:
{
lean_object* v_name_921_; lean_object* v_defValue_922_; lean_object* v_map_923_; lean_object* v___x_924_; 
v_name_921_ = lean_ctor_get(v_opt_920_, 0);
v_defValue_922_ = lean_ctor_get(v_opt_920_, 1);
v_map_923_ = lean_ctor_get(v_opts_919_, 0);
v___x_924_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_923_, v_name_921_);
if (lean_obj_tag(v___x_924_) == 0)
{
uint8_t v___x_925_; 
v___x_925_ = lean_unbox(v_defValue_922_);
return v___x_925_;
}
else
{
lean_object* v_val_926_; 
v_val_926_ = lean_ctor_get(v___x_924_, 0);
lean_inc(v_val_926_);
lean_dec_ref_known(v___x_924_, 1);
if (lean_obj_tag(v_val_926_) == 1)
{
uint8_t v_v_927_; 
v_v_927_ = lean_ctor_get_uint8(v_val_926_, 0);
lean_dec_ref_known(v_val_926_, 0);
return v_v_927_;
}
else
{
uint8_t v___x_928_; 
lean_dec(v_val_926_);
v___x_928_ = lean_unbox(v_defValue_922_);
return v___x_928_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_opts_929_, lean_object* v_opt_930_){
_start:
{
uint8_t v_res_931_; lean_object* v_r_932_; 
v_res_931_ = lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(v_opts_929_, v_opt_930_);
lean_dec_ref(v_opt_930_);
lean_dec_ref(v_opts_929_);
v_r_932_ = lean_box(v_res_931_);
return v_r_932_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(lean_object* v_msgData_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_){
_start:
{
lean_object* v___x_939_; lean_object* v_env_940_; lean_object* v___x_941_; lean_object* v_mctx_942_; lean_object* v_lctx_943_; lean_object* v_options_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_939_ = lean_st_ref_get(v___y_937_);
v_env_940_ = lean_ctor_get(v___x_939_, 0);
lean_inc_ref(v_env_940_);
lean_dec(v___x_939_);
v___x_941_ = lean_st_ref_get(v___y_935_);
v_mctx_942_ = lean_ctor_get(v___x_941_, 0);
lean_inc_ref(v_mctx_942_);
lean_dec(v___x_941_);
v_lctx_943_ = lean_ctor_get(v___y_934_, 2);
v_options_944_ = lean_ctor_get(v___y_936_, 2);
lean_inc_ref(v_options_944_);
lean_inc_ref(v_lctx_943_);
v___x_945_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_945_, 0, v_env_940_);
lean_ctor_set(v___x_945_, 1, v_mctx_942_);
lean_ctor_set(v___x_945_, 2, v_lctx_943_);
lean_ctor_set(v___x_945_, 3, v_options_944_);
v___x_946_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_946_, 0, v___x_945_);
lean_ctor_set(v___x_946_, 1, v_msgData_933_);
v___x_947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_947_, 0, v___x_946_);
return v___x_947_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_msgData_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_){
_start:
{
lean_object* v_res_954_; 
v_res_954_ = lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(v_msgData_948_, v___y_949_, v___y_950_, v___y_951_, v___y_952_);
lean_dec(v___y_952_);
lean_dec_ref(v___y_951_);
lean_dec(v___y_950_);
lean_dec_ref(v___y_949_);
return v_res_954_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1(lean_object* v_ref_956_, lean_object* v_msgData_957_, uint8_t v_severity_958_, uint8_t v_isSilent_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v___y_966_; uint8_t v___y_967_; lean_object* v___y_968_; uint8_t v___y_969_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v___y_972_; lean_object* v___y_973_; lean_object* v___y_974_; lean_object* v___y_1002_; lean_object* v___y_1003_; lean_object* v___y_1004_; uint8_t v___y_1005_; uint8_t v___y_1006_; uint8_t v___y_1007_; lean_object* v___y_1008_; lean_object* v___y_1009_; lean_object* v___y_1027_; lean_object* v___y_1028_; uint8_t v___y_1029_; lean_object* v___y_1030_; lean_object* v___y_1031_; uint8_t v___y_1032_; uint8_t v___y_1033_; lean_object* v___y_1034_; lean_object* v___y_1038_; lean_object* v___y_1039_; lean_object* v___y_1040_; uint8_t v___y_1041_; uint8_t v___y_1042_; lean_object* v___y_1043_; uint8_t v___y_1044_; uint8_t v___x_1049_; lean_object* v___y_1051_; lean_object* v___y_1052_; lean_object* v___y_1053_; lean_object* v___y_1054_; uint8_t v___y_1055_; uint8_t v___y_1056_; uint8_t v___y_1057_; uint8_t v___y_1059_; uint8_t v___x_1074_; 
v___x_1049_ = 2;
v___x_1074_ = l_Lean_instBEqMessageSeverity_beq(v_severity_958_, v___x_1049_);
if (v___x_1074_ == 0)
{
v___y_1059_ = v___x_1074_;
goto v___jp_1058_;
}
else
{
uint8_t v___x_1075_; 
lean_inc_ref(v_msgData_957_);
v___x_1075_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_957_);
v___y_1059_ = v___x_1075_;
goto v___jp_1058_;
}
v___jp_965_:
{
lean_object* v___x_975_; lean_object* v_currNamespace_976_; lean_object* v_openDecls_977_; lean_object* v_env_978_; lean_object* v_nextMacroScope_979_; lean_object* v_ngen_980_; lean_object* v_auxDeclNGen_981_; lean_object* v_traceState_982_; lean_object* v_cache_983_; lean_object* v_messages_984_; lean_object* v_infoState_985_; lean_object* v_snapshotTasks_986_; lean_object* v___x_988_; uint8_t v_isShared_989_; uint8_t v_isSharedCheck_1000_; 
v___x_975_ = lean_st_ref_take(v___y_974_);
v_currNamespace_976_ = lean_ctor_get(v___y_973_, 6);
v_openDecls_977_ = lean_ctor_get(v___y_973_, 7);
v_env_978_ = lean_ctor_get(v___x_975_, 0);
v_nextMacroScope_979_ = lean_ctor_get(v___x_975_, 1);
v_ngen_980_ = lean_ctor_get(v___x_975_, 2);
v_auxDeclNGen_981_ = lean_ctor_get(v___x_975_, 3);
v_traceState_982_ = lean_ctor_get(v___x_975_, 4);
v_cache_983_ = lean_ctor_get(v___x_975_, 5);
v_messages_984_ = lean_ctor_get(v___x_975_, 6);
v_infoState_985_ = lean_ctor_get(v___x_975_, 7);
v_snapshotTasks_986_ = lean_ctor_get(v___x_975_, 8);
v_isSharedCheck_1000_ = !lean_is_exclusive(v___x_975_);
if (v_isSharedCheck_1000_ == 0)
{
v___x_988_ = v___x_975_;
v_isShared_989_ = v_isSharedCheck_1000_;
goto v_resetjp_987_;
}
else
{
lean_inc(v_snapshotTasks_986_);
lean_inc(v_infoState_985_);
lean_inc(v_messages_984_);
lean_inc(v_cache_983_);
lean_inc(v_traceState_982_);
lean_inc(v_auxDeclNGen_981_);
lean_inc(v_ngen_980_);
lean_inc(v_nextMacroScope_979_);
lean_inc(v_env_978_);
lean_dec(v___x_975_);
v___x_988_ = lean_box(0);
v_isShared_989_ = v_isSharedCheck_1000_;
goto v_resetjp_987_;
}
v_resetjp_987_:
{
lean_object* v___x_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_995_; 
lean_inc(v_openDecls_977_);
lean_inc(v_currNamespace_976_);
v___x_990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_990_, 0, v_currNamespace_976_);
lean_ctor_set(v___x_990_, 1, v_openDecls_977_);
v___x_991_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_991_, 0, v___x_990_);
lean_ctor_set(v___x_991_, 1, v___y_966_);
lean_inc_ref(v___y_970_);
lean_inc_ref(v___y_968_);
v___x_992_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_992_, 0, v___y_968_);
lean_ctor_set(v___x_992_, 1, v___y_972_);
lean_ctor_set(v___x_992_, 2, v___y_971_);
lean_ctor_set(v___x_992_, 3, v___y_970_);
lean_ctor_set(v___x_992_, 4, v___x_991_);
lean_ctor_set_uint8(v___x_992_, sizeof(void*)*5, v___y_969_);
lean_ctor_set_uint8(v___x_992_, sizeof(void*)*5 + 1, v___y_967_);
lean_ctor_set_uint8(v___x_992_, sizeof(void*)*5 + 2, v_isSilent_959_);
v___x_993_ = l_Lean_MessageLog_add(v___x_992_, v_messages_984_);
if (v_isShared_989_ == 0)
{
lean_ctor_set(v___x_988_, 6, v___x_993_);
v___x_995_ = v___x_988_;
goto v_reusejp_994_;
}
else
{
lean_object* v_reuseFailAlloc_999_; 
v_reuseFailAlloc_999_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_999_, 0, v_env_978_);
lean_ctor_set(v_reuseFailAlloc_999_, 1, v_nextMacroScope_979_);
lean_ctor_set(v_reuseFailAlloc_999_, 2, v_ngen_980_);
lean_ctor_set(v_reuseFailAlloc_999_, 3, v_auxDeclNGen_981_);
lean_ctor_set(v_reuseFailAlloc_999_, 4, v_traceState_982_);
lean_ctor_set(v_reuseFailAlloc_999_, 5, v_cache_983_);
lean_ctor_set(v_reuseFailAlloc_999_, 6, v___x_993_);
lean_ctor_set(v_reuseFailAlloc_999_, 7, v_infoState_985_);
lean_ctor_set(v_reuseFailAlloc_999_, 8, v_snapshotTasks_986_);
v___x_995_ = v_reuseFailAlloc_999_;
goto v_reusejp_994_;
}
v_reusejp_994_:
{
lean_object* v___x_996_; lean_object* v___x_997_; lean_object* v___x_998_; 
v___x_996_ = lean_st_ref_set(v___y_974_, v___x_995_);
v___x_997_ = lean_box(0);
v___x_998_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_998_, 0, v___x_997_);
return v___x_998_;
}
}
}
v___jp_1001_:
{
lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v_a_1012_; lean_object* v___x_1014_; uint8_t v_isShared_1015_; uint8_t v_isSharedCheck_1025_; 
v___x_1010_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_957_);
v___x_1011_ = lp_proofwidgets_Lean_addMessageContextFull___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__4(v___x_1010_, v___y_960_, v___y_961_, v___y_962_, v___y_963_);
v_a_1012_ = lean_ctor_get(v___x_1011_, 0);
v_isSharedCheck_1025_ = !lean_is_exclusive(v___x_1011_);
if (v_isSharedCheck_1025_ == 0)
{
v___x_1014_ = v___x_1011_;
v_isShared_1015_ = v_isSharedCheck_1025_;
goto v_resetjp_1013_;
}
else
{
lean_inc(v_a_1012_);
lean_dec(v___x_1011_);
v___x_1014_ = lean_box(0);
v_isShared_1015_ = v_isSharedCheck_1025_;
goto v_resetjp_1013_;
}
v_resetjp_1013_:
{
lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; 
lean_inc_ref_n(v___y_1003_, 2);
v___x_1016_ = l_Lean_FileMap_toPosition(v___y_1003_, v___y_1008_);
lean_dec(v___y_1008_);
v___x_1017_ = l_Lean_FileMap_toPosition(v___y_1003_, v___y_1009_);
lean_dec(v___y_1009_);
v___x_1018_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1018_, 0, v___x_1017_);
v___x_1019_ = ((lean_object*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___closed__0));
if (v___y_1007_ == 0)
{
lean_del_object(v___x_1014_);
lean_dec_ref(v___y_1002_);
v___y_966_ = v_a_1012_;
v___y_967_ = v___y_1005_;
v___y_968_ = v___y_1004_;
v___y_969_ = v___y_1006_;
v___y_970_ = v___x_1019_;
v___y_971_ = v___x_1018_;
v___y_972_ = v___x_1016_;
v___y_973_ = v___y_962_;
v___y_974_ = v___y_963_;
goto v___jp_965_;
}
else
{
uint8_t v___x_1020_; 
lean_inc(v_a_1012_);
v___x_1020_ = l_Lean_MessageData_hasTag(v___y_1002_, v_a_1012_);
if (v___x_1020_ == 0)
{
lean_object* v___x_1021_; lean_object* v___x_1023_; 
lean_dec_ref_known(v___x_1018_, 1);
lean_dec_ref(v___x_1016_);
lean_dec(v_a_1012_);
v___x_1021_ = lean_box(0);
if (v_isShared_1015_ == 0)
{
lean_ctor_set(v___x_1014_, 0, v___x_1021_);
v___x_1023_ = v___x_1014_;
goto v_reusejp_1022_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v___x_1021_);
v___x_1023_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1022_;
}
v_reusejp_1022_:
{
return v___x_1023_;
}
}
else
{
lean_del_object(v___x_1014_);
v___y_966_ = v_a_1012_;
v___y_967_ = v___y_1005_;
v___y_968_ = v___y_1004_;
v___y_969_ = v___y_1006_;
v___y_970_ = v___x_1019_;
v___y_971_ = v___x_1018_;
v___y_972_ = v___x_1016_;
v___y_973_ = v___y_962_;
v___y_974_ = v___y_963_;
goto v___jp_965_;
}
}
}
}
v___jp_1026_:
{
lean_object* v___x_1035_; 
v___x_1035_ = l_Lean_Syntax_getTailPos_x3f(v___y_1031_, v___y_1032_);
lean_dec(v___y_1031_);
if (lean_obj_tag(v___x_1035_) == 0)
{
lean_inc(v___y_1034_);
v___y_1002_ = v___y_1027_;
v___y_1003_ = v___y_1028_;
v___y_1004_ = v___y_1030_;
v___y_1005_ = v___y_1029_;
v___y_1006_ = v___y_1032_;
v___y_1007_ = v___y_1033_;
v___y_1008_ = v___y_1034_;
v___y_1009_ = v___y_1034_;
goto v___jp_1001_;
}
else
{
lean_object* v_val_1036_; 
v_val_1036_ = lean_ctor_get(v___x_1035_, 0);
lean_inc(v_val_1036_);
lean_dec_ref_known(v___x_1035_, 1);
v___y_1002_ = v___y_1027_;
v___y_1003_ = v___y_1028_;
v___y_1004_ = v___y_1030_;
v___y_1005_ = v___y_1029_;
v___y_1006_ = v___y_1032_;
v___y_1007_ = v___y_1033_;
v___y_1008_ = v___y_1034_;
v___y_1009_ = v_val_1036_;
goto v___jp_1001_;
}
}
v___jp_1037_:
{
lean_object* v_ref_1045_; lean_object* v___x_1046_; 
v_ref_1045_ = l_Lean_replaceRef(v_ref_956_, v___y_1043_);
v___x_1046_ = l_Lean_Syntax_getPos_x3f(v_ref_1045_, v___y_1041_);
if (lean_obj_tag(v___x_1046_) == 0)
{
lean_object* v___x_1047_; 
v___x_1047_ = lean_unsigned_to_nat(0u);
v___y_1027_ = v___y_1038_;
v___y_1028_ = v___y_1039_;
v___y_1029_ = v___y_1044_;
v___y_1030_ = v___y_1040_;
v___y_1031_ = v_ref_1045_;
v___y_1032_ = v___y_1041_;
v___y_1033_ = v___y_1042_;
v___y_1034_ = v___x_1047_;
goto v___jp_1026_;
}
else
{
lean_object* v_val_1048_; 
v_val_1048_ = lean_ctor_get(v___x_1046_, 0);
lean_inc(v_val_1048_);
lean_dec_ref_known(v___x_1046_, 1);
v___y_1027_ = v___y_1038_;
v___y_1028_ = v___y_1039_;
v___y_1029_ = v___y_1044_;
v___y_1030_ = v___y_1040_;
v___y_1031_ = v_ref_1045_;
v___y_1032_ = v___y_1041_;
v___y_1033_ = v___y_1042_;
v___y_1034_ = v_val_1048_;
goto v___jp_1026_;
}
}
v___jp_1050_:
{
if (v___y_1057_ == 0)
{
v___y_1038_ = v___y_1051_;
v___y_1039_ = v___y_1052_;
v___y_1040_ = v___y_1053_;
v___y_1041_ = v___y_1056_;
v___y_1042_ = v___y_1055_;
v___y_1043_ = v___y_1054_;
v___y_1044_ = v_severity_958_;
goto v___jp_1037_;
}
else
{
v___y_1038_ = v___y_1051_;
v___y_1039_ = v___y_1052_;
v___y_1040_ = v___y_1053_;
v___y_1041_ = v___y_1056_;
v___y_1042_ = v___y_1055_;
v___y_1043_ = v___y_1054_;
v___y_1044_ = v___x_1049_;
goto v___jp_1037_;
}
}
v___jp_1058_:
{
if (v___y_1059_ == 0)
{
lean_object* v_fileName_1060_; lean_object* v_fileMap_1061_; lean_object* v_options_1062_; lean_object* v_ref_1063_; uint8_t v_suppressElabErrors_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___f_1067_; uint8_t v___x_1068_; uint8_t v___x_1069_; 
v_fileName_1060_ = lean_ctor_get(v___y_962_, 0);
v_fileMap_1061_ = lean_ctor_get(v___y_962_, 1);
v_options_1062_ = lean_ctor_get(v___y_962_, 2);
v_ref_1063_ = lean_ctor_get(v___y_962_, 5);
v_suppressElabErrors_1064_ = lean_ctor_get_uint8(v___y_962_, sizeof(void*)*14 + 1);
v___x_1065_ = lean_box(v___y_1059_);
v___x_1066_ = lean_box(v_suppressElabErrors_1064_);
v___f_1067_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1067_, 0, v___x_1065_);
lean_closure_set(v___f_1067_, 1, v___x_1066_);
v___x_1068_ = 1;
v___x_1069_ = l_Lean_instBEqMessageSeverity_beq(v_severity_958_, v___x_1068_);
if (v___x_1069_ == 0)
{
v___y_1051_ = v___f_1067_;
v___y_1052_ = v_fileMap_1061_;
v___y_1053_ = v_fileName_1060_;
v___y_1054_ = v_ref_1063_;
v___y_1055_ = v_suppressElabErrors_1064_;
v___y_1056_ = v___y_1059_;
v___y_1057_ = v___x_1069_;
goto v___jp_1050_;
}
else
{
lean_object* v___x_1070_; uint8_t v___x_1071_; 
v___x_1070_ = l_Lean_warningAsError;
v___x_1071_ = lp_proofwidgets_Lean_Option_get___at___00Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1_spec__5(v_options_1062_, v___x_1070_);
v___y_1051_ = v___f_1067_;
v___y_1052_ = v_fileMap_1061_;
v___y_1053_ = v_fileName_1060_;
v___y_1054_ = v_ref_1063_;
v___y_1055_ = v_suppressElabErrors_1064_;
v___y_1056_ = v___y_1059_;
v___y_1057_ = v___x_1071_;
goto v___jp_1050_;
}
}
else
{
lean_object* v___x_1072_; lean_object* v___x_1073_; 
lean_dec_ref(v_msgData_957_);
v___x_1072_ = lean_box(0);
v___x_1073_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1073_, 0, v___x_1072_);
return v___x_1073_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1___boxed(lean_object* v_ref_1076_, lean_object* v_msgData_1077_, lean_object* v_severity_1078_, lean_object* v_isSilent_1079_, lean_object* v___y_1080_, lean_object* v___y_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_){
_start:
{
uint8_t v_severity_boxed_1085_; uint8_t v_isSilent_boxed_1086_; lean_object* v_res_1087_; 
v_severity_boxed_1085_ = lean_unbox(v_severity_1078_);
v_isSilent_boxed_1086_ = lean_unbox(v_isSilent_1079_);
v_res_1087_ = lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_ref_1076_, v_msgData_1077_, v_severity_boxed_1085_, v_isSilent_boxed_1086_, v___y_1080_, v___y_1081_, v___y_1082_, v___y_1083_);
lean_dec(v___y_1083_);
lean_dec_ref(v___y_1082_);
lean_dec(v___y_1081_);
lean_dec_ref(v___y_1080_);
lean_dec(v_ref_1076_);
return v_res_1087_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0(lean_object* v_msgData_1088_, uint8_t v_severity_1089_, uint8_t v_isSilent_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_){
_start:
{
lean_object* v_ref_1096_; lean_object* v___x_1097_; 
v_ref_1096_ = lean_ctor_get(v___y_1093_, 5);
v___x_1097_ = lp_proofwidgets_Lean_logAt___at___00Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0_spec__1(v_ref_1096_, v_msgData_1088_, v_severity_1089_, v_isSilent_1090_, v___y_1091_, v___y_1092_, v___y_1093_, v___y_1094_);
return v___x_1097_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0___boxed(lean_object* v_msgData_1098_, lean_object* v_severity_1099_, lean_object* v_isSilent_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
uint8_t v_severity_boxed_1106_; uint8_t v_isSilent_boxed_1107_; lean_object* v_res_1108_; 
v_severity_boxed_1106_ = lean_unbox(v_severity_1099_);
v_isSilent_boxed_1107_ = lean_unbox(v_isSilent_1100_);
v_res_1108_ = lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0(v_msgData_1098_, v_severity_boxed_1106_, v_isSilent_boxed_1107_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec(v___y_1102_);
lean_dec_ref(v___y_1101_);
return v_res_1108_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0(lean_object* v_msgData_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
uint8_t v___x_1115_; uint8_t v___x_1116_; lean_object* v___x_1117_; 
v___x_1115_ = 1;
v___x_1116_ = 0;
v___x_1117_ = lp_proofwidgets_Lean_log___at___00Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0_spec__0(v_msgData_1109_, v___x_1115_, v___x_1116_, v___y_1110_, v___y_1111_, v___y_1112_, v___y_1113_);
return v___x_1117_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0___boxed(lean_object* v_msgData_1118_, lean_object* v___y_1119_, lean_object* v___y_1120_, lean_object* v___y_1121_, lean_object* v___y_1122_, lean_object* v___y_1123_){
_start:
{
lean_object* v_res_1124_; 
v_res_1124_ = lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0(v_msgData_1118_, v___y_1119_, v___y_1120_, v___y_1121_, v___y_1122_);
lean_dec(v___y_1122_);
lean_dec_ref(v___y_1121_);
lean_dec(v___y_1120_);
lean_dec_ref(v___y_1119_);
return v_res_1124_;
}
}
static uint64_t _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1131_; uint64_t v___x_1132_; 
v___x_1131_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1132_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_1131_);
return v___x_1132_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
uint64_t v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; 
v___x_1133_ = lean_uint64_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1134_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1135_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1135_, 0, v___x_1134_);
lean_ctor_set_uint64(v___x_1135_, sizeof(void*)*1, v___x_1133_);
return v___x_1135_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1136_; 
v___x_1136_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1136_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1137_; lean_object* v___x_1138_; 
v___x_1137_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1138_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1138_, 0, v___x_1137_);
return v___x_1138_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; 
v___x_1139_ = lean_unsigned_to_nat(32u);
v___x_1140_ = lean_mk_empty_array_with_capacity(v___x_1139_);
v___x_1141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1141_, 0, v___x_1140_);
return v___x_1141_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1142_; lean_object* v___x_1143_; 
v___x_1142_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1143_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1143_, 0, v___x_1142_);
lean_ctor_set(v___x_1143_, 1, v___x_1142_);
lean_ctor_set(v___x_1143_, 2, v___x_1142_);
lean_ctor_set(v___x_1143_, 3, v___x_1142_);
lean_ctor_set(v___x_1143_, 4, v___x_1142_);
lean_ctor_set(v___x_1143_, 5, v___x_1142_);
return v___x_1143_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1144_; lean_object* v___x_1145_; 
v___x_1144_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1145_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1145_, 0, v___x_1144_);
lean_ctor_set(v___x_1145_, 1, v___x_1144_);
lean_ctor_set(v___x_1145_, 2, v___x_1144_);
lean_ctor_set(v___x_1145_, 3, v___x_1144_);
lean_ctor_set(v___x_1145_, 4, v___x_1144_);
return v___x_1145_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1149_; lean_object* v___x_1150_; 
v___x_1149_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__9_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1150_ = l_Lean_MessageData_ofFormat(v___x_1149_);
return v___x_1150_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(lean_object* v___x_1152_, lean_object* v___x_1153_, lean_object* v___x_1154_, lean_object* v_decl_1155_, lean_object* v_x_1156_, uint8_t v_x_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_){
_start:
{
uint8_t v___x_1161_; uint8_t v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; size_t v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___y_1181_; lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1161_ = 0;
v___x_1162_ = 1;
v___x_1163_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1164_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__4_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1165_ = lean_unsigned_to_nat(32u);
v___x_1166_ = lean_mk_empty_array_with_capacity(v___x_1165_);
v___x_1167_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1168_ = ((size_t)5ULL);
lean_inc_n(v___x_1152_, 6);
v___x_1169_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1169_, 0, v___x_1167_);
lean_ctor_set(v___x_1169_, 1, v___x_1166_);
lean_ctor_set(v___x_1169_, 2, v___x_1152_);
lean_ctor_set(v___x_1169_, 3, v___x_1152_);
lean_ctor_set_usize(v___x_1169_, 4, v___x_1168_);
v___x_1170_ = lean_box(1);
lean_inc_ref(v___x_1169_);
v___x_1171_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1171_, 0, v___x_1164_);
lean_ctor_set(v___x_1171_, 1, v___x_1169_);
lean_ctor_set(v___x_1171_, 2, v___x_1170_);
v___x_1172_ = lean_mk_empty_array_with_capacity(v___x_1152_);
v___x_1173_ = lean_box(0);
lean_inc(v___x_1153_);
v___x_1174_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1174_, 0, v___x_1163_);
lean_ctor_set(v___x_1174_, 1, v___x_1153_);
lean_ctor_set(v___x_1174_, 2, v___x_1171_);
lean_ctor_set(v___x_1174_, 3, v___x_1172_);
lean_ctor_set(v___x_1174_, 4, v___x_1173_);
lean_ctor_set(v___x_1174_, 5, v___x_1152_);
lean_ctor_set(v___x_1174_, 6, v___x_1173_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*7, v___x_1161_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*7 + 1, v___x_1161_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*7 + 2, v___x_1161_);
lean_ctor_set_uint8(v___x_1174_, sizeof(void*)*7 + 3, v___x_1162_);
v___x_1175_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1175_, 0, v___x_1152_);
lean_ctor_set(v___x_1175_, 1, v___x_1152_);
lean_ctor_set(v___x_1175_, 2, v___x_1152_);
lean_ctor_set(v___x_1175_, 3, v___x_1152_);
lean_ctor_set(v___x_1175_, 4, v___x_1164_);
lean_ctor_set(v___x_1175_, 5, v___x_1164_);
lean_ctor_set(v___x_1175_, 6, v___x_1164_);
lean_ctor_set(v___x_1175_, 7, v___x_1164_);
lean_ctor_set(v___x_1175_, 8, v___x_1164_);
lean_ctor_set(v___x_1175_, 9, v___x_1164_);
v___x_1176_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__6_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1177_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__7_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1178_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1175_);
lean_ctor_set(v___x_1178_, 1, v___x_1176_);
lean_ctor_set(v___x_1178_, 2, v___x_1153_);
lean_ctor_set(v___x_1178_, 3, v___x_1169_);
lean_ctor_set(v___x_1178_, 4, v___x_1177_);
v___x_1179_ = lean_st_mk_ref(v___x_1178_);
v___x_1191_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__10_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1192_ = lp_proofwidgets_Lean_logWarning___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__0(v___x_1191_, v___x_1174_, v___x_1179_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1192_) == 0)
{
lean_object* v___x_1194_; uint8_t v_isShared_1195_; uint8_t v_isSharedCheck_1235_; 
v_isSharedCheck_1235_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1235_ == 0)
{
lean_object* v_unused_1236_; 
v_unused_1236_ = lean_ctor_get(v___x_1192_, 0);
lean_dec(v_unused_1236_);
v___x_1194_ = v___x_1192_;
v_isShared_1195_ = v_isSharedCheck_1235_;
goto v_resetjp_1193_;
}
else
{
lean_dec(v___x_1192_);
v___x_1194_ = lean_box(0);
v_isShared_1195_ = v_isSharedCheck_1235_;
goto v_resetjp_1193_;
}
v_resetjp_1193_:
{
lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v___x_1196_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_cancellableSuffix));
lean_inc(v_decl_1155_);
v___x_1197_ = l_Lean_Name_append(v_decl_1155_, v___x_1196_);
v___x_1198_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__11_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1199_ = l_Lean_Name_mkStr2(v___x_1154_, v___x_1198_);
v___x_1200_ = lean_box(0);
v___x_1201_ = l_Lean_mkConst(v_decl_1155_, v___x_1200_);
v___x_1202_ = lean_unsigned_to_nat(1u);
v___x_1203_ = lean_mk_empty_array_with_capacity(v___x_1202_);
v___x_1204_ = lean_array_push(v___x_1203_, v___x_1201_);
v___x_1205_ = l_Lean_Meta_mkAppM(v___x_1199_, v___x_1204_, v___x_1174_, v___x_1179_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1205_) == 0)
{
lean_object* v_a_1206_; lean_object* v___x_1207_; 
v_a_1206_ = lean_ctor_get(v___x_1205_, 0);
lean_inc_n(v_a_1206_, 2);
lean_dec_ref_known(v___x_1205_, 1);
lean_inc(v___y_1159_);
lean_inc_ref(v___y_1158_);
lean_inc(v___x_1179_);
v___x_1207_ = lean_infer_type(v_a_1206_, v___x_1174_, v___x_1179_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1207_) == 0)
{
lean_object* v_a_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; uint8_t v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1215_; 
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
lean_inc(v_a_1208_);
lean_dec_ref_known(v___x_1207_, 1);
lean_inc_n(v___x_1197_, 2);
v___x_1209_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1209_, 0, v___x_1197_);
lean_ctor_set(v___x_1209_, 1, v___x_1200_);
lean_ctor_set(v___x_1209_, 2, v_a_1208_);
v___x_1210_ = lean_box(0);
v___x_1211_ = 1;
v___x_1212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1212_, 0, v___x_1197_);
lean_ctor_set(v___x_1212_, 1, v___x_1200_);
v___x_1213_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_1213_, 0, v___x_1209_);
lean_ctor_set(v___x_1213_, 1, v_a_1206_);
lean_ctor_set(v___x_1213_, 2, v___x_1210_);
lean_ctor_set(v___x_1213_, 3, v___x_1212_);
lean_ctor_set_uint8(v___x_1213_, sizeof(void*)*4, v___x_1211_);
if (v_isShared_1195_ == 0)
{
lean_ctor_set_tag(v___x_1194_, 1);
lean_ctor_set(v___x_1194_, 0, v___x_1213_);
v___x_1215_ = v___x_1194_;
goto v_reusejp_1214_;
}
else
{
lean_object* v_reuseFailAlloc_1218_; 
v_reuseFailAlloc_1218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1218_, 0, v___x_1213_);
v___x_1215_ = v_reuseFailAlloc_1218_;
goto v_reusejp_1214_;
}
v_reusejp_1214_:
{
lean_object* v___x_1216_; 
v___x_1216_ = l_Lean_addAndCompile(v___x_1215_, v___x_1162_, v___x_1161_, v___y_1158_, v___y_1159_);
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_object* v___x_1217_; 
lean_dec_ref_known(v___x_1216_, 1);
v___x_1217_ = l_Lean_Server_registerRpcProcedure(v___x_1197_, v___y_1158_, v___y_1159_);
v___y_1181_ = v___x_1217_;
goto v___jp_1180_;
}
else
{
lean_dec(v___x_1197_);
v___y_1181_ = v___x_1216_;
goto v___jp_1180_;
}
}
}
else
{
lean_object* v_a_1219_; lean_object* v___x_1221_; uint8_t v_isShared_1222_; uint8_t v_isSharedCheck_1226_; 
lean_dec(v_a_1206_);
lean_dec(v___x_1197_);
lean_del_object(v___x_1194_);
lean_dec(v___x_1179_);
v_a_1219_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1226_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1226_ == 0)
{
v___x_1221_ = v___x_1207_;
v_isShared_1222_ = v_isSharedCheck_1226_;
goto v_resetjp_1220_;
}
else
{
lean_inc(v_a_1219_);
lean_dec(v___x_1207_);
v___x_1221_ = lean_box(0);
v_isShared_1222_ = v_isSharedCheck_1226_;
goto v_resetjp_1220_;
}
v_resetjp_1220_:
{
lean_object* v___x_1224_; 
if (v_isShared_1222_ == 0)
{
v___x_1224_ = v___x_1221_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v_a_1219_);
v___x_1224_ = v_reuseFailAlloc_1225_;
goto v_reusejp_1223_;
}
v_reusejp_1223_:
{
return v___x_1224_;
}
}
}
}
else
{
lean_object* v_a_1227_; lean_object* v___x_1229_; uint8_t v_isShared_1230_; uint8_t v_isSharedCheck_1234_; 
lean_dec(v___x_1197_);
lean_del_object(v___x_1194_);
lean_dec(v___x_1179_);
lean_dec_ref_known(v___x_1174_, 7);
v_a_1227_ = lean_ctor_get(v___x_1205_, 0);
v_isSharedCheck_1234_ = !lean_is_exclusive(v___x_1205_);
if (v_isSharedCheck_1234_ == 0)
{
v___x_1229_ = v___x_1205_;
v_isShared_1230_ = v_isSharedCheck_1234_;
goto v_resetjp_1228_;
}
else
{
lean_inc(v_a_1227_);
lean_dec(v___x_1205_);
v___x_1229_ = lean_box(0);
v_isShared_1230_ = v_isSharedCheck_1234_;
goto v_resetjp_1228_;
}
v_resetjp_1228_:
{
lean_object* v___x_1232_; 
if (v_isShared_1230_ == 0)
{
v___x_1232_ = v___x_1229_;
goto v_reusejp_1231_;
}
else
{
lean_object* v_reuseFailAlloc_1233_; 
v_reuseFailAlloc_1233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1233_, 0, v_a_1227_);
v___x_1232_ = v_reuseFailAlloc_1233_;
goto v_reusejp_1231_;
}
v_reusejp_1231_:
{
return v___x_1232_;
}
}
}
}
}
else
{
lean_dec_ref_known(v___x_1174_, 7);
lean_dec(v_decl_1155_);
lean_dec_ref(v___x_1154_);
v___y_1181_ = v___x_1192_;
goto v___jp_1180_;
}
v___jp_1180_:
{
if (lean_obj_tag(v___y_1181_) == 0)
{
lean_object* v_a_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1190_; 
v_a_1182_ = lean_ctor_get(v___y_1181_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___y_1181_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1184_ = v___y_1181_;
v_isShared_1185_ = v_isSharedCheck_1190_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_a_1182_);
lean_dec(v___y_1181_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1190_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
lean_object* v___x_1186_; lean_object* v___x_1188_; 
v___x_1186_ = lean_st_ref_get(v___x_1179_);
lean_dec(v___x_1179_);
lean_dec(v___x_1186_);
if (v_isShared_1185_ == 0)
{
v___x_1188_ = v___x_1184_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1182_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
else
{
lean_dec(v___x_1179_);
return v___y_1181_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object* v___x_1237_, lean_object* v___x_1238_, lean_object* v___x_1239_, lean_object* v_decl_1240_, lean_object* v_x_1241_, lean_object* v_x_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_){
_start:
{
uint8_t v_x_5453__boxed_1246_; lean_object* v_res_1247_; 
v_x_5453__boxed_1246_ = lean_unbox(v_x_1242_);
v_res_1247_ = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(v___x_1237_, v___x_1238_, v___x_1239_, v_decl_1240_, v_x_1241_, v_x_5453__boxed_1246_, v___y_1243_, v___y_1244_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec(v_x_1241_);
return v_res_1247_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1248_; 
v___x_1248_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1248_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1249_; lean_object* v___x_1250_; 
v___x_1249_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__0);
v___x_1250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1250_, 0, v___x_1249_);
return v___x_1250_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; 
v___x_1251_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_1252_ = lean_unsigned_to_nat(0u);
v___x_1253_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1252_);
lean_ctor_set(v___x_1253_, 1, v___x_1252_);
lean_ctor_set(v___x_1253_, 2, v___x_1252_);
lean_ctor_set(v___x_1253_, 3, v___x_1252_);
lean_ctor_set(v___x_1253_, 4, v___x_1251_);
lean_ctor_set(v___x_1253_, 5, v___x_1251_);
lean_ctor_set(v___x_1253_, 6, v___x_1251_);
lean_ctor_set(v___x_1253_, 7, v___x_1251_);
lean_ctor_set(v___x_1253_, 8, v___x_1251_);
lean_ctor_set(v___x_1253_, 9, v___x_1251_);
return v___x_1253_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3(void){
_start:
{
size_t v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v___x_1254_ = ((size_t)5ULL);
v___x_1255_ = lean_unsigned_to_nat(0u);
v___x_1256_ = lean_unsigned_to_nat(32u);
v___x_1257_ = lean_mk_empty_array_with_capacity(v___x_1256_);
v___x_1258_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__0___closed__5_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1259_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1259_, 0, v___x_1258_);
lean_ctor_set(v___x_1259_, 1, v___x_1257_);
lean_ctor_set(v___x_1259_, 2, v___x_1255_);
lean_ctor_set(v___x_1259_, 3, v___x_1255_);
lean_ctor_set_usize(v___x_1259_, 4, v___x_1254_);
return v___x_1259_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4(void){
_start:
{
lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; 
v___x_1260_ = lean_box(1);
v___x_1261_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__3);
v___x_1262_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__1);
v___x_1263_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
lean_ctor_set(v___x_1263_, 1, v___x_1261_);
lean_ctor_set(v___x_1263_, 2, v___x_1260_);
return v___x_1263_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2(lean_object* v_msgData_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_){
_start:
{
lean_object* v___x_1268_; lean_object* v_env_1269_; lean_object* v_options_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; 
v___x_1268_ = lean_st_ref_get(v___y_1266_);
v_env_1269_ = lean_ctor_get(v___x_1268_, 0);
lean_inc_ref(v_env_1269_);
lean_dec(v___x_1268_);
v_options_1270_ = lean_ctor_get(v___y_1265_, 2);
v___x_1271_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__2);
v___x_1272_ = lean_obj_once(&lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4, &lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4_once, _init_lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___closed__4);
lean_inc_ref(v_options_1270_);
v___x_1273_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1273_, 0, v_env_1269_);
lean_ctor_set(v___x_1273_, 1, v___x_1271_);
lean_ctor_set(v___x_1273_, 2, v___x_1272_);
lean_ctor_set(v___x_1273_, 3, v_options_1270_);
v___x_1274_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1273_);
lean_ctor_set(v___x_1274_, 1, v_msgData_1264_);
v___x_1275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1274_);
return v___x_1275_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2___boxed(lean_object* v_msgData_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
lean_object* v_res_1280_; 
v_res_1280_ = lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2(v_msgData_1276_, v___y_1277_, v___y_1278_);
lean_dec(v___y_1278_);
lean_dec_ref(v___y_1277_);
return v_res_1280_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg(lean_object* v_msg_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_){
_start:
{
lean_object* v_ref_1285_; lean_object* v___x_1286_; lean_object* v_a_1287_; lean_object* v___x_1289_; uint8_t v_isShared_1290_; uint8_t v_isSharedCheck_1295_; 
v_ref_1285_ = lean_ctor_get(v___y_1282_, 5);
v___x_1286_ = lp_proofwidgets_Lean_addMessageContextPartial___at___00Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1_spec__2(v_msg_1281_, v___y_1282_, v___y_1283_);
v_a_1287_ = lean_ctor_get(v___x_1286_, 0);
v_isSharedCheck_1295_ = !lean_is_exclusive(v___x_1286_);
if (v_isSharedCheck_1295_ == 0)
{
v___x_1289_ = v___x_1286_;
v_isShared_1290_ = v_isSharedCheck_1295_;
goto v_resetjp_1288_;
}
else
{
lean_inc(v_a_1287_);
lean_dec(v___x_1286_);
v___x_1289_ = lean_box(0);
v_isShared_1290_ = v_isSharedCheck_1295_;
goto v_resetjp_1288_;
}
v_resetjp_1288_:
{
lean_object* v___x_1291_; lean_object* v___x_1293_; 
lean_inc(v_ref_1285_);
v___x_1291_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1291_, 0, v_ref_1285_);
lean_ctor_set(v___x_1291_, 1, v_a_1287_);
if (v_isShared_1290_ == 0)
{
lean_ctor_set_tag(v___x_1289_, 1);
lean_ctor_set(v___x_1289_, 0, v___x_1291_);
v___x_1293_ = v___x_1289_;
goto v_reusejp_1292_;
}
else
{
lean_object* v_reuseFailAlloc_1294_; 
v_reuseFailAlloc_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1294_, 0, v___x_1291_);
v___x_1293_ = v_reuseFailAlloc_1294_;
goto v_reusejp_1292_;
}
v_reusejp_1292_:
{
return v___x_1293_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg___boxed(lean_object* v_msg_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_){
_start:
{
lean_object* v_res_1300_; 
v_res_1300_ = lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg(v_msg_1296_, v___y_1297_, v___y_1298_);
lean_dec(v___y_1298_);
lean_dec_ref(v___y_1297_);
return v_res_1300_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1302_; lean_object* v___x_1303_; 
v___x_1302_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__0_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1303_ = l_Lean_stringToMessageData(v___x_1302_);
return v___x_1303_;
}
}
static lean_object* _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_1305_; lean_object* v___x_1306_; 
v___x_1305_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__2_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1306_ = l_Lean_stringToMessageData(v___x_1305_);
return v___x_1306_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(lean_object* v___x_1307_, lean_object* v_decl_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_){
_start:
{
lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; lean_object* v___x_1316_; lean_object* v___x_1317_; 
v___x_1312_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1313_ = l_Lean_MessageData_ofName(v___x_1307_);
v___x_1314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1314_, 0, v___x_1312_);
lean_ctor_set(v___x_1314_, 1, v___x_1313_);
v___x_1315_ = lean_obj_once(&lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_, &lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__once, _init_lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1___closed__3_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_);
v___x_1316_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1316_, 0, v___x_1314_);
lean_ctor_set(v___x_1316_, 1, v___x_1315_);
v___x_1317_ = lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg(v___x_1316_, v___y_1309_, v___y_1310_);
return v___x_1317_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object* v___x_1318_, lean_object* v_decl_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
lean_object* v_res_1323_; 
v_res_1323_ = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___lam__1_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(v___x_1318_, v_decl_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
lean_dec(v_decl_1319_);
return v_res_1323_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_1389_; lean_object* v___x_1390_; 
v___x_1389_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn___closed__25_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_));
v___x_1390_ = l_Lean_registerBuiltinAttribute(v___x_1389_);
return v___x_1390_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2____boxed(lean_object* v_a_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_();
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1(lean_object* v_00_u03b1_1393_, lean_object* v_msg_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_){
_start:
{
lean_object* v___x_1398_; 
v___x_1398_ = lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___redArg(v_msg_1394_, v___y_1395_, v___y_1396_);
return v___x_1398_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1___boxed(lean_object* v_00_u03b1_1399_, lean_object* v_msg_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_, lean_object* v___y_1403_){
_start:
{
lean_object* v_res_1404_; 
v_res_1404_ = lp_proofwidgets_Lean_throwError___at___00__private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2__spec__1(v_00_u03b1_1399_, v_msg_1400_, v___y_1401_, v___y_1402_);
lean_dec(v___y_1402_);
lean_dec_ref(v___y_1401_);
return v_res_1404_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Compat(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Cancellable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Compat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_Rpc_RequestHandling(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Cancellable(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_Rpc_RequestHandling(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_2133826679____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_proofwidgets_ProofWidgets_runningRequests = lean_io_result_get_value(res);
lean_mark_persistent(lp_proofwidgets_ProofWidgets_runningRequests);
lean_dec_ref(res);
lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped = _init_lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_cancelRequest___rpc__wrapped);
lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped = _init_lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_checkRequest___rpc__wrapped);
res = lp_proofwidgets___private_ProofWidgets_Cancellable_0__ProofWidgets_initFn_00___x40_ProofWidgets_Cancellable_680169540____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Server_Rpc_RequestHandling(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Compat(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Cancellable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_Rpc_RequestHandling(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Compat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Cancellable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Cancellable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Cancellable(builtin);
}
#ifdef __cplusplus
}
#endif
