// Lean compiler output
// Module: ProofWidgets.Component.RefreshComponent
// Imports: public import Init public meta import Init public import ProofWidgets.Component.Panel.Basic public import ProofWidgets.Data.Html public import ProofWidgets.Util
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
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* lean_io_basemutex_lock(lean_object*);
lean_object* lean_io_basemutex_unlock(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps;
lean_object* lean_mk_thunk(lean_object*);
lean_object* lean_io_promise_new();
lean_object* lean_io_promise_result_opt(lean_object*);
lean_object* l_Std_Mutex_new___redArg(lean_object*);
lean_object* l_IO_CancelToken_new();
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Server_WithRpcRef_mk___redArg(lean_object*);
uint64_t lean_string_hash(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcEncode___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
extern lean_object* l_Lean_interruptExceptionId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* lean_thunk_pure(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_swap(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_io_promise_resolve(lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_InteractiveMessage;
lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_EIO_catchExceptions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_BaseIO_asTask___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Json_getNat_x3f(lean_object*);
lean_object* l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcDecode___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_thunk_get_own(lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCostly___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
lean_object* l_IO_sleep(uint32_t);
uint8_t l_Lean_Server_RequestCancellationToken_wasCancelledByCancelRequest(lean_object*);
lean_object* l_IO_CancelToken_set(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_instInhabitedHtml_default;
lean_object* lean_task_pure(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "html"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "idx"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__spec__0(lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_enc_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_enc_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml___closed__2_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "ProofWidgets"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "RefreshComponent"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "RefreshRef"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 189, 138, 64, 141, 73, 78)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(32, 13, 196, 154, 160, 200, 148, 11)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instTypeNameRefreshRef = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__3_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "state"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "oldIdx"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_enc_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_enc_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "awaitRefresh"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 189, 138, 64, 141, 73, 78)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(123, 11, 193, 53, 247, 206, 238, 85)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__2_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped;
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "IO"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "CancelToken"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(2, 76, 19, 202, 4, 69, 238, 60)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(145, 130, 218, 72, 57, 23, 234, 205)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instTypeNameCancelToken__proofWidgets = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__2_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3__value;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "cancelTk"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_(lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31__value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_ = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31__value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__0_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__1_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__1_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__2_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps___closed__2_value;
static const lean_ctor_object lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__0 = (const lean_object*)&lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__0_value;
static const lean_ctor_object lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__1 = (const lean_object*)&lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "monitor"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(181, 180, 15, 10, 207, 227, 48, 81)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9__value),LEAN_SCALAR_PTR_LITERAL(173, 174, 189, 138, 64, 141, 73, 78)}};
static const lean_ctor_object lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value_aux_1),((lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(63, 132, 179, 212, 11, 143, 84, 85)}};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1_value;
static const lean_closure_object lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__2_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1553, .m_capacity = 1553, .m_length = 1552, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import o from\"react\";import{useRpcSession as a,EnvPosContext as i,useAsyncPersistent as s,mapRpcError as l,importWidgetModule as c,RpcPtr as m}from\"@leanprover/infoview\";async function f(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,s]=i.element,l={};for(const[e,t]of r)l[e]=t;const c=await Promise.all(s.map(async e=>await f(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===c.length\?n.createElement(e,l):n.createElement(e,l,c)}if(\"component\"in i){const[e,t,r,s]=i.component,l=await Promise.all(s.map(async e=>await f(o,a,e))),m={...r,pos:a},u=await c(o,a,e);if(!(t in u))throw new Error(`Module '${e}' does not export '${t}'`);return 0===l.length\?n.createElement(u[t],m):n.createElement(u[t],m,l)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function u({html:o}){const c=a(),m=n.useContext(i),u=s(()=>f(c,m,o),[c,m,o]);return\"resolved\"===u.state\?u.value:\"rejected\"===u.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",l(u.error).message]}):t(r,{})}function d(e){const r=a(),[n,i]=o.useState({text:\"\"});return o.useEffect(()=>{let t=!1;const n=new AbortController;return r.call(\"ProofWidgets.RefreshComponent.monitor\",e,{abortSignal:n.signal}),async function n(o){const a=await r.call(\"ProofWidgets.RefreshComponent.awaitRefresh\",{oldIdx:o,state:e.state});if(!t&&a)return i(a.html),n(a.idx)}(0),()=>{t=!0,n.abort()}},[m.toKey(e.state)]),t(u,{html:n})}export{d as default};"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent___closed__0_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2;
static const lean_string_object lp_proofwidgets_ProofWidgets_RefreshComponent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_RefreshComponent___closed__3_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_update(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_update___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0___boxed(lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_mkRefreshComponent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponent___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "This component was cancelled"};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__0_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "\n            An error occurred in the mkRefreshComponentM thread:\n            "};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__3_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__4_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(lean_object* v_t_1_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_thunk_get_own(v_t_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg___boxed(lean_object* v_t_4_, lean_object* v_a_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(v_t_4_);
lean_dec_ref(v_t_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk(lean_object* v_00_u03b1_7_, lean_object* v_t_8_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(v_t_8_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___boxed(lean_object* v_00_u03b1_11_, lean_object* v_t_12_, lean_object* v_a_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk(v_00_u03b1_11_, v_t_12_);
lean_dec_ref(v_t_12_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(lean_object* v_j_15_, lean_object* v_k_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = l_Lean_Json_getObjValD(v_j_15_, v_k_16_);
v___x_18_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0___boxed(lean_object* v_j_19_, lean_object* v_k_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_j_19_, v_k_20_);
lean_dec_ref(v_k_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_(lean_object* v_json_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v_a_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v_a_30_; lean_object* v___x_32_; uint8_t v_isShared_33_; uint8_t v_isSharedCheck_38_; 
v___x_25_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_));
lean_inc(v_json_24_);
v___x_26_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_24_, v___x_25_);
v_a_27_ = lean_ctor_get(v___x_26_, 0);
lean_inc(v_a_27_);
lean_dec_ref(v___x_26_);
v___x_28_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_));
v___x_29_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_24_, v___x_28_);
v_a_30_ = lean_ctor_get(v___x_29_, 0);
v_isSharedCheck_38_ = !lean_is_exclusive(v___x_29_);
if (v_isSharedCheck_38_ == 0)
{
v___x_32_ = v___x_29_;
v_isShared_33_ = v_isSharedCheck_38_;
goto v_resetjp_31_;
}
else
{
lean_inc(v_a_30_);
lean_dec(v___x_29_);
v___x_32_ = lean_box(0);
v_isShared_33_ = v_isSharedCheck_38_;
goto v_resetjp_31_;
}
v_resetjp_31_:
{
lean_object* v___x_34_; lean_object* v___x_36_; 
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v_a_27_);
lean_ctor_set(v___x_34_, 1, v_a_30_);
if (v_isShared_33_ == 0)
{
lean_ctor_set(v___x_32_, 0, v___x_34_);
v___x_36_ = v___x_32_;
goto v_reusejp_35_;
}
else
{
lean_object* v_reuseFailAlloc_37_; 
v_reuseFailAlloc_37_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_37_, 0, v___x_34_);
v___x_36_ = v_reuseFailAlloc_37_;
goto v_reusejp_35_;
}
v_reusejp_35_:
{
return v___x_36_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__spec__0(lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
if (lean_obj_tag(v_a_41_) == 0)
{
lean_object* v___x_43_; 
v___x_43_ = lean_array_to_list(v_a_42_);
return v___x_43_;
}
else
{
lean_object* v_head_44_; lean_object* v_tail_45_; lean_object* v___x_46_; 
v_head_44_ = lean_ctor_get(v_a_41_, 0);
lean_inc(v_head_44_);
v_tail_45_ = lean_ctor_get(v_a_41_, 1);
lean_inc(v_tail_45_);
lean_dec_ref_known(v_a_41_, 2);
v___x_46_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_42_, v_head_44_);
v_a_41_ = v_tail_45_;
v_a_42_ = v___x_46_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_(lean_object* v_x_50_){
_start:
{
lean_object* v_html_51_; lean_object* v_idx_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_70_; 
v_html_51_ = lean_ctor_get(v_x_50_, 0);
v_idx_52_ = lean_ctor_get(v_x_50_, 1);
v_isSharedCheck_70_ = !lean_is_exclusive(v_x_50_);
if (v_isSharedCheck_70_ == 0)
{
v___x_54_ = v_x_50_;
v_isShared_55_ = v_isSharedCheck_70_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_idx_52_);
lean_inc(v_html_51_);
lean_dec(v_x_50_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_70_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_56_; lean_object* v___x_58_; 
v___x_56_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_));
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v_html_51_);
lean_ctor_set(v___x_54_, 0, v___x_56_);
v___x_58_ = v___x_54_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_69_; 
v_reuseFailAlloc_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_69_, 0, v___x_56_);
lean_ctor_set(v_reuseFailAlloc_69_, 1, v_html_51_);
v___x_58_ = v_reuseFailAlloc_69_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_59_ = lean_box(0);
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_58_);
lean_ctor_set(v___x_60_, 1, v___x_59_);
v___x_61_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_));
v___x_62_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_62_, 0, v___x_61_);
lean_ctor_set(v___x_62_, 1, v_idx_52_);
v___x_63_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
lean_ctor_set(v___x_63_, 1, v___x_59_);
v___x_64_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_64_, 0, v___x_63_);
lean_ctor_set(v___x_64_, 1, v___x_59_);
v___x_65_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_60_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
v___x_66_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_));
v___x_67_ = lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__spec__0(v___x_65_, v___x_66_);
v___x_68_ = l_Lean_Json_mkObj(v___x_67_);
lean_dec(v___x_67_);
return v___x_68_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_enc_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(lean_object* v_a_73_, lean_object* v_a_74_){
_start:
{
lean_object* v_html_75_; lean_object* v_idx_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_96_; 
v_html_75_ = lean_ctor_get(v_a_73_, 0);
v_idx_76_ = lean_ctor_get(v_a_73_, 1);
v_isSharedCheck_96_ = !lean_is_exclusive(v_a_73_);
if (v_isSharedCheck_96_ == 0)
{
v___x_78_ = v_a_73_;
v_isShared_79_ = v_isSharedCheck_96_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_idx_76_);
lean_inc(v_html_75_);
lean_dec(v_a_73_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_96_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_80_; lean_object* v_fst_81_; lean_object* v_snd_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_95_; 
v___x_80_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_html_75_, v_a_74_);
v_fst_81_ = lean_ctor_get(v___x_80_, 0);
v_snd_82_ = lean_ctor_get(v___x_80_, 1);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_80_);
if (v_isSharedCheck_95_ == 0)
{
v___x_84_ = v___x_80_;
v_isShared_85_ = v_isSharedCheck_95_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_snd_82_);
lean_inc(v_fst_81_);
lean_dec(v___x_80_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_95_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_89_; 
v___x_86_ = l_Lean_JsonNumber_fromNat(v_idx_76_);
v___x_87_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
if (v_isShared_79_ == 0)
{
lean_ctor_set(v___x_78_, 1, v___x_87_);
lean_ctor_set(v___x_78_, 0, v_fst_81_);
v___x_89_ = v___x_78_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v_fst_81_);
lean_ctor_set(v_reuseFailAlloc_94_, 1, v___x_87_);
v___x_89_ = v_reuseFailAlloc_94_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
lean_object* v___x_90_; lean_object* v___x_92_; 
v___x_90_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_(v___x_89_);
if (v_isShared_85_ == 0)
{
lean_ctor_set(v___x_84_, 0, v___x_90_);
v___x_92_ = v___x_84_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v___x_90_);
lean_ctor_set(v_reuseFailAlloc_93_, 1, v_snd_82_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___redArg(lean_object* v_x_97_){
_start:
{
lean_inc_ref(v_x_97_);
return v_x_97_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object* v_x_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___redArg(v_x_98_);
lean_dec_ref(v_x_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0(lean_object* v_00_u03b1_100_, lean_object* v_x_101_, lean_object* v___y_102_){
_start:
{
lean_inc_ref(v_x_101_);
return v_x_101_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0___boxed(lean_object* v_00_u03b1_103_, lean_object* v_x_104_, lean_object* v___y_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_proofwidgets_MonadExcept_ofExcept___at___00ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1__spec__0(v_00_u03b1_103_, v_x_104_, v___y_105_);
lean_dec_ref(v___y_105_);
lean_dec_ref(v_x_104_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(lean_object* v_j_107_, lean_object* v_a_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12_(v_j_107_);
if (lean_obj_tag(v___x_109_) == 0)
{
lean_object* v_a_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_117_; 
v_a_110_ = lean_ctor_get(v___x_109_, 0);
v_isSharedCheck_117_ = !lean_is_exclusive(v___x_109_);
if (v_isSharedCheck_117_ == 0)
{
v___x_112_ = v___x_109_;
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_a_110_);
lean_dec(v___x_109_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_117_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_115_; 
if (v_isShared_113_ == 0)
{
v___x_115_ = v___x_112_;
goto v_reusejp_114_;
}
else
{
lean_object* v_reuseFailAlloc_116_; 
v_reuseFailAlloc_116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_116_, 0, v_a_110_);
v___x_115_ = v_reuseFailAlloc_116_;
goto v_reusejp_114_;
}
v_reusejp_114_:
{
return v___x_115_;
}
}
}
else
{
lean_object* v_a_118_; lean_object* v_html_119_; lean_object* v_idx_120_; lean_object* v___x_122_; uint8_t v_isShared_123_; uint8_t v_isSharedCheck_154_; 
v_a_118_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_a_118_);
lean_dec_ref_known(v___x_109_, 1);
v_html_119_ = lean_ctor_get(v_a_118_, 0);
v_idx_120_ = lean_ctor_get(v_a_118_, 1);
v_isSharedCheck_154_ = !lean_is_exclusive(v_a_118_);
if (v_isSharedCheck_154_ == 0)
{
v___x_122_ = v_a_118_;
v_isShared_123_ = v_isSharedCheck_154_;
goto v_resetjp_121_;
}
else
{
lean_inc(v_idx_120_);
lean_inc(v_html_119_);
lean_dec(v_a_118_);
v___x_122_ = lean_box(0);
v_isShared_123_ = v_isSharedCheck_154_;
goto v_resetjp_121_;
}
v_resetjp_121_:
{
lean_object* v___x_124_; 
v___x_124_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_dec_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_html_119_, v_a_108_);
if (lean_obj_tag(v___x_124_) == 0)
{
lean_object* v_a_125_; lean_object* v___x_127_; uint8_t v_isShared_128_; uint8_t v_isSharedCheck_132_; 
lean_del_object(v___x_122_);
lean_dec(v_idx_120_);
v_a_125_ = lean_ctor_get(v___x_124_, 0);
v_isSharedCheck_132_ = !lean_is_exclusive(v___x_124_);
if (v_isSharedCheck_132_ == 0)
{
v___x_127_ = v___x_124_;
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
else
{
lean_inc(v_a_125_);
lean_dec(v___x_124_);
v___x_127_ = lean_box(0);
v_isShared_128_ = v_isSharedCheck_132_;
goto v_resetjp_126_;
}
v_resetjp_126_:
{
lean_object* v___x_130_; 
if (v_isShared_128_ == 0)
{
v___x_130_ = v___x_127_;
goto v_reusejp_129_;
}
else
{
lean_object* v_reuseFailAlloc_131_; 
v_reuseFailAlloc_131_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_131_, 0, v_a_125_);
v___x_130_ = v_reuseFailAlloc_131_;
goto v_reusejp_129_;
}
v_reusejp_129_:
{
return v___x_130_;
}
}
}
else
{
lean_object* v_a_133_; lean_object* v___x_134_; 
v_a_133_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_a_133_);
lean_dec_ref_known(v___x_124_, 1);
v___x_134_ = l_Lean_Json_getNat_x3f(v_idx_120_);
if (lean_obj_tag(v___x_134_) == 0)
{
lean_object* v_a_135_; lean_object* v___x_137_; uint8_t v_isShared_138_; uint8_t v_isSharedCheck_142_; 
lean_dec(v_a_133_);
lean_del_object(v___x_122_);
v_a_135_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_142_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_142_ == 0)
{
v___x_137_ = v___x_134_;
v_isShared_138_ = v_isSharedCheck_142_;
goto v_resetjp_136_;
}
else
{
lean_inc(v_a_135_);
lean_dec(v___x_134_);
v___x_137_ = lean_box(0);
v_isShared_138_ = v_isSharedCheck_142_;
goto v_resetjp_136_;
}
v_resetjp_136_:
{
lean_object* v___x_140_; 
if (v_isShared_138_ == 0)
{
v___x_140_ = v___x_137_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v_a_135_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
}
else
{
lean_object* v_a_143_; lean_object* v___x_145_; uint8_t v_isShared_146_; uint8_t v_isSharedCheck_153_; 
v_a_143_ = lean_ctor_get(v___x_134_, 0);
v_isSharedCheck_153_ = !lean_is_exclusive(v___x_134_);
if (v_isSharedCheck_153_ == 0)
{
v___x_145_ = v___x_134_;
v_isShared_146_ = v_isSharedCheck_153_;
goto v_resetjp_144_;
}
else
{
lean_inc(v_a_143_);
lean_dec(v___x_134_);
v___x_145_ = lean_box(0);
v_isShared_146_ = v_isSharedCheck_153_;
goto v_resetjp_144_;
}
v_resetjp_144_:
{
lean_object* v___x_148_; 
if (v_isShared_123_ == 0)
{
lean_ctor_set(v___x_122_, 1, v_a_143_);
lean_ctor_set(v___x_122_, 0, v_a_133_);
v___x_148_ = v___x_122_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_a_133_);
lean_ctor_set(v_reuseFailAlloc_152_, 1, v_a_143_);
v___x_148_ = v_reuseFailAlloc_152_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
lean_object* v___x_150_; 
if (v_isShared_146_ == 0)
{
lean_ctor_set(v___x_145_, 0, v___x_148_);
v___x_150_ = v___x_145_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_148_);
v___x_150_ = v_reuseFailAlloc_151_;
goto v_reusejp_149_;
}
v_reusejp_149_:
{
return v___x_150_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1____boxed(lean_object* v_j_155_, lean_object* v_a_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_dec_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(v_j_155_, v_a_156_);
lean_dec_ref(v_a_156_);
return v_res_157_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0(void){
_start:
{
lean_object* v___x_164_; lean_object* v___x_165_; 
v___x_164_ = lp_proofwidgets_ProofWidgets_instInhabitedHtml_default;
v___x_165_ = lean_thunk_pure(v___x_164_);
return v___x_165_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1(void){
_start:
{
lean_object* v___x_166_; lean_object* v___x_167_; 
v___x_166_ = lean_box(0);
v___x_167_ = lean_task_pure(v___x_166_);
return v___x_167_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2(void){
_start:
{
lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
v___x_168_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1, &lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__1);
v___x_169_ = lean_unsigned_to_nat(0u);
v___x_170_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0, &lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__0);
v___x_171_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_171_, 0, v___x_170_);
lean_ctor_set(v___x_171_, 1, v___x_169_);
lean_ctor_set(v___x_171_, 2, v___x_168_);
return v___x_171_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default(void){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2, &lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default___closed__2);
return v___x_172_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState(void){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default;
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_(lean_object* v_json_185_){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v_a_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_199_; 
v___x_186_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
lean_inc(v_json_185_);
v___x_187_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_185_, v___x_186_);
v_a_188_ = lean_ctor_get(v___x_187_, 0);
lean_inc(v_a_188_);
lean_dec_ref(v___x_187_);
v___x_189_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
v___x_190_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_185_, v___x_189_);
v_a_191_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_199_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_199_ == 0)
{
v___x_193_ = v___x_190_;
v_isShared_194_ = v_isSharedCheck_199_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_190_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_199_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_195_; lean_object* v___x_197_; 
v___x_195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_195_, 0, v_a_188_);
lean_ctor_set(v___x_195_, 1, v_a_191_);
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 0, v___x_195_);
v___x_197_ = v___x_193_;
goto v_reusejp_196_;
}
else
{
lean_object* v_reuseFailAlloc_198_; 
v_reuseFailAlloc_198_ = lean_alloc_ctor(1, 1, 0);
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
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_(lean_object* v_x_202_){
_start:
{
lean_object* v_state_203_; lean_object* v_oldIdx_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_222_; 
v_state_203_ = lean_ctor_get(v_x_202_, 0);
v_oldIdx_204_ = lean_ctor_get(v_x_202_, 1);
v_isSharedCheck_222_ = !lean_is_exclusive(v_x_202_);
if (v_isSharedCheck_222_ == 0)
{
v___x_206_ = v_x_202_;
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_oldIdx_204_);
lean_inc(v_state_203_);
lean_dec(v_x_202_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_210_; 
v___x_208_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 1, v_state_203_);
lean_ctor_set(v___x_206_, 0, v___x_208_);
v___x_210_ = v___x_206_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_221_; 
v_reuseFailAlloc_221_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_221_, 0, v___x_208_);
lean_ctor_set(v_reuseFailAlloc_221_, 1, v_state_203_);
v___x_210_ = v_reuseFailAlloc_221_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; lean_object* v___x_217_; lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___x_220_; 
v___x_211_ = lean_box(0);
v___x_212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_210_);
lean_ctor_set(v___x_212_, 1, v___x_211_);
v___x_213_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
v___x_214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
lean_ctor_set(v___x_214_, 1, v_oldIdx_204_);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_214_);
lean_ctor_set(v___x_215_, 1, v___x_211_);
v___x_216_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_215_);
lean_ctor_set(v___x_216_, 1, v___x_211_);
v___x_217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_217_, 0, v___x_212_);
lean_ctor_set(v___x_217_, 1, v___x_216_);
v___x_218_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_));
v___x_219_ = lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__spec__0(v___x_217_, v___x_218_);
v___x_220_ = l_Lean_Json_mkObj(v___x_219_);
lean_dec(v___x_219_);
return v___x_220_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_enc_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(lean_object* v_a_225_, lean_object* v_a_226_){
_start:
{
lean_object* v_state_227_; lean_object* v_oldIdx_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_249_; 
v_state_227_ = lean_ctor_get(v_a_225_, 0);
v_oldIdx_228_ = lean_ctor_get(v_a_225_, 1);
v_isSharedCheck_249_ = !lean_is_exclusive(v_a_225_);
if (v_isSharedCheck_249_ == 0)
{
v___x_230_ = v_a_225_;
v_isShared_231_ = v_isSharedCheck_249_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_oldIdx_228_);
lean_inc(v_state_227_);
lean_dec(v_a_225_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_249_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v_fst_234_; lean_object* v_snd_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_248_; 
v___x_232_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_));
v___x_233_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcEncode___redArg(v___x_232_, v_state_227_, v_a_226_);
lean_dec_ref(v_state_227_);
v_fst_234_ = lean_ctor_get(v___x_233_, 0);
v_snd_235_ = lean_ctor_get(v___x_233_, 1);
v_isSharedCheck_248_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_248_ == 0)
{
v___x_237_ = v___x_233_;
v_isShared_238_ = v_isSharedCheck_248_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_snd_235_);
lean_inc(v_fst_234_);
lean_dec(v___x_233_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_248_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_242_; 
v___x_239_ = l_Lean_JsonNumber_fromNat(v_oldIdx_228_);
v___x_240_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_240_, 0, v___x_239_);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 1, v___x_240_);
lean_ctor_set(v___x_230_, 0, v_fst_234_);
v___x_242_ = v___x_230_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_fst_234_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v___x_240_);
v___x_242_ = v_reuseFailAlloc_247_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_243_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_31_(v___x_242_);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_243_);
v___x_245_ = v___x_237_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_246_, 1, v_snd_235_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(lean_object* v_j_250_, lean_object* v_a_251_){
_start:
{
lean_object* v___x_252_; 
v___x_252_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_(v_j_250_);
if (lean_obj_tag(v___x_252_) == 0)
{
lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_260_; 
v_a_253_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_260_ == 0)
{
v___x_255_ = v___x_252_;
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_252_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_260_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_258_; 
if (v_isShared_256_ == 0)
{
v___x_258_ = v___x_255_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v_a_253_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
else
{
lean_object* v_a_261_; lean_object* v_state_262_; lean_object* v_oldIdx_263_; lean_object* v___x_265_; uint8_t v_isShared_266_; uint8_t v_isSharedCheck_298_; 
v_a_261_ = lean_ctor_get(v___x_252_, 0);
lean_inc(v_a_261_);
lean_dec_ref_known(v___x_252_, 1);
v_state_262_ = lean_ctor_get(v_a_261_, 0);
v_oldIdx_263_ = lean_ctor_get(v_a_261_, 1);
v_isSharedCheck_298_ = !lean_is_exclusive(v_a_261_);
if (v_isSharedCheck_298_ == 0)
{
v___x_265_ = v_a_261_;
v_isShared_266_ = v_isSharedCheck_298_;
goto v_resetjp_264_;
}
else
{
lean_inc(v_oldIdx_263_);
lean_inc(v_state_262_);
lean_dec(v_a_261_);
v___x_265_ = lean_box(0);
v_isShared_266_ = v_isSharedCheck_298_;
goto v_resetjp_264_;
}
v_resetjp_264_:
{
lean_object* v___x_267_; lean_object* v___x_268_; 
v___x_267_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_));
v___x_268_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcDecode___redArg(v___x_267_, v_state_262_, v_a_251_);
if (lean_obj_tag(v___x_268_) == 0)
{
lean_object* v_a_269_; lean_object* v___x_271_; uint8_t v_isShared_272_; uint8_t v_isSharedCheck_276_; 
lean_del_object(v___x_265_);
lean_dec(v_oldIdx_263_);
v_a_269_ = lean_ctor_get(v___x_268_, 0);
v_isSharedCheck_276_ = !lean_is_exclusive(v___x_268_);
if (v_isSharedCheck_276_ == 0)
{
v___x_271_ = v___x_268_;
v_isShared_272_ = v_isSharedCheck_276_;
goto v_resetjp_270_;
}
else
{
lean_inc(v_a_269_);
lean_dec(v___x_268_);
v___x_271_ = lean_box(0);
v_isShared_272_ = v_isSharedCheck_276_;
goto v_resetjp_270_;
}
v_resetjp_270_:
{
lean_object* v___x_274_; 
if (v_isShared_272_ == 0)
{
v___x_274_ = v___x_271_;
goto v_reusejp_273_;
}
else
{
lean_object* v_reuseFailAlloc_275_; 
v_reuseFailAlloc_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_275_, 0, v_a_269_);
v___x_274_ = v_reuseFailAlloc_275_;
goto v_reusejp_273_;
}
v_reusejp_273_:
{
return v___x_274_;
}
}
}
else
{
lean_object* v_a_277_; lean_object* v___x_278_; 
v_a_277_ = lean_ctor_get(v___x_268_, 0);
lean_inc(v_a_277_);
lean_dec_ref_known(v___x_268_, 1);
v___x_278_ = l_Lean_Json_getNat_x3f(v_oldIdx_263_);
if (lean_obj_tag(v___x_278_) == 0)
{
lean_object* v_a_279_; lean_object* v___x_281_; uint8_t v_isShared_282_; uint8_t v_isSharedCheck_286_; 
lean_dec(v_a_277_);
lean_del_object(v___x_265_);
v_a_279_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_286_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_286_ == 0)
{
v___x_281_ = v___x_278_;
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
else
{
lean_inc(v_a_279_);
lean_dec(v___x_278_);
v___x_281_ = lean_box(0);
v_isShared_282_ = v_isSharedCheck_286_;
goto v_resetjp_280_;
}
v_resetjp_280_:
{
lean_object* v___x_284_; 
if (v_isShared_282_ == 0)
{
v___x_284_ = v___x_281_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_a_279_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
}
else
{
lean_object* v_a_287_; lean_object* v___x_289_; uint8_t v_isShared_290_; uint8_t v_isSharedCheck_297_; 
v_a_287_ = lean_ctor_get(v___x_278_, 0);
v_isSharedCheck_297_ = !lean_is_exclusive(v___x_278_);
if (v_isSharedCheck_297_ == 0)
{
v___x_289_ = v___x_278_;
v_isShared_290_ = v_isSharedCheck_297_;
goto v_resetjp_288_;
}
else
{
lean_inc(v_a_287_);
lean_dec(v___x_278_);
v___x_289_ = lean_box(0);
v_isShared_290_ = v_isSharedCheck_297_;
goto v_resetjp_288_;
}
v_resetjp_288_:
{
lean_object* v___x_292_; 
if (v_isShared_266_ == 0)
{
lean_ctor_set(v___x_265_, 1, v_a_287_);
lean_ctor_set(v___x_265_, 0, v_a_277_);
v___x_292_ = v___x_265_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_296_; 
v_reuseFailAlloc_296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_296_, 0, v_a_277_);
lean_ctor_set(v_reuseFailAlloc_296_, 1, v_a_287_);
v___x_292_ = v_reuseFailAlloc_296_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
lean_object* v___x_294_; 
if (v_isShared_290_ == 0)
{
lean_ctor_set(v___x_289_, 0, v___x_292_);
v___x_294_ = v___x_289_;
goto v_reusejp_293_;
}
else
{
lean_object* v_reuseFailAlloc_295_; 
v_reuseFailAlloc_295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_295_, 0, v___x_292_);
v___x_294_ = v_reuseFailAlloc_295_;
goto v_reusejp_293_;
}
v_reusejp_293_:
{
return v___x_294_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1____boxed(lean_object* v_j_299_, lean_object* v_a_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(v_j_299_, v_a_300_);
lean_dec_ref(v_a_300_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0(lean_object* v_mutex_308_, lean_object* v_a_x3f_309_){
_start:
{
lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_311_ = lean_io_basemutex_unlock(v_mutex_308_);
v___x_312_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_312_, 0, v___x_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0___boxed(lean_object* v_mutex_313_, lean_object* v_a_x3f_314_, lean_object* v___y_315_){
_start:
{
lean_object* v_res_316_; 
v_res_316_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0(v_mutex_313_, v_a_x3f_314_);
lean_dec(v_a_x3f_314_);
lean_dec(v_mutex_313_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(lean_object* v_mutex_317_, lean_object* v_k_318_, lean_object* v___y_319_){
_start:
{
lean_object* v_ref_321_; lean_object* v_mutex_322_; lean_object* v___x_323_; lean_object* v___x_324_; 
v_ref_321_ = lean_ctor_get(v_mutex_317_, 0);
lean_inc(v_ref_321_);
v_mutex_322_ = lean_ctor_get(v_mutex_317_, 1);
lean_inc(v_mutex_322_);
lean_dec_ref(v_mutex_317_);
v___x_323_ = lean_io_basemutex_lock(v_mutex_322_);
lean_inc_ref(v___y_319_);
v___x_324_ = lean_apply_3(v_k_318_, v_ref_321_, v___y_319_, lean_box(0));
if (lean_obj_tag(v___x_324_) == 0)
{
lean_object* v_a_325_; lean_object* v___x_327_; uint8_t v_isShared_328_; uint8_t v_isSharedCheck_341_; 
v_a_325_ = lean_ctor_get(v___x_324_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_341_ == 0)
{
v___x_327_ = v___x_324_;
v_isShared_328_ = v_isSharedCheck_341_;
goto v_resetjp_326_;
}
else
{
lean_inc(v_a_325_);
lean_dec(v___x_324_);
v___x_327_ = lean_box(0);
v_isShared_328_ = v_isSharedCheck_341_;
goto v_resetjp_326_;
}
v_resetjp_326_:
{
lean_object* v___x_330_; 
lean_inc(v_a_325_);
if (v_isShared_328_ == 0)
{
lean_ctor_set_tag(v___x_327_, 1);
v___x_330_ = v___x_327_;
goto v_reusejp_329_;
}
else
{
lean_object* v_reuseFailAlloc_340_; 
v_reuseFailAlloc_340_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_340_, 0, v_a_325_);
v___x_330_ = v_reuseFailAlloc_340_;
goto v_reusejp_329_;
}
v_reusejp_329_:
{
lean_object* v___x_331_; lean_object* v___x_333_; uint8_t v_isShared_334_; uint8_t v_isSharedCheck_338_; 
v___x_331_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0(v_mutex_322_, v___x_330_);
lean_dec_ref(v___x_330_);
lean_dec(v_mutex_322_);
v_isSharedCheck_338_ = !lean_is_exclusive(v___x_331_);
if (v_isSharedCheck_338_ == 0)
{
lean_object* v_unused_339_; 
v_unused_339_ = lean_ctor_get(v___x_331_, 0);
lean_dec(v_unused_339_);
v___x_333_ = v___x_331_;
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
else
{
lean_dec(v___x_331_);
v___x_333_ = lean_box(0);
v_isShared_334_ = v_isSharedCheck_338_;
goto v_resetjp_332_;
}
v_resetjp_332_:
{
lean_object* v___x_336_; 
if (v_isShared_334_ == 0)
{
lean_ctor_set(v___x_333_, 0, v_a_325_);
v___x_336_ = v___x_333_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v_a_325_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
}
}
else
{
lean_object* v_a_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_346_; uint8_t v_isShared_347_; uint8_t v_isSharedCheck_351_; 
v_a_342_ = lean_ctor_get(v___x_324_, 0);
lean_inc(v_a_342_);
lean_dec_ref_known(v___x_324_, 1);
v___x_343_ = lean_box(0);
v___x_344_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___lam__0(v_mutex_322_, v___x_343_);
lean_dec(v_mutex_322_);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_344_);
if (v_isSharedCheck_351_ == 0)
{
lean_object* v_unused_352_; 
v_unused_352_ = lean_ctor_get(v___x_344_, 0);
lean_dec(v_unused_352_);
v___x_346_ = v___x_344_;
v_isShared_347_ = v_isSharedCheck_351_;
goto v_resetjp_345_;
}
else
{
lean_dec(v___x_344_);
v___x_346_ = lean_box(0);
v_isShared_347_ = v_isSharedCheck_351_;
goto v_resetjp_345_;
}
v_resetjp_345_:
{
lean_object* v___x_349_; 
if (v_isShared_347_ == 0)
{
lean_ctor_set_tag(v___x_346_, 1);
lean_ctor_set(v___x_346_, 0, v_a_342_);
v___x_349_ = v___x_346_;
goto v_reusejp_348_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v_a_342_);
v___x_349_ = v_reuseFailAlloc_350_;
goto v_reusejp_348_;
}
v_reusejp_348_:
{
return v___x_349_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg___boxed(lean_object* v_mutex_353_, lean_object* v_k_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(v_mutex_353_, v_k_354_, v___y_355_);
lean_dec_ref(v___y_355_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0(lean_object* v_00_u03b1_358_, lean_object* v_00_u03b2_359_, lean_object* v_mutex_360_, lean_object* v_k_361_, lean_object* v___y_362_){
_start:
{
lean_object* v___x_364_; 
v___x_364_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(v_mutex_360_, v_k_361_, v___y_362_);
return v___x_364_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___boxed(lean_object* v_00_u03b1_365_, lean_object* v_00_u03b2_366_, lean_object* v_mutex_367_, lean_object* v_k_368_, lean_object* v___y_369_, lean_object* v___y_370_){
_start:
{
lean_object* v_res_371_; 
v_res_371_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0(v_00_u03b1_365_, v_00_u03b2_366_, v_mutex_367_, v_k_368_, v___y_369_);
lean_dec_ref(v___y_369_);
return v_res_371_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0(lean_object* v___y_372_, lean_object* v___y_373_){
_start:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = lean_st_ref_get(v___y_372_);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v___x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0___boxed(lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_){
_start:
{
lean_object* v_res_380_; 
v_res_380_ = lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__0(v___y_377_, v___y_378_);
lean_dec_ref(v___y_378_);
lean_dec(v___y_377_);
return v_res_380_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1(lean_object* v_val_381_, lean_object* v___f_382_, lean_object* v_x_383_, lean_object* v___y_384_){
_start:
{
if (lean_obj_tag(v_x_383_) == 0)
{
lean_object* v___x_386_; lean_object* v___x_387_; 
lean_dec_ref(v___f_382_);
lean_dec(v_val_381_);
v___x_386_ = lean_box(0);
v___x_387_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_387_, 0, v___x_386_);
return v___x_387_;
}
else
{
lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_415_; 
v_isSharedCheck_415_ = !lean_is_exclusive(v_x_383_);
if (v_isSharedCheck_415_ == 0)
{
lean_object* v_unused_416_; 
v_unused_416_ = lean_ctor_get(v_x_383_, 0);
lean_dec(v_unused_416_);
v___x_389_ = v_x_383_;
v_isShared_390_ = v_isSharedCheck_415_;
goto v_resetjp_388_;
}
else
{
lean_dec(v_x_383_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_415_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_391_; 
v___x_391_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(v_val_381_, v___f_382_, v___y_384_);
if (lean_obj_tag(v___x_391_) == 0)
{
lean_object* v_a_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_406_; 
v_a_392_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_406_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_406_ == 0)
{
v___x_394_ = v___x_391_;
v_isShared_395_ = v_isSharedCheck_406_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_a_392_);
lean_dec(v___x_391_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_406_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
lean_object* v_curr_396_; lean_object* v_idx_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_401_; 
v_curr_396_ = lean_ctor_get(v_a_392_, 0);
lean_inc_ref(v_curr_396_);
v_idx_397_ = lean_ctor_get(v_a_392_, 1);
lean_inc(v_idx_397_);
lean_dec(v_a_392_);
v___x_398_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(v_curr_396_);
lean_dec_ref(v_curr_396_);
v___x_399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
lean_ctor_set(v___x_399_, 1, v_idx_397_);
if (v_isShared_390_ == 0)
{
lean_ctor_set(v___x_389_, 0, v___x_399_);
v___x_401_ = v___x_389_;
goto v_reusejp_400_;
}
else
{
lean_object* v_reuseFailAlloc_405_; 
v_reuseFailAlloc_405_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_405_, 0, v___x_399_);
v___x_401_ = v_reuseFailAlloc_405_;
goto v_reusejp_400_;
}
v_reusejp_400_:
{
lean_object* v___x_403_; 
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 0, v___x_401_);
v___x_403_ = v___x_394_;
goto v_reusejp_402_;
}
else
{
lean_object* v_reuseFailAlloc_404_; 
v_reuseFailAlloc_404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_404_, 0, v___x_401_);
v___x_403_ = v_reuseFailAlloc_404_;
goto v_reusejp_402_;
}
v_reusejp_402_:
{
return v___x_403_;
}
}
}
}
else
{
lean_object* v_a_407_; lean_object* v___x_409_; uint8_t v_isShared_410_; uint8_t v_isSharedCheck_414_; 
lean_del_object(v___x_389_);
v_a_407_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_414_ == 0)
{
v___x_409_ = v___x_391_;
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
else
{
lean_inc(v_a_407_);
lean_dec(v___x_391_);
v___x_409_ = lean_box(0);
v_isShared_410_ = v_isSharedCheck_414_;
goto v_resetjp_408_;
}
v_resetjp_408_:
{
lean_object* v___x_412_; 
if (v_isShared_410_ == 0)
{
v___x_412_ = v___x_409_;
goto v_reusejp_411_;
}
else
{
lean_object* v_reuseFailAlloc_413_; 
v_reuseFailAlloc_413_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_413_, 0, v_a_407_);
v___x_412_ = v_reuseFailAlloc_413_;
goto v_reusejp_411_;
}
v_reusejp_411_:
{
return v___x_412_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1___boxed(lean_object* v_val_417_, lean_object* v___f_418_, lean_object* v_x_419_, lean_object* v___y_420_, lean_object* v___y_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1(v_val_417_, v___f_418_, v_x_419_, v___y_420_);
lean_dec_ref(v___y_420_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2(lean_object* v_curr_423_, lean_object* v_idx_424_, lean_object* v___y_425_){
_start:
{
lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_427_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__IO_forceThunk___redArg(v_curr_423_);
v___x_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_428_, 0, v___x_427_);
lean_ctor_set(v___x_428_, 1, v_idx_424_);
v___x_429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
v___x_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2___boxed(lean_object* v_curr_431_, lean_object* v_idx_432_, lean_object* v___y_433_, lean_object* v___y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2(v_curr_431_, v_idx_432_, v___y_433_);
lean_dec_ref(v___y_433_);
lean_dec_ref(v_curr_431_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh(lean_object* v_ps_437_, lean_object* v_a_438_){
_start:
{
lean_object* v_state_440_; lean_object* v_oldIdx_441_; lean_object* v_val_442_; lean_object* v___f_443_; lean_object* v___x_444_; 
v_state_440_ = lean_ctor_get(v_ps_437_, 0);
lean_inc_ref(v_state_440_);
v_oldIdx_441_ = lean_ctor_get(v_ps_437_, 1);
lean_inc(v_oldIdx_441_);
lean_dec_ref(v_ps_437_);
v_val_442_ = lean_ctor_get(v_state_440_, 0);
lean_inc_n(v_val_442_, 2);
lean_dec_ref(v_state_440_);
v___f_443_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___closed__0));
v___x_444_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshComponent_awaitRefresh_spec__0___redArg(v_val_442_, v___f_443_, v_a_438_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_object* v_a_445_; lean_object* v_curr_446_; lean_object* v_idx_447_; lean_object* v_next_448_; uint8_t v___x_449_; 
v_a_445_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_a_445_);
lean_dec_ref_known(v___x_444_, 1);
v_curr_446_ = lean_ctor_get(v_a_445_, 0);
lean_inc_ref(v_curr_446_);
v_idx_447_ = lean_ctor_get(v_a_445_, 1);
lean_inc(v_idx_447_);
v_next_448_ = lean_ctor_get(v_a_445_, 2);
lean_inc_ref(v_next_448_);
lean_dec(v_a_445_);
v___x_449_ = lean_nat_dec_lt(v_oldIdx_441_, v_idx_447_);
lean_dec(v_oldIdx_441_);
if (v___x_449_ == 0)
{
lean_object* v___f_450_; lean_object* v___x_451_; 
lean_dec(v_idx_447_);
lean_dec_ref(v_curr_446_);
v___f_450_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__1___boxed), 5, 2);
lean_closure_set(v___f_450_, 0, v_val_442_);
lean_closure_set(v___f_450_, 1, v___f_443_);
v___x_451_ = l_Lean_Server_RequestM_mapTaskCostly___redArg(v_next_448_, v___f_450_, v_a_438_);
return v___x_451_;
}
else
{
lean_object* v___f_452_; lean_object* v___x_453_; 
lean_dec_ref(v_next_448_);
lean_dec(v_val_442_);
v___f_452_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___lam__2___boxed), 4, 2);
lean_closure_set(v___f_452_, 0, v_curr_446_);
lean_closure_set(v___f_452_, 1, v_idx_447_);
v___x_453_ = l_Lean_Server_RequestM_asTask___redArg(v___f_452_, v_a_438_);
return v___x_453_;
}
}
else
{
lean_object* v_a_454_; lean_object* v___x_456_; uint8_t v_isShared_457_; uint8_t v_isSharedCheck_461_; 
lean_dec(v_val_442_);
lean_dec(v_oldIdx_441_);
v_a_454_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_461_ == 0)
{
v___x_456_ = v___x_444_;
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
else
{
lean_inc(v_a_454_);
lean_dec(v___x_444_);
v___x_456_ = lean_box(0);
v_isShared_457_ = v_isSharedCheck_461_;
goto v_resetjp_455_;
}
v_resetjp_455_:
{
lean_object* v___x_459_; 
if (v_isShared_457_ == 0)
{
v___x_459_ = v___x_456_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v_a_454_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___boxed(lean_object* v_ps_462_, lean_object* v_a_463_, lean_object* v_a_464_){
_start:
{
lean_object* v_res_465_; 
v_res_465_ = lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh(v_ps_462_, v_a_463_);
lean_dec_ref(v_a_463_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0(lean_object* v___y_466_){
_start:
{
lean_inc(v___y_466_);
return v___y_466_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__0(v___y_467_);
lean_dec(v___y_467_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1(lean_object* v_x_469_){
_start:
{
if (lean_obj_tag(v_x_469_) == 0)
{
lean_object* v___x_470_; 
v___x_470_ = lean_box(0);
return v___x_470_;
}
else
{
lean_object* v_val_471_; 
v_val_471_ = lean_ctor_get(v_x_469_, 0);
lean_inc(v_val_471_);
return v_val_471_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1___boxed(lean_object* v_x_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1(v_x_472_);
lean_dec(v_x_472_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_474_, lean_object* v_x_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_476_, 0, v_x_475_);
lean_ctor_set(v___x_476_, 1, v_expireTime_474_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2(lean_object* v_val_477_, lean_object* v___f_478_, lean_object* v_x_479_, lean_object* v___y_480_){
_start:
{
if (lean_obj_tag(v_x_479_) == 0)
{
lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
lean_dec_ref(v___f_478_);
v_a_482_ = lean_ctor_get(v_x_479_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v_x_479_);
if (v_isSharedCheck_489_ == 0)
{
v___x_484_ = v_x_479_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v_x_479_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
if (v_isShared_485_ == 0)
{
lean_ctor_set_tag(v___x_484_, 1);
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_a_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
else
{
lean_object* v_a_490_; lean_object* v___x_492_; uint8_t v_isShared_493_; uint8_t v_isSharedCheck_528_; 
v_a_490_ = lean_ctor_get(v_x_479_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v_x_479_);
if (v_isSharedCheck_528_ == 0)
{
v___x_492_ = v_x_479_;
v_isShared_493_ = v_isSharedCheck_528_;
goto v_resetjp_491_;
}
else
{
lean_inc(v_a_490_);
lean_dec(v_x_479_);
v___x_492_ = lean_box(0);
v_isShared_493_ = v_isSharedCheck_528_;
goto v_resetjp_491_;
}
v_resetjp_491_:
{
lean_object* v___x_494_; lean_object* v_objects_495_; lean_object* v_expireTime_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_527_; 
v___x_494_ = lean_st_ref_take(v_val_477_);
v_objects_495_ = lean_ctor_get(v___x_494_, 0);
v_expireTime_496_ = lean_ctor_get(v___x_494_, 1);
v_isSharedCheck_527_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_527_ == 0)
{
v___x_498_ = v___x_494_;
v_isShared_499_ = v_isSharedCheck_527_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_expireTime_496_);
lean_inc(v_objects_495_);
lean_dec(v___x_494_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_527_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___f_500_; lean_object* v_fst_502_; lean_object* v_snd_503_; 
v___f_500_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_500_, 0, v_expireTime_496_);
if (lean_obj_tag(v_a_490_) == 0)
{
lean_object* v___x_515_; 
v___x_515_ = lean_box(0);
v_fst_502_ = v___x_515_;
v_snd_503_ = v_objects_495_;
goto v___jp_501_;
}
else
{
lean_object* v_val_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_526_; 
v_val_516_ = lean_ctor_get(v_a_490_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v_a_490_);
if (v_isSharedCheck_526_ == 0)
{
v___x_518_ = v_a_490_;
v_isShared_519_ = v_isSharedCheck_526_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_val_516_);
lean_dec(v_a_490_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_526_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_520_; lean_object* v_fst_521_; lean_object* v_snd_522_; lean_object* v___x_524_; 
v___x_520_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableVersionedHtml_enc_00___x40_ProofWidgets_Component_RefreshComponent_4220679497____hygCtx___hyg_1_(v_val_516_, v_objects_495_);
v_fst_521_ = lean_ctor_get(v___x_520_, 0);
lean_inc(v_fst_521_);
v_snd_522_ = lean_ctor_get(v___x_520_, 1);
lean_inc(v_snd_522_);
lean_dec_ref(v___x_520_);
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 0, v_fst_521_);
v___x_524_ = v___x_518_;
goto v_reusejp_523_;
}
else
{
lean_object* v_reuseFailAlloc_525_; 
v_reuseFailAlloc_525_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_525_, 0, v_fst_521_);
v___x_524_ = v_reuseFailAlloc_525_;
goto v_reusejp_523_;
}
v_reusejp_523_:
{
v_fst_502_ = v___x_524_;
v_snd_503_ = v_snd_522_;
goto v___jp_501_;
}
}
}
v___jp_501_:
{
lean_object* v___x_504_; lean_object* v___x_506_; 
v___x_504_ = lp_proofwidgets_Lean_Option_toJson___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__1(v_fst_502_);
lean_dec(v_fst_502_);
if (v_isShared_499_ == 0)
{
lean_ctor_set(v___x_498_, 1, v_snd_503_);
lean_ctor_set(v___x_498_, 0, v___x_504_);
v___x_506_ = v___x_498_;
goto v_reusejp_505_;
}
else
{
lean_object* v_reuseFailAlloc_514_; 
v_reuseFailAlloc_514_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_514_, 0, v___x_504_);
lean_ctor_set(v_reuseFailAlloc_514_, 1, v_snd_503_);
v___x_506_ = v_reuseFailAlloc_514_;
goto v_reusejp_505_;
}
v_reusejp_505_:
{
lean_object* v___x_507_; lean_object* v_fst_508_; lean_object* v_snd_509_; lean_object* v___x_510_; lean_object* v___x_512_; 
v___x_507_ = l_Prod_map___redArg(v___f_478_, v___f_500_, v___x_506_);
v_fst_508_ = lean_ctor_get(v___x_507_, 0);
lean_inc(v_fst_508_);
v_snd_509_ = lean_ctor_get(v___x_507_, 1);
lean_inc(v_snd_509_);
lean_dec_ref(v___x_507_);
v___x_510_ = lean_st_ref_set(v_val_477_, v_snd_509_);
if (v_isShared_493_ == 0)
{
lean_ctor_set_tag(v___x_492_, 0);
lean_ctor_set(v___x_492_, 0, v_fst_508_);
v___x_512_ = v___x_492_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_fst_508_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_529_, lean_object* v___f_530_, lean_object* v_x_531_, lean_object* v___y_532_, lean_object* v___y_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2(v_val_529_, v___f_530_, v_x_531_, v___y_532_);
lean_dec_ref(v___y_532_);
lean_dec(v_val_529_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_535_, uint64_t v_k_536_){
_start:
{
if (lean_obj_tag(v_t_535_) == 0)
{
lean_object* v_k_537_; lean_object* v_v_538_; lean_object* v_l_539_; lean_object* v_r_540_; uint64_t v___x_541_; uint8_t v___x_542_; 
v_k_537_ = lean_ctor_get(v_t_535_, 1);
v_v_538_ = lean_ctor_get(v_t_535_, 2);
v_l_539_ = lean_ctor_get(v_t_535_, 3);
v_r_540_ = lean_ctor_get(v_t_535_, 4);
v___x_541_ = lean_unbox_uint64(v_k_537_);
v___x_542_ = lean_uint64_dec_lt(v_k_536_, v___x_541_);
if (v___x_542_ == 0)
{
uint64_t v___x_543_; uint8_t v___x_544_; 
v___x_543_ = lean_unbox_uint64(v_k_537_);
v___x_544_ = lean_uint64_dec_eq(v_k_536_, v___x_543_);
if (v___x_544_ == 0)
{
v_t_535_ = v_r_540_;
goto _start;
}
else
{
lean_object* v___x_546_; 
lean_inc(v_v_538_);
v___x_546_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_546_, 0, v_v_538_);
return v___x_546_;
}
}
else
{
v_t_535_ = v_l_539_;
goto _start;
}
}
else
{
lean_object* v___x_548_; 
v___x_548_ = lean_box(0);
return v___x_548_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_549_, lean_object* v_k_550_){
_start:
{
uint64_t v_k_boxed_551_; lean_object* v_res_552_; 
v_k_boxed_551_ = lean_unbox_uint64(v_k_550_);
lean_dec_ref(v_k_550_);
v_res_552_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(v_t_549_, v_k_boxed_551_);
lean_dec(v_t_549_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3(lean_object* v_method_560_, lean_object* v_handler_561_, lean_object* v___f_562_, uint64_t v_seshId_563_, lean_object* v_j_564_, lean_object* v___y_565_){
_start:
{
lean_object* v_rpcSessions_567_; lean_object* v___x_568_; 
v_rpcSessions_567_ = lean_ctor_get(v___y_565_, 0);
v___x_568_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_567_, v_seshId_563_);
if (lean_obj_tag(v___x_568_) == 1)
{
lean_object* v_val_569_; lean_object* v___x_570_; lean_object* v_objects_571_; lean_object* v___x_572_; 
v_val_569_ = lean_ctor_get(v___x_568_, 0);
lean_inc(v_val_569_);
lean_dec_ref_known(v___x_568_, 1);
v___x_570_ = lean_st_ref_get(v_val_569_);
v_objects_571_ = lean_ctor_get(v___x_570_, 0);
lean_inc_ref(v_objects_571_);
lean_dec(v___x_570_);
lean_inc(v_j_564_);
v___x_572_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableAwaitRefreshParams_dec_00___x40_ProofWidgets_Component_RefreshComponent_311896448____hygCtx___hyg_1_(v_j_564_, v_objects_571_);
lean_dec_ref(v_objects_571_);
if (lean_obj_tag(v___x_572_) == 0)
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_593_; 
lean_dec(v_val_569_);
lean_dec_ref(v___f_562_);
lean_dec_ref(v_handler_561_);
v_a_573_ = lean_ctor_get(v___x_572_, 0);
v_isSharedCheck_593_ = !lean_is_exclusive(v___x_572_);
if (v_isSharedCheck_593_ == 0)
{
v___x_575_ = v___x_572_;
v_isShared_576_ = v_isSharedCheck_593_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_572_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_593_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
uint8_t v___x_577_; lean_object* v___x_578_; uint8_t v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_591_; 
v___x_577_ = 3;
v___x_578_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_579_ = 1;
v___x_580_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_560_, v___x_579_);
v___x_581_ = lean_string_append(v___x_578_, v___x_580_);
lean_dec_ref(v___x_580_);
v___x_582_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_583_ = lean_string_append(v___x_581_, v___x_582_);
v___x_584_ = l_Lean_Json_compress(v_j_564_);
v___x_585_ = lean_string_append(v___x_583_, v___x_584_);
lean_dec_ref(v___x_584_);
v___x_586_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_587_ = lean_string_append(v___x_585_, v___x_586_);
v___x_588_ = lean_string_append(v___x_587_, v_a_573_);
lean_dec(v_a_573_);
v___x_589_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_589_, 0, v___x_588_);
lean_ctor_set_uint8(v___x_589_, sizeof(void*)*1, v___x_577_);
if (v_isShared_576_ == 0)
{
lean_ctor_set_tag(v___x_575_, 1);
lean_ctor_set(v___x_575_, 0, v___x_589_);
v___x_591_ = v___x_575_;
goto v_reusejp_590_;
}
else
{
lean_object* v_reuseFailAlloc_592_; 
v_reuseFailAlloc_592_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_592_, 0, v___x_589_);
v___x_591_ = v_reuseFailAlloc_592_;
goto v_reusejp_590_;
}
v_reusejp_590_:
{
return v___x_591_;
}
}
}
else
{
lean_object* v_a_594_; lean_object* v___x_595_; 
lean_dec(v_j_564_);
lean_dec(v_method_560_);
v_a_594_ = lean_ctor_get(v___x_572_, 0);
lean_inc(v_a_594_);
lean_dec_ref_known(v___x_572_, 1);
lean_inc_ref(v___y_565_);
v___x_595_ = lean_apply_3(v_handler_561_, v_a_594_, v___y_565_, lean_box(0));
if (lean_obj_tag(v___x_595_) == 0)
{
lean_object* v_a_596_; lean_object* v___f_597_; lean_object* v___x_598_; 
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_a_596_);
lean_dec_ref_known(v___x_595_, 1);
v___f_597_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_597_, 0, v_val_569_);
lean_closure_set(v___f_597_, 1, v___f_562_);
v___x_598_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_596_, v___f_597_, v___y_565_);
return v___x_598_;
}
else
{
lean_object* v_a_599_; lean_object* v___x_601_; uint8_t v_isShared_602_; uint8_t v_isSharedCheck_606_; 
lean_dec(v_val_569_);
lean_dec_ref(v___f_562_);
v_a_599_ = lean_ctor_get(v___x_595_, 0);
v_isSharedCheck_606_ = !lean_is_exclusive(v___x_595_);
if (v_isSharedCheck_606_ == 0)
{
v___x_601_ = v___x_595_;
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
else
{
lean_inc(v_a_599_);
lean_dec(v___x_595_);
v___x_601_ = lean_box(0);
v_isShared_602_ = v_isSharedCheck_606_;
goto v_resetjp_600_;
}
v_resetjp_600_:
{
lean_object* v___x_604_; 
if (v_isShared_602_ == 0)
{
v___x_604_ = v___x_601_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v_a_599_);
v___x_604_ = v_reuseFailAlloc_605_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
return v___x_604_;
}
}
}
}
}
else
{
lean_object* v___x_607_; lean_object* v___x_608_; 
lean_dec(v___x_568_);
lean_dec(v_j_564_);
lean_dec_ref(v___f_562_);
lean_dec_ref(v_handler_561_);
lean_dec(v_method_560_);
v___x_607_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_608_, 0, v___x_607_);
return v___x_608_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_609_, lean_object* v_handler_610_, lean_object* v___f_611_, lean_object* v_seshId_612_, lean_object* v_j_613_, lean_object* v___y_614_, lean_object* v___y_615_){
_start:
{
uint64_t v_seshId_boxed_616_; lean_object* v_res_617_; 
v_seshId_boxed_616_ = lean_unbox_uint64(v_seshId_612_);
lean_dec_ref(v_seshId_612_);
v_res_617_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3(v_method_609_, v_handler_610_, v___f_611_, v_seshId_boxed_616_, v_j_613_, v___y_614_);
lean_dec_ref(v___y_614_);
return v_res_617_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0(lean_object* v_method_619_, lean_object* v_handler_620_){
_start:
{
lean_object* v___f_621_; lean_object* v___f_622_; 
v___f_621_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___closed__0));
v___f_622_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_622_, 0, v_method_619_);
lean_closure_set(v___f_622_, 1, v_handler_620_);
lean_closure_set(v___f_622_, 2, v___f_621_);
return v___f_622_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; 
v___x_629_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__2));
v___x_630_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__1));
v___x_631_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0(v___x_630_, v___x_629_);
return v___x_631_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped(void){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3, &lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped___closed__3);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_633_, lean_object* v_t_634_, uint64_t v_k_635_){
_start:
{
lean_object* v___x_636_; 
v___x_636_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(v_t_634_, v_k_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_637_, lean_object* v_t_638_, lean_object* v_k_639_){
_start:
{
uint64_t v_k_boxed_640_; lean_object* v_res_641_; 
v_k_boxed_640_ = lean_unbox_uint64(v_k_639_);
lean_dec_ref(v_k_639_);
v_res_641_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0(v_00_u03b4_637_, v_t_638_, v_k_boxed_640_);
lean_dec(v_t_638_);
return v_res_641_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_(lean_object* v_json_650_){
_start:
{
lean_object* v___x_651_; lean_object* v___x_652_; lean_object* v_a_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v_a_656_; lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_664_; 
v___x_651_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
lean_inc(v_json_650_);
v___x_652_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_650_, v___x_651_);
v_a_653_ = lean_ctor_get(v___x_652_, 0);
lean_inc(v_a_653_);
lean_dec_ref(v___x_652_);
v___x_654_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_));
v___x_655_ = lp_proofwidgets_Lean_Json_getObjValAs_x3f___at___00ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_12__spec__0(v_json_650_, v___x_654_);
v_a_656_ = lean_ctor_get(v___x_655_, 0);
v_isSharedCheck_664_ = !lean_is_exclusive(v___x_655_);
if (v_isSharedCheck_664_ == 0)
{
v___x_658_ = v___x_655_;
v_isShared_659_ = v_isSharedCheck_664_;
goto v_resetjp_657_;
}
else
{
lean_inc(v_a_656_);
lean_dec(v___x_655_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_664_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_660_; lean_object* v___x_662_; 
v___x_660_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_660_, 0, v_a_653_);
lean_ctor_set(v___x_660_, 1, v_a_656_);
if (v_isShared_659_ == 0)
{
lean_ctor_set(v___x_658_, 0, v___x_660_);
v___x_662_ = v___x_658_;
goto v_reusejp_661_;
}
else
{
lean_object* v_reuseFailAlloc_663_; 
v_reuseFailAlloc_663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_663_, 0, v___x_660_);
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
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_(lean_object* v_x_667_){
_start:
{
lean_object* v_state_668_; lean_object* v_cancelTk_669_; lean_object* v___x_671_; uint8_t v_isShared_672_; uint8_t v_isSharedCheck_687_; 
v_state_668_ = lean_ctor_get(v_x_667_, 0);
v_cancelTk_669_ = lean_ctor_get(v_x_667_, 1);
v_isSharedCheck_687_ = !lean_is_exclusive(v_x_667_);
if (v_isSharedCheck_687_ == 0)
{
v___x_671_ = v_x_667_;
v_isShared_672_ = v_isSharedCheck_687_;
goto v_resetjp_670_;
}
else
{
lean_inc(v_cancelTk_669_);
lean_inc(v_state_668_);
lean_dec(v_x_667_);
v___x_671_ = lean_box(0);
v_isShared_672_ = v_isSharedCheck_687_;
goto v_resetjp_670_;
}
v_resetjp_670_:
{
lean_object* v___x_673_; lean_object* v___x_675_; 
v___x_673_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_3442624324____hygCtx___hyg_12_));
if (v_isShared_672_ == 0)
{
lean_ctor_set(v___x_671_, 1, v_state_668_);
lean_ctor_set(v___x_671_, 0, v___x_673_);
v___x_675_ = v___x_671_;
goto v_reusejp_674_;
}
else
{
lean_object* v_reuseFailAlloc_686_; 
v_reuseFailAlloc_686_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_686_, 0, v___x_673_);
lean_ctor_set(v_reuseFailAlloc_686_, 1, v_state_668_);
v___x_675_ = v_reuseFailAlloc_686_;
goto v_reusejp_674_;
}
v_reusejp_674_:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; lean_object* v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; 
v___x_676_ = lean_box(0);
v___x_677_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_677_, 0, v___x_675_);
lean_ctor_set(v___x_677_, 1, v___x_676_);
v___x_678_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_));
v___x_679_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_679_, 0, v___x_678_);
lean_ctor_set(v___x_679_, 1, v_cancelTk_669_);
v___x_680_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
lean_ctor_set(v___x_680_, 1, v___x_676_);
v___x_681_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
lean_ctor_set(v___x_681_, 1, v___x_676_);
v___x_682_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_682_, 0, v___x_677_);
lean_ctor_set(v___x_682_, 1, v___x_681_);
v___x_683_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_));
v___x_684_ = lp_proofwidgets___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31__spec__0(v___x_682_, v___x_683_);
v___x_685_ = l_Lean_Json_mkObj(v___x_684_);
lean_dec(v___x_684_);
return v___x_685_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(lean_object* v_a_690_, lean_object* v_a_691_){
_start:
{
lean_object* v_state_692_; lean_object* v_cancelTk_693_; lean_object* v___x_694_; lean_object* v___x_695_; lean_object* v_fst_696_; lean_object* v_snd_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_716_; 
v_state_692_ = lean_ctor_get(v_a_690_, 0);
v_cancelTk_693_ = lean_ctor_get(v_a_690_, 1);
v___x_694_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_));
v___x_695_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcEncode___redArg(v___x_694_, v_state_692_, v_a_691_);
v_fst_696_ = lean_ctor_get(v___x_695_, 0);
v_snd_697_ = lean_ctor_get(v___x_695_, 1);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_695_);
if (v_isSharedCheck_716_ == 0)
{
v___x_699_ = v___x_695_;
v_isShared_700_ = v_isSharedCheck_716_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_snd_697_);
lean_inc(v_fst_696_);
lean_dec(v___x_695_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_716_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v_fst_703_; lean_object* v_snd_704_; lean_object* v___x_706_; uint8_t v_isShared_707_; uint8_t v_isSharedCheck_715_; 
v___x_701_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_));
v___x_702_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcEncode___redArg(v___x_701_, v_cancelTk_693_, v_snd_697_);
v_fst_703_ = lean_ctor_get(v___x_702_, 0);
v_snd_704_ = lean_ctor_get(v___x_702_, 1);
v_isSharedCheck_715_ = !lean_is_exclusive(v___x_702_);
if (v_isSharedCheck_715_ == 0)
{
v___x_706_ = v___x_702_;
v_isShared_707_ = v_isSharedCheck_715_;
goto v_resetjp_705_;
}
else
{
lean_inc(v_snd_704_);
lean_inc(v_fst_703_);
lean_dec(v___x_702_);
v___x_706_ = lean_box(0);
v_isShared_707_ = v_isSharedCheck_715_;
goto v_resetjp_705_;
}
v_resetjp_705_:
{
lean_object* v___x_709_; 
if (v_isShared_700_ == 0)
{
lean_ctor_set(v___x_699_, 1, v_fst_703_);
v___x_709_ = v___x_699_;
goto v_reusejp_708_;
}
else
{
lean_object* v_reuseFailAlloc_714_; 
v_reuseFailAlloc_714_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_714_, 0, v_fst_696_);
lean_ctor_set(v_reuseFailAlloc_714_, 1, v_fst_703_);
v___x_709_ = v_reuseFailAlloc_714_;
goto v_reusejp_708_;
}
v_reusejp_708_:
{
lean_object* v___x_710_; lean_object* v___x_712_; 
v___x_710_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_31_(v___x_709_);
if (v_isShared_707_ == 0)
{
lean_ctor_set(v___x_706_, 0, v___x_710_);
v___x_712_ = v___x_706_;
goto v_reusejp_711_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v___x_710_);
lean_ctor_set(v_reuseFailAlloc_713_, 1, v_snd_704_);
v___x_712_ = v_reuseFailAlloc_713_;
goto v_reusejp_711_;
}
v_reusejp_711_:
{
return v___x_712_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed(lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(v_a_717_, v_a_718_);
lean_dec_ref(v_a_717_);
return v_res_719_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(lean_object* v_j_720_, lean_object* v_a_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instFromJsonRpcEncodablePacket_fromJson_00___x40_ProofWidgets_Component_RefreshComponent_565092539____hygCtx___hyg_12_(v_j_720_);
if (lean_obj_tag(v___x_722_) == 0)
{
lean_object* v_a_723_; lean_object* v___x_725_; uint8_t v_isShared_726_; uint8_t v_isSharedCheck_730_; 
v_a_723_ = lean_ctor_get(v___x_722_, 0);
v_isSharedCheck_730_ = !lean_is_exclusive(v___x_722_);
if (v_isSharedCheck_730_ == 0)
{
v___x_725_ = v___x_722_;
v_isShared_726_ = v_isSharedCheck_730_;
goto v_resetjp_724_;
}
else
{
lean_inc(v_a_723_);
lean_dec(v___x_722_);
v___x_725_ = lean_box(0);
v_isShared_726_ = v_isSharedCheck_730_;
goto v_resetjp_724_;
}
v_resetjp_724_:
{
lean_object* v___x_728_; 
if (v_isShared_726_ == 0)
{
v___x_728_ = v___x_725_;
goto v_reusejp_727_;
}
else
{
lean_object* v_reuseFailAlloc_729_; 
v_reuseFailAlloc_729_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_729_, 0, v_a_723_);
v___x_728_ = v_reuseFailAlloc_729_;
goto v_reusejp_727_;
}
v_reusejp_727_:
{
return v___x_728_;
}
}
}
else
{
lean_object* v_a_731_; lean_object* v_state_732_; lean_object* v_cancelTk_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_769_; 
v_a_731_ = lean_ctor_get(v___x_722_, 0);
lean_inc(v_a_731_);
lean_dec_ref_known(v___x_722_, 1);
v_state_732_ = lean_ctor_get(v_a_731_, 0);
v_cancelTk_733_ = lean_ctor_get(v_a_731_, 1);
v_isSharedCheck_769_ = !lean_is_exclusive(v_a_731_);
if (v_isSharedCheck_769_ == 0)
{
v___x_735_ = v_a_731_;
v_isShared_736_ = v_isSharedCheck_769_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_cancelTk_733_);
lean_inc(v_state_732_);
lean_dec(v_a_731_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_769_;
goto v_resetjp_734_;
}
v_resetjp_734_:
{
lean_object* v___x_737_; lean_object* v___x_738_; 
v___x_737_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_4067551214____hygCtx___hyg_9_));
v___x_738_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcDecode___redArg(v___x_737_, v_state_732_, v_a_721_);
if (lean_obj_tag(v___x_738_) == 0)
{
lean_object* v_a_739_; lean_object* v___x_741_; uint8_t v_isShared_742_; uint8_t v_isSharedCheck_746_; 
lean_del_object(v___x_735_);
lean_dec(v_cancelTk_733_);
v_a_739_ = lean_ctor_get(v___x_738_, 0);
v_isSharedCheck_746_ = !lean_is_exclusive(v___x_738_);
if (v_isSharedCheck_746_ == 0)
{
v___x_741_ = v___x_738_;
v_isShared_742_ = v_isSharedCheck_746_;
goto v_resetjp_740_;
}
else
{
lean_inc(v_a_739_);
lean_dec(v___x_738_);
v___x_741_ = lean_box(0);
v_isShared_742_ = v_isSharedCheck_746_;
goto v_resetjp_740_;
}
v_resetjp_740_:
{
lean_object* v___x_744_; 
if (v_isShared_742_ == 0)
{
v___x_744_ = v___x_741_;
goto v_reusejp_743_;
}
else
{
lean_object* v_reuseFailAlloc_745_; 
v_reuseFailAlloc_745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_745_, 0, v_a_739_);
v___x_744_ = v_reuseFailAlloc_745_;
goto v_reusejp_743_;
}
v_reusejp_743_:
{
return v___x_744_;
}
}
}
else
{
lean_object* v_a_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
v_a_747_ = lean_ctor_get(v___x_738_, 0);
lean_inc(v_a_747_);
lean_dec_ref_known(v___x_738_, 1);
v___x_748_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instImpl_00___x40_ProofWidgets_Component_RefreshComponent_1045526817____hygCtx___hyg_3_));
v___x_749_ = l_Lean_Server_instRpcEncodableWithRpcRefOfTypeName_rpcDecode___redArg(v___x_748_, v_cancelTk_733_, v_a_721_);
if (lean_obj_tag(v___x_749_) == 0)
{
lean_object* v_a_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_757_; 
lean_dec(v_a_747_);
lean_del_object(v___x_735_);
v_a_750_ = lean_ctor_get(v___x_749_, 0);
v_isSharedCheck_757_ = !lean_is_exclusive(v___x_749_);
if (v_isSharedCheck_757_ == 0)
{
v___x_752_ = v___x_749_;
v_isShared_753_ = v_isSharedCheck_757_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_a_750_);
lean_dec(v___x_749_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_757_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_755_; 
if (v_isShared_753_ == 0)
{
v___x_755_ = v___x_752_;
goto v_reusejp_754_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v_a_750_);
v___x_755_ = v_reuseFailAlloc_756_;
goto v_reusejp_754_;
}
v_reusejp_754_:
{
return v___x_755_;
}
}
}
else
{
lean_object* v_a_758_; lean_object* v___x_760_; uint8_t v_isShared_761_; uint8_t v_isSharedCheck_768_; 
v_a_758_ = lean_ctor_get(v___x_749_, 0);
v_isSharedCheck_768_ = !lean_is_exclusive(v___x_749_);
if (v_isSharedCheck_768_ == 0)
{
v___x_760_ = v___x_749_;
v_isShared_761_ = v_isSharedCheck_768_;
goto v_resetjp_759_;
}
else
{
lean_inc(v_a_758_);
lean_dec(v___x_749_);
v___x_760_ = lean_box(0);
v_isShared_761_ = v_isSharedCheck_768_;
goto v_resetjp_759_;
}
v_resetjp_759_:
{
lean_object* v___x_763_; 
if (v_isShared_736_ == 0)
{
lean_ctor_set(v___x_735_, 1, v_a_758_);
lean_ctor_set(v___x_735_, 0, v_a_747_);
v___x_763_ = v___x_735_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_a_747_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v_a_758_);
v___x_763_ = v_reuseFailAlloc_767_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
lean_object* v___x_765_; 
if (v_isShared_761_ == 0)
{
lean_ctor_set(v___x_760_, 0, v___x_763_);
v___x_765_ = v___x_760_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v___x_763_);
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
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed(lean_object* v_j_770_, lean_object* v_a_771_){
_start:
{
lean_object* v_res_772_; 
v_res_772_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(v_j_770_, v_a_771_);
lean_dec_ref(v_a_771_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg(lean_object* v_p_784_, lean_object* v___y_785_){
_start:
{
uint32_t v___x_787_; lean_object* v___x_788_; lean_object* v_cancelTk_789_; uint8_t v___x_790_; 
v___x_787_ = 1000;
v___x_788_ = l_IO_sleep(v___x_787_);
v_cancelTk_789_ = lean_ctor_get(v___y_785_, 4);
v___x_790_ = l_Lean_Server_RequestCancellationToken_wasCancelledByCancelRequest(v_cancelTk_789_);
if (v___x_790_ == 0)
{
goto _start;
}
else
{
lean_object* v_cancelTk_792_; lean_object* v_val_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; 
v_cancelTk_792_ = lean_ctor_get(v_p_784_, 1);
v_val_793_ = lean_ctor_get(v_cancelTk_792_, 0);
v___x_794_ = l_IO_CancelToken_set(v_val_793_);
v___x_795_ = ((lean_object*)(lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___closed__1));
v___x_796_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_796_, 0, v___x_795_);
return v___x_796_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg___boxed(lean_object* v_p_797_, lean_object* v___y_798_, lean_object* v___y_799_){
_start:
{
lean_object* v_res_800_; 
v_res_800_ = lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg(v_p_797_, v___y_798_);
lean_dec_ref(v___y_798_);
lean_dec_ref(v_p_797_);
return v_res_800_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0(lean_object* v_p_801_, lean_object* v___x_802_, lean_object* v___y_803_){
_start:
{
lean_object* v___x_805_; 
v___x_805_ = lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg(v_p_801_, v___y_803_);
if (lean_obj_tag(v___x_805_) == 0)
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_818_; 
v_a_806_ = lean_ctor_get(v___x_805_, 0);
v_isSharedCheck_818_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_818_ == 0)
{
v___x_808_ = v___x_805_;
v_isShared_809_ = v_isSharedCheck_818_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_805_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_818_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v_fst_810_; 
v_fst_810_ = lean_ctor_get(v_a_806_, 0);
lean_inc(v_fst_810_);
lean_dec(v_a_806_);
if (lean_obj_tag(v_fst_810_) == 0)
{
lean_object* v___x_812_; 
if (v_isShared_809_ == 0)
{
lean_ctor_set(v___x_808_, 0, v___x_802_);
v___x_812_ = v___x_808_;
goto v_reusejp_811_;
}
else
{
lean_object* v_reuseFailAlloc_813_; 
v_reuseFailAlloc_813_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_813_, 0, v___x_802_);
v___x_812_ = v_reuseFailAlloc_813_;
goto v_reusejp_811_;
}
v_reusejp_811_:
{
return v___x_812_;
}
}
else
{
lean_object* v_val_814_; lean_object* v___x_816_; 
v_val_814_ = lean_ctor_get(v_fst_810_, 0);
lean_inc(v_val_814_);
lean_dec_ref_known(v_fst_810_, 1);
if (v_isShared_809_ == 0)
{
lean_ctor_set(v___x_808_, 0, v_val_814_);
v___x_816_ = v___x_808_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v_val_814_);
v___x_816_ = v_reuseFailAlloc_817_;
goto v_reusejp_815_;
}
v_reusejp_815_:
{
return v___x_816_;
}
}
}
}
else
{
lean_object* v_a_819_; lean_object* v___x_821_; uint8_t v_isShared_822_; uint8_t v_isSharedCheck_826_; 
v_a_819_ = lean_ctor_get(v___x_805_, 0);
v_isSharedCheck_826_ = !lean_is_exclusive(v___x_805_);
if (v_isSharedCheck_826_ == 0)
{
v___x_821_ = v___x_805_;
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
else
{
lean_inc(v_a_819_);
lean_dec(v___x_805_);
v___x_821_ = lean_box(0);
v_isShared_822_ = v_isSharedCheck_826_;
goto v_resetjp_820_;
}
v_resetjp_820_:
{
lean_object* v___x_824_; 
if (v_isShared_822_ == 0)
{
v___x_824_ = v___x_821_;
goto v_reusejp_823_;
}
else
{
lean_object* v_reuseFailAlloc_825_; 
v_reuseFailAlloc_825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_825_, 0, v_a_819_);
v___x_824_ = v_reuseFailAlloc_825_;
goto v_reusejp_823_;
}
v_reusejp_823_:
{
return v___x_824_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0___boxed(lean_object* v_p_827_, lean_object* v___x_828_, lean_object* v___y_829_, lean_object* v___y_830_){
_start:
{
lean_object* v_res_831_; 
v_res_831_ = lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0(v_p_827_, v___x_828_, v___y_829_);
lean_dec_ref(v___y_829_);
lean_dec_ref(v_p_827_);
return v_res_831_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor(lean_object* v_p_832_, lean_object* v_a_833_){
_start:
{
lean_object* v___x_835_; lean_object* v___f_836_; lean_object* v___x_837_; 
v___x_835_ = lean_box(0);
v___f_836_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___lam__0___boxed), 4, 2);
lean_closure_set(v___f_836_, 0, v_p_832_);
lean_closure_set(v___f_836_, 1, v___x_835_);
v___x_837_ = l_Lean_Server_RequestM_asTask___redArg(v___f_836_, v_a_833_);
return v___x_837_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___boxed(lean_object* v_p_838_, lean_object* v_a_839_, lean_object* v_a_840_){
_start:
{
lean_object* v_res_841_; 
v_res_841_ = lp_proofwidgets_ProofWidgets_RefreshComponent_monitor(v_p_838_, v_a_839_);
lean_dec_ref(v_a_839_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0(lean_object* v_p_842_, lean_object* v_inst_843_, lean_object* v_a_844_, lean_object* v___y_845_){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___redArg(v_p_842_, v___y_845_);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0___boxed(lean_object* v_p_848_, lean_object* v_inst_849_, lean_object* v_a_850_, lean_object* v___y_851_, lean_object* v___y_852_){
_start:
{
lean_object* v_res_853_; 
v_res_853_ = lp_proofwidgets___private_Init_While_0__repeatM_erased___at___00ProofWidgets_RefreshComponent_monitor_spec__0(v_p_848_, v_inst_849_, v_a_850_, v___y_851_);
lean_dec_ref(v___y_851_);
lean_dec_ref(v_a_850_);
lean_dec_ref(v_p_848_);
return v_res_853_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2(lean_object* v_val_854_, lean_object* v___f_855_, lean_object* v_x_856_, lean_object* v___y_857_){
_start:
{
if (lean_obj_tag(v_x_856_) == 0)
{
lean_object* v_a_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_866_; 
lean_dec_ref(v___f_855_);
v_a_859_ = lean_ctor_get(v_x_856_, 0);
v_isSharedCheck_866_ = !lean_is_exclusive(v_x_856_);
if (v_isSharedCheck_866_ == 0)
{
v___x_861_ = v_x_856_;
v_isShared_862_ = v_isSharedCheck_866_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_a_859_);
lean_dec(v_x_856_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_866_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v___x_864_; 
if (v_isShared_862_ == 0)
{
lean_ctor_set_tag(v___x_861_, 1);
v___x_864_ = v___x_861_;
goto v_reusejp_863_;
}
else
{
lean_object* v_reuseFailAlloc_865_; 
v_reuseFailAlloc_865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_865_, 0, v_a_859_);
v___x_864_ = v_reuseFailAlloc_865_;
goto v_reusejp_863_;
}
v_reusejp_863_:
{
return v___x_864_;
}
}
}
else
{
lean_object* v___x_868_; uint8_t v_isShared_869_; uint8_t v_isSharedCheck_889_; 
v_isSharedCheck_889_ = !lean_is_exclusive(v_x_856_);
if (v_isSharedCheck_889_ == 0)
{
lean_object* v_unused_890_; 
v_unused_890_ = lean_ctor_get(v_x_856_, 0);
lean_dec(v_unused_890_);
v___x_868_ = v_x_856_;
v_isShared_869_ = v_isSharedCheck_889_;
goto v_resetjp_867_;
}
else
{
lean_dec(v_x_856_);
v___x_868_ = lean_box(0);
v_isShared_869_ = v_isSharedCheck_889_;
goto v_resetjp_867_;
}
v_resetjp_867_:
{
lean_object* v___x_870_; lean_object* v_objects_871_; lean_object* v_expireTime_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_888_; 
v___x_870_ = lean_st_ref_take(v_val_854_);
v_objects_871_ = lean_ctor_get(v___x_870_, 0);
v_expireTime_872_ = lean_ctor_get(v___x_870_, 1);
v_isSharedCheck_888_ = !lean_is_exclusive(v___x_870_);
if (v_isSharedCheck_888_ == 0)
{
v___x_874_ = v___x_870_;
v_isShared_875_ = v_isSharedCheck_888_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_expireTime_872_);
lean_inc(v_objects_871_);
lean_dec(v___x_870_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_888_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v___f_876_; lean_object* v___x_877_; lean_object* v___x_879_; 
v___f_876_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_876_, 0, v_expireTime_872_);
v___x_877_ = lean_box(0);
if (v_isShared_875_ == 0)
{
lean_ctor_set(v___x_874_, 1, v_objects_871_);
lean_ctor_set(v___x_874_, 0, v___x_877_);
v___x_879_ = v___x_874_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_887_; 
v_reuseFailAlloc_887_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_887_, 0, v___x_877_);
lean_ctor_set(v_reuseFailAlloc_887_, 1, v_objects_871_);
v___x_879_ = v_reuseFailAlloc_887_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
lean_object* v___x_880_; lean_object* v_fst_881_; lean_object* v_snd_882_; lean_object* v___x_883_; lean_object* v___x_885_; 
v___x_880_ = l_Prod_map___redArg(v___f_855_, v___f_876_, v___x_879_);
v_fst_881_ = lean_ctor_get(v___x_880_, 0);
lean_inc(v_fst_881_);
v_snd_882_ = lean_ctor_get(v___x_880_, 1);
lean_inc(v_snd_882_);
lean_dec_ref(v___x_880_);
v___x_883_ = lean_st_ref_set(v_val_854_, v_snd_882_);
if (v_isShared_869_ == 0)
{
lean_ctor_set_tag(v___x_868_, 0);
lean_ctor_set(v___x_868_, 0, v_fst_881_);
v___x_885_ = v___x_868_;
goto v_reusejp_884_;
}
else
{
lean_object* v_reuseFailAlloc_886_; 
v_reuseFailAlloc_886_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_886_, 0, v_fst_881_);
v___x_885_ = v_reuseFailAlloc_886_;
goto v_reusejp_884_;
}
v_reusejp_884_:
{
return v___x_885_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_891_, lean_object* v___f_892_, lean_object* v_x_893_, lean_object* v___y_894_, lean_object* v___y_895_){
_start:
{
lean_object* v_res_896_; 
v_res_896_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2(v_val_891_, v___f_892_, v_x_893_, v___y_894_);
lean_dec_ref(v___y_894_);
lean_dec(v_val_891_);
return v_res_896_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0(lean_object* v_method_897_, lean_object* v_handler_898_, lean_object* v___f_899_, uint64_t v_seshId_900_, lean_object* v_j_901_, lean_object* v___y_902_){
_start:
{
lean_object* v_rpcSessions_904_; lean_object* v___x_905_; 
v_rpcSessions_904_ = lean_ctor_get(v___y_902_, 0);
v___x_905_ = lp_proofwidgets_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_904_, v_seshId_900_);
if (lean_obj_tag(v___x_905_) == 1)
{
lean_object* v_val_906_; lean_object* v___x_907_; lean_object* v_objects_908_; lean_object* v___x_909_; 
v_val_906_ = lean_ctor_get(v___x_905_, 0);
lean_inc(v_val_906_);
lean_dec_ref_known(v___x_905_, 1);
v___x_907_ = lean_st_ref_get(v_val_906_);
v_objects_908_ = lean_ctor_get(v___x_907_, 0);
lean_inc_ref(v_objects_908_);
lean_dec(v___x_907_);
lean_inc(v_j_901_);
v___x_909_ = lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_dec_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1_(v_j_901_, v_objects_908_);
lean_dec_ref(v_objects_908_);
if (lean_obj_tag(v___x_909_) == 0)
{
lean_object* v_a_910_; lean_object* v___x_912_; uint8_t v_isShared_913_; uint8_t v_isSharedCheck_930_; 
lean_dec(v_val_906_);
lean_dec_ref(v___f_899_);
lean_dec_ref(v_handler_898_);
v_a_910_ = lean_ctor_get(v___x_909_, 0);
v_isSharedCheck_930_ = !lean_is_exclusive(v___x_909_);
if (v_isSharedCheck_930_ == 0)
{
v___x_912_ = v___x_909_;
v_isShared_913_ = v_isSharedCheck_930_;
goto v_resetjp_911_;
}
else
{
lean_inc(v_a_910_);
lean_dec(v___x_909_);
v___x_912_ = lean_box(0);
v_isShared_913_ = v_isSharedCheck_930_;
goto v_resetjp_911_;
}
v_resetjp_911_:
{
uint8_t v___x_914_; lean_object* v___x_915_; uint8_t v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_928_; 
v___x_914_ = 3;
v___x_915_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_916_ = 1;
v___x_917_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_897_, v___x_916_);
v___x_918_ = lean_string_append(v___x_915_, v___x_917_);
lean_dec_ref(v___x_917_);
v___x_919_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_920_ = lean_string_append(v___x_918_, v___x_919_);
v___x_921_ = l_Lean_Json_compress(v_j_901_);
v___x_922_ = lean_string_append(v___x_920_, v___x_921_);
lean_dec_ref(v___x_921_);
v___x_923_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_924_ = lean_string_append(v___x_922_, v___x_923_);
v___x_925_ = lean_string_append(v___x_924_, v_a_910_);
lean_dec(v_a_910_);
v___x_926_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_926_, 0, v___x_925_);
lean_ctor_set_uint8(v___x_926_, sizeof(void*)*1, v___x_914_);
if (v_isShared_913_ == 0)
{
lean_ctor_set_tag(v___x_912_, 1);
lean_ctor_set(v___x_912_, 0, v___x_926_);
v___x_928_ = v___x_912_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v___x_926_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
else
{
lean_object* v_a_931_; lean_object* v___x_932_; 
lean_dec(v_j_901_);
lean_dec(v_method_897_);
v_a_931_ = lean_ctor_get(v___x_909_, 0);
lean_inc(v_a_931_);
lean_dec_ref_known(v___x_909_, 1);
lean_inc_ref(v___y_902_);
v___x_932_ = lean_apply_3(v_handler_898_, v_a_931_, v___y_902_, lean_box(0));
if (lean_obj_tag(v___x_932_) == 0)
{
lean_object* v_a_933_; lean_object* v___f_934_; lean_object* v___x_935_; 
v_a_933_ = lean_ctor_get(v___x_932_, 0);
lean_inc(v_a_933_);
lean_dec_ref_known(v___x_932_, 1);
v___f_934_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_934_, 0, v_val_906_);
lean_closure_set(v___f_934_, 1, v___f_899_);
v___x_935_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_933_, v___f_934_, v___y_902_);
return v___x_935_;
}
else
{
lean_object* v_a_936_; lean_object* v___x_938_; uint8_t v_isShared_939_; uint8_t v_isSharedCheck_943_; 
lean_dec(v_val_906_);
lean_dec_ref(v___f_899_);
v_a_936_ = lean_ctor_get(v___x_932_, 0);
v_isSharedCheck_943_ = !lean_is_exclusive(v___x_932_);
if (v_isSharedCheck_943_ == 0)
{
v___x_938_ = v___x_932_;
v_isShared_939_ = v_isSharedCheck_943_;
goto v_resetjp_937_;
}
else
{
lean_inc(v_a_936_);
lean_dec(v___x_932_);
v___x_938_ = lean_box(0);
v_isShared_939_ = v_isSharedCheck_943_;
goto v_resetjp_937_;
}
v_resetjp_937_:
{
lean_object* v___x_941_; 
if (v_isShared_939_ == 0)
{
v___x_941_ = v___x_938_;
goto v_reusejp_940_;
}
else
{
lean_object* v_reuseFailAlloc_942_; 
v_reuseFailAlloc_942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_942_, 0, v_a_936_);
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
else
{
lean_object* v___x_944_; lean_object* v___x_945_; 
lean_dec(v___x_905_);
lean_dec(v_j_901_);
lean_dec_ref(v___f_899_);
lean_dec_ref(v_handler_898_);
lean_dec(v_method_897_);
v___x_944_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_945_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
return v___x_945_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v_method_946_, lean_object* v_handler_947_, lean_object* v___f_948_, lean_object* v_seshId_949_, lean_object* v_j_950_, lean_object* v___y_951_, lean_object* v___y_952_){
_start:
{
uint64_t v_seshId_boxed_953_; lean_object* v_res_954_; 
v_seshId_boxed_953_ = lean_unbox_uint64(v_seshId_949_);
lean_dec_ref(v_seshId_949_);
v_res_954_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0(v_method_946_, v_handler_947_, v___f_948_, v_seshId_boxed_953_, v_j_950_, v___y_951_);
lean_dec_ref(v___y_951_);
return v_res_954_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0(lean_object* v_method_955_, lean_object* v_handler_956_){
_start:
{
lean_object* v___f_957_; lean_object* v___f_958_; 
v___f_957_ = ((lean_object*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped_spec__0___closed__0));
v___f_958_ = lean_alloc_closure((void*)(lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0___lam__0___boxed), 7, 3);
lean_closure_set(v___f_958_, 0, v_method_955_);
lean_closure_set(v___f_958_, 1, v_handler_956_);
lean_closure_set(v___f_958_, 2, v___f_957_);
return v___f_958_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3(void){
_start:
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; 
v___x_965_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__2));
v___x_966_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__1));
v___x_967_ = lp_proofwidgets_Lean_Server_wrapRpcProcedure___at___00ProofWidgets_RefreshComponent_monitor___rpc__wrapped_spec__0(v___x_966_, v___x_965_);
return v___x_967_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped(void){
_start:
{
lean_object* v___x_968_; 
v___x_968_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3, &lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped___closed__3);
return v___x_968_;
}
}
static uint64_t _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1(void){
_start:
{
lean_object* v___x_970_; uint64_t v___x_971_; 
v___x_970_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent___closed__0));
v___x_971_ = lean_string_hash(v___x_970_);
return v___x_971_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2(void){
_start:
{
uint64_t v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; 
v___x_972_ = lean_uint64_once(&lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1, &lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__1);
v___x_973_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent___closed__0));
v___x_974_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_974_, 0, v___x_973_);
lean_ctor_set_uint64(v___x_974_, sizeof(void*)*1, v___x_972_);
return v___x_974_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4(void){
_start:
{
lean_object* v___x_976_; lean_object* v___x_977_; lean_object* v___x_978_; 
v___x_976_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent___closed__3));
v___x_977_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2, &lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__2);
v___x_978_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_978_, 0, v___x_977_);
lean_ctor_set(v___x_978_, 1, v___x_976_);
return v___x_978_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_RefreshComponent(void){
_start:
{
lean_object* v___x_979_; 
v___x_979_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4, &lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4_once, _init_lp_proofwidgets_ProofWidgets_RefreshComponent___closed__4);
return v___x_979_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new(lean_object* v_initial_980_){
_start:
{
lean_object* v___x_982_; lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
v___x_982_ = lean_io_promise_new();
v___x_983_ = lean_unsigned_to_nat(1u);
v___x_984_ = lean_io_promise_result_opt(v___x_982_);
v___x_985_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_985_, 0, v_initial_980_);
lean_ctor_set(v___x_985_, 1, v___x_983_);
lean_ctor_set(v___x_985_, 2, v___x_984_);
v___x_986_ = l_Std_Mutex_new___redArg(v___x_985_);
v___x_987_ = l_IO_CancelToken_new();
v___x_988_ = lean_st_mk_ref(v___x_982_);
v___x_989_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_989_, 0, v___x_986_);
lean_ctor_set(v___x_989_, 1, v___x_987_);
lean_ctor_set(v___x_989_, 2, v___x_988_);
return v___x_989_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new___boxed(lean_object* v_initial_990_, lean_object* v_a_991_){
_start:
{
lean_object* v_res_992_; 
v_res_992_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new(v_initial_990_);
return v_res_992_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg(lean_object* v_mutex_993_, lean_object* v_k_994_){
_start:
{
lean_object* v_ref_996_; lean_object* v_mutex_997_; lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
v_ref_996_ = lean_ctor_get(v_mutex_993_, 0);
lean_inc(v_ref_996_);
v_mutex_997_ = lean_ctor_get(v_mutex_993_, 1);
lean_inc(v_mutex_997_);
lean_dec_ref(v_mutex_993_);
v___x_998_ = lean_io_basemutex_lock(v_mutex_997_);
v___x_999_ = lean_apply_2(v_k_994_, v_ref_996_, lean_box(0));
v___x_1000_ = lean_io_basemutex_unlock(v_mutex_997_);
lean_dec(v_mutex_997_);
return v___x_999_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg___boxed(lean_object* v_mutex_1001_, lean_object* v_k_1002_, lean_object* v___y_1003_){
_start:
{
lean_object* v_res_1004_; 
v_res_1004_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg(v_mutex_1001_, v_k_1002_);
return v_res_1004_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0(lean_object* v_00_u03b1_1005_, lean_object* v_00_u03b2_1006_, lean_object* v_mutex_1007_, lean_object* v_k_1008_){
_start:
{
lean_object* v___x_1010_; 
v___x_1010_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg(v_mutex_1007_, v_k_1008_);
return v___x_1010_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___boxed(lean_object* v_00_u03b1_1011_, lean_object* v_00_u03b2_1012_, lean_object* v_mutex_1013_, lean_object* v_k_1014_, lean_object* v___y_1015_){
_start:
{
lean_object* v_res_1016_; 
v_res_1016_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0(v_00_u03b1_1011_, v_00_u03b2_1012_, v_mutex_1013_, v_k_1014_);
return v_res_1016_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0(lean_object* v_promise_1017_, lean_object* v_val_1018_, lean_object* v_html_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v_idx_1024_; lean_object* v___x_1026_; uint8_t v_isShared_1027_; uint8_t v_isSharedCheck_1035_; 
v___x_1022_ = lean_st_ref_get(v___y_1020_);
lean_inc(v_val_1018_);
v___x_1023_ = lean_st_ref_swap(v_promise_1017_, v_val_1018_);
v_idx_1024_ = lean_ctor_get(v___x_1022_, 1);
v_isSharedCheck_1035_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1035_ == 0)
{
lean_object* v_unused_1036_; lean_object* v_unused_1037_; 
v_unused_1036_ = lean_ctor_get(v___x_1022_, 2);
lean_dec(v_unused_1036_);
v_unused_1037_ = lean_ctor_get(v___x_1022_, 0);
lean_dec(v_unused_1037_);
v___x_1026_ = v___x_1022_;
v_isShared_1027_ = v_isSharedCheck_1035_;
goto v_resetjp_1025_;
}
else
{
lean_inc(v_idx_1024_);
lean_dec(v___x_1022_);
v___x_1026_ = lean_box(0);
v_isShared_1027_ = v_isSharedCheck_1035_;
goto v_resetjp_1025_;
}
v_resetjp_1025_:
{
lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1032_; 
v___x_1028_ = lean_unsigned_to_nat(1u);
v___x_1029_ = lean_nat_add(v_idx_1024_, v___x_1028_);
lean_dec(v_idx_1024_);
v___x_1030_ = lean_io_promise_result_opt(v_val_1018_);
lean_dec(v_val_1018_);
if (v_isShared_1027_ == 0)
{
lean_ctor_set(v___x_1026_, 2, v___x_1030_);
lean_ctor_set(v___x_1026_, 1, v___x_1029_);
lean_ctor_set(v___x_1026_, 0, v_html_1019_);
v___x_1032_ = v___x_1026_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1034_; 
v_reuseFailAlloc_1034_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1034_, 0, v_html_1019_);
lean_ctor_set(v_reuseFailAlloc_1034_, 1, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1034_, 2, v___x_1030_);
v___x_1032_ = v_reuseFailAlloc_1034_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
lean_object* v___x_1033_; 
v___x_1033_ = lean_st_ref_set(v___y_1020_, v___x_1032_);
return v___x_1023_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0___boxed(lean_object* v_promise_1038_, lean_object* v_val_1039_, lean_object* v_html_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
lean_object* v_res_1043_; 
v_res_1043_ = lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0(v_promise_1038_, v_val_1039_, v_html_1040_, v___y_1041_);
lean_dec(v___y_1041_);
lean_dec(v_promise_1038_);
return v_res_1043_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy(lean_object* v_token_1044_, lean_object* v_html_1045_){
_start:
{
lean_object* v_state_1047_; lean_object* v_promise_1048_; lean_object* v___x_1049_; lean_object* v___f_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; lean_object* v___x_1053_; 
v_state_1047_ = lean_ctor_get(v_token_1044_, 0);
lean_inc_ref(v_state_1047_);
v_promise_1048_ = lean_ctor_get(v_token_1044_, 2);
lean_inc(v_promise_1048_);
lean_dec_ref(v_token_1044_);
v___x_1049_ = lean_io_promise_new();
v___f_1050_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___lam__0___boxed), 5, 3);
lean_closure_set(v___f_1050_, 0, v_promise_1048_);
lean_closure_set(v___f_1050_, 1, v___x_1049_);
lean_closure_set(v___f_1050_, 2, v_html_1045_);
v___x_1051_ = lp_proofwidgets_Std_Mutex_atomically___at___00ProofWidgets_RefreshToken_updateLazy_spec__0___redArg(v_state_1047_, v___f_1050_);
v___x_1052_ = lean_box(0);
v___x_1053_ = lean_io_promise_resolve(v___x_1052_, v___x_1051_);
lean_dec(v___x_1051_);
return v___x_1053_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy___boxed(lean_object* v_token_1054_, lean_object* v_html_1055_, lean_object* v_a_1056_){
_start:
{
lean_object* v_res_1057_; 
v_res_1057_ = lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy(v_token_1054_, v_html_1055_);
return v_res_1057_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_update(lean_object* v_token_1058_, lean_object* v_html_1059_){
_start:
{
lean_object* v___x_1061_; lean_object* v___x_1062_; 
v___x_1061_ = lean_thunk_pure(v_html_1059_);
v___x_1062_ = lp_proofwidgets_ProofWidgets_RefreshToken_updateLazy(v_token_1058_, v___x_1061_);
return v___x_1062_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_update___boxed(lean_object* v_token_1063_, lean_object* v_html_1064_, lean_object* v_a_1065_){
_start:
{
lean_object* v_res_1066_; 
v_res_1066_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_token_1063_, v_html_1064_);
return v_res_1066_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0(lean_object* v_c_1067_, lean_object* v_props_1068_, lean_object* v_children_1069_){
_start:
{
lean_object* v_toModule_1070_; lean_object* v_export_1071_; lean_object* v_javascript_1072_; uint64_t v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; 
v_toModule_1070_ = lean_ctor_get(v_c_1067_, 0);
v_export_1071_ = lean_ctor_get(v_c_1067_, 1);
v_javascript_1072_ = lean_ctor_get(v_toModule_1070_, 0);
v___x_1073_ = lean_string_hash(v_javascript_1072_);
v___x_1074_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instRpcEncodableProps_enc_00___x40_ProofWidgets_Component_RefreshComponent_1995342541____hygCtx___hyg_1____boxed), 2, 1);
lean_closure_set(v___x_1074_, 0, v_props_1068_);
lean_inc_ref(v_export_1071_);
v___x_1075_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_1075_, 0, v_export_1071_);
lean_ctor_set(v___x_1075_, 1, v___x_1074_);
lean_ctor_set(v___x_1075_, 2, v_children_1069_);
lean_ctor_set_uint64(v___x_1075_, sizeof(void*)*3, v___x_1073_);
return v___x_1075_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0___boxed(lean_object* v_c_1076_, lean_object* v_props_1077_, lean_object* v_children_1078_){
_start:
{
lean_object* v_res_1079_; 
v_res_1079_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0(v_c_1076_, v_props_1077_, v_children_1078_);
lean_dec_ref(v_c_1076_);
return v_res_1079_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0(lean_object* v_initial_1080_, lean_object* v_x_1081_){
_start:
{
lean_inc_ref(v_initial_1080_);
return v_initial_1080_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0___boxed(lean_object* v_initial_1082_, lean_object* v_x_1083_){
_start:
{
lean_object* v_res_1084_; 
v_res_1084_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0(v_initial_1082_, v_x_1083_);
lean_dec_ref(v_initial_1082_);
return v_res_1084_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent(lean_object* v_initial_1087_){
_start:
{
lean_object* v___f_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v_state_1092_; lean_object* v_cancelTk_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; 
v___f_1089_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponent___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1089_, 0, v_initial_1087_);
v___x_1090_ = lean_mk_thunk(v___f_1089_);
v___x_1091_ = lp_proofwidgets___private_ProofWidgets_Component_RefreshComponent_0__ProofWidgets_RefreshToken_new(v___x_1090_);
v_state_1092_ = lean_ctor_get(v___x_1091_, 0);
lean_inc_ref(v_state_1092_);
v_cancelTk_1093_ = lean_ctor_get(v___x_1091_, 1);
lean_inc_ref(v_cancelTk_1093_);
v___x_1094_ = l_Lean_Server_WithRpcRef_mk___redArg(v_state_1092_);
v___x_1095_ = l_Lean_Server_WithRpcRef_mk___redArg(v_cancelTk_1093_);
v___x_1096_ = lp_proofwidgets_ProofWidgets_RefreshComponent;
v___x_1097_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1097_, 0, v___x_1094_);
lean_ctor_set(v___x_1097_, 1, v___x_1095_);
v___x_1098_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_mkRefreshComponent___closed__0));
v___x_1099_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponent_spec__0(v___x_1096_, v___x_1097_, v___x_1098_);
v___x_1100_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1100_, 0, v___x_1099_);
lean_ctor_set(v___x_1100_, 1, v___x_1091_);
return v___x_1100_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent___boxed(lean_object* v_initial_1101_, lean_object* v_a_1102_){
_start:
{
lean_object* v_res_1103_; 
v_res_1103_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent(v_initial_1101_);
return v_res_1103_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0(lean_object* v_snd_1104_, lean_object* v_x_1105_){
_start:
{
lean_object* v_fileName_1106_; lean_object* v_fileMap_1107_; lean_object* v_options_1108_; lean_object* v_currRecDepth_1109_; lean_object* v_maxRecDepth_1110_; lean_object* v_ref_1111_; lean_object* v_currNamespace_1112_; lean_object* v_openDecls_1113_; lean_object* v_initHeartbeats_1114_; lean_object* v_maxHeartbeats_1115_; lean_object* v_quotContext_1116_; lean_object* v_currMacroScope_1117_; uint8_t v_diag_1118_; uint8_t v_suppressElabErrors_1119_; lean_object* v_inheritedTraceOptions_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1129_; 
v_fileName_1106_ = lean_ctor_get(v_x_1105_, 0);
v_fileMap_1107_ = lean_ctor_get(v_x_1105_, 1);
v_options_1108_ = lean_ctor_get(v_x_1105_, 2);
v_currRecDepth_1109_ = lean_ctor_get(v_x_1105_, 3);
v_maxRecDepth_1110_ = lean_ctor_get(v_x_1105_, 4);
v_ref_1111_ = lean_ctor_get(v_x_1105_, 5);
v_currNamespace_1112_ = lean_ctor_get(v_x_1105_, 6);
v_openDecls_1113_ = lean_ctor_get(v_x_1105_, 7);
v_initHeartbeats_1114_ = lean_ctor_get(v_x_1105_, 8);
v_maxHeartbeats_1115_ = lean_ctor_get(v_x_1105_, 9);
v_quotContext_1116_ = lean_ctor_get(v_x_1105_, 10);
v_currMacroScope_1117_ = lean_ctor_get(v_x_1105_, 11);
v_diag_1118_ = lean_ctor_get_uint8(v_x_1105_, sizeof(void*)*14);
v_suppressElabErrors_1119_ = lean_ctor_get_uint8(v_x_1105_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1120_ = lean_ctor_get(v_x_1105_, 13);
v_isSharedCheck_1129_ = !lean_is_exclusive(v_x_1105_);
if (v_isSharedCheck_1129_ == 0)
{
lean_object* v_unused_1130_; 
v_unused_1130_ = lean_ctor_get(v_x_1105_, 12);
lean_dec(v_unused_1130_);
v___x_1122_ = v_x_1105_;
v_isShared_1123_ = v_isSharedCheck_1129_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_inheritedTraceOptions_1120_);
lean_inc(v_currMacroScope_1117_);
lean_inc(v_quotContext_1116_);
lean_inc(v_maxHeartbeats_1115_);
lean_inc(v_initHeartbeats_1114_);
lean_inc(v_openDecls_1113_);
lean_inc(v_currNamespace_1112_);
lean_inc(v_ref_1111_);
lean_inc(v_maxRecDepth_1110_);
lean_inc(v_currRecDepth_1109_);
lean_inc(v_options_1108_);
lean_inc(v_fileMap_1107_);
lean_inc(v_fileName_1106_);
lean_dec(v_x_1105_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1129_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
lean_object* v_cancelTk_1124_; lean_object* v___x_1125_; lean_object* v___x_1127_; 
v_cancelTk_1124_ = lean_ctor_get(v_snd_1104_, 1);
lean_inc_ref(v_cancelTk_1124_);
v___x_1125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1125_, 0, v_cancelTk_1124_);
if (v_isShared_1123_ == 0)
{
lean_ctor_set(v___x_1122_, 12, v___x_1125_);
v___x_1127_ = v___x_1122_;
goto v_reusejp_1126_;
}
else
{
lean_object* v_reuseFailAlloc_1128_; 
v_reuseFailAlloc_1128_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_1128_, 0, v_fileName_1106_);
lean_ctor_set(v_reuseFailAlloc_1128_, 1, v_fileMap_1107_);
lean_ctor_set(v_reuseFailAlloc_1128_, 2, v_options_1108_);
lean_ctor_set(v_reuseFailAlloc_1128_, 3, v_currRecDepth_1109_);
lean_ctor_set(v_reuseFailAlloc_1128_, 4, v_maxRecDepth_1110_);
lean_ctor_set(v_reuseFailAlloc_1128_, 5, v_ref_1111_);
lean_ctor_set(v_reuseFailAlloc_1128_, 6, v_currNamespace_1112_);
lean_ctor_set(v_reuseFailAlloc_1128_, 7, v_openDecls_1113_);
lean_ctor_set(v_reuseFailAlloc_1128_, 8, v_initHeartbeats_1114_);
lean_ctor_set(v_reuseFailAlloc_1128_, 9, v_maxHeartbeats_1115_);
lean_ctor_set(v_reuseFailAlloc_1128_, 10, v_quotContext_1116_);
lean_ctor_set(v_reuseFailAlloc_1128_, 11, v_currMacroScope_1117_);
lean_ctor_set(v_reuseFailAlloc_1128_, 12, v___x_1125_);
lean_ctor_set(v_reuseFailAlloc_1128_, 13, v_inheritedTraceOptions_1120_);
lean_ctor_set_uint8(v_reuseFailAlloc_1128_, sizeof(void*)*14, v_diag_1118_);
lean_ctor_set_uint8(v_reuseFailAlloc_1128_, sizeof(void*)*14 + 1, v_suppressElabErrors_1119_);
v___x_1127_ = v_reuseFailAlloc_1128_;
goto v_reusejp_1126_;
}
v_reusejp_1126_:
{
return v___x_1127_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0___boxed(lean_object* v_snd_1131_, lean_object* v_x_1132_){
_start:
{
lean_object* v_res_1133_; 
v_res_1133_ = lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0(v_snd_1131_, v_x_1132_);
lean_dec_ref(v_snd_1131_);
return v_res_1133_;
}
}
static lean_object* _init_lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5(void){
_start:
{
lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; 
v___x_1141_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__4));
v___x_1142_ = lean_unsigned_to_nat(2u);
v___x_1143_ = lean_mk_empty_array_with_capacity(v___x_1142_);
v___x_1144_ = lean_array_push(v___x_1143_, v___x_1141_);
return v___x_1144_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1(lean_object* v_snd_1145_, lean_object* v___x_1146_, lean_object* v_ex_1147_){
_start:
{
if (lean_obj_tag(v_ex_1147_) == 1)
{
lean_object* v_id_1149_; lean_object* v___x_1150_; uint8_t v___x_1151_; 
lean_dec_ref(v___x_1146_);
v_id_1149_ = lean_ctor_get(v_ex_1147_, 0);
lean_inc(v_id_1149_);
lean_dec_ref_known(v_ex_1147_, 2);
v___x_1150_ = l_Lean_interruptExceptionId;
v___x_1151_ = l_Lean_instBEqInternalExceptionId_beq(v_id_1149_, v___x_1150_);
lean_dec(v_id_1149_);
if (v___x_1151_ == 0)
{
lean_object* v___x_1152_; 
lean_dec_ref(v_snd_1145_);
v___x_1152_ = lean_box(0);
return v___x_1152_;
}
else
{
lean_object* v___x_1153_; lean_object* v___x_1154_; 
v___x_1153_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__1));
v___x_1154_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_snd_1145_, v___x_1153_);
return v___x_1154_;
}
}
else
{
lean_object* v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; 
v___x_1155_ = l_Lean_Exception_toMessageData(v_ex_1147_);
v___x_1156_ = l_Lean_Server_WithRpcRef_mk___redArg(v___x_1155_);
v___x_1157_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__2));
v___x_1158_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_RefreshComponent_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ProofWidgets_Component_RefreshComponent_1021021771____hygCtx___hyg_31_));
v___x_1159_ = lp_proofwidgets_ProofWidgets_InteractiveMessage;
v___x_1160_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(v___x_1146_, v___x_1159_, v___x_1156_, v___x_1158_);
v___x_1161_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5, &lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5_once, _init_lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___closed__5);
v___x_1162_ = lean_array_push(v___x_1161_, v___x_1160_);
v___x_1163_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1163_, 0, v___x_1157_);
lean_ctor_set(v___x_1163_, 1, v___x_1158_);
lean_ctor_set(v___x_1163_, 2, v___x_1162_);
v___x_1164_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_snd_1145_, v___x_1163_);
return v___x_1164_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___boxed(lean_object* v_snd_1165_, lean_object* v___x_1166_, lean_object* v_ex_1167_, lean_object* v___y_1168_){
_start:
{
lean_object* v_res_1169_; 
v_res_1169_ = lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1(v_snd_1165_, v___x_1166_, v_ex_1167_);
return v_res_1169_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__2(lean_object* v_toPure_1170_, lean_object* v_fst_1171_, lean_object* v_____r_1172_){
_start:
{
lean_object* v___x_1173_; 
v___x_1173_ = lean_apply_2(v_toPure_1170_, lean_box(0), v_fst_1171_);
return v___x_1173_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__3(lean_object* v_toFunctor_1174_, lean_object* v___f_1175_, lean_object* v_inst_1176_, lean_object* v_toBind_1177_, lean_object* v___f_1178_, lean_object* v_____do__lift_1179_){
_start:
{
lean_object* v_mapConst_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; 
v_mapConst_1180_ = lean_ctor_get(v_toFunctor_1174_, 1);
lean_inc(v_mapConst_1180_);
lean_dec_ref(v_toFunctor_1174_);
v___x_1181_ = lean_alloc_closure((void*)(l_EIO_catchExceptions___boxed), 5, 4);
lean_closure_set(v___x_1181_, 0, lean_box(0));
lean_closure_set(v___x_1181_, 1, lean_box(0));
lean_closure_set(v___x_1181_, 2, v_____do__lift_1179_);
lean_closure_set(v___x_1181_, 3, v___f_1175_);
v___x_1182_ = lean_unsigned_to_nat(9u);
v___x_1183_ = lean_alloc_closure((void*)(l_BaseIO_asTask___boxed), 4, 3);
lean_closure_set(v___x_1183_, 0, lean_box(0));
lean_closure_set(v___x_1183_, 1, v___x_1181_);
lean_closure_set(v___x_1183_, 2, v___x_1182_);
v___x_1184_ = lean_apply_2(v_inst_1176_, lean_box(0), v___x_1183_);
v___x_1185_ = lean_box(0);
v___x_1186_ = lean_apply_4(v_mapConst_1180_, lean_box(0), lean_box(0), v___x_1185_, v___x_1184_);
v___x_1187_ = lean_apply_4(v_toBind_1177_, lean_box(0), lean_box(0), v___x_1186_, v___f_1178_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__4(lean_object* v___x_1188_, lean_object* v_toPure_1189_, lean_object* v_toFunctor_1190_, lean_object* v_inst_1191_, lean_object* v_toBind_1192_, lean_object* v_k_1193_, lean_object* v_inst_1194_, lean_object* v_inst_1195_, lean_object* v_____x_1196_){
_start:
{
lean_object* v_fst_1197_; lean_object* v_snd_1198_; lean_object* v___f_1199_; lean_object* v___f_1200_; lean_object* v___f_1201_; lean_object* v___f_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v_mkAct_1205_; lean_object* v___x_1206_; 
v_fst_1197_ = lean_ctor_get(v_____x_1196_, 0);
lean_inc(v_fst_1197_);
v_snd_1198_ = lean_ctor_get(v_____x_1196_, 1);
lean_inc_n(v_snd_1198_, 3);
lean_dec_ref(v_____x_1196_);
v___f_1199_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1199_, 0, v_snd_1198_);
v___f_1200_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_1200_, 0, v_snd_1198_);
lean_closure_set(v___f_1200_, 1, v___x_1188_);
v___f_1201_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__2), 3, 2);
lean_closure_set(v___f_1201_, 0, v_toPure_1189_);
lean_closure_set(v___f_1201_, 1, v_fst_1197_);
lean_inc(v_toBind_1192_);
v___f_1202_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__3), 6, 5);
lean_closure_set(v___f_1202_, 0, v_toFunctor_1190_);
lean_closure_set(v___f_1202_, 1, v___f_1200_);
lean_closure_set(v___f_1202_, 2, v_inst_1191_);
lean_closure_set(v___f_1202_, 3, v_toBind_1192_);
lean_closure_set(v___f_1202_, 4, v___f_1201_);
v___x_1203_ = lean_apply_1(v_k_1193_, v_snd_1198_);
v___x_1204_ = lean_apply_3(v_inst_1194_, lean_box(0), v___f_1199_, v___x_1203_);
v_mkAct_1205_ = lean_apply_2(v_inst_1195_, lean_box(0), v___x_1204_);
v___x_1206_ = lean_apply_4(v_toBind_1192_, lean_box(0), lean_box(0), v_mkAct_1205_, v___f_1202_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg(lean_object* v_inst_1207_, lean_object* v_inst_1208_, lean_object* v_inst_1209_, lean_object* v_inst_1210_, lean_object* v_initial_1211_, lean_object* v_k_1212_){
_start:
{
lean_object* v___x_1213_; lean_object* v_toApplicative_1214_; lean_object* v_toBind_1215_; lean_object* v_toFunctor_1216_; lean_object* v_toPure_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___f_1220_; lean_object* v___x_1221_; 
v___x_1213_ = lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps;
v_toApplicative_1214_ = lean_ctor_get(v_inst_1207_, 0);
lean_inc_ref(v_toApplicative_1214_);
v_toBind_1215_ = lean_ctor_get(v_inst_1207_, 1);
lean_inc_n(v_toBind_1215_, 2);
lean_dec_ref(v_inst_1207_);
v_toFunctor_1216_ = lean_ctor_get(v_toApplicative_1214_, 0);
lean_inc_ref(v_toFunctor_1216_);
v_toPure_1217_ = lean_ctor_get(v_toApplicative_1214_, 1);
lean_inc(v_toPure_1217_);
lean_dec_ref(v_toApplicative_1214_);
v___x_1218_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponent___boxed), 2, 1);
lean_closure_set(v___x_1218_, 0, v_initial_1211_);
lean_inc(v_inst_1208_);
v___x_1219_ = lean_apply_2(v_inst_1208_, lean_box(0), v___x_1218_);
v___f_1220_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg___lam__4), 9, 8);
lean_closure_set(v___f_1220_, 0, v___x_1213_);
lean_closure_set(v___f_1220_, 1, v_toPure_1217_);
lean_closure_set(v___f_1220_, 2, v_toFunctor_1216_);
lean_closure_set(v___f_1220_, 3, v_inst_1208_);
lean_closure_set(v___f_1220_, 4, v_toBind_1215_);
lean_closure_set(v___f_1220_, 5, v_k_1212_);
lean_closure_set(v___f_1220_, 6, v_inst_1210_);
lean_closure_set(v___f_1220_, 7, v_inst_1209_);
v___x_1221_ = lean_apply_4(v_toBind_1215_, lean_box(0), lean_box(0), v___x_1219_, v___f_1220_);
return v___x_1221_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponentM(lean_object* v_m_1222_, lean_object* v_inst_1223_, lean_object* v_inst_1224_, lean_object* v_inst_1225_, lean_object* v_inst_1226_, lean_object* v_initial_1227_, lean_object* v_k_1228_){
_start:
{
lean_object* v___x_1229_; 
v___x_1229_ = lp_proofwidgets_ProofWidgets_mkRefreshComponentM___redArg(v_inst_1223_, v_inst_1224_, v_inst_1225_, v_inst_1226_, v_initial_1227_, v_k_1228_);
return v___x_1229_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default = _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState_default);
lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState = _init_lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_RefreshComponent_instInhabitedRefreshState);
lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped = _init_lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_RefreshComponent_awaitRefresh___rpc__wrapped);
lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped = _init_lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_RefreshComponent_monitor___rpc__wrapped);
lp_proofwidgets_ProofWidgets_RefreshComponent = _init_lp_proofwidgets_ProofWidgets_RefreshComponent();
lean_mark_persistent(lp_proofwidgets_ProofWidgets_RefreshComponent);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Component_RefreshComponent(builtin);
}
#ifdef __cplusplus
}
#endif
