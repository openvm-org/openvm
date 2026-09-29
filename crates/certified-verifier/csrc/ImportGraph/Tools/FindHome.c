// Lean compiler output
// Module: ImportGraph.Tools.FindHome
// Imports: public import Init public meta import Init public meta import Lean.Elab.Command public meta import Lean.Widget.UserWidget public meta import ImportGraph.Imports.RequiredModules public meta import ImportGraph.Imports.ImportGraph public meta import ImportGraph.Graph.TransitiveClosure
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
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint64_t lean_string_hash(lean_object*);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Name_fromJson_x3f(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Server_documentUriFromModule_x3f(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_invalidParams(lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_maxView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Array_eraseIdx___redArg(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_Environment_importGraph(lean_object*);
lean_object* lp_importGraph_Lean_NameMap_transitiveClosure(lean_object*);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_importGraph_Lean_Name_requiredModules(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Widget_WidgetInstance_ofHash___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Name_findHome(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Name_findHome___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_getModuleUri___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "couldn't find URI for module '"};
static const lean_object* lp_importGraph_getModuleUri___lam__0___closed__0 = (const lean_object*)&lp_importGraph_getModuleUri___lam__0___closed__0_value;
static const lean_string_object lp_importGraph_getModuleUri___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_importGraph_getModuleUri___lam__0___closed__1 = (const lean_object*)&lp_importGraph_getModuleUri___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_getModuleUri___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "getModuleUri"};
static const lean_object* lp_importGraph_getModuleUri___rpc__wrapped___closed__0 = (const lean_object*)&lp_importGraph_getModuleUri___rpc__wrapped___closed__0_value;
static const lean_ctor_object lp_importGraph_getModuleUri___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_getModuleUri___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 151, 91, 132, 175, 17, 228, 58)}};
static const lean_object* lp_importGraph_getModuleUri___rpc__wrapped___closed__1 = (const lean_object*)&lp_importGraph_getModuleUri___rpc__wrapped___closed__1_value;
static const lean_closure_object lp_importGraph_getModuleUri___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_getModuleUri___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_getModuleUri___rpc__wrapped___closed__2 = (const lean_object*)&lp_importGraph_getModuleUri___rpc__wrapped___closed__2_value;
static lean_once_cell_t lp_importGraph_getModuleUri___rpc__wrapped___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_getModuleUri___rpc__wrapped___closed__3;
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___rpc__wrapped;
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_importGraph_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "modName"};
static const lean_object* lp_importGraph_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_ = (const lean_object*)&lp_importGraph_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__value;
LEAN_EXPORT lean_object* lp_importGraph_instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_(lean_object*);
static const lean_closure_object lp_importGraph_instFromJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_instFromJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_ = (const lean_object*)&lp_importGraph_instFromJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__value;
LEAN_EXPORT const lean_object* lp_importGraph_instFromJsonRpcEncodablePacket_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_ = (const lean_object*)&lp_importGraph_instFromJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__value;
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__spec__0(lean_object*, lean_object*);
static const lean_array_object lp_importGraph_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_ = (const lean_object*)&lp_importGraph_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__value;
LEAN_EXPORT lean_object* lp_importGraph_instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_(lean_object*);
static const lean_closure_object lp_importGraph_instToJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_instToJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_ = (const lean_object*)&lp_importGraph_instToJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__value;
LEAN_EXPORT const lean_object* lp_importGraph_instToJsonRpcEncodablePacket_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_ = (const lean_object*)&lp_importGraph_instToJsonRpcEncodablePacket___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__value;
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec___redArg_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
static const lean_closure_object lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__0 = (const lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__0_value;
static const lean_closure_object lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__1 = (const lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__1_value;
static const lean_ctor_object lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__0_value),((lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__1_value)}};
static const lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__2 = (const lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__2_value;
LEAN_EXPORT const lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps = (const lean_object*)&lp_importGraph_instRpcEncodableGoToModuleLinkProps___closed__2_value;
static const lean_string_object lp_importGraph_GoToModuleLink___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 581, .m_capacity = 581, .m_length = 580, .m_data = "\n    import * as React from 'react'\n    import { EditorContext, useRpcSession } from '@leanprover/infoview'\n\n    export default function(props) {\n      const ec = React.useContext(EditorContext)\n      const rs = useRpcSession()\n      return React.createElement('a',\n        {\n          className: 'link pointer dim',\n          onClick: async () => {\n            try {\n              const uri = await rs.call('getModuleUri', props.modName)\n              ec.revealPosition({ uri, line: 0, character: 0 })\n            } catch {}\n          }\n        },\n        props.modName)\n    }\n  "};
static const lean_object* lp_importGraph_GoToModuleLink___closed__0 = (const lean_object*)&lp_importGraph_GoToModuleLink___closed__0_value;
static lean_once_cell_t lp_importGraph_GoToModuleLink___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_importGraph_GoToModuleLink___closed__1;
static lean_once_cell_t lp_importGraph_GoToModuleLink___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_GoToModuleLink___closed__2;
LEAN_EXPORT lean_object* lp_importGraph_GoToModuleLink;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "command#find_home!_"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__0 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__0_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 21, 19, 151, 126, 234, 188, 102)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__1 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__1_value;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__2 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__2_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__3 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__3_value;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "#find_home"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__4 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__4_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__4_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__5 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__5_value;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__6 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__6_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__7 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__7_value;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "!"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__8 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__8_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__8_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__9 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__9_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__7_value),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__9_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__10 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__10_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__3_value),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__5_value),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__10_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__11 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__11_value;
static const lean_string_object lp_importGraph_command_x23find__home_x21___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__12 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__12_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__12_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__13 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__13_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__13_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__14 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__14_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__3_value),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__11_value),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__14_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__15 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__15_value;
static const lean_ctor_object lp_importGraph_command_x23find__home_x21___00__closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__15_value)}};
static const lean_object* lp_importGraph_command_x23find__home_x21___00__closed__16 = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__16_value;
LEAN_EXPORT const lean_object* lp_importGraph_command_x23find__home_x21__ = (const lean_object*)&lp_importGraph_command_x23find__home_x21___00__closed__16_value;
static lean_once_cell_t lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4;
static lean_once_cell_t lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5;
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___closed__0 = (const lean_object*)&lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___closed__0 = (const lean_object*)&lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2(lean_object* v_a_1_, lean_object* v_b_2_, lean_object* v_as_3_, size_t v_i_4_, size_t v_stop_5_){
_start:
{
uint8_t v___x_6_; 
v___x_6_ = lean_usize_dec_eq(v_i_4_, v_stop_5_);
if (v___x_6_ == 0)
{
uint8_t v___x_7_; uint8_t v___y_9_; lean_object* v___x_13_; uint8_t v___x_14_; 
v___x_7_ = 1;
v___x_13_ = lean_array_uget_borrowed(v_as_3_, v_i_4_);
v___x_14_ = lean_name_eq(v_a_1_, v___x_13_);
if (v___x_14_ == 0)
{
uint8_t v___x_15_; 
v___x_15_ = l_Lean_NameSet_contains(v_b_2_, v___x_13_);
v___y_9_ = v___x_15_;
goto v___jp_8_;
}
else
{
v___y_9_ = v___x_14_;
goto v___jp_8_;
}
v___jp_8_:
{
if (v___y_9_ == 0)
{
return v___x_7_;
}
else
{
if (v___x_6_ == 0)
{
size_t v___x_10_; size_t v___x_11_; 
v___x_10_ = ((size_t)1ULL);
v___x_11_ = lean_usize_add(v_i_4_, v___x_10_);
v_i_4_ = v___x_11_;
goto _start;
}
else
{
return v___x_7_;
}
}
}
}
else
{
uint8_t v___x_16_; 
v___x_16_ = 0;
return v___x_16_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2___boxed(lean_object* v_a_17_, lean_object* v_b_18_, lean_object* v_as_19_, lean_object* v_i_20_, lean_object* v_stop_21_){
_start:
{
size_t v_i_boxed_22_; size_t v_stop_boxed_23_; uint8_t v_res_24_; lean_object* v_r_25_; 
v_i_boxed_22_ = lean_unbox_usize(v_i_20_);
lean_dec(v_i_20_);
v_stop_boxed_23_ = lean_unbox_usize(v_stop_21_);
lean_dec(v_stop_21_);
v_res_24_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2(v_a_17_, v_b_18_, v_as_19_, v_i_boxed_22_, v_stop_boxed_23_);
lean_dec_ref(v_as_19_);
lean_dec(v_b_18_);
lean_dec(v_a_17_);
v_r_25_ = lean_box(v_res_24_);
return v_r_25_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(lean_object* v___x_26_, lean_object* v_init_27_, lean_object* v_x_28_){
_start:
{
lean_object* v_d_31_; 
if (lean_obj_tag(v_x_28_) == 0)
{
lean_object* v_k_34_; lean_object* v_v_35_; lean_object* v_l_36_; lean_object* v_r_37_; lean_object* v___x_38_; lean_object* v_a_39_; 
v_k_34_ = lean_ctor_get(v_x_28_, 1);
lean_inc(v_k_34_);
v_v_35_ = lean_ctor_get(v_x_28_, 2);
lean_inc(v_v_35_);
v_l_36_ = lean_ctor_get(v_x_28_, 3);
lean_inc(v_l_36_);
v_r_37_ = lean_ctor_get(v_x_28_, 4);
lean_inc(v_r_37_);
lean_dec_ref_known(v_x_28_, 5);
v___x_38_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(v___x_26_, v_init_27_, v_l_36_);
v_a_39_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_a_39_);
if (lean_obj_tag(v_a_39_) == 0)
{
lean_object* v_a_40_; 
lean_dec_ref(v___x_38_);
lean_dec(v_r_37_);
lean_dec(v_v_35_);
lean_dec(v_k_34_);
v_a_40_ = lean_ctor_get(v_a_39_, 0);
lean_inc(v_a_40_);
lean_dec_ref_known(v_a_39_, 1);
v_d_31_ = v_a_40_;
goto v___jp_30_;
}
else
{
lean_object* v_a_41_; lean_object* v___x_45_; lean_object* v___x_46_; uint8_t v___x_47_; 
v_a_41_ = lean_ctor_get(v_a_39_, 0);
lean_inc(v_a_41_);
lean_dec_ref_known(v_a_39_, 1);
v___x_45_ = lean_unsigned_to_nat(0u);
v___x_46_ = lean_array_get_size(v___x_26_);
v___x_47_ = lean_nat_dec_lt(v___x_45_, v___x_46_);
if (v___x_47_ == 0)
{
lean_dec_ref(v___x_38_);
lean_dec(v_v_35_);
goto v___jp_42_;
}
else
{
if (v___x_47_ == 0)
{
lean_dec_ref(v___x_38_);
lean_dec(v_v_35_);
goto v___jp_42_;
}
else
{
size_t v___x_48_; size_t v___x_49_; uint8_t v___x_50_; 
v___x_48_ = ((size_t)0ULL);
v___x_49_ = lean_usize_of_nat(v___x_46_);
v___x_50_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Lean_Name_findHome_spec__2(v_k_34_, v_v_35_, v___x_26_, v___x_48_, v___x_49_);
lean_dec(v_v_35_);
if (v___x_50_ == 0)
{
lean_dec_ref(v___x_38_);
goto v___jp_42_;
}
else
{
lean_object* v_a_51_; 
lean_dec(v_a_41_);
lean_dec(v_k_34_);
v_a_51_ = lean_ctor_get(v___x_38_, 0);
lean_inc(v_a_51_);
lean_dec_ref(v___x_38_);
if (lean_obj_tag(v_a_51_) == 0)
{
lean_object* v_a_52_; 
lean_dec(v_r_37_);
v_a_52_ = lean_ctor_get(v_a_51_, 0);
lean_inc(v_a_52_);
lean_dec_ref_known(v_a_51_, 1);
v_d_31_ = v_a_52_;
goto v___jp_30_;
}
else
{
lean_object* v_a_53_; 
v_a_53_ = lean_ctor_get(v_a_51_, 0);
lean_inc(v_a_53_);
lean_dec_ref_known(v_a_51_, 1);
v_init_27_ = v_a_53_;
v_x_28_ = v_r_37_;
goto _start;
}
}
}
}
v___jp_42_:
{
lean_object* v___x_43_; 
v___x_43_ = l_Lean_NameSet_insert(v_a_41_, v_k_34_);
v_init_27_ = v___x_43_;
v_x_28_ = v_r_37_;
goto _start;
}
}
}
else
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_55_, 0, v_init_27_);
v___x_56_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
return v___x_56_;
}
v___jp_30_:
{
lean_object* v___x_32_; lean_object* v___x_33_; 
v___x_32_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_32_, 0, v_d_31_);
v___x_33_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
return v___x_33_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg___boxed(lean_object* v___x_57_, lean_object* v_init_58_, lean_object* v_x_59_, lean_object* v___y_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(v___x_57_, v_init_58_, v_x_59_);
lean_dec_ref(v___x_57_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0_spec__0(lean_object* v_init_62_, lean_object* v_x_63_){
_start:
{
if (lean_obj_tag(v_x_63_) == 0)
{
lean_object* v_k_64_; lean_object* v_l_65_; lean_object* v_r_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v_k_64_ = lean_ctor_get(v_x_63_, 1);
lean_inc(v_k_64_);
v_l_65_ = lean_ctor_get(v_x_63_, 3);
lean_inc(v_l_65_);
v_r_66_ = lean_ctor_get(v_x_63_, 4);
lean_inc(v_r_66_);
lean_dec_ref_known(v_x_63_, 5);
v___x_67_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0_spec__0(v_init_62_, v_l_65_);
v___x_68_ = lean_array_push(v___x_67_, v_k_64_);
v_init_62_ = v___x_68_;
v_x_63_ = v_r_66_;
goto _start;
}
else
{
return v_init_62_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(lean_object* v_k_70_, lean_object* v_t_71_){
_start:
{
if (lean_obj_tag(v_t_71_) == 0)
{
lean_object* v_k_72_; lean_object* v_v_73_; lean_object* v_l_74_; lean_object* v_r_75_; lean_object* v___x_77_; uint8_t v_isShared_78_; uint8_t v_isSharedCheck_729_; 
v_k_72_ = lean_ctor_get(v_t_71_, 1);
v_v_73_ = lean_ctor_get(v_t_71_, 2);
v_l_74_ = lean_ctor_get(v_t_71_, 3);
v_r_75_ = lean_ctor_get(v_t_71_, 4);
v_isSharedCheck_729_ = !lean_is_exclusive(v_t_71_);
if (v_isSharedCheck_729_ == 0)
{
lean_object* v_unused_730_; 
v_unused_730_ = lean_ctor_get(v_t_71_, 0);
lean_dec(v_unused_730_);
v___x_77_ = v_t_71_;
v_isShared_78_ = v_isSharedCheck_729_;
goto v_resetjp_76_;
}
else
{
lean_inc(v_r_75_);
lean_inc(v_l_74_);
lean_inc(v_v_73_);
lean_inc(v_k_72_);
lean_dec(v_t_71_);
v___x_77_ = lean_box(0);
v_isShared_78_ = v_isSharedCheck_729_;
goto v_resetjp_76_;
}
v_resetjp_76_:
{
uint8_t v___x_79_; 
v___x_79_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_70_, v_k_72_);
switch(v___x_79_)
{
case 0:
{
lean_object* v_impl_80_; lean_object* v___x_81_; 
v_impl_80_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(v_k_70_, v_l_74_);
v___x_81_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_80_) == 0)
{
if (lean_obj_tag(v_r_75_) == 0)
{
lean_object* v_size_82_; lean_object* v_size_83_; lean_object* v_k_84_; lean_object* v_v_85_; lean_object* v_l_86_; lean_object* v_r_87_; lean_object* v___x_88_; lean_object* v___x_89_; uint8_t v___x_90_; 
v_size_82_ = lean_ctor_get(v_impl_80_, 0);
lean_inc(v_size_82_);
v_size_83_ = lean_ctor_get(v_r_75_, 0);
v_k_84_ = lean_ctor_get(v_r_75_, 1);
v_v_85_ = lean_ctor_get(v_r_75_, 2);
v_l_86_ = lean_ctor_get(v_r_75_, 3);
lean_inc(v_l_86_);
v_r_87_ = lean_ctor_get(v_r_75_, 4);
v___x_88_ = lean_unsigned_to_nat(3u);
v___x_89_ = lean_nat_mul(v___x_88_, v_size_82_);
v___x_90_ = lean_nat_dec_lt(v___x_89_, v_size_83_);
lean_dec(v___x_89_);
if (v___x_90_ == 0)
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_94_; 
lean_dec(v_l_86_);
v___x_91_ = lean_nat_add(v___x_81_, v_size_82_);
lean_dec(v_size_82_);
v___x_92_ = lean_nat_add(v___x_91_, v_size_83_);
lean_dec(v___x_91_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 3, v_impl_80_);
lean_ctor_set(v___x_77_, 0, v___x_92_);
v___x_94_ = v___x_77_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v___x_92_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_95_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_95_, 3, v_impl_80_);
lean_ctor_set(v_reuseFailAlloc_95_, 4, v_r_75_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
else
{
lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_159_; 
lean_inc(v_r_87_);
lean_inc(v_v_85_);
lean_inc(v_k_84_);
lean_inc(v_size_83_);
v_isSharedCheck_159_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_159_ == 0)
{
lean_object* v_unused_160_; lean_object* v_unused_161_; lean_object* v_unused_162_; lean_object* v_unused_163_; lean_object* v_unused_164_; 
v_unused_160_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_160_);
v_unused_161_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_161_);
v_unused_162_ = lean_ctor_get(v_r_75_, 2);
lean_dec(v_unused_162_);
v_unused_163_ = lean_ctor_get(v_r_75_, 1);
lean_dec(v_unused_163_);
v_unused_164_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_164_);
v___x_97_ = v_r_75_;
v_isShared_98_ = v_isSharedCheck_159_;
goto v_resetjp_96_;
}
else
{
lean_dec(v_r_75_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_159_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v_size_99_; lean_object* v_k_100_; lean_object* v_v_101_; lean_object* v_l_102_; lean_object* v_r_103_; lean_object* v_size_104_; lean_object* v___x_105_; lean_object* v___x_106_; uint8_t v___x_107_; 
v_size_99_ = lean_ctor_get(v_l_86_, 0);
v_k_100_ = lean_ctor_get(v_l_86_, 1);
v_v_101_ = lean_ctor_get(v_l_86_, 2);
v_l_102_ = lean_ctor_get(v_l_86_, 3);
v_r_103_ = lean_ctor_get(v_l_86_, 4);
v_size_104_ = lean_ctor_get(v_r_87_, 0);
v___x_105_ = lean_unsigned_to_nat(2u);
v___x_106_ = lean_nat_mul(v___x_105_, v_size_104_);
v___x_107_ = lean_nat_dec_lt(v_size_99_, v___x_106_);
lean_dec(v___x_106_);
if (v___x_107_ == 0)
{
lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_135_; 
lean_inc(v_r_103_);
lean_inc(v_l_102_);
lean_inc(v_v_101_);
lean_inc(v_k_100_);
v_isSharedCheck_135_ = !lean_is_exclusive(v_l_86_);
if (v_isSharedCheck_135_ == 0)
{
lean_object* v_unused_136_; lean_object* v_unused_137_; lean_object* v_unused_138_; lean_object* v_unused_139_; lean_object* v_unused_140_; 
v_unused_136_ = lean_ctor_get(v_l_86_, 4);
lean_dec(v_unused_136_);
v_unused_137_ = lean_ctor_get(v_l_86_, 3);
lean_dec(v_unused_137_);
v_unused_138_ = lean_ctor_get(v_l_86_, 2);
lean_dec(v_unused_138_);
v_unused_139_ = lean_ctor_get(v_l_86_, 1);
lean_dec(v_unused_139_);
v_unused_140_ = lean_ctor_get(v_l_86_, 0);
lean_dec(v_unused_140_);
v___x_109_ = v_l_86_;
v_isShared_110_ = v_isSharedCheck_135_;
goto v_resetjp_108_;
}
else
{
lean_dec(v_l_86_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_135_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___y_114_; lean_object* v___y_115_; lean_object* v___y_116_; lean_object* v___y_125_; 
v___x_111_ = lean_nat_add(v___x_81_, v_size_82_);
lean_dec(v_size_82_);
v___x_112_ = lean_nat_add(v___x_111_, v_size_83_);
lean_dec(v_size_83_);
if (lean_obj_tag(v_l_102_) == 0)
{
lean_object* v_size_133_; 
v_size_133_ = lean_ctor_get(v_l_102_, 0);
lean_inc(v_size_133_);
v___y_125_ = v_size_133_;
goto v___jp_124_;
}
else
{
lean_object* v___x_134_; 
v___x_134_ = lean_unsigned_to_nat(0u);
v___y_125_ = v___x_134_;
goto v___jp_124_;
}
v___jp_113_:
{
lean_object* v___x_117_; lean_object* v___x_119_; 
v___x_117_ = lean_nat_add(v___y_115_, v___y_116_);
lean_dec(v___y_116_);
lean_dec(v___y_115_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 4, v_r_87_);
lean_ctor_set(v___x_109_, 3, v_r_103_);
lean_ctor_set(v___x_109_, 2, v_v_85_);
lean_ctor_set(v___x_109_, 1, v_k_84_);
lean_ctor_set(v___x_109_, 0, v___x_117_);
v___x_119_ = v___x_109_;
goto v_reusejp_118_;
}
else
{
lean_object* v_reuseFailAlloc_123_; 
v_reuseFailAlloc_123_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_123_, 0, v___x_117_);
lean_ctor_set(v_reuseFailAlloc_123_, 1, v_k_84_);
lean_ctor_set(v_reuseFailAlloc_123_, 2, v_v_85_);
lean_ctor_set(v_reuseFailAlloc_123_, 3, v_r_103_);
lean_ctor_set(v_reuseFailAlloc_123_, 4, v_r_87_);
v___x_119_ = v_reuseFailAlloc_123_;
goto v_reusejp_118_;
}
v_reusejp_118_:
{
lean_object* v___x_121_; 
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 4, v___x_119_);
lean_ctor_set(v___x_97_, 3, v___y_114_);
lean_ctor_set(v___x_97_, 2, v_v_101_);
lean_ctor_set(v___x_97_, 1, v_k_100_);
lean_ctor_set(v___x_97_, 0, v___x_112_);
v___x_121_ = v___x_97_;
goto v_reusejp_120_;
}
else
{
lean_object* v_reuseFailAlloc_122_; 
v_reuseFailAlloc_122_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_122_, 0, v___x_112_);
lean_ctor_set(v_reuseFailAlloc_122_, 1, v_k_100_);
lean_ctor_set(v_reuseFailAlloc_122_, 2, v_v_101_);
lean_ctor_set(v_reuseFailAlloc_122_, 3, v___y_114_);
lean_ctor_set(v_reuseFailAlloc_122_, 4, v___x_119_);
v___x_121_ = v_reuseFailAlloc_122_;
goto v_reusejp_120_;
}
v_reusejp_120_:
{
return v___x_121_;
}
}
}
v___jp_124_:
{
lean_object* v___x_126_; lean_object* v___x_128_; 
v___x_126_ = lean_nat_add(v___x_111_, v___y_125_);
lean_dec(v___y_125_);
lean_dec(v___x_111_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_l_102_);
lean_ctor_set(v___x_77_, 3, v_impl_80_);
lean_ctor_set(v___x_77_, 0, v___x_126_);
v___x_128_ = v___x_77_;
goto v_reusejp_127_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v___x_126_);
lean_ctor_set(v_reuseFailAlloc_132_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_132_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_132_, 3, v_impl_80_);
lean_ctor_set(v_reuseFailAlloc_132_, 4, v_l_102_);
v___x_128_ = v_reuseFailAlloc_132_;
goto v_reusejp_127_;
}
v_reusejp_127_:
{
lean_object* v___x_129_; 
v___x_129_ = lean_nat_add(v___x_81_, v_size_104_);
if (lean_obj_tag(v_r_103_) == 0)
{
lean_object* v_size_130_; 
v_size_130_ = lean_ctor_get(v_r_103_, 0);
lean_inc(v_size_130_);
v___y_114_ = v___x_128_;
v___y_115_ = v___x_129_;
v___y_116_ = v_size_130_;
goto v___jp_113_;
}
else
{
lean_object* v___x_131_; 
v___x_131_ = lean_unsigned_to_nat(0u);
v___y_114_ = v___x_128_;
v___y_115_ = v___x_129_;
v___y_116_ = v___x_131_;
goto v___jp_113_;
}
}
}
}
}
else
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_145_; 
lean_del_object(v___x_77_);
v___x_141_ = lean_nat_add(v___x_81_, v_size_82_);
lean_dec(v_size_82_);
v___x_142_ = lean_nat_add(v___x_141_, v_size_83_);
lean_dec(v_size_83_);
v___x_143_ = lean_nat_add(v___x_141_, v_size_99_);
lean_dec(v___x_141_);
lean_inc_ref(v_impl_80_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 4, v_l_86_);
lean_ctor_set(v___x_97_, 3, v_impl_80_);
lean_ctor_set(v___x_97_, 2, v_v_73_);
lean_ctor_set(v___x_97_, 1, v_k_72_);
lean_ctor_set(v___x_97_, 0, v___x_143_);
v___x_145_ = v___x_97_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v___x_143_);
lean_ctor_set(v_reuseFailAlloc_158_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_158_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_158_, 3, v_impl_80_);
lean_ctor_set(v_reuseFailAlloc_158_, 4, v_l_86_);
v___x_145_ = v_reuseFailAlloc_158_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
lean_object* v___x_147_; uint8_t v_isShared_148_; uint8_t v_isSharedCheck_152_; 
v_isSharedCheck_152_ = !lean_is_exclusive(v_impl_80_);
if (v_isSharedCheck_152_ == 0)
{
lean_object* v_unused_153_; lean_object* v_unused_154_; lean_object* v_unused_155_; lean_object* v_unused_156_; lean_object* v_unused_157_; 
v_unused_153_ = lean_ctor_get(v_impl_80_, 4);
lean_dec(v_unused_153_);
v_unused_154_ = lean_ctor_get(v_impl_80_, 3);
lean_dec(v_unused_154_);
v_unused_155_ = lean_ctor_get(v_impl_80_, 2);
lean_dec(v_unused_155_);
v_unused_156_ = lean_ctor_get(v_impl_80_, 1);
lean_dec(v_unused_156_);
v_unused_157_ = lean_ctor_get(v_impl_80_, 0);
lean_dec(v_unused_157_);
v___x_147_ = v_impl_80_;
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
else
{
lean_dec(v_impl_80_);
v___x_147_ = lean_box(0);
v_isShared_148_ = v_isSharedCheck_152_;
goto v_resetjp_146_;
}
v_resetjp_146_:
{
lean_object* v___x_150_; 
if (v_isShared_148_ == 0)
{
lean_ctor_set(v___x_147_, 4, v_r_87_);
lean_ctor_set(v___x_147_, 3, v___x_145_);
lean_ctor_set(v___x_147_, 2, v_v_85_);
lean_ctor_set(v___x_147_, 1, v_k_84_);
lean_ctor_set(v___x_147_, 0, v___x_142_);
v___x_150_ = v___x_147_;
goto v_reusejp_149_;
}
else
{
lean_object* v_reuseFailAlloc_151_; 
v_reuseFailAlloc_151_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_151_, 0, v___x_142_);
lean_ctor_set(v_reuseFailAlloc_151_, 1, v_k_84_);
lean_ctor_set(v_reuseFailAlloc_151_, 2, v_v_85_);
lean_ctor_set(v_reuseFailAlloc_151_, 3, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_151_, 4, v_r_87_);
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
else
{
lean_object* v_size_165_; lean_object* v___x_166_; lean_object* v___x_168_; 
v_size_165_ = lean_ctor_get(v_impl_80_, 0);
lean_inc(v_size_165_);
v___x_166_ = lean_nat_add(v___x_81_, v_size_165_);
lean_dec(v_size_165_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 3, v_impl_80_);
lean_ctor_set(v___x_77_, 0, v___x_166_);
v___x_168_ = v___x_77_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v___x_166_);
lean_ctor_set(v_reuseFailAlloc_169_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_169_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_169_, 3, v_impl_80_);
lean_ctor_set(v_reuseFailAlloc_169_, 4, v_r_75_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
else
{
if (lean_obj_tag(v_r_75_) == 0)
{
lean_object* v_l_170_; 
v_l_170_ = lean_ctor_get(v_r_75_, 3);
lean_inc(v_l_170_);
if (lean_obj_tag(v_l_170_) == 0)
{
lean_object* v_r_171_; 
v_r_171_ = lean_ctor_get(v_r_75_, 4);
lean_inc(v_r_171_);
if (lean_obj_tag(v_r_171_) == 0)
{
lean_object* v_size_172_; lean_object* v_k_173_; lean_object* v_v_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_187_; 
v_size_172_ = lean_ctor_get(v_r_75_, 0);
v_k_173_ = lean_ctor_get(v_r_75_, 1);
v_v_174_ = lean_ctor_get(v_r_75_, 2);
v_isSharedCheck_187_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_187_ == 0)
{
lean_object* v_unused_188_; lean_object* v_unused_189_; 
v_unused_188_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_188_);
v_unused_189_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_189_);
v___x_176_ = v_r_75_;
v_isShared_177_ = v_isSharedCheck_187_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_v_174_);
lean_inc(v_k_173_);
lean_inc(v_size_172_);
lean_dec(v_r_75_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_187_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v_size_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_182_; 
v_size_178_ = lean_ctor_get(v_l_170_, 0);
v___x_179_ = lean_nat_add(v___x_81_, v_size_172_);
lean_dec(v_size_172_);
v___x_180_ = lean_nat_add(v___x_81_, v_size_178_);
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 4, v_l_170_);
lean_ctor_set(v___x_176_, 3, v_impl_80_);
lean_ctor_set(v___x_176_, 2, v_v_73_);
lean_ctor_set(v___x_176_, 1, v_k_72_);
lean_ctor_set(v___x_176_, 0, v___x_180_);
v___x_182_ = v___x_176_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_impl_80_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_l_170_);
v___x_182_ = v_reuseFailAlloc_186_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
lean_object* v___x_184_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_r_171_);
lean_ctor_set(v___x_77_, 3, v___x_182_);
lean_ctor_set(v___x_77_, 2, v_v_174_);
lean_ctor_set(v___x_77_, 1, v_k_173_);
lean_ctor_set(v___x_77_, 0, v___x_179_);
v___x_184_ = v___x_77_;
goto v_reusejp_183_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_179_);
lean_ctor_set(v_reuseFailAlloc_185_, 1, v_k_173_);
lean_ctor_set(v_reuseFailAlloc_185_, 2, v_v_174_);
lean_ctor_set(v_reuseFailAlloc_185_, 3, v___x_182_);
lean_ctor_set(v_reuseFailAlloc_185_, 4, v_r_171_);
v___x_184_ = v_reuseFailAlloc_185_;
goto v_reusejp_183_;
}
v_reusejp_183_:
{
return v___x_184_;
}
}
}
}
else
{
lean_object* v_k_190_; lean_object* v_v_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_214_; 
v_k_190_ = lean_ctor_get(v_r_75_, 1);
v_v_191_ = lean_ctor_get(v_r_75_, 2);
v_isSharedCheck_214_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_214_ == 0)
{
lean_object* v_unused_215_; lean_object* v_unused_216_; lean_object* v_unused_217_; 
v_unused_215_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_215_);
v_unused_216_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_216_);
v_unused_217_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_217_);
v___x_193_ = v_r_75_;
v_isShared_194_ = v_isSharedCheck_214_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_v_191_);
lean_inc(v_k_190_);
lean_dec(v_r_75_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_214_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v_k_195_; lean_object* v_v_196_; lean_object* v___x_198_; uint8_t v_isShared_199_; uint8_t v_isSharedCheck_210_; 
v_k_195_ = lean_ctor_get(v_l_170_, 1);
v_v_196_ = lean_ctor_get(v_l_170_, 2);
v_isSharedCheck_210_ = !lean_is_exclusive(v_l_170_);
if (v_isSharedCheck_210_ == 0)
{
lean_object* v_unused_211_; lean_object* v_unused_212_; lean_object* v_unused_213_; 
v_unused_211_ = lean_ctor_get(v_l_170_, 4);
lean_dec(v_unused_211_);
v_unused_212_ = lean_ctor_get(v_l_170_, 3);
lean_dec(v_unused_212_);
v_unused_213_ = lean_ctor_get(v_l_170_, 0);
lean_dec(v_unused_213_);
v___x_198_ = v_l_170_;
v_isShared_199_ = v_isSharedCheck_210_;
goto v_resetjp_197_;
}
else
{
lean_inc(v_v_196_);
lean_inc(v_k_195_);
lean_dec(v_l_170_);
v___x_198_ = lean_box(0);
v_isShared_199_ = v_isSharedCheck_210_;
goto v_resetjp_197_;
}
v_resetjp_197_:
{
lean_object* v___x_200_; lean_object* v___x_202_; 
v___x_200_ = lean_unsigned_to_nat(3u);
if (v_isShared_199_ == 0)
{
lean_ctor_set(v___x_198_, 4, v_r_171_);
lean_ctor_set(v___x_198_, 3, v_r_171_);
lean_ctor_set(v___x_198_, 2, v_v_73_);
lean_ctor_set(v___x_198_, 1, v_k_72_);
lean_ctor_set(v___x_198_, 0, v___x_81_);
v___x_202_ = v___x_198_;
goto v_reusejp_201_;
}
else
{
lean_object* v_reuseFailAlloc_209_; 
v_reuseFailAlloc_209_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_209_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_209_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_209_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_209_, 3, v_r_171_);
lean_ctor_set(v_reuseFailAlloc_209_, 4, v_r_171_);
v___x_202_ = v_reuseFailAlloc_209_;
goto v_reusejp_201_;
}
v_reusejp_201_:
{
lean_object* v___x_204_; 
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 3, v_r_171_);
lean_ctor_set(v___x_193_, 0, v___x_81_);
v___x_204_ = v___x_193_;
goto v_reusejp_203_;
}
else
{
lean_object* v_reuseFailAlloc_208_; 
v_reuseFailAlloc_208_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_208_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_208_, 1, v_k_190_);
lean_ctor_set(v_reuseFailAlloc_208_, 2, v_v_191_);
lean_ctor_set(v_reuseFailAlloc_208_, 3, v_r_171_);
lean_ctor_set(v_reuseFailAlloc_208_, 4, v_r_171_);
v___x_204_ = v_reuseFailAlloc_208_;
goto v_reusejp_203_;
}
v_reusejp_203_:
{
lean_object* v___x_206_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v___x_204_);
lean_ctor_set(v___x_77_, 3, v___x_202_);
lean_ctor_set(v___x_77_, 2, v_v_196_);
lean_ctor_set(v___x_77_, 1, v_k_195_);
lean_ctor_set(v___x_77_, 0, v___x_200_);
v___x_206_ = v___x_77_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_207_; 
v_reuseFailAlloc_207_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_207_, 0, v___x_200_);
lean_ctor_set(v_reuseFailAlloc_207_, 1, v_k_195_);
lean_ctor_set(v_reuseFailAlloc_207_, 2, v_v_196_);
lean_ctor_set(v_reuseFailAlloc_207_, 3, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_207_, 4, v___x_204_);
v___x_206_ = v_reuseFailAlloc_207_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
return v___x_206_;
}
}
}
}
}
}
}
else
{
lean_object* v_r_218_; 
v_r_218_ = lean_ctor_get(v_r_75_, 4);
lean_inc(v_r_218_);
if (lean_obj_tag(v_r_218_) == 0)
{
lean_object* v_k_219_; lean_object* v_v_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_231_; 
v_k_219_ = lean_ctor_get(v_r_75_, 1);
v_v_220_ = lean_ctor_get(v_r_75_, 2);
v_isSharedCheck_231_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_231_ == 0)
{
lean_object* v_unused_232_; lean_object* v_unused_233_; lean_object* v_unused_234_; 
v_unused_232_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_232_);
v_unused_233_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_233_);
v_unused_234_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_234_);
v___x_222_ = v_r_75_;
v_isShared_223_ = v_isSharedCheck_231_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_v_220_);
lean_inc(v_k_219_);
lean_dec(v_r_75_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_231_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_224_; lean_object* v___x_226_; 
v___x_224_ = lean_unsigned_to_nat(3u);
if (v_isShared_223_ == 0)
{
lean_ctor_set(v___x_222_, 4, v_l_170_);
lean_ctor_set(v___x_222_, 2, v_v_73_);
lean_ctor_set(v___x_222_, 1, v_k_72_);
lean_ctor_set(v___x_222_, 0, v___x_81_);
v___x_226_ = v___x_222_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_230_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_230_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_230_, 3, v_l_170_);
lean_ctor_set(v_reuseFailAlloc_230_, 4, v_l_170_);
v___x_226_ = v_reuseFailAlloc_230_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
lean_object* v___x_228_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_r_218_);
lean_ctor_set(v___x_77_, 3, v___x_226_);
lean_ctor_set(v___x_77_, 2, v_v_220_);
lean_ctor_set(v___x_77_, 1, v_k_219_);
lean_ctor_set(v___x_77_, 0, v___x_224_);
v___x_228_ = v___x_77_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v___x_224_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v_k_219_);
lean_ctor_set(v_reuseFailAlloc_229_, 2, v_v_220_);
lean_ctor_set(v_reuseFailAlloc_229_, 3, v___x_226_);
lean_ctor_set(v_reuseFailAlloc_229_, 4, v_r_218_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
}
}
else
{
lean_object* v_size_235_; lean_object* v_k_236_; lean_object* v_v_237_; lean_object* v___x_239_; uint8_t v_isShared_240_; uint8_t v_isSharedCheck_248_; 
v_size_235_ = lean_ctor_get(v_r_75_, 0);
v_k_236_ = lean_ctor_get(v_r_75_, 1);
v_v_237_ = lean_ctor_get(v_r_75_, 2);
v_isSharedCheck_248_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_248_ == 0)
{
lean_object* v_unused_249_; lean_object* v_unused_250_; 
v_unused_249_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_249_);
v_unused_250_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_250_);
v___x_239_ = v_r_75_;
v_isShared_240_ = v_isSharedCheck_248_;
goto v_resetjp_238_;
}
else
{
lean_inc(v_v_237_);
lean_inc(v_k_236_);
lean_inc(v_size_235_);
lean_dec(v_r_75_);
v___x_239_ = lean_box(0);
v_isShared_240_ = v_isSharedCheck_248_;
goto v_resetjp_238_;
}
v_resetjp_238_:
{
lean_object* v___x_242_; 
if (v_isShared_240_ == 0)
{
lean_ctor_set(v___x_239_, 3, v_r_218_);
v___x_242_ = v___x_239_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v_size_235_);
lean_ctor_set(v_reuseFailAlloc_247_, 1, v_k_236_);
lean_ctor_set(v_reuseFailAlloc_247_, 2, v_v_237_);
lean_ctor_set(v_reuseFailAlloc_247_, 3, v_r_218_);
lean_ctor_set(v_reuseFailAlloc_247_, 4, v_r_218_);
v___x_242_ = v_reuseFailAlloc_247_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_243_ = lean_unsigned_to_nat(2u);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v___x_242_);
lean_ctor_set(v___x_77_, 3, v_r_218_);
lean_ctor_set(v___x_77_, 0, v___x_243_);
v___x_245_ = v___x_77_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_246_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_246_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_246_, 3, v_r_218_);
lean_ctor_set(v_reuseFailAlloc_246_, 4, v___x_242_);
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
else
{
lean_object* v___x_252_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 3, v_r_75_);
lean_ctor_set(v___x_77_, 0, v___x_81_);
v___x_252_ = v___x_77_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_253_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_253_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_253_, 3, v_r_75_);
lean_ctor_set(v_reuseFailAlloc_253_, 4, v_r_75_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
case 1:
{
lean_del_object(v___x_77_);
lean_dec(v_v_73_);
lean_dec(v_k_72_);
if (lean_obj_tag(v_l_74_) == 0)
{
if (lean_obj_tag(v_r_75_) == 0)
{
lean_object* v_size_254_; lean_object* v_k_255_; lean_object* v_v_256_; lean_object* v_l_257_; lean_object* v_r_258_; lean_object* v_size_259_; lean_object* v_k_260_; lean_object* v_v_261_; lean_object* v_l_262_; lean_object* v_r_263_; lean_object* v___x_264_; uint8_t v___x_265_; 
v_size_254_ = lean_ctor_get(v_l_74_, 0);
v_k_255_ = lean_ctor_get(v_l_74_, 1);
v_v_256_ = lean_ctor_get(v_l_74_, 2);
v_l_257_ = lean_ctor_get(v_l_74_, 3);
v_r_258_ = lean_ctor_get(v_l_74_, 4);
lean_inc(v_r_258_);
v_size_259_ = lean_ctor_get(v_r_75_, 0);
v_k_260_ = lean_ctor_get(v_r_75_, 1);
v_v_261_ = lean_ctor_get(v_r_75_, 2);
v_l_262_ = lean_ctor_get(v_r_75_, 3);
lean_inc(v_l_262_);
v_r_263_ = lean_ctor_get(v_r_75_, 4);
v___x_264_ = lean_unsigned_to_nat(1u);
v___x_265_ = lean_nat_dec_lt(v_size_254_, v_size_259_);
if (v___x_265_ == 0)
{
lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_401_; 
lean_inc(v_l_257_);
lean_inc(v_v_256_);
lean_inc(v_k_255_);
v_isSharedCheck_401_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_401_ == 0)
{
lean_object* v_unused_402_; lean_object* v_unused_403_; lean_object* v_unused_404_; lean_object* v_unused_405_; lean_object* v_unused_406_; 
v_unused_402_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_402_);
v_unused_403_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_403_);
v_unused_404_ = lean_ctor_get(v_l_74_, 2);
lean_dec(v_unused_404_);
v_unused_405_ = lean_ctor_get(v_l_74_, 1);
lean_dec(v_unused_405_);
v_unused_406_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_406_);
v___x_267_ = v_l_74_;
v_isShared_268_ = v_isSharedCheck_401_;
goto v_resetjp_266_;
}
else
{
lean_dec(v_l_74_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_401_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_269_; lean_object* v_tree_270_; 
v___x_269_ = l_Std_DTreeMap_Internal_Impl_maxView___redArg(v_k_255_, v_v_256_, v_l_257_, v_r_258_);
v_tree_270_ = lean_ctor_get(v___x_269_, 2);
lean_inc(v_tree_270_);
if (lean_obj_tag(v_tree_270_) == 0)
{
lean_object* v_k_271_; lean_object* v_v_272_; lean_object* v_size_273_; lean_object* v___x_274_; lean_object* v___x_275_; uint8_t v___x_276_; 
v_k_271_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_k_271_);
v_v_272_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_v_272_);
lean_dec_ref(v___x_269_);
v_size_273_ = lean_ctor_get(v_tree_270_, 0);
v___x_274_ = lean_unsigned_to_nat(3u);
v___x_275_ = lean_nat_mul(v___x_274_, v_size_273_);
v___x_276_ = lean_nat_dec_lt(v___x_275_, v_size_259_);
lean_dec(v___x_275_);
if (v___x_276_ == 0)
{
lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_280_; 
lean_dec(v_l_262_);
v___x_277_ = lean_nat_add(v___x_264_, v_size_273_);
v___x_278_ = lean_nat_add(v___x_277_, v_size_259_);
lean_dec(v___x_277_);
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v_r_75_);
lean_ctor_set(v___x_267_, 3, v_tree_270_);
lean_ctor_set(v___x_267_, 2, v_v_272_);
lean_ctor_set(v___x_267_, 1, v_k_271_);
lean_ctor_set(v___x_267_, 0, v___x_278_);
v___x_280_ = v___x_267_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_278_);
lean_ctor_set(v_reuseFailAlloc_281_, 1, v_k_271_);
lean_ctor_set(v_reuseFailAlloc_281_, 2, v_v_272_);
lean_ctor_set(v_reuseFailAlloc_281_, 3, v_tree_270_);
lean_ctor_set(v_reuseFailAlloc_281_, 4, v_r_75_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
else
{
lean_object* v___x_283_; uint8_t v_isShared_284_; uint8_t v_isSharedCheck_336_; 
lean_inc(v_r_263_);
lean_inc(v_v_261_);
lean_inc(v_k_260_);
lean_inc(v_size_259_);
v_isSharedCheck_336_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_336_ == 0)
{
lean_object* v_unused_337_; lean_object* v_unused_338_; lean_object* v_unused_339_; lean_object* v_unused_340_; lean_object* v_unused_341_; 
v_unused_337_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_337_);
v_unused_338_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_338_);
v_unused_339_ = lean_ctor_get(v_r_75_, 2);
lean_dec(v_unused_339_);
v_unused_340_ = lean_ctor_get(v_r_75_, 1);
lean_dec(v_unused_340_);
v_unused_341_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_341_);
v___x_283_ = v_r_75_;
v_isShared_284_ = v_isSharedCheck_336_;
goto v_resetjp_282_;
}
else
{
lean_dec(v_r_75_);
v___x_283_ = lean_box(0);
v_isShared_284_ = v_isSharedCheck_336_;
goto v_resetjp_282_;
}
v_resetjp_282_:
{
lean_object* v_size_285_; lean_object* v_k_286_; lean_object* v_v_287_; lean_object* v_l_288_; lean_object* v_r_289_; lean_object* v_size_290_; lean_object* v___x_291_; lean_object* v___x_292_; uint8_t v___x_293_; 
v_size_285_ = lean_ctor_get(v_l_262_, 0);
v_k_286_ = lean_ctor_get(v_l_262_, 1);
v_v_287_ = lean_ctor_get(v_l_262_, 2);
v_l_288_ = lean_ctor_get(v_l_262_, 3);
v_r_289_ = lean_ctor_get(v_l_262_, 4);
v_size_290_ = lean_ctor_get(v_r_263_, 0);
v___x_291_ = lean_unsigned_to_nat(2u);
v___x_292_ = lean_nat_mul(v___x_291_, v_size_290_);
v___x_293_ = lean_nat_dec_lt(v_size_285_, v___x_292_);
lean_dec(v___x_292_);
if (v___x_293_ == 0)
{
lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_321_; 
lean_inc(v_r_289_);
lean_inc(v_l_288_);
lean_inc(v_v_287_);
lean_inc(v_k_286_);
v_isSharedCheck_321_ = !lean_is_exclusive(v_l_262_);
if (v_isSharedCheck_321_ == 0)
{
lean_object* v_unused_322_; lean_object* v_unused_323_; lean_object* v_unused_324_; lean_object* v_unused_325_; lean_object* v_unused_326_; 
v_unused_322_ = lean_ctor_get(v_l_262_, 4);
lean_dec(v_unused_322_);
v_unused_323_ = lean_ctor_get(v_l_262_, 3);
lean_dec(v_unused_323_);
v_unused_324_ = lean_ctor_get(v_l_262_, 2);
lean_dec(v_unused_324_);
v_unused_325_ = lean_ctor_get(v_l_262_, 1);
lean_dec(v_unused_325_);
v_unused_326_ = lean_ctor_get(v_l_262_, 0);
lean_dec(v_unused_326_);
v___x_295_ = v_l_262_;
v_isShared_296_ = v_isSharedCheck_321_;
goto v_resetjp_294_;
}
else
{
lean_dec(v_l_262_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_321_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___y_300_; lean_object* v___y_301_; lean_object* v___y_302_; lean_object* v___y_311_; 
v___x_297_ = lean_nat_add(v___x_264_, v_size_273_);
v___x_298_ = lean_nat_add(v___x_297_, v_size_259_);
lean_dec(v_size_259_);
if (lean_obj_tag(v_l_288_) == 0)
{
lean_object* v_size_319_; 
v_size_319_ = lean_ctor_get(v_l_288_, 0);
lean_inc(v_size_319_);
v___y_311_ = v_size_319_;
goto v___jp_310_;
}
else
{
lean_object* v___x_320_; 
v___x_320_ = lean_unsigned_to_nat(0u);
v___y_311_ = v___x_320_;
goto v___jp_310_;
}
v___jp_299_:
{
lean_object* v___x_303_; lean_object* v___x_305_; 
v___x_303_ = lean_nat_add(v___y_300_, v___y_302_);
lean_dec(v___y_302_);
lean_dec(v___y_300_);
if (v_isShared_296_ == 0)
{
lean_ctor_set(v___x_295_, 4, v_r_263_);
lean_ctor_set(v___x_295_, 3, v_r_289_);
lean_ctor_set(v___x_295_, 2, v_v_261_);
lean_ctor_set(v___x_295_, 1, v_k_260_);
lean_ctor_set(v___x_295_, 0, v___x_303_);
v___x_305_ = v___x_295_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_309_; 
v_reuseFailAlloc_309_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_309_, 0, v___x_303_);
lean_ctor_set(v_reuseFailAlloc_309_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_309_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_309_, 3, v_r_289_);
lean_ctor_set(v_reuseFailAlloc_309_, 4, v_r_263_);
v___x_305_ = v_reuseFailAlloc_309_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
lean_object* v___x_307_; 
if (v_isShared_284_ == 0)
{
lean_ctor_set(v___x_283_, 4, v___x_305_);
lean_ctor_set(v___x_283_, 3, v___y_301_);
lean_ctor_set(v___x_283_, 2, v_v_287_);
lean_ctor_set(v___x_283_, 1, v_k_286_);
lean_ctor_set(v___x_283_, 0, v___x_298_);
v___x_307_ = v___x_283_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v___x_298_);
lean_ctor_set(v_reuseFailAlloc_308_, 1, v_k_286_);
lean_ctor_set(v_reuseFailAlloc_308_, 2, v_v_287_);
lean_ctor_set(v_reuseFailAlloc_308_, 3, v___y_301_);
lean_ctor_set(v_reuseFailAlloc_308_, 4, v___x_305_);
v___x_307_ = v_reuseFailAlloc_308_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
return v___x_307_;
}
}
}
v___jp_310_:
{
lean_object* v___x_312_; lean_object* v___x_314_; 
v___x_312_ = lean_nat_add(v___x_297_, v___y_311_);
lean_dec(v___y_311_);
lean_dec(v___x_297_);
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v_l_288_);
lean_ctor_set(v___x_267_, 3, v_tree_270_);
lean_ctor_set(v___x_267_, 2, v_v_272_);
lean_ctor_set(v___x_267_, 1, v_k_271_);
lean_ctor_set(v___x_267_, 0, v___x_312_);
v___x_314_ = v___x_267_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_318_; 
v_reuseFailAlloc_318_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_318_, 0, v___x_312_);
lean_ctor_set(v_reuseFailAlloc_318_, 1, v_k_271_);
lean_ctor_set(v_reuseFailAlloc_318_, 2, v_v_272_);
lean_ctor_set(v_reuseFailAlloc_318_, 3, v_tree_270_);
lean_ctor_set(v_reuseFailAlloc_318_, 4, v_l_288_);
v___x_314_ = v_reuseFailAlloc_318_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
lean_object* v___x_315_; 
v___x_315_ = lean_nat_add(v___x_264_, v_size_290_);
if (lean_obj_tag(v_r_289_) == 0)
{
lean_object* v_size_316_; 
v_size_316_ = lean_ctor_get(v_r_289_, 0);
lean_inc(v_size_316_);
v___y_300_ = v___x_315_;
v___y_301_ = v___x_314_;
v___y_302_ = v_size_316_;
goto v___jp_299_;
}
else
{
lean_object* v___x_317_; 
v___x_317_ = lean_unsigned_to_nat(0u);
v___y_300_ = v___x_315_;
v___y_301_ = v___x_314_;
v___y_302_ = v___x_317_;
goto v___jp_299_;
}
}
}
}
}
else
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_331_; 
v___x_327_ = lean_nat_add(v___x_264_, v_size_273_);
v___x_328_ = lean_nat_add(v___x_327_, v_size_259_);
lean_dec(v_size_259_);
v___x_329_ = lean_nat_add(v___x_327_, v_size_285_);
lean_dec(v___x_327_);
if (v_isShared_284_ == 0)
{
lean_ctor_set(v___x_283_, 4, v_l_262_);
lean_ctor_set(v___x_283_, 3, v_tree_270_);
lean_ctor_set(v___x_283_, 2, v_v_272_);
lean_ctor_set(v___x_283_, 1, v_k_271_);
lean_ctor_set(v___x_283_, 0, v___x_329_);
v___x_331_ = v___x_283_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v___x_329_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v_k_271_);
lean_ctor_set(v_reuseFailAlloc_335_, 2, v_v_272_);
lean_ctor_set(v_reuseFailAlloc_335_, 3, v_tree_270_);
lean_ctor_set(v_reuseFailAlloc_335_, 4, v_l_262_);
v___x_331_ = v_reuseFailAlloc_335_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
lean_object* v___x_333_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v_r_263_);
lean_ctor_set(v___x_267_, 3, v___x_331_);
lean_ctor_set(v___x_267_, 2, v_v_261_);
lean_ctor_set(v___x_267_, 1, v_k_260_);
lean_ctor_set(v___x_267_, 0, v___x_328_);
v___x_333_ = v___x_267_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_334_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_334_, 3, v___x_331_);
lean_ctor_set(v_reuseFailAlloc_334_, 4, v_r_263_);
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
}
}
else
{
lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_395_; 
lean_inc(v_r_263_);
lean_inc(v_v_261_);
lean_inc(v_k_260_);
lean_inc(v_size_259_);
v_isSharedCheck_395_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_395_ == 0)
{
lean_object* v_unused_396_; lean_object* v_unused_397_; lean_object* v_unused_398_; lean_object* v_unused_399_; lean_object* v_unused_400_; 
v_unused_396_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_396_);
v_unused_397_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_397_);
v_unused_398_ = lean_ctor_get(v_r_75_, 2);
lean_dec(v_unused_398_);
v_unused_399_ = lean_ctor_get(v_r_75_, 1);
lean_dec(v_unused_399_);
v_unused_400_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_400_);
v___x_343_ = v_r_75_;
v_isShared_344_ = v_isSharedCheck_395_;
goto v_resetjp_342_;
}
else
{
lean_dec(v_r_75_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_395_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
if (lean_obj_tag(v_l_262_) == 0)
{
if (lean_obj_tag(v_r_263_) == 0)
{
lean_object* v_k_345_; lean_object* v_v_346_; lean_object* v_size_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_351_; 
v_k_345_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_k_345_);
v_v_346_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_v_346_);
lean_dec_ref(v___x_269_);
v_size_347_ = lean_ctor_get(v_l_262_, 0);
v___x_348_ = lean_nat_add(v___x_264_, v_size_259_);
lean_dec(v_size_259_);
v___x_349_ = lean_nat_add(v___x_264_, v_size_347_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 4, v_l_262_);
lean_ctor_set(v___x_343_, 3, v_tree_270_);
lean_ctor_set(v___x_343_, 2, v_v_346_);
lean_ctor_set(v___x_343_, 1, v_k_345_);
lean_ctor_set(v___x_343_, 0, v___x_349_);
v___x_351_ = v___x_343_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v___x_349_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v_k_345_);
lean_ctor_set(v_reuseFailAlloc_355_, 2, v_v_346_);
lean_ctor_set(v_reuseFailAlloc_355_, 3, v_tree_270_);
lean_ctor_set(v_reuseFailAlloc_355_, 4, v_l_262_);
v___x_351_ = v_reuseFailAlloc_355_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
lean_object* v___x_353_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v_r_263_);
lean_ctor_set(v___x_267_, 3, v___x_351_);
lean_ctor_set(v___x_267_, 2, v_v_261_);
lean_ctor_set(v___x_267_, 1, v_k_260_);
lean_ctor_set(v___x_267_, 0, v___x_348_);
v___x_353_ = v___x_267_;
goto v_reusejp_352_;
}
else
{
lean_object* v_reuseFailAlloc_354_; 
v_reuseFailAlloc_354_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_354_, 0, v___x_348_);
lean_ctor_set(v_reuseFailAlloc_354_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_354_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_354_, 3, v___x_351_);
lean_ctor_set(v_reuseFailAlloc_354_, 4, v_r_263_);
v___x_353_ = v_reuseFailAlloc_354_;
goto v_reusejp_352_;
}
v_reusejp_352_:
{
return v___x_353_;
}
}
}
else
{
lean_object* v_k_356_; lean_object* v_v_357_; lean_object* v_k_358_; lean_object* v_v_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_373_; 
lean_dec(v_size_259_);
v_k_356_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_k_356_);
v_v_357_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_v_357_);
lean_dec_ref(v___x_269_);
v_k_358_ = lean_ctor_get(v_l_262_, 1);
v_v_359_ = lean_ctor_get(v_l_262_, 2);
v_isSharedCheck_373_ = !lean_is_exclusive(v_l_262_);
if (v_isSharedCheck_373_ == 0)
{
lean_object* v_unused_374_; lean_object* v_unused_375_; lean_object* v_unused_376_; 
v_unused_374_ = lean_ctor_get(v_l_262_, 4);
lean_dec(v_unused_374_);
v_unused_375_ = lean_ctor_get(v_l_262_, 3);
lean_dec(v_unused_375_);
v_unused_376_ = lean_ctor_get(v_l_262_, 0);
lean_dec(v_unused_376_);
v___x_361_ = v_l_262_;
v_isShared_362_ = v_isSharedCheck_373_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_v_359_);
lean_inc(v_k_358_);
lean_dec(v_l_262_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_373_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v___x_363_; lean_object* v___x_365_; 
v___x_363_ = lean_unsigned_to_nat(3u);
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 4, v_r_263_);
lean_ctor_set(v___x_361_, 3, v_r_263_);
lean_ctor_set(v___x_361_, 2, v_v_357_);
lean_ctor_set(v___x_361_, 1, v_k_356_);
lean_ctor_set(v___x_361_, 0, v___x_264_);
v___x_365_ = v___x_361_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_k_356_);
lean_ctor_set(v_reuseFailAlloc_372_, 2, v_v_357_);
lean_ctor_set(v_reuseFailAlloc_372_, 3, v_r_263_);
lean_ctor_set(v_reuseFailAlloc_372_, 4, v_r_263_);
v___x_365_ = v_reuseFailAlloc_372_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
lean_object* v___x_367_; 
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 3, v_r_263_);
lean_ctor_set(v___x_343_, 0, v___x_264_);
v___x_367_ = v___x_343_;
goto v_reusejp_366_;
}
else
{
lean_object* v_reuseFailAlloc_371_; 
v_reuseFailAlloc_371_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_371_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_371_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_371_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_371_, 3, v_r_263_);
lean_ctor_set(v_reuseFailAlloc_371_, 4, v_r_263_);
v___x_367_ = v_reuseFailAlloc_371_;
goto v_reusejp_366_;
}
v_reusejp_366_:
{
lean_object* v___x_369_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v___x_367_);
lean_ctor_set(v___x_267_, 3, v___x_365_);
lean_ctor_set(v___x_267_, 2, v_v_359_);
lean_ctor_set(v___x_267_, 1, v_k_358_);
lean_ctor_set(v___x_267_, 0, v___x_363_);
v___x_369_ = v___x_267_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v___x_363_);
lean_ctor_set(v_reuseFailAlloc_370_, 1, v_k_358_);
lean_ctor_set(v_reuseFailAlloc_370_, 2, v_v_359_);
lean_ctor_set(v_reuseFailAlloc_370_, 3, v___x_365_);
lean_ctor_set(v_reuseFailAlloc_370_, 4, v___x_367_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_263_) == 0)
{
lean_object* v_k_377_; lean_object* v_v_378_; lean_object* v___x_379_; lean_object* v___x_381_; 
lean_dec(v_size_259_);
v_k_377_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_k_377_);
v_v_378_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_v_378_);
lean_dec_ref(v___x_269_);
v___x_379_ = lean_unsigned_to_nat(3u);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 4, v_l_262_);
lean_ctor_set(v___x_343_, 2, v_v_378_);
lean_ctor_set(v___x_343_, 1, v_k_377_);
lean_ctor_set(v___x_343_, 0, v___x_264_);
v___x_381_ = v___x_343_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_385_, 1, v_k_377_);
lean_ctor_set(v_reuseFailAlloc_385_, 2, v_v_378_);
lean_ctor_set(v_reuseFailAlloc_385_, 3, v_l_262_);
lean_ctor_set(v_reuseFailAlloc_385_, 4, v_l_262_);
v___x_381_ = v_reuseFailAlloc_385_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
lean_object* v___x_383_; 
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v_r_263_);
lean_ctor_set(v___x_267_, 3, v___x_381_);
lean_ctor_set(v___x_267_, 2, v_v_261_);
lean_ctor_set(v___x_267_, 1, v_k_260_);
lean_ctor_set(v___x_267_, 0, v___x_379_);
v___x_383_ = v___x_267_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v___x_379_);
lean_ctor_set(v_reuseFailAlloc_384_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_384_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_384_, 3, v___x_381_);
lean_ctor_set(v_reuseFailAlloc_384_, 4, v_r_263_);
v___x_383_ = v_reuseFailAlloc_384_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
return v___x_383_;
}
}
}
else
{
lean_object* v_k_386_; lean_object* v_v_387_; lean_object* v___x_389_; 
v_k_386_ = lean_ctor_get(v___x_269_, 0);
lean_inc(v_k_386_);
v_v_387_ = lean_ctor_get(v___x_269_, 1);
lean_inc(v_v_387_);
lean_dec_ref(v___x_269_);
if (v_isShared_344_ == 0)
{
lean_ctor_set(v___x_343_, 3, v_r_263_);
v___x_389_ = v___x_343_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_394_; 
v_reuseFailAlloc_394_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_394_, 0, v_size_259_);
lean_ctor_set(v_reuseFailAlloc_394_, 1, v_k_260_);
lean_ctor_set(v_reuseFailAlloc_394_, 2, v_v_261_);
lean_ctor_set(v_reuseFailAlloc_394_, 3, v_r_263_);
lean_ctor_set(v_reuseFailAlloc_394_, 4, v_r_263_);
v___x_389_ = v_reuseFailAlloc_394_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_390_ = lean_unsigned_to_nat(2u);
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 4, v___x_389_);
lean_ctor_set(v___x_267_, 3, v_r_263_);
lean_ctor_set(v___x_267_, 2, v_v_387_);
lean_ctor_set(v___x_267_, 1, v_k_386_);
lean_ctor_set(v___x_267_, 0, v___x_390_);
v___x_392_ = v___x_267_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_k_386_);
lean_ctor_set(v_reuseFailAlloc_393_, 2, v_v_387_);
lean_ctor_set(v_reuseFailAlloc_393_, 3, v_r_263_);
lean_ctor_set(v_reuseFailAlloc_393_, 4, v___x_389_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
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
lean_object* v___x_408_; uint8_t v_isShared_409_; uint8_t v_isSharedCheck_559_; 
lean_inc(v_r_263_);
lean_inc(v_v_261_);
lean_inc(v_k_260_);
v_isSharedCheck_559_ = !lean_is_exclusive(v_r_75_);
if (v_isSharedCheck_559_ == 0)
{
lean_object* v_unused_560_; lean_object* v_unused_561_; lean_object* v_unused_562_; lean_object* v_unused_563_; lean_object* v_unused_564_; 
v_unused_560_ = lean_ctor_get(v_r_75_, 4);
lean_dec(v_unused_560_);
v_unused_561_ = lean_ctor_get(v_r_75_, 3);
lean_dec(v_unused_561_);
v_unused_562_ = lean_ctor_get(v_r_75_, 2);
lean_dec(v_unused_562_);
v_unused_563_ = lean_ctor_get(v_r_75_, 1);
lean_dec(v_unused_563_);
v_unused_564_ = lean_ctor_get(v_r_75_, 0);
lean_dec(v_unused_564_);
v___x_408_ = v_r_75_;
v_isShared_409_ = v_isSharedCheck_559_;
goto v_resetjp_407_;
}
else
{
lean_dec(v_r_75_);
v___x_408_ = lean_box(0);
v_isShared_409_ = v_isSharedCheck_559_;
goto v_resetjp_407_;
}
v_resetjp_407_:
{
lean_object* v___x_410_; lean_object* v_tree_411_; 
v___x_410_ = l_Std_DTreeMap_Internal_Impl_minView___redArg(v_k_260_, v_v_261_, v_l_262_, v_r_263_);
v_tree_411_ = lean_ctor_get(v___x_410_, 2);
lean_inc(v_tree_411_);
if (lean_obj_tag(v_tree_411_) == 0)
{
lean_object* v_k_412_; lean_object* v_v_413_; lean_object* v_size_414_; lean_object* v___x_415_; lean_object* v___x_416_; uint8_t v___x_417_; 
v_k_412_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_k_412_);
v_v_413_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_v_413_);
lean_dec_ref(v___x_410_);
v_size_414_ = lean_ctor_get(v_tree_411_, 0);
v___x_415_ = lean_unsigned_to_nat(3u);
v___x_416_ = lean_nat_mul(v___x_415_, v_size_414_);
v___x_417_ = lean_nat_dec_lt(v___x_416_, v_size_254_);
lean_dec(v___x_416_);
if (v___x_417_ == 0)
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_421_; 
lean_dec(v_r_258_);
v___x_418_ = lean_nat_add(v___x_264_, v_size_254_);
v___x_419_ = lean_nat_add(v___x_418_, v_size_414_);
lean_dec(v___x_418_);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_tree_411_);
lean_ctor_set(v___x_408_, 3, v_l_74_);
lean_ctor_set(v___x_408_, 2, v_v_413_);
lean_ctor_set(v___x_408_, 1, v_k_412_);
lean_ctor_set(v___x_408_, 0, v___x_419_);
v___x_421_ = v___x_408_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_419_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v_k_412_);
lean_ctor_set(v_reuseFailAlloc_422_, 2, v_v_413_);
lean_ctor_set(v_reuseFailAlloc_422_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_422_, 4, v_tree_411_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
else
{
lean_object* v___x_424_; uint8_t v_isShared_425_; uint8_t v_isSharedCheck_488_; 
lean_inc(v_l_257_);
lean_inc(v_v_256_);
lean_inc(v_k_255_);
lean_inc(v_size_254_);
v_isSharedCheck_488_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_488_ == 0)
{
lean_object* v_unused_489_; lean_object* v_unused_490_; lean_object* v_unused_491_; lean_object* v_unused_492_; lean_object* v_unused_493_; 
v_unused_489_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_489_);
v_unused_490_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_490_);
v_unused_491_ = lean_ctor_get(v_l_74_, 2);
lean_dec(v_unused_491_);
v_unused_492_ = lean_ctor_get(v_l_74_, 1);
lean_dec(v_unused_492_);
v_unused_493_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_493_);
v___x_424_ = v_l_74_;
v_isShared_425_ = v_isSharedCheck_488_;
goto v_resetjp_423_;
}
else
{
lean_dec(v_l_74_);
v___x_424_ = lean_box(0);
v_isShared_425_ = v_isSharedCheck_488_;
goto v_resetjp_423_;
}
v_resetjp_423_:
{
lean_object* v_size_426_; lean_object* v_size_427_; lean_object* v_k_428_; lean_object* v_v_429_; lean_object* v_l_430_; lean_object* v_r_431_; lean_object* v___x_432_; lean_object* v___x_433_; uint8_t v___x_434_; 
v_size_426_ = lean_ctor_get(v_l_257_, 0);
v_size_427_ = lean_ctor_get(v_r_258_, 0);
v_k_428_ = lean_ctor_get(v_r_258_, 1);
v_v_429_ = lean_ctor_get(v_r_258_, 2);
v_l_430_ = lean_ctor_get(v_r_258_, 3);
v_r_431_ = lean_ctor_get(v_r_258_, 4);
v___x_432_ = lean_unsigned_to_nat(2u);
v___x_433_ = lean_nat_mul(v___x_432_, v_size_426_);
v___x_434_ = lean_nat_dec_lt(v_size_427_, v___x_433_);
lean_dec(v___x_433_);
if (v___x_434_ == 0)
{
lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_472_; 
lean_inc(v_r_431_);
lean_inc(v_l_430_);
lean_inc(v_v_429_);
lean_inc(v_k_428_);
lean_del_object(v___x_424_);
v_isSharedCheck_472_ = !lean_is_exclusive(v_r_258_);
if (v_isSharedCheck_472_ == 0)
{
lean_object* v_unused_473_; lean_object* v_unused_474_; lean_object* v_unused_475_; lean_object* v_unused_476_; lean_object* v_unused_477_; 
v_unused_473_ = lean_ctor_get(v_r_258_, 4);
lean_dec(v_unused_473_);
v_unused_474_ = lean_ctor_get(v_r_258_, 3);
lean_dec(v_unused_474_);
v_unused_475_ = lean_ctor_get(v_r_258_, 2);
lean_dec(v_unused_475_);
v_unused_476_ = lean_ctor_get(v_r_258_, 1);
lean_dec(v_unused_476_);
v_unused_477_ = lean_ctor_get(v_r_258_, 0);
lean_dec(v_unused_477_);
v___x_436_ = v_r_258_;
v_isShared_437_ = v_isSharedCheck_472_;
goto v_resetjp_435_;
}
else
{
lean_dec(v_r_258_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_472_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___y_441_; lean_object* v___y_442_; lean_object* v___y_443_; lean_object* v___x_460_; lean_object* v___y_462_; 
v___x_438_ = lean_nat_add(v___x_264_, v_size_254_);
lean_dec(v_size_254_);
v___x_439_ = lean_nat_add(v___x_438_, v_size_414_);
lean_dec(v___x_438_);
v___x_460_ = lean_nat_add(v___x_264_, v_size_426_);
if (lean_obj_tag(v_l_430_) == 0)
{
lean_object* v_size_470_; 
v_size_470_ = lean_ctor_get(v_l_430_, 0);
lean_inc(v_size_470_);
v___y_462_ = v_size_470_;
goto v___jp_461_;
}
else
{
lean_object* v___x_471_; 
v___x_471_ = lean_unsigned_to_nat(0u);
v___y_462_ = v___x_471_;
goto v___jp_461_;
}
v___jp_440_:
{
lean_object* v___x_444_; lean_object* v___x_446_; 
v___x_444_ = lean_nat_add(v___y_442_, v___y_443_);
lean_dec(v___y_443_);
lean_dec(v___y_442_);
lean_inc_ref(v_tree_411_);
if (v_isShared_437_ == 0)
{
lean_ctor_set(v___x_436_, 4, v_tree_411_);
lean_ctor_set(v___x_436_, 3, v_r_431_);
lean_ctor_set(v___x_436_, 2, v_v_413_);
lean_ctor_set(v___x_436_, 1, v_k_412_);
lean_ctor_set(v___x_436_, 0, v___x_444_);
v___x_446_ = v___x_436_;
goto v_reusejp_445_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v___x_444_);
lean_ctor_set(v_reuseFailAlloc_459_, 1, v_k_412_);
lean_ctor_set(v_reuseFailAlloc_459_, 2, v_v_413_);
lean_ctor_set(v_reuseFailAlloc_459_, 3, v_r_431_);
lean_ctor_set(v_reuseFailAlloc_459_, 4, v_tree_411_);
v___x_446_ = v_reuseFailAlloc_459_;
goto v_reusejp_445_;
}
v_reusejp_445_:
{
lean_object* v___x_448_; uint8_t v_isShared_449_; uint8_t v_isSharedCheck_453_; 
v_isSharedCheck_453_ = !lean_is_exclusive(v_tree_411_);
if (v_isSharedCheck_453_ == 0)
{
lean_object* v_unused_454_; lean_object* v_unused_455_; lean_object* v_unused_456_; lean_object* v_unused_457_; lean_object* v_unused_458_; 
v_unused_454_ = lean_ctor_get(v_tree_411_, 4);
lean_dec(v_unused_454_);
v_unused_455_ = lean_ctor_get(v_tree_411_, 3);
lean_dec(v_unused_455_);
v_unused_456_ = lean_ctor_get(v_tree_411_, 2);
lean_dec(v_unused_456_);
v_unused_457_ = lean_ctor_get(v_tree_411_, 1);
lean_dec(v_unused_457_);
v_unused_458_ = lean_ctor_get(v_tree_411_, 0);
lean_dec(v_unused_458_);
v___x_448_ = v_tree_411_;
v_isShared_449_ = v_isSharedCheck_453_;
goto v_resetjp_447_;
}
else
{
lean_dec(v_tree_411_);
v___x_448_ = lean_box(0);
v_isShared_449_ = v_isSharedCheck_453_;
goto v_resetjp_447_;
}
v_resetjp_447_:
{
lean_object* v___x_451_; 
if (v_isShared_449_ == 0)
{
lean_ctor_set(v___x_448_, 4, v___x_446_);
lean_ctor_set(v___x_448_, 3, v___y_441_);
lean_ctor_set(v___x_448_, 2, v_v_429_);
lean_ctor_set(v___x_448_, 1, v_k_428_);
lean_ctor_set(v___x_448_, 0, v___x_439_);
v___x_451_ = v___x_448_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v___x_439_);
lean_ctor_set(v_reuseFailAlloc_452_, 1, v_k_428_);
lean_ctor_set(v_reuseFailAlloc_452_, 2, v_v_429_);
lean_ctor_set(v_reuseFailAlloc_452_, 3, v___y_441_);
lean_ctor_set(v_reuseFailAlloc_452_, 4, v___x_446_);
v___x_451_ = v_reuseFailAlloc_452_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
return v___x_451_;
}
}
}
}
v___jp_461_:
{
lean_object* v___x_463_; lean_object* v___x_465_; 
v___x_463_ = lean_nat_add(v___x_460_, v___y_462_);
lean_dec(v___y_462_);
lean_dec(v___x_460_);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_l_430_);
lean_ctor_set(v___x_408_, 3, v_l_257_);
lean_ctor_set(v___x_408_, 2, v_v_256_);
lean_ctor_set(v___x_408_, 1, v_k_255_);
lean_ctor_set(v___x_408_, 0, v___x_463_);
v___x_465_ = v___x_408_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_463_);
lean_ctor_set(v_reuseFailAlloc_469_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_469_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_469_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_469_, 4, v_l_430_);
v___x_465_ = v_reuseFailAlloc_469_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
lean_object* v___x_466_; 
v___x_466_ = lean_nat_add(v___x_264_, v_size_414_);
if (lean_obj_tag(v_r_431_) == 0)
{
lean_object* v_size_467_; 
v_size_467_ = lean_ctor_get(v_r_431_, 0);
lean_inc(v_size_467_);
v___y_441_ = v___x_465_;
v___y_442_ = v___x_466_;
v___y_443_ = v_size_467_;
goto v___jp_440_;
}
else
{
lean_object* v___x_468_; 
v___x_468_ = lean_unsigned_to_nat(0u);
v___y_441_ = v___x_465_;
v___y_442_ = v___x_466_;
v___y_443_ = v___x_468_;
goto v___jp_440_;
}
}
}
}
}
else
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_478_ = lean_nat_add(v___x_264_, v_size_254_);
lean_dec(v_size_254_);
v___x_479_ = lean_nat_add(v___x_478_, v_size_414_);
lean_dec(v___x_478_);
v___x_480_ = lean_nat_add(v___x_264_, v_size_414_);
v___x_481_ = lean_nat_add(v___x_480_, v_size_427_);
lean_dec(v___x_480_);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_tree_411_);
lean_ctor_set(v___x_408_, 3, v_r_258_);
lean_ctor_set(v___x_408_, 2, v_v_413_);
lean_ctor_set(v___x_408_, 1, v_k_412_);
lean_ctor_set(v___x_408_, 0, v___x_481_);
v___x_483_ = v___x_408_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_487_; 
v_reuseFailAlloc_487_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_487_, 0, v___x_481_);
lean_ctor_set(v_reuseFailAlloc_487_, 1, v_k_412_);
lean_ctor_set(v_reuseFailAlloc_487_, 2, v_v_413_);
lean_ctor_set(v_reuseFailAlloc_487_, 3, v_r_258_);
lean_ctor_set(v_reuseFailAlloc_487_, 4, v_tree_411_);
v___x_483_ = v_reuseFailAlloc_487_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_485_; 
if (v_isShared_425_ == 0)
{
lean_ctor_set(v___x_424_, 4, v___x_483_);
lean_ctor_set(v___x_424_, 0, v___x_479_);
v___x_485_ = v___x_424_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_479_);
lean_ctor_set(v_reuseFailAlloc_486_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_486_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_486_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_486_, 4, v___x_483_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_l_257_) == 0)
{
lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_517_; 
lean_inc_ref(v_l_257_);
lean_inc(v_v_256_);
lean_inc(v_k_255_);
lean_inc(v_size_254_);
v_isSharedCheck_517_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_517_ == 0)
{
lean_object* v_unused_518_; lean_object* v_unused_519_; lean_object* v_unused_520_; lean_object* v_unused_521_; lean_object* v_unused_522_; 
v_unused_518_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_518_);
v_unused_519_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_519_);
v_unused_520_ = lean_ctor_get(v_l_74_, 2);
lean_dec(v_unused_520_);
v_unused_521_ = lean_ctor_get(v_l_74_, 1);
lean_dec(v_unused_521_);
v_unused_522_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_522_);
v___x_495_ = v_l_74_;
v_isShared_496_ = v_isSharedCheck_517_;
goto v_resetjp_494_;
}
else
{
lean_dec(v_l_74_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_517_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
if (lean_obj_tag(v_r_258_) == 0)
{
lean_object* v_k_497_; lean_object* v_v_498_; lean_object* v_size_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_503_; 
v_k_497_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_k_497_);
v_v_498_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_v_498_);
lean_dec_ref(v___x_410_);
v_size_499_ = lean_ctor_get(v_r_258_, 0);
v___x_500_ = lean_nat_add(v___x_264_, v_size_254_);
lean_dec(v_size_254_);
v___x_501_ = lean_nat_add(v___x_264_, v_size_499_);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_tree_411_);
lean_ctor_set(v___x_408_, 3, v_r_258_);
lean_ctor_set(v___x_408_, 2, v_v_498_);
lean_ctor_set(v___x_408_, 1, v_k_497_);
lean_ctor_set(v___x_408_, 0, v___x_501_);
v___x_503_ = v___x_408_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_507_; 
v_reuseFailAlloc_507_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_507_, 0, v___x_501_);
lean_ctor_set(v_reuseFailAlloc_507_, 1, v_k_497_);
lean_ctor_set(v_reuseFailAlloc_507_, 2, v_v_498_);
lean_ctor_set(v_reuseFailAlloc_507_, 3, v_r_258_);
lean_ctor_set(v_reuseFailAlloc_507_, 4, v_tree_411_);
v___x_503_ = v_reuseFailAlloc_507_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
lean_object* v___x_505_; 
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 4, v___x_503_);
lean_ctor_set(v___x_495_, 0, v___x_500_);
v___x_505_ = v___x_495_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v___x_500_);
lean_ctor_set(v_reuseFailAlloc_506_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_506_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_506_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_506_, 4, v___x_503_);
v___x_505_ = v_reuseFailAlloc_506_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
return v___x_505_;
}
}
}
else
{
lean_object* v_k_508_; lean_object* v_v_509_; lean_object* v___x_510_; lean_object* v___x_512_; 
lean_dec(v_size_254_);
v_k_508_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_k_508_);
v_v_509_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_v_509_);
lean_dec_ref(v___x_410_);
v___x_510_ = lean_unsigned_to_nat(3u);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_r_258_);
lean_ctor_set(v___x_408_, 3, v_r_258_);
lean_ctor_set(v___x_408_, 2, v_v_509_);
lean_ctor_set(v___x_408_, 1, v_k_508_);
lean_ctor_set(v___x_408_, 0, v___x_264_);
v___x_512_ = v___x_408_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_516_, 1, v_k_508_);
lean_ctor_set(v_reuseFailAlloc_516_, 2, v_v_509_);
lean_ctor_set(v_reuseFailAlloc_516_, 3, v_r_258_);
lean_ctor_set(v_reuseFailAlloc_516_, 4, v_r_258_);
v___x_512_ = v_reuseFailAlloc_516_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
lean_object* v___x_514_; 
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 4, v___x_512_);
lean_ctor_set(v___x_495_, 0, v___x_510_);
v___x_514_ = v___x_495_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v___x_510_);
lean_ctor_set(v_reuseFailAlloc_515_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_515_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_515_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_515_, 4, v___x_512_);
v___x_514_ = v_reuseFailAlloc_515_;
goto v_reusejp_513_;
}
v_reusejp_513_:
{
return v___x_514_;
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_258_) == 0)
{
lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_547_; 
lean_inc(v_l_257_);
lean_inc(v_v_256_);
lean_inc(v_k_255_);
v_isSharedCheck_547_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_547_ == 0)
{
lean_object* v_unused_548_; lean_object* v_unused_549_; lean_object* v_unused_550_; lean_object* v_unused_551_; lean_object* v_unused_552_; 
v_unused_548_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_548_);
v_unused_549_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_549_);
v_unused_550_ = lean_ctor_get(v_l_74_, 2);
lean_dec(v_unused_550_);
v_unused_551_ = lean_ctor_get(v_l_74_, 1);
lean_dec(v_unused_551_);
v_unused_552_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_552_);
v___x_524_ = v_l_74_;
v_isShared_525_ = v_isSharedCheck_547_;
goto v_resetjp_523_;
}
else
{
lean_dec(v_l_74_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_547_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v_k_526_; lean_object* v_v_527_; lean_object* v_k_528_; lean_object* v_v_529_; lean_object* v___x_531_; uint8_t v_isShared_532_; uint8_t v_isSharedCheck_543_; 
v_k_526_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_k_526_);
v_v_527_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_v_527_);
lean_dec_ref(v___x_410_);
v_k_528_ = lean_ctor_get(v_r_258_, 1);
v_v_529_ = lean_ctor_get(v_r_258_, 2);
v_isSharedCheck_543_ = !lean_is_exclusive(v_r_258_);
if (v_isSharedCheck_543_ == 0)
{
lean_object* v_unused_544_; lean_object* v_unused_545_; lean_object* v_unused_546_; 
v_unused_544_ = lean_ctor_get(v_r_258_, 4);
lean_dec(v_unused_544_);
v_unused_545_ = lean_ctor_get(v_r_258_, 3);
lean_dec(v_unused_545_);
v_unused_546_ = lean_ctor_get(v_r_258_, 0);
lean_dec(v_unused_546_);
v___x_531_ = v_r_258_;
v_isShared_532_ = v_isSharedCheck_543_;
goto v_resetjp_530_;
}
else
{
lean_inc(v_v_529_);
lean_inc(v_k_528_);
lean_dec(v_r_258_);
v___x_531_ = lean_box(0);
v_isShared_532_ = v_isSharedCheck_543_;
goto v_resetjp_530_;
}
v_resetjp_530_:
{
lean_object* v___x_533_; lean_object* v___x_535_; 
v___x_533_ = lean_unsigned_to_nat(3u);
if (v_isShared_532_ == 0)
{
lean_ctor_set(v___x_531_, 4, v_l_257_);
lean_ctor_set(v___x_531_, 3, v_l_257_);
lean_ctor_set(v___x_531_, 2, v_v_256_);
lean_ctor_set(v___x_531_, 1, v_k_255_);
lean_ctor_set(v___x_531_, 0, v___x_264_);
v___x_535_ = v___x_531_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_542_; 
v_reuseFailAlloc_542_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_542_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_542_, 1, v_k_255_);
lean_ctor_set(v_reuseFailAlloc_542_, 2, v_v_256_);
lean_ctor_set(v_reuseFailAlloc_542_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_542_, 4, v_l_257_);
v___x_535_ = v_reuseFailAlloc_542_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
lean_object* v___x_537_; 
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_l_257_);
lean_ctor_set(v___x_408_, 3, v_l_257_);
lean_ctor_set(v___x_408_, 2, v_v_527_);
lean_ctor_set(v___x_408_, 1, v_k_526_);
lean_ctor_set(v___x_408_, 0, v___x_264_);
v___x_537_ = v___x_408_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_541_, 1, v_k_526_);
lean_ctor_set(v_reuseFailAlloc_541_, 2, v_v_527_);
lean_ctor_set(v_reuseFailAlloc_541_, 3, v_l_257_);
lean_ctor_set(v_reuseFailAlloc_541_, 4, v_l_257_);
v___x_537_ = v_reuseFailAlloc_541_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
lean_object* v___x_539_; 
if (v_isShared_525_ == 0)
{
lean_ctor_set(v___x_524_, 4, v___x_537_);
lean_ctor_set(v___x_524_, 3, v___x_535_);
lean_ctor_set(v___x_524_, 2, v_v_529_);
lean_ctor_set(v___x_524_, 1, v_k_528_);
lean_ctor_set(v___x_524_, 0, v___x_533_);
v___x_539_ = v___x_524_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v___x_533_);
lean_ctor_set(v_reuseFailAlloc_540_, 1, v_k_528_);
lean_ctor_set(v_reuseFailAlloc_540_, 2, v_v_529_);
lean_ctor_set(v_reuseFailAlloc_540_, 3, v___x_535_);
lean_ctor_set(v_reuseFailAlloc_540_, 4, v___x_537_);
v___x_539_ = v_reuseFailAlloc_540_;
goto v_reusejp_538_;
}
v_reusejp_538_:
{
return v___x_539_;
}
}
}
}
}
}
else
{
lean_object* v_k_553_; lean_object* v_v_554_; lean_object* v___x_555_; lean_object* v___x_557_; 
v_k_553_ = lean_ctor_get(v___x_410_, 0);
lean_inc(v_k_553_);
v_v_554_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_v_554_);
lean_dec_ref(v___x_410_);
v___x_555_ = lean_unsigned_to_nat(2u);
if (v_isShared_409_ == 0)
{
lean_ctor_set(v___x_408_, 4, v_r_258_);
lean_ctor_set(v___x_408_, 3, v_l_74_);
lean_ctor_set(v___x_408_, 2, v_v_554_);
lean_ctor_set(v___x_408_, 1, v_k_553_);
lean_ctor_set(v___x_408_, 0, v___x_555_);
v___x_557_ = v___x_408_;
goto v_reusejp_556_;
}
else
{
lean_object* v_reuseFailAlloc_558_; 
v_reuseFailAlloc_558_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_558_, 0, v___x_555_);
lean_ctor_set(v_reuseFailAlloc_558_, 1, v_k_553_);
lean_ctor_set(v_reuseFailAlloc_558_, 2, v_v_554_);
lean_ctor_set(v_reuseFailAlloc_558_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_558_, 4, v_r_258_);
v___x_557_ = v_reuseFailAlloc_558_;
goto v_reusejp_556_;
}
v_reusejp_556_:
{
return v___x_557_;
}
}
}
}
}
}
}
else
{
return v_l_74_;
}
}
else
{
return v_r_75_;
}
}
default: 
{
lean_object* v_impl_565_; lean_object* v___x_566_; 
v_impl_565_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(v_k_70_, v_r_75_);
v___x_566_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_565_) == 0)
{
if (lean_obj_tag(v_l_74_) == 0)
{
lean_object* v_size_567_; lean_object* v_size_568_; lean_object* v_k_569_; lean_object* v_v_570_; lean_object* v_l_571_; lean_object* v_r_572_; lean_object* v___x_573_; lean_object* v___x_574_; uint8_t v___x_575_; 
v_size_567_ = lean_ctor_get(v_impl_565_, 0);
lean_inc(v_size_567_);
v_size_568_ = lean_ctor_get(v_l_74_, 0);
v_k_569_ = lean_ctor_get(v_l_74_, 1);
v_v_570_ = lean_ctor_get(v_l_74_, 2);
v_l_571_ = lean_ctor_get(v_l_74_, 3);
v_r_572_ = lean_ctor_get(v_l_74_, 4);
lean_inc(v_r_572_);
v___x_573_ = lean_unsigned_to_nat(3u);
v___x_574_ = lean_nat_mul(v___x_573_, v_size_567_);
v___x_575_ = lean_nat_dec_lt(v___x_574_, v_size_568_);
lean_dec(v___x_574_);
if (v___x_575_ == 0)
{
lean_object* v___x_576_; lean_object* v___x_577_; lean_object* v___x_579_; 
lean_dec(v_r_572_);
v___x_576_ = lean_nat_add(v___x_566_, v_size_568_);
v___x_577_ = lean_nat_add(v___x_576_, v_size_567_);
lean_dec(v_size_567_);
lean_dec(v___x_576_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_impl_565_);
lean_ctor_set(v___x_77_, 0, v___x_577_);
v___x_579_ = v___x_77_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v___x_577_);
lean_ctor_set(v_reuseFailAlloc_580_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_580_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_580_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_580_, 4, v_impl_565_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
else
{
lean_object* v___x_582_; uint8_t v_isShared_583_; uint8_t v_isSharedCheck_646_; 
lean_inc(v_l_571_);
lean_inc(v_v_570_);
lean_inc(v_k_569_);
lean_inc(v_size_568_);
v_isSharedCheck_646_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_646_ == 0)
{
lean_object* v_unused_647_; lean_object* v_unused_648_; lean_object* v_unused_649_; lean_object* v_unused_650_; lean_object* v_unused_651_; 
v_unused_647_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_647_);
v_unused_648_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_648_);
v_unused_649_ = lean_ctor_get(v_l_74_, 2);
lean_dec(v_unused_649_);
v_unused_650_ = lean_ctor_get(v_l_74_, 1);
lean_dec(v_unused_650_);
v_unused_651_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_651_);
v___x_582_ = v_l_74_;
v_isShared_583_ = v_isSharedCheck_646_;
goto v_resetjp_581_;
}
else
{
lean_dec(v_l_74_);
v___x_582_ = lean_box(0);
v_isShared_583_ = v_isSharedCheck_646_;
goto v_resetjp_581_;
}
v_resetjp_581_:
{
lean_object* v_size_584_; lean_object* v_size_585_; lean_object* v_k_586_; lean_object* v_v_587_; lean_object* v_l_588_; lean_object* v_r_589_; lean_object* v___x_590_; lean_object* v___x_591_; uint8_t v___x_592_; 
v_size_584_ = lean_ctor_get(v_l_571_, 0);
v_size_585_ = lean_ctor_get(v_r_572_, 0);
v_k_586_ = lean_ctor_get(v_r_572_, 1);
v_v_587_ = lean_ctor_get(v_r_572_, 2);
v_l_588_ = lean_ctor_get(v_r_572_, 3);
v_r_589_ = lean_ctor_get(v_r_572_, 4);
v___x_590_ = lean_unsigned_to_nat(2u);
v___x_591_ = lean_nat_mul(v___x_590_, v_size_584_);
v___x_592_ = lean_nat_dec_lt(v_size_585_, v___x_591_);
lean_dec(v___x_591_);
if (v___x_592_ == 0)
{
lean_object* v___x_594_; uint8_t v_isShared_595_; uint8_t v_isSharedCheck_621_; 
lean_inc(v_r_589_);
lean_inc(v_l_588_);
lean_inc(v_v_587_);
lean_inc(v_k_586_);
v_isSharedCheck_621_ = !lean_is_exclusive(v_r_572_);
if (v_isSharedCheck_621_ == 0)
{
lean_object* v_unused_622_; lean_object* v_unused_623_; lean_object* v_unused_624_; lean_object* v_unused_625_; lean_object* v_unused_626_; 
v_unused_622_ = lean_ctor_get(v_r_572_, 4);
lean_dec(v_unused_622_);
v_unused_623_ = lean_ctor_get(v_r_572_, 3);
lean_dec(v_unused_623_);
v_unused_624_ = lean_ctor_get(v_r_572_, 2);
lean_dec(v_unused_624_);
v_unused_625_ = lean_ctor_get(v_r_572_, 1);
lean_dec(v_unused_625_);
v_unused_626_ = lean_ctor_get(v_r_572_, 0);
lean_dec(v_unused_626_);
v___x_594_ = v_r_572_;
v_isShared_595_ = v_isSharedCheck_621_;
goto v_resetjp_593_;
}
else
{
lean_dec(v_r_572_);
v___x_594_ = lean_box(0);
v_isShared_595_ = v_isSharedCheck_621_;
goto v_resetjp_593_;
}
v_resetjp_593_:
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_601_; lean_object* v___x_609_; lean_object* v___y_611_; 
v___x_596_ = lean_nat_add(v___x_566_, v_size_568_);
lean_dec(v_size_568_);
v___x_597_ = lean_nat_add(v___x_596_, v_size_567_);
lean_dec(v___x_596_);
v___x_609_ = lean_nat_add(v___x_566_, v_size_584_);
if (lean_obj_tag(v_l_588_) == 0)
{
lean_object* v_size_619_; 
v_size_619_ = lean_ctor_get(v_l_588_, 0);
lean_inc(v_size_619_);
v___y_611_ = v_size_619_;
goto v___jp_610_;
}
else
{
lean_object* v___x_620_; 
v___x_620_ = lean_unsigned_to_nat(0u);
v___y_611_ = v___x_620_;
goto v___jp_610_;
}
v___jp_598_:
{
lean_object* v___x_602_; lean_object* v___x_604_; 
v___x_602_ = lean_nat_add(v___y_599_, v___y_601_);
lean_dec(v___y_601_);
lean_dec(v___y_599_);
if (v_isShared_595_ == 0)
{
lean_ctor_set(v___x_594_, 4, v_impl_565_);
lean_ctor_set(v___x_594_, 3, v_r_589_);
lean_ctor_set(v___x_594_, 2, v_v_73_);
lean_ctor_set(v___x_594_, 1, v_k_72_);
lean_ctor_set(v___x_594_, 0, v___x_602_);
v___x_604_ = v___x_594_;
goto v_reusejp_603_;
}
else
{
lean_object* v_reuseFailAlloc_608_; 
v_reuseFailAlloc_608_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_608_, 0, v___x_602_);
lean_ctor_set(v_reuseFailAlloc_608_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_608_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_608_, 3, v_r_589_);
lean_ctor_set(v_reuseFailAlloc_608_, 4, v_impl_565_);
v___x_604_ = v_reuseFailAlloc_608_;
goto v_reusejp_603_;
}
v_reusejp_603_:
{
lean_object* v___x_606_; 
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 4, v___x_604_);
lean_ctor_set(v___x_582_, 3, v___y_600_);
lean_ctor_set(v___x_582_, 2, v_v_587_);
lean_ctor_set(v___x_582_, 1, v_k_586_);
lean_ctor_set(v___x_582_, 0, v___x_597_);
v___x_606_ = v___x_582_;
goto v_reusejp_605_;
}
else
{
lean_object* v_reuseFailAlloc_607_; 
v_reuseFailAlloc_607_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_607_, 0, v___x_597_);
lean_ctor_set(v_reuseFailAlloc_607_, 1, v_k_586_);
lean_ctor_set(v_reuseFailAlloc_607_, 2, v_v_587_);
lean_ctor_set(v_reuseFailAlloc_607_, 3, v___y_600_);
lean_ctor_set(v_reuseFailAlloc_607_, 4, v___x_604_);
v___x_606_ = v_reuseFailAlloc_607_;
goto v_reusejp_605_;
}
v_reusejp_605_:
{
return v___x_606_;
}
}
}
v___jp_610_:
{
lean_object* v___x_612_; lean_object* v___x_614_; 
v___x_612_ = lean_nat_add(v___x_609_, v___y_611_);
lean_dec(v___y_611_);
lean_dec(v___x_609_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_l_588_);
lean_ctor_set(v___x_77_, 3, v_l_571_);
lean_ctor_set(v___x_77_, 2, v_v_570_);
lean_ctor_set(v___x_77_, 1, v_k_569_);
lean_ctor_set(v___x_77_, 0, v___x_612_);
v___x_614_ = v___x_77_;
goto v_reusejp_613_;
}
else
{
lean_object* v_reuseFailAlloc_618_; 
v_reuseFailAlloc_618_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_618_, 0, v___x_612_);
lean_ctor_set(v_reuseFailAlloc_618_, 1, v_k_569_);
lean_ctor_set(v_reuseFailAlloc_618_, 2, v_v_570_);
lean_ctor_set(v_reuseFailAlloc_618_, 3, v_l_571_);
lean_ctor_set(v_reuseFailAlloc_618_, 4, v_l_588_);
v___x_614_ = v_reuseFailAlloc_618_;
goto v_reusejp_613_;
}
v_reusejp_613_:
{
lean_object* v___x_615_; 
v___x_615_ = lean_nat_add(v___x_566_, v_size_567_);
lean_dec(v_size_567_);
if (lean_obj_tag(v_r_589_) == 0)
{
lean_object* v_size_616_; 
v_size_616_ = lean_ctor_get(v_r_589_, 0);
lean_inc(v_size_616_);
v___y_599_ = v___x_615_;
v___y_600_ = v___x_614_;
v___y_601_ = v_size_616_;
goto v___jp_598_;
}
else
{
lean_object* v___x_617_; 
v___x_617_ = lean_unsigned_to_nat(0u);
v___y_599_ = v___x_615_;
v___y_600_ = v___x_614_;
v___y_601_ = v___x_617_;
goto v___jp_598_;
}
}
}
}
}
else
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_632_; 
lean_del_object(v___x_77_);
v___x_627_ = lean_nat_add(v___x_566_, v_size_568_);
lean_dec(v_size_568_);
v___x_628_ = lean_nat_add(v___x_627_, v_size_567_);
lean_dec(v___x_627_);
v___x_629_ = lean_nat_add(v___x_566_, v_size_567_);
lean_dec(v_size_567_);
v___x_630_ = lean_nat_add(v___x_629_, v_size_585_);
lean_dec(v___x_629_);
lean_inc_ref(v_impl_565_);
if (v_isShared_583_ == 0)
{
lean_ctor_set(v___x_582_, 4, v_impl_565_);
lean_ctor_set(v___x_582_, 3, v_r_572_);
lean_ctor_set(v___x_582_, 2, v_v_73_);
lean_ctor_set(v___x_582_, 1, v_k_72_);
lean_ctor_set(v___x_582_, 0, v___x_630_);
v___x_632_ = v___x_582_;
goto v_reusejp_631_;
}
else
{
lean_object* v_reuseFailAlloc_645_; 
v_reuseFailAlloc_645_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_645_, 0, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_645_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_645_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_645_, 3, v_r_572_);
lean_ctor_set(v_reuseFailAlloc_645_, 4, v_impl_565_);
v___x_632_ = v_reuseFailAlloc_645_;
goto v_reusejp_631_;
}
v_reusejp_631_:
{
lean_object* v___x_634_; uint8_t v_isShared_635_; uint8_t v_isSharedCheck_639_; 
v_isSharedCheck_639_ = !lean_is_exclusive(v_impl_565_);
if (v_isSharedCheck_639_ == 0)
{
lean_object* v_unused_640_; lean_object* v_unused_641_; lean_object* v_unused_642_; lean_object* v_unused_643_; lean_object* v_unused_644_; 
v_unused_640_ = lean_ctor_get(v_impl_565_, 4);
lean_dec(v_unused_640_);
v_unused_641_ = lean_ctor_get(v_impl_565_, 3);
lean_dec(v_unused_641_);
v_unused_642_ = lean_ctor_get(v_impl_565_, 2);
lean_dec(v_unused_642_);
v_unused_643_ = lean_ctor_get(v_impl_565_, 1);
lean_dec(v_unused_643_);
v_unused_644_ = lean_ctor_get(v_impl_565_, 0);
lean_dec(v_unused_644_);
v___x_634_ = v_impl_565_;
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
else
{
lean_dec(v_impl_565_);
v___x_634_ = lean_box(0);
v_isShared_635_ = v_isSharedCheck_639_;
goto v_resetjp_633_;
}
v_resetjp_633_:
{
lean_object* v___x_637_; 
if (v_isShared_635_ == 0)
{
lean_ctor_set(v___x_634_, 4, v___x_632_);
lean_ctor_set(v___x_634_, 3, v_l_571_);
lean_ctor_set(v___x_634_, 2, v_v_570_);
lean_ctor_set(v___x_634_, 1, v_k_569_);
lean_ctor_set(v___x_634_, 0, v___x_628_);
v___x_637_ = v___x_634_;
goto v_reusejp_636_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v___x_628_);
lean_ctor_set(v_reuseFailAlloc_638_, 1, v_k_569_);
lean_ctor_set(v_reuseFailAlloc_638_, 2, v_v_570_);
lean_ctor_set(v_reuseFailAlloc_638_, 3, v_l_571_);
lean_ctor_set(v_reuseFailAlloc_638_, 4, v___x_632_);
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
}
}
}
else
{
lean_object* v_size_652_; lean_object* v___x_653_; lean_object* v___x_655_; 
v_size_652_ = lean_ctor_get(v_impl_565_, 0);
lean_inc(v_size_652_);
v___x_653_ = lean_nat_add(v___x_566_, v_size_652_);
lean_dec(v_size_652_);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_impl_565_);
lean_ctor_set(v___x_77_, 0, v___x_653_);
v___x_655_ = v___x_77_;
goto v_reusejp_654_;
}
else
{
lean_object* v_reuseFailAlloc_656_; 
v_reuseFailAlloc_656_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_656_, 0, v___x_653_);
lean_ctor_set(v_reuseFailAlloc_656_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_656_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_656_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_656_, 4, v_impl_565_);
v___x_655_ = v_reuseFailAlloc_656_;
goto v_reusejp_654_;
}
v_reusejp_654_:
{
return v___x_655_;
}
}
}
else
{
if (lean_obj_tag(v_l_74_) == 0)
{
lean_object* v_l_657_; 
v_l_657_ = lean_ctor_get(v_l_74_, 3);
if (lean_obj_tag(v_l_657_) == 0)
{
lean_object* v_r_658_; 
lean_inc_ref(v_l_657_);
v_r_658_ = lean_ctor_get(v_l_74_, 4);
lean_inc(v_r_658_);
if (lean_obj_tag(v_r_658_) == 0)
{
lean_object* v_size_659_; lean_object* v_k_660_; lean_object* v_v_661_; lean_object* v___x_663_; uint8_t v_isShared_664_; uint8_t v_isSharedCheck_674_; 
v_size_659_ = lean_ctor_get(v_l_74_, 0);
v_k_660_ = lean_ctor_get(v_l_74_, 1);
v_v_661_ = lean_ctor_get(v_l_74_, 2);
v_isSharedCheck_674_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_674_ == 0)
{
lean_object* v_unused_675_; lean_object* v_unused_676_; 
v_unused_675_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_675_);
v_unused_676_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_676_);
v___x_663_ = v_l_74_;
v_isShared_664_ = v_isSharedCheck_674_;
goto v_resetjp_662_;
}
else
{
lean_inc(v_v_661_);
lean_inc(v_k_660_);
lean_inc(v_size_659_);
lean_dec(v_l_74_);
v___x_663_ = lean_box(0);
v_isShared_664_ = v_isSharedCheck_674_;
goto v_resetjp_662_;
}
v_resetjp_662_:
{
lean_object* v_size_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_669_; 
v_size_665_ = lean_ctor_get(v_r_658_, 0);
v___x_666_ = lean_nat_add(v___x_566_, v_size_659_);
lean_dec(v_size_659_);
v___x_667_ = lean_nat_add(v___x_566_, v_size_665_);
if (v_isShared_664_ == 0)
{
lean_ctor_set(v___x_663_, 4, v_impl_565_);
lean_ctor_set(v___x_663_, 3, v_r_658_);
lean_ctor_set(v___x_663_, 2, v_v_73_);
lean_ctor_set(v___x_663_, 1, v_k_72_);
lean_ctor_set(v___x_663_, 0, v___x_667_);
v___x_669_ = v___x_663_;
goto v_reusejp_668_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v___x_667_);
lean_ctor_set(v_reuseFailAlloc_673_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_673_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_673_, 3, v_r_658_);
lean_ctor_set(v_reuseFailAlloc_673_, 4, v_impl_565_);
v___x_669_ = v_reuseFailAlloc_673_;
goto v_reusejp_668_;
}
v_reusejp_668_:
{
lean_object* v___x_671_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v___x_669_);
lean_ctor_set(v___x_77_, 3, v_l_657_);
lean_ctor_set(v___x_77_, 2, v_v_661_);
lean_ctor_set(v___x_77_, 1, v_k_660_);
lean_ctor_set(v___x_77_, 0, v___x_666_);
v___x_671_ = v___x_77_;
goto v_reusejp_670_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_666_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_k_660_);
lean_ctor_set(v_reuseFailAlloc_672_, 2, v_v_661_);
lean_ctor_set(v_reuseFailAlloc_672_, 3, v_l_657_);
lean_ctor_set(v_reuseFailAlloc_672_, 4, v___x_669_);
v___x_671_ = v_reuseFailAlloc_672_;
goto v_reusejp_670_;
}
v_reusejp_670_:
{
return v___x_671_;
}
}
}
}
else
{
lean_object* v_k_677_; lean_object* v_v_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_689_; 
v_k_677_ = lean_ctor_get(v_l_74_, 1);
v_v_678_ = lean_ctor_get(v_l_74_, 2);
v_isSharedCheck_689_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_689_ == 0)
{
lean_object* v_unused_690_; lean_object* v_unused_691_; lean_object* v_unused_692_; 
v_unused_690_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_690_);
v_unused_691_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_691_);
v_unused_692_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_692_);
v___x_680_ = v_l_74_;
v_isShared_681_ = v_isSharedCheck_689_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_v_678_);
lean_inc(v_k_677_);
lean_dec(v_l_74_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_689_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_682_; lean_object* v___x_684_; 
v___x_682_ = lean_unsigned_to_nat(3u);
if (v_isShared_681_ == 0)
{
lean_ctor_set(v___x_680_, 3, v_r_658_);
lean_ctor_set(v___x_680_, 2, v_v_73_);
lean_ctor_set(v___x_680_, 1, v_k_72_);
lean_ctor_set(v___x_680_, 0, v___x_566_);
v___x_684_ = v___x_680_;
goto v_reusejp_683_;
}
else
{
lean_object* v_reuseFailAlloc_688_; 
v_reuseFailAlloc_688_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_688_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_688_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_688_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_688_, 3, v_r_658_);
lean_ctor_set(v_reuseFailAlloc_688_, 4, v_r_658_);
v___x_684_ = v_reuseFailAlloc_688_;
goto v_reusejp_683_;
}
v_reusejp_683_:
{
lean_object* v___x_686_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v___x_684_);
lean_ctor_set(v___x_77_, 3, v_l_657_);
lean_ctor_set(v___x_77_, 2, v_v_678_);
lean_ctor_set(v___x_77_, 1, v_k_677_);
lean_ctor_set(v___x_77_, 0, v___x_682_);
v___x_686_ = v___x_77_;
goto v_reusejp_685_;
}
else
{
lean_object* v_reuseFailAlloc_687_; 
v_reuseFailAlloc_687_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_687_, 0, v___x_682_);
lean_ctor_set(v_reuseFailAlloc_687_, 1, v_k_677_);
lean_ctor_set(v_reuseFailAlloc_687_, 2, v_v_678_);
lean_ctor_set(v_reuseFailAlloc_687_, 3, v_l_657_);
lean_ctor_set(v_reuseFailAlloc_687_, 4, v___x_684_);
v___x_686_ = v_reuseFailAlloc_687_;
goto v_reusejp_685_;
}
v_reusejp_685_:
{
return v___x_686_;
}
}
}
}
}
else
{
lean_object* v_r_693_; 
v_r_693_ = lean_ctor_get(v_l_74_, 4);
lean_inc(v_r_693_);
if (lean_obj_tag(v_r_693_) == 0)
{
lean_object* v_k_694_; lean_object* v_v_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_718_; 
lean_inc(v_l_657_);
v_k_694_ = lean_ctor_get(v_l_74_, 1);
v_v_695_ = lean_ctor_get(v_l_74_, 2);
v_isSharedCheck_718_ = !lean_is_exclusive(v_l_74_);
if (v_isSharedCheck_718_ == 0)
{
lean_object* v_unused_719_; lean_object* v_unused_720_; lean_object* v_unused_721_; 
v_unused_719_ = lean_ctor_get(v_l_74_, 4);
lean_dec(v_unused_719_);
v_unused_720_ = lean_ctor_get(v_l_74_, 3);
lean_dec(v_unused_720_);
v_unused_721_ = lean_ctor_get(v_l_74_, 0);
lean_dec(v_unused_721_);
v___x_697_ = v_l_74_;
v_isShared_698_ = v_isSharedCheck_718_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_v_695_);
lean_inc(v_k_694_);
lean_dec(v_l_74_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_718_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
lean_object* v_k_699_; lean_object* v_v_700_; lean_object* v___x_702_; uint8_t v_isShared_703_; uint8_t v_isSharedCheck_714_; 
v_k_699_ = lean_ctor_get(v_r_693_, 1);
v_v_700_ = lean_ctor_get(v_r_693_, 2);
v_isSharedCheck_714_ = !lean_is_exclusive(v_r_693_);
if (v_isSharedCheck_714_ == 0)
{
lean_object* v_unused_715_; lean_object* v_unused_716_; lean_object* v_unused_717_; 
v_unused_715_ = lean_ctor_get(v_r_693_, 4);
lean_dec(v_unused_715_);
v_unused_716_ = lean_ctor_get(v_r_693_, 3);
lean_dec(v_unused_716_);
v_unused_717_ = lean_ctor_get(v_r_693_, 0);
lean_dec(v_unused_717_);
v___x_702_ = v_r_693_;
v_isShared_703_ = v_isSharedCheck_714_;
goto v_resetjp_701_;
}
else
{
lean_inc(v_v_700_);
lean_inc(v_k_699_);
lean_dec(v_r_693_);
v___x_702_ = lean_box(0);
v_isShared_703_ = v_isSharedCheck_714_;
goto v_resetjp_701_;
}
v_resetjp_701_:
{
lean_object* v___x_704_; lean_object* v___x_706_; 
v___x_704_ = lean_unsigned_to_nat(3u);
if (v_isShared_703_ == 0)
{
lean_ctor_set(v___x_702_, 4, v_l_657_);
lean_ctor_set(v___x_702_, 3, v_l_657_);
lean_ctor_set(v___x_702_, 2, v_v_695_);
lean_ctor_set(v___x_702_, 1, v_k_694_);
lean_ctor_set(v___x_702_, 0, v___x_566_);
v___x_706_ = v___x_702_;
goto v_reusejp_705_;
}
else
{
lean_object* v_reuseFailAlloc_713_; 
v_reuseFailAlloc_713_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_713_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_713_, 1, v_k_694_);
lean_ctor_set(v_reuseFailAlloc_713_, 2, v_v_695_);
lean_ctor_set(v_reuseFailAlloc_713_, 3, v_l_657_);
lean_ctor_set(v_reuseFailAlloc_713_, 4, v_l_657_);
v___x_706_ = v_reuseFailAlloc_713_;
goto v_reusejp_705_;
}
v_reusejp_705_:
{
lean_object* v___x_708_; 
if (v_isShared_698_ == 0)
{
lean_ctor_set(v___x_697_, 4, v_l_657_);
lean_ctor_set(v___x_697_, 2, v_v_73_);
lean_ctor_set(v___x_697_, 1, v_k_72_);
lean_ctor_set(v___x_697_, 0, v___x_566_);
v___x_708_ = v___x_697_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_712_; 
v_reuseFailAlloc_712_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_712_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_712_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_712_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_712_, 3, v_l_657_);
lean_ctor_set(v_reuseFailAlloc_712_, 4, v_l_657_);
v___x_708_ = v_reuseFailAlloc_712_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
lean_object* v___x_710_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v___x_708_);
lean_ctor_set(v___x_77_, 3, v___x_706_);
lean_ctor_set(v___x_77_, 2, v_v_700_);
lean_ctor_set(v___x_77_, 1, v_k_699_);
lean_ctor_set(v___x_77_, 0, v___x_704_);
v___x_710_ = v___x_77_;
goto v_reusejp_709_;
}
else
{
lean_object* v_reuseFailAlloc_711_; 
v_reuseFailAlloc_711_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_711_, 0, v___x_704_);
lean_ctor_set(v_reuseFailAlloc_711_, 1, v_k_699_);
lean_ctor_set(v_reuseFailAlloc_711_, 2, v_v_700_);
lean_ctor_set(v_reuseFailAlloc_711_, 3, v___x_706_);
lean_ctor_set(v_reuseFailAlloc_711_, 4, v___x_708_);
v___x_710_ = v_reuseFailAlloc_711_;
goto v_reusejp_709_;
}
v_reusejp_709_:
{
return v___x_710_;
}
}
}
}
}
}
else
{
lean_object* v___x_722_; lean_object* v___x_724_; 
v___x_722_ = lean_unsigned_to_nat(2u);
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_r_693_);
lean_ctor_set(v___x_77_, 0, v___x_722_);
v___x_724_ = v___x_77_;
goto v_reusejp_723_;
}
else
{
lean_object* v_reuseFailAlloc_725_; 
v_reuseFailAlloc_725_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_725_, 0, v___x_722_);
lean_ctor_set(v_reuseFailAlloc_725_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_725_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_725_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_725_, 4, v_r_693_);
v___x_724_ = v_reuseFailAlloc_725_;
goto v_reusejp_723_;
}
v_reusejp_723_:
{
return v___x_724_;
}
}
}
}
else
{
lean_object* v___x_727_; 
if (v_isShared_78_ == 0)
{
lean_ctor_set(v___x_77_, 4, v_l_74_);
lean_ctor_set(v___x_77_, 0, v___x_566_);
v___x_727_ = v___x_77_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v___x_566_);
lean_ctor_set(v_reuseFailAlloc_728_, 1, v_k_72_);
lean_ctor_set(v_reuseFailAlloc_728_, 2, v_v_73_);
lean_ctor_set(v_reuseFailAlloc_728_, 3, v_l_74_);
lean_ctor_set(v_reuseFailAlloc_728_, 4, v_l_74_);
v___x_727_ = v_reuseFailAlloc_728_;
goto v_reusejp_726_;
}
v_reusejp_726_:
{
return v___x_727_;
}
}
}
}
}
}
}
else
{
return v_t_71_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg___boxed(lean_object* v_k_731_, lean_object* v_t_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(v_k_731_, v_t_732_);
lean_dec(v_k_731_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(lean_object* v_a_734_, lean_object* v___x_735_, lean_object* v_init_736_, lean_object* v_x_737_){
_start:
{
lean_object* v_d_740_; 
if (lean_obj_tag(v_x_737_) == 0)
{
lean_object* v_k_743_; lean_object* v_l_744_; lean_object* v_r_745_; lean_object* v___x_746_; lean_object* v_a_747_; 
v_k_743_ = lean_ctor_get(v_x_737_, 1);
v_l_744_ = lean_ctor_get(v_x_737_, 3);
v_r_745_ = lean_ctor_get(v_x_737_, 4);
v___x_746_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(v_a_734_, v___x_735_, v_init_736_, v_l_744_);
v_a_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_a_747_);
if (lean_obj_tag(v_a_747_) == 0)
{
lean_object* v_a_748_; 
lean_dec_ref(v___x_746_);
v_a_748_ = lean_ctor_get(v_a_747_, 0);
lean_inc(v_a_748_);
lean_dec_ref_known(v_a_747_, 1);
v_d_740_ = v_a_748_;
goto v___jp_739_;
}
else
{
lean_object* v_a_749_; lean_object* v___y_751_; lean_object* v___x_759_; 
v_a_749_ = lean_ctor_get(v_a_747_, 0);
lean_inc(v_a_749_);
lean_dec_ref_known(v_a_747_, 1);
v___x_759_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___x_735_, v_k_743_);
if (lean_obj_tag(v___x_759_) == 0)
{
lean_object* v___x_760_; 
v___x_760_ = l_Lean_NameSet_empty;
v___y_751_ = v___x_760_;
goto v___jp_750_;
}
else
{
lean_object* v_val_761_; 
v_val_761_ = lean_ctor_get(v___x_759_, 0);
lean_inc(v_val_761_);
lean_dec_ref_known(v___x_759_, 1);
v___y_751_ = v_val_761_;
goto v___jp_750_;
}
v___jp_750_:
{
uint8_t v___x_752_; 
v___x_752_ = l_Lean_NameSet_contains(v___y_751_, v_a_734_);
lean_dec(v___y_751_);
if (v___x_752_ == 0)
{
lean_object* v_a_753_; 
lean_dec(v_a_749_);
v_a_753_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_a_753_);
lean_dec_ref(v___x_746_);
if (lean_obj_tag(v_a_753_) == 0)
{
lean_object* v_a_754_; 
v_a_754_ = lean_ctor_get(v_a_753_, 0);
lean_inc(v_a_754_);
lean_dec_ref_known(v_a_753_, 1);
v_d_740_ = v_a_754_;
goto v___jp_739_;
}
else
{
lean_object* v_a_755_; 
v_a_755_ = lean_ctor_get(v_a_753_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v_a_753_, 1);
v_init_736_ = v_a_755_;
v_x_737_ = v_r_745_;
goto _start;
}
}
else
{
lean_object* v___x_757_; 
lean_dec_ref(v___x_746_);
v___x_757_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(v_k_743_, v_a_749_);
v_init_736_ = v___x_757_;
v_x_737_ = v_r_745_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_762_; lean_object* v___x_763_; 
v___x_762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_762_, 0, v_init_736_);
v___x_763_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_763_, 0, v___x_762_);
return v___x_763_;
}
v___jp_739_:
{
lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_741_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_741_, 0, v_d_740_);
v___x_742_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_742_, 0, v___x_741_);
return v___x_742_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg___boxed(lean_object* v_a_764_, lean_object* v___x_765_, lean_object* v_init_766_, lean_object* v_x_767_, lean_object* v___y_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(v_a_764_, v___x_765_, v_init_766_, v_x_767_);
lean_dec(v_x_767_);
lean_dec(v___x_765_);
lean_dec(v_a_764_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5(lean_object* v___x_770_, lean_object* v_init_771_, lean_object* v_x_772_, lean_object* v___y_773_, lean_object* v___y_774_){
_start:
{
lean_object* v_d_777_; 
if (lean_obj_tag(v_x_772_) == 0)
{
lean_object* v_k_780_; lean_object* v_l_781_; lean_object* v_r_782_; lean_object* v___x_783_; lean_object* v_a_784_; 
v_k_780_ = lean_ctor_get(v_x_772_, 1);
v_l_781_ = lean_ctor_get(v_x_772_, 3);
v_r_782_ = lean_ctor_get(v_x_772_, 4);
v___x_783_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5(v___x_770_, v_init_771_, v_l_781_, v___y_773_, v___y_774_);
v_a_784_ = lean_ctor_get(v___x_783_, 0);
lean_inc(v_a_784_);
lean_dec_ref(v___x_783_);
if (lean_obj_tag(v_a_784_) == 0)
{
lean_object* v_a_785_; 
v_a_785_ = lean_ctor_get(v_a_784_, 0);
lean_inc(v_a_785_);
lean_dec_ref_known(v_a_784_, 1);
v_d_777_ = v_a_785_;
goto v___jp_776_;
}
else
{
lean_object* v_a_786_; lean_object* v___x_787_; lean_object* v_a_788_; lean_object* v_a_789_; 
v_a_786_ = lean_ctor_get(v_a_784_, 0);
lean_inc_n(v_a_786_, 2);
lean_dec_ref_known(v_a_784_, 1);
v___x_787_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(v_k_780_, v___x_770_, v_a_786_, v_a_786_);
lean_dec(v_a_786_);
v_a_788_ = lean_ctor_get(v___x_787_, 0);
lean_inc(v_a_788_);
lean_dec_ref(v___x_787_);
v_a_789_ = lean_ctor_get(v_a_788_, 0);
lean_inc(v_a_789_);
lean_dec(v_a_788_);
v_init_771_ = v_a_789_;
v_x_772_ = v_r_782_;
goto _start;
}
}
else
{
lean_object* v___x_791_; lean_object* v___x_792_; 
v___x_791_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_791_, 0, v_init_771_);
v___x_792_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_792_, 0, v___x_791_);
return v___x_792_;
}
v___jp_776_:
{
lean_object* v___x_778_; lean_object* v___x_779_; 
v___x_778_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_778_, 0, v_d_777_);
v___x_779_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_779_, 0, v___x_778_);
return v___x_779_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5___boxed(lean_object* v___x_793_, lean_object* v_init_794_, lean_object* v_x_795_, lean_object* v___y_796_, lean_object* v___y_797_, lean_object* v___y_798_){
_start:
{
lean_object* v_res_799_; 
v_res_799_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5(v___x_793_, v_init_794_, v_x_795_, v___y_796_, v___y_797_);
lean_dec(v___y_797_);
lean_dec_ref(v___y_796_);
lean_dec(v_x_795_);
lean_dec(v___x_793_);
return v_res_799_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3(lean_object* v_xs_800_, lean_object* v_v_801_, lean_object* v_i_802_){
_start:
{
lean_object* v___x_803_; uint8_t v___x_804_; 
v___x_803_ = lean_array_get_size(v_xs_800_);
v___x_804_ = lean_nat_dec_lt(v_i_802_, v___x_803_);
if (v___x_804_ == 0)
{
lean_object* v___x_805_; 
lean_dec(v_i_802_);
v___x_805_ = lean_box(0);
return v___x_805_;
}
else
{
lean_object* v___x_806_; uint8_t v___x_807_; 
v___x_806_ = lean_array_fget_borrowed(v_xs_800_, v_i_802_);
v___x_807_ = lean_name_eq(v___x_806_, v_v_801_);
if (v___x_807_ == 0)
{
lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_808_ = lean_unsigned_to_nat(1u);
v___x_809_ = lean_nat_add(v_i_802_, v___x_808_);
lean_dec(v_i_802_);
v_i_802_ = v___x_809_;
goto _start;
}
else
{
lean_object* v___x_811_; 
v___x_811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_811_, 0, v_i_802_);
return v___x_811_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3___boxed(lean_object* v_xs_812_, lean_object* v_v_813_, lean_object* v_i_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3(v_xs_812_, v_v_813_, v_i_814_);
lean_dec(v_v_813_);
lean_dec_ref(v_xs_812_);
return v_res_815_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2(lean_object* v_xs_816_, lean_object* v_v_817_){
_start:
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = lean_unsigned_to_nat(0u);
v___x_819_ = lp_importGraph_Array_idxOfAux___at___00Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2_spec__3(v_xs_816_, v_v_817_, v___x_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2___boxed(lean_object* v_xs_820_, lean_object* v_v_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2(v_xs_820_, v_v_821_);
lean_dec(v_v_821_);
lean_dec_ref(v_xs_820_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1(lean_object* v_as_823_, lean_object* v_a_824_){
_start:
{
lean_object* v___x_825_; 
v___x_825_ = lp_importGraph_Array_finIdxOf_x3f___at___00Array_erase___at___00Lean_Name_findHome_spec__1_spec__2(v_as_823_, v_a_824_);
if (lean_obj_tag(v___x_825_) == 0)
{
return v_as_823_;
}
else
{
lean_object* v_val_826_; lean_object* v___x_827_; 
v_val_826_ = lean_ctor_get(v___x_825_, 0);
lean_inc(v_val_826_);
lean_dec_ref_known(v___x_825_, 1);
v___x_827_ = l_Array_eraseIdx___redArg(v_as_823_, v_val_826_);
return v___x_827_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1___boxed(lean_object* v_as_828_, lean_object* v_a_829_){
_start:
{
lean_object* v_res_830_; 
v_res_830_ = lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1(v_as_828_, v_a_829_);
lean_dec(v_a_829_);
return v_res_830_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Name_findHome(lean_object* v_n_831_, lean_object* v_env_832_, lean_object* v_a_833_, lean_object* v_a_834_){
_start:
{
lean_object* v___y_837_; lean_object* v_a_838_; lean_object* v___y_850_; lean_object* v___y_851_; lean_object* v___y_852_; lean_object* v___y_865_; 
if (lean_obj_tag(v_env_832_) == 1)
{
lean_object* v_val_870_; lean_object* v___x_871_; lean_object* v_mainModule_872_; 
v_val_870_ = lean_ctor_get(v_env_832_, 0);
v___x_871_ = l_Lean_Environment_header(v_val_870_);
v_mainModule_872_ = lean_ctor_get(v___x_871_, 0);
lean_inc(v_mainModule_872_);
lean_dec_ref(v___x_871_);
v___y_865_ = v_mainModule_872_;
goto v___jp_864_;
}
else
{
lean_object* v___x_873_; 
v___x_873_ = lean_box(0);
v___y_865_ = v___x_873_;
goto v___jp_864_;
}
v___jp_836_:
{
lean_object* v___x_839_; lean_object* v_a_840_; lean_object* v___x_842_; uint8_t v_isShared_843_; uint8_t v_isSharedCheck_848_; 
lean_inc(v_a_838_);
v___x_839_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__5(v___y_837_, v_a_838_, v_a_838_, v_a_833_, v_a_834_);
lean_dec(v_a_838_);
lean_dec(v___y_837_);
v_a_840_ = lean_ctor_get(v___x_839_, 0);
v_isSharedCheck_848_ = !lean_is_exclusive(v___x_839_);
if (v_isSharedCheck_848_ == 0)
{
v___x_842_ = v___x_839_;
v_isShared_843_ = v_isSharedCheck_848_;
goto v_resetjp_841_;
}
else
{
lean_inc(v_a_840_);
lean_dec(v___x_839_);
v___x_842_ = lean_box(0);
v_isShared_843_ = v_isSharedCheck_848_;
goto v_resetjp_841_;
}
v_resetjp_841_:
{
lean_object* v_a_844_; lean_object* v___x_846_; 
v_a_844_ = lean_ctor_get(v_a_840_, 0);
lean_inc(v_a_844_);
lean_dec(v_a_840_);
if (v_isShared_843_ == 0)
{
lean_ctor_set(v___x_842_, 0, v_a_844_);
v___x_846_ = v___x_842_;
goto v_reusejp_845_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_a_844_);
v___x_846_ = v_reuseFailAlloc_847_;
goto v_reusejp_845_;
}
v_reusejp_845_:
{
return v___x_846_;
}
}
}
v___jp_849_:
{
lean_object* v___x_853_; lean_object* v_env_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v_a_862_; lean_object* v_a_863_; 
v___x_853_ = lean_st_ref_get(v_a_834_);
v_env_854_ = lean_ctor_get(v___x_853_, 0);
lean_inc_ref(v_env_854_);
lean_dec(v___x_853_);
v___x_855_ = lean_mk_empty_array_with_capacity(v___y_852_);
lean_dec(v___y_852_);
v___x_856_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0_spec__0(v___x_855_, v___y_850_);
v___x_857_ = lp_importGraph_Array_erase___at___00Lean_Name_findHome_spec__1(v___x_856_, v___y_851_);
lean_dec(v___y_851_);
v___x_858_ = lp_importGraph_Lean_Environment_importGraph(v_env_854_);
lean_dec_ref(v_env_854_);
v___x_859_ = lp_importGraph_Lean_NameMap_transitiveClosure(v___x_858_);
v___x_860_ = l_Lean_NameSet_empty;
lean_inc(v___x_859_);
v___x_861_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(v___x_857_, v___x_860_, v___x_859_);
lean_dec_ref(v___x_857_);
v_a_862_ = lean_ctor_get(v___x_861_, 0);
lean_inc(v_a_862_);
lean_dec_ref(v___x_861_);
v_a_863_ = lean_ctor_get(v_a_862_, 0);
lean_inc(v_a_863_);
lean_dec(v_a_862_);
v___y_837_ = v___x_859_;
v_a_838_ = v_a_863_;
goto v___jp_836_;
}
v___jp_864_:
{
lean_object* v___x_866_; 
v___x_866_ = lp_importGraph_Lean_Name_requiredModules(v_n_831_, v_a_833_, v_a_834_);
if (lean_obj_tag(v___x_866_) == 0)
{
lean_object* v_a_867_; 
v_a_867_ = lean_ctor_get(v___x_866_, 0);
lean_inc(v_a_867_);
lean_dec_ref_known(v___x_866_, 1);
if (lean_obj_tag(v_a_867_) == 0)
{
lean_object* v_size_868_; 
v_size_868_ = lean_ctor_get(v_a_867_, 0);
lean_inc(v_size_868_);
v___y_850_ = v_a_867_;
v___y_851_ = v___y_865_;
v___y_852_ = v_size_868_;
goto v___jp_849_;
}
else
{
lean_object* v___x_869_; 
v___x_869_ = lean_unsigned_to_nat(0u);
v___y_850_ = v_a_867_;
v___y_851_ = v___y_865_;
v___y_852_ = v___x_869_;
goto v___jp_849_;
}
}
else
{
lean_dec(v___y_865_);
return v___x_866_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Name_findHome___boxed(lean_object* v_n_874_, lean_object* v_env_875_, lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_importGraph_Lean_Name_findHome(v_n_874_, v_env_875_, v_a_876_, v_a_877_);
lean_dec(v_a_877_);
lean_dec_ref(v_a_876_);
lean_dec(v_env_875_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0(lean_object* v_init_880_, lean_object* v_t_881_){
_start:
{
lean_object* v___x_882_; 
v___x_882_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_Name_findHome_spec__0_spec__0(v_init_880_, v_t_881_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3(lean_object* v_00_u03b2_883_, lean_object* v_k_884_, lean_object* v_t_885_, lean_object* v_h_886_){
_start:
{
lean_object* v___x_887_; 
v___x_887_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___redArg(v_k_884_, v_t_885_);
return v___x_887_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3___boxed(lean_object* v_00_u03b2_888_, lean_object* v_k_889_, lean_object* v_t_890_, lean_object* v_h_891_){
_start:
{
lean_object* v_res_892_; 
v_res_892_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Name_findHome_spec__3(v_00_u03b2_888_, v_k_889_, v_t_890_, v_h_891_);
lean_dec(v_k_889_);
return v_res_892_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4(lean_object* v_a_893_, lean_object* v___x_894_, lean_object* v_init_895_, lean_object* v_x_896_, lean_object* v___y_897_, lean_object* v___y_898_){
_start:
{
lean_object* v___x_900_; 
v___x_900_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___redArg(v_a_893_, v___x_894_, v_init_895_, v_x_896_);
return v___x_900_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4___boxed(lean_object* v_a_901_, lean_object* v___x_902_, lean_object* v_init_903_, lean_object* v_x_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_){
_start:
{
lean_object* v_res_908_; 
v_res_908_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__4(v_a_901_, v___x_902_, v_init_903_, v_x_904_, v___y_905_, v___y_906_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
lean_dec(v_x_904_);
lean_dec(v___x_902_);
lean_dec(v_a_901_);
return v_res_908_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6(lean_object* v___x_909_, lean_object* v_init_910_, lean_object* v_x_911_, lean_object* v___y_912_, lean_object* v___y_913_){
_start:
{
lean_object* v___x_915_; 
v___x_915_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___redArg(v___x_909_, v_init_910_, v_x_911_);
return v___x_915_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6___boxed(lean_object* v___x_916_, lean_object* v_init_917_, lean_object* v_x_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_){
_start:
{
lean_object* v_res_922_; 
v_res_922_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00Lean_Name_findHome_spec__6(v___x_916_, v_init_917_, v_x_918_, v___y_919_, v___y_920_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec_ref(v___x_916_);
return v_res_922_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___lam__0(lean_object* v_modName_925_, lean_object* v___y_926_){
_start:
{
lean_object* v___x_928_; 
lean_inc(v_modName_925_);
v___x_928_ = l_Lean_Server_documentUriFromModule_x3f(v_modName_925_);
if (lean_obj_tag(v___x_928_) == 0)
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_947_; 
v_a_929_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_947_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_947_ == 0)
{
v___x_931_ = v___x_928_;
v_isShared_932_ = v_isSharedCheck_947_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v___x_928_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_947_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
if (lean_obj_tag(v_a_929_) == 1)
{
lean_object* v_val_933_; lean_object* v___x_935_; 
lean_dec(v_modName_925_);
v_val_933_ = lean_ctor_get(v_a_929_, 0);
lean_inc(v_val_933_);
lean_dec_ref_known(v_a_929_, 1);
if (v_isShared_932_ == 0)
{
lean_ctor_set(v___x_931_, 0, v_val_933_);
v___x_935_ = v___x_931_;
goto v_reusejp_934_;
}
else
{
lean_object* v_reuseFailAlloc_936_; 
v_reuseFailAlloc_936_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_936_, 0, v_val_933_);
v___x_935_ = v_reuseFailAlloc_936_;
goto v_reusejp_934_;
}
v_reusejp_934_:
{
return v___x_935_;
}
}
else
{
lean_object* v___x_937_; uint8_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_945_; 
lean_dec(v_a_929_);
v___x_937_ = ((lean_object*)(lp_importGraph_getModuleUri___lam__0___closed__0));
v___x_938_ = 1;
v___x_939_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_modName_925_, v___x_938_);
v___x_940_ = lean_string_append(v___x_937_, v___x_939_);
lean_dec_ref(v___x_939_);
v___x_941_ = ((lean_object*)(lp_importGraph_getModuleUri___lam__0___closed__1));
v___x_942_ = lean_string_append(v___x_940_, v___x_941_);
v___x_943_ = l_Lean_Server_RequestError_invalidParams(v___x_942_);
if (v_isShared_932_ == 0)
{
lean_ctor_set_tag(v___x_931_, 1);
lean_ctor_set(v___x_931_, 0, v___x_943_);
v___x_945_ = v___x_931_;
goto v_reusejp_944_;
}
else
{
lean_object* v_reuseFailAlloc_946_; 
v_reuseFailAlloc_946_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_946_, 0, v___x_943_);
v___x_945_ = v_reuseFailAlloc_946_;
goto v_reusejp_944_;
}
v_reusejp_944_:
{
return v___x_945_;
}
}
}
}
else
{
lean_object* v_a_948_; lean_object* v___x_950_; uint8_t v_isShared_951_; uint8_t v_isSharedCheck_956_; 
lean_dec(v_modName_925_);
v_a_948_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_956_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_956_ == 0)
{
v___x_950_ = v___x_928_;
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
else
{
lean_inc(v_a_948_);
lean_dec(v___x_928_);
v___x_950_ = lean_box(0);
v_isShared_951_ = v_isSharedCheck_956_;
goto v_resetjp_949_;
}
v_resetjp_949_:
{
lean_object* v___x_952_; lean_object* v___x_954_; 
v___x_952_ = l_Lean_Server_RequestError_ofIoError(v_a_948_);
if (v_isShared_951_ == 0)
{
lean_ctor_set(v___x_950_, 0, v___x_952_);
v___x_954_ = v___x_950_;
goto v_reusejp_953_;
}
else
{
lean_object* v_reuseFailAlloc_955_; 
v_reuseFailAlloc_955_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_955_, 0, v___x_952_);
v___x_954_ = v_reuseFailAlloc_955_;
goto v_reusejp_953_;
}
v_reusejp_953_:
{
return v___x_954_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___lam__0___boxed(lean_object* v_modName_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_importGraph_getModuleUri___lam__0(v_modName_957_, v___y_958_);
lean_dec_ref(v___y_958_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri(lean_object* v_modName_961_, lean_object* v_a_962_){
_start:
{
lean_object* v___f_964_; lean_object* v___x_965_; 
v___f_964_ = lean_alloc_closure((void*)(lp_importGraph_getModuleUri___lam__0___boxed), 3, 1);
lean_closure_set(v___f_964_, 0, v_modName_961_);
v___x_965_ = l_Lean_Server_RequestM_asTask___redArg(v___f_964_, v_a_962_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_getModuleUri___boxed(lean_object* v_modName_966_, lean_object* v_a_967_, lean_object* v_a_968_){
_start:
{
lean_object* v_res_969_; 
v_res_969_ = lp_importGraph_getModuleUri(v_modName_966_, v_a_967_);
lean_dec_ref(v_a_967_);
return v_res_969_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0(lean_object* v___y_970_){
_start:
{
lean_inc(v___y_970_);
return v___y_970_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_971_){
_start:
{
lean_object* v_res_972_; 
v_res_972_ = lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__0(v___y_971_);
lean_dec(v___y_971_);
return v_res_972_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_973_, uint64_t v_k_974_){
_start:
{
if (lean_obj_tag(v_t_973_) == 0)
{
lean_object* v_k_975_; lean_object* v_v_976_; lean_object* v_l_977_; lean_object* v_r_978_; uint64_t v___x_979_; uint8_t v___x_980_; 
v_k_975_ = lean_ctor_get(v_t_973_, 1);
v_v_976_ = lean_ctor_get(v_t_973_, 2);
v_l_977_ = lean_ctor_get(v_t_973_, 3);
v_r_978_ = lean_ctor_get(v_t_973_, 4);
v___x_979_ = lean_unbox_uint64(v_k_975_);
v___x_980_ = lean_uint64_dec_lt(v_k_974_, v___x_979_);
if (v___x_980_ == 0)
{
uint64_t v___x_981_; uint8_t v___x_982_; 
v___x_981_ = lean_unbox_uint64(v_k_975_);
v___x_982_ = lean_uint64_dec_eq(v_k_974_, v___x_981_);
if (v___x_982_ == 0)
{
v_t_973_ = v_r_978_;
goto _start;
}
else
{
lean_object* v___x_984_; 
lean_inc(v_v_976_);
v___x_984_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_984_, 0, v_v_976_);
return v___x_984_;
}
}
else
{
v_t_973_ = v_l_977_;
goto _start;
}
}
else
{
lean_object* v___x_986_; 
v___x_986_ = lean_box(0);
return v___x_986_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_987_, lean_object* v_k_988_){
_start:
{
uint64_t v_k_boxed_989_; lean_object* v_res_990_; 
v_k_boxed_989_ = lean_unbox_uint64(v_k_988_);
lean_dec_ref(v_k_988_);
v_res_990_ = lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg(v_t_987_, v_k_boxed_989_);
lean_dec(v_t_987_);
return v_res_990_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_991_, lean_object* v_x_992_){
_start:
{
lean_object* v___x_993_; 
v___x_993_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_993_, 0, v_x_992_);
lean_ctor_set(v___x_993_, 1, v_expireTime_991_);
return v___x_993_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2(lean_object* v_val_994_, lean_object* v___f_995_, lean_object* v_x_996_, lean_object* v___y_997_){
_start:
{
if (lean_obj_tag(v_x_996_) == 0)
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
lean_dec_ref(v___f_995_);
v_a_999_ = lean_ctor_get(v_x_996_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v_x_996_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_1001_ = v_x_996_;
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v_x_996_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
lean_object* v___x_1004_; 
if (v_isShared_1002_ == 0)
{
lean_ctor_set_tag(v___x_1001_, 1);
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
else
{
lean_object* v_a_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1030_; 
v_a_1007_ = lean_ctor_get(v_x_996_, 0);
v_isSharedCheck_1030_ = !lean_is_exclusive(v_x_996_);
if (v_isSharedCheck_1030_ == 0)
{
v___x_1009_ = v_x_996_;
v_isShared_1010_ = v_isSharedCheck_1030_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_a_1007_);
lean_dec(v_x_996_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1030_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1011_; lean_object* v_objects_1012_; lean_object* v_expireTime_1013_; lean_object* v___x_1015_; uint8_t v_isShared_1016_; uint8_t v_isSharedCheck_1029_; 
v___x_1011_ = lean_st_ref_take(v_val_994_);
v_objects_1012_ = lean_ctor_get(v___x_1011_, 0);
v_expireTime_1013_ = lean_ctor_get(v___x_1011_, 1);
v_isSharedCheck_1029_ = !lean_is_exclusive(v___x_1011_);
if (v_isSharedCheck_1029_ == 0)
{
v___x_1015_ = v___x_1011_;
v_isShared_1016_ = v_isSharedCheck_1029_;
goto v_resetjp_1014_;
}
else
{
lean_inc(v_expireTime_1013_);
lean_inc(v_objects_1012_);
lean_dec(v___x_1011_);
v___x_1015_ = lean_box(0);
v_isShared_1016_ = v_isSharedCheck_1029_;
goto v_resetjp_1014_;
}
v_resetjp_1014_:
{
lean_object* v___f_1017_; lean_object* v___x_1019_; 
v___f_1017_ = lean_alloc_closure((void*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_1017_, 0, v_expireTime_1013_);
if (v_isShared_1010_ == 0)
{
lean_ctor_set_tag(v___x_1009_, 3);
v___x_1019_ = v___x_1009_;
goto v_reusejp_1018_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v_a_1007_);
v___x_1019_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1018_;
}
v_reusejp_1018_:
{
lean_object* v___x_1021_; 
if (v_isShared_1016_ == 0)
{
lean_ctor_set(v___x_1015_, 1, v_objects_1012_);
lean_ctor_set(v___x_1015_, 0, v___x_1019_);
v___x_1021_ = v___x_1015_;
goto v_reusejp_1020_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1027_, 0, v___x_1019_);
lean_ctor_set(v_reuseFailAlloc_1027_, 1, v_objects_1012_);
v___x_1021_ = v_reuseFailAlloc_1027_;
goto v_reusejp_1020_;
}
v_reusejp_1020_:
{
lean_object* v___x_1022_; lean_object* v_fst_1023_; lean_object* v_snd_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; 
v___x_1022_ = l_Prod_map___redArg(v___f_995_, v___f_1017_, v___x_1021_);
v_fst_1023_ = lean_ctor_get(v___x_1022_, 0);
lean_inc(v_fst_1023_);
v_snd_1024_ = lean_ctor_get(v___x_1022_, 1);
lean_inc(v_snd_1024_);
lean_dec_ref(v___x_1022_);
v___x_1025_ = lean_st_ref_set(v_val_994_, v_snd_1024_);
v___x_1026_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1026_, 0, v_fst_1023_);
return v___x_1026_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_1031_, lean_object* v___f_1032_, lean_object* v_x_1033_, lean_object* v___y_1034_, lean_object* v___y_1035_){
_start:
{
lean_object* v_res_1036_; 
v_res_1036_ = lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2(v_val_1031_, v___f_1032_, v_x_1033_, v___y_1034_);
lean_dec_ref(v___y_1034_);
lean_dec(v_val_1031_);
return v_res_1036_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3(lean_object* v_method_1044_, lean_object* v_handler_1045_, lean_object* v___f_1046_, uint64_t v_seshId_1047_, lean_object* v_j_1048_, lean_object* v___y_1049_){
_start:
{
lean_object* v_rpcSessions_1051_; lean_object* v___x_1052_; 
v_rpcSessions_1051_ = lean_ctor_get(v___y_1049_, 0);
v___x_1052_ = lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_1051_, v_seshId_1047_);
if (lean_obj_tag(v___x_1052_) == 1)
{
lean_object* v_val_1053_; lean_object* v___x_1054_; lean_object* v___x_1055_; 
v_val_1053_ = lean_ctor_get(v___x_1052_, 0);
lean_inc(v_val_1053_);
lean_dec_ref_known(v___x_1052_, 1);
v___x_1054_ = lean_st_ref_get(v_val_1053_);
lean_dec(v___x_1054_);
lean_inc(v_j_1048_);
v___x_1055_ = l_Lean_Name_fromJson_x3f(v_j_1048_);
if (lean_obj_tag(v___x_1055_) == 0)
{
lean_object* v_a_1056_; lean_object* v___x_1058_; uint8_t v_isShared_1059_; uint8_t v_isSharedCheck_1076_; 
lean_dec(v_val_1053_);
lean_dec_ref(v___f_1046_);
lean_dec_ref(v_handler_1045_);
v_a_1056_ = lean_ctor_get(v___x_1055_, 0);
v_isSharedCheck_1076_ = !lean_is_exclusive(v___x_1055_);
if (v_isSharedCheck_1076_ == 0)
{
v___x_1058_ = v___x_1055_;
v_isShared_1059_ = v_isSharedCheck_1076_;
goto v_resetjp_1057_;
}
else
{
lean_inc(v_a_1056_);
lean_dec(v___x_1055_);
v___x_1058_ = lean_box(0);
v_isShared_1059_ = v_isSharedCheck_1076_;
goto v_resetjp_1057_;
}
v_resetjp_1057_:
{
uint8_t v___x_1060_; lean_object* v___x_1061_; uint8_t v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1074_; 
v___x_1060_ = 3;
v___x_1061_ = ((lean_object*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_1062_ = 1;
v___x_1063_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_1044_, v___x_1062_);
v___x_1064_ = lean_string_append(v___x_1061_, v___x_1063_);
lean_dec_ref(v___x_1063_);
v___x_1065_ = ((lean_object*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_1066_ = lean_string_append(v___x_1064_, v___x_1065_);
v___x_1067_ = l_Lean_Json_compress(v_j_1048_);
v___x_1068_ = lean_string_append(v___x_1066_, v___x_1067_);
lean_dec_ref(v___x_1067_);
v___x_1069_ = ((lean_object*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_1070_ = lean_string_append(v___x_1068_, v___x_1069_);
v___x_1071_ = lean_string_append(v___x_1070_, v_a_1056_);
lean_dec(v_a_1056_);
v___x_1072_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_1072_, 0, v___x_1071_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*1, v___x_1060_);
if (v_isShared_1059_ == 0)
{
lean_ctor_set_tag(v___x_1058_, 1);
lean_ctor_set(v___x_1058_, 0, v___x_1072_);
v___x_1074_ = v___x_1058_;
goto v_reusejp_1073_;
}
else
{
lean_object* v_reuseFailAlloc_1075_; 
v_reuseFailAlloc_1075_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1075_, 0, v___x_1072_);
v___x_1074_ = v_reuseFailAlloc_1075_;
goto v_reusejp_1073_;
}
v_reusejp_1073_:
{
return v___x_1074_;
}
}
}
else
{
lean_object* v_a_1077_; lean_object* v___x_1078_; 
lean_dec(v_j_1048_);
lean_dec(v_method_1044_);
v_a_1077_ = lean_ctor_get(v___x_1055_, 0);
lean_inc(v_a_1077_);
lean_dec_ref_known(v___x_1055_, 1);
lean_inc_ref(v___y_1049_);
v___x_1078_ = lean_apply_3(v_handler_1045_, v_a_1077_, v___y_1049_, lean_box(0));
if (lean_obj_tag(v___x_1078_) == 0)
{
lean_object* v_a_1079_; lean_object* v___f_1080_; lean_object* v___x_1081_; 
v_a_1079_ = lean_ctor_get(v___x_1078_, 0);
lean_inc(v_a_1079_);
lean_dec_ref_known(v___x_1078_, 1);
v___f_1080_ = lean_alloc_closure((void*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_1080_, 0, v_val_1053_);
lean_closure_set(v___f_1080_, 1, v___f_1046_);
v___x_1081_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_1079_, v___f_1080_, v___y_1049_);
return v___x_1081_;
}
else
{
lean_object* v_a_1082_; lean_object* v___x_1084_; uint8_t v_isShared_1085_; uint8_t v_isSharedCheck_1089_; 
lean_dec(v_val_1053_);
lean_dec_ref(v___f_1046_);
v_a_1082_ = lean_ctor_get(v___x_1078_, 0);
v_isSharedCheck_1089_ = !lean_is_exclusive(v___x_1078_);
if (v_isSharedCheck_1089_ == 0)
{
v___x_1084_ = v___x_1078_;
v_isShared_1085_ = v_isSharedCheck_1089_;
goto v_resetjp_1083_;
}
else
{
lean_inc(v_a_1082_);
lean_dec(v___x_1078_);
v___x_1084_ = lean_box(0);
v_isShared_1085_ = v_isSharedCheck_1089_;
goto v_resetjp_1083_;
}
v_resetjp_1083_:
{
lean_object* v___x_1087_; 
if (v_isShared_1085_ == 0)
{
v___x_1087_ = v___x_1084_;
goto v_reusejp_1086_;
}
else
{
lean_object* v_reuseFailAlloc_1088_; 
v_reuseFailAlloc_1088_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1088_, 0, v_a_1082_);
v___x_1087_ = v_reuseFailAlloc_1088_;
goto v_reusejp_1086_;
}
v_reusejp_1086_:
{
return v___x_1087_;
}
}
}
}
}
else
{
lean_object* v___x_1090_; lean_object* v___x_1091_; 
lean_dec(v___x_1052_);
lean_dec(v_j_1048_);
lean_dec_ref(v___f_1046_);
lean_dec_ref(v_handler_1045_);
lean_dec(v_method_1044_);
v___x_1090_ = ((lean_object*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_1091_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1091_, 0, v___x_1090_);
return v___x_1091_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_1092_, lean_object* v_handler_1093_, lean_object* v___f_1094_, lean_object* v_seshId_1095_, lean_object* v_j_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_){
_start:
{
uint64_t v_seshId_boxed_1099_; lean_object* v_res_1100_; 
v_seshId_boxed_1099_ = lean_unbox_uint64(v_seshId_1095_);
lean_dec_ref(v_seshId_1095_);
v_res_1100_ = lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3(v_method_1092_, v_handler_1093_, v___f_1094_, v_seshId_boxed_1099_, v_j_1096_, v___y_1097_);
lean_dec_ref(v___y_1097_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0(lean_object* v_method_1102_, lean_object* v_handler_1103_){
_start:
{
lean_object* v___f_1104_; lean_object* v___f_1105_; 
v___f_1104_ = ((lean_object*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___closed__0));
v___f_1105_ = lean_alloc_closure((void*)(lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_1105_, 0, v_method_1102_);
lean_closure_set(v___f_1105_, 1, v_handler_1103_);
lean_closure_set(v___f_1105_, 2, v___f_1104_);
return v___f_1105_;
}
}
static lean_object* _init_lp_importGraph_getModuleUri___rpc__wrapped___closed__3(void){
_start:
{
lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; 
v___x_1110_ = ((lean_object*)(lp_importGraph_getModuleUri___rpc__wrapped___closed__2));
v___x_1111_ = ((lean_object*)(lp_importGraph_getModuleUri___rpc__wrapped___closed__1));
v___x_1112_ = lp_importGraph_Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0(v___x_1111_, v___x_1110_);
return v___x_1112_;
}
}
static lean_object* _init_lp_importGraph_getModuleUri___rpc__wrapped(void){
_start:
{
lean_object* v___x_1113_; 
v___x_1113_ = lean_obj_once(&lp_importGraph_getModuleUri___rpc__wrapped___closed__3, &lp_importGraph_getModuleUri___rpc__wrapped___closed__3_once, _init_lp_importGraph_getModuleUri___rpc__wrapped___closed__3);
return v___x_1113_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___redArg(lean_object* v_x_1114_){
_start:
{
lean_inc_ref(v_x_1114_);
return v_x_1114_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___redArg___boxed(lean_object* v_x_1115_){
_start:
{
lean_object* v_res_1116_; 
v_res_1116_ = lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___redArg(v_x_1115_);
lean_dec_ref(v_x_1115_);
return v_res_1116_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1(lean_object* v_00_u03b1_1117_, lean_object* v_x_1118_, lean_object* v___y_1119_){
_start:
{
lean_inc_ref(v_x_1118_);
return v_x_1118_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1___boxed(lean_object* v_00_u03b1_1120_, lean_object* v_x_1121_, lean_object* v___y_1122_){
_start:
{
lean_object* v_res_1123_; 
v_res_1123_ = lp_importGraph_MonadExcept_ofExcept___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__1(v_00_u03b1_1120_, v_x_1121_, v___y_1122_);
lean_dec_ref(v___y_1122_);
lean_dec_ref(v_x_1121_);
return v_res_1123_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_1124_, lean_object* v_t_1125_, uint64_t v_k_1126_){
_start:
{
lean_object* v___x_1127_; 
v___x_1127_ = lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___redArg(v_t_1125_, v_k_1126_);
return v___x_1127_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_1128_, lean_object* v_t_1129_, lean_object* v_k_1130_){
_start:
{
uint64_t v_k_boxed_1131_; lean_object* v_res_1132_; 
v_k_boxed_1131_ = lean_unbox_uint64(v_k_1130_);
lean_dec_ref(v_k_1130_);
v_res_1132_ = lp_importGraph_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00getModuleUri___rpc__wrapped_spec__0_spec__0(v_00_u03b4_1128_, v_t_1129_, v_k_boxed_1131_);
lean_dec(v_t_1129_);
return v_res_1132_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0(lean_object* v_j_1133_, lean_object* v_k_1134_){
_start:
{
lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1135_ = l_Lean_Json_getObjValD(v_j_1133_, v_k_1134_);
v___x_1136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1136_, 0, v___x_1135_);
return v___x_1136_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0___boxed(lean_object* v_j_1137_, lean_object* v_k_1138_){
_start:
{
lean_object* v_res_1139_; 
v_res_1139_ = lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0(v_j_1137_, v_k_1138_);
lean_dec_ref(v_k_1138_);
return v_res_1139_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_(lean_object* v_json_1141_){
_start:
{
lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v_a_1144_; lean_object* v___x_1146_; uint8_t v_isShared_1147_; uint8_t v_isSharedCheck_1151_; 
v___x_1142_ = ((lean_object*)(lp_importGraph_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_));
v___x_1143_ = lp_importGraph_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10__spec__0(v_json_1141_, v___x_1142_);
v_a_1144_ = lean_ctor_get(v___x_1143_, 0);
v_isSharedCheck_1151_ = !lean_is_exclusive(v___x_1143_);
if (v_isSharedCheck_1151_ == 0)
{
v___x_1146_ = v___x_1143_;
v_isShared_1147_ = v_isSharedCheck_1151_;
goto v_resetjp_1145_;
}
else
{
lean_inc(v_a_1144_);
lean_dec(v___x_1143_);
v___x_1146_ = lean_box(0);
v_isShared_1147_ = v_isSharedCheck_1151_;
goto v_resetjp_1145_;
}
v_resetjp_1145_:
{
lean_object* v___x_1149_; 
if (v_isShared_1147_ == 0)
{
v___x_1149_ = v___x_1146_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1150_; 
v_reuseFailAlloc_1150_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1150_, 0, v_a_1144_);
v___x_1149_ = v_reuseFailAlloc_1150_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
return v___x_1149_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__spec__0(lean_object* v_a_1154_, lean_object* v_a_1155_){
_start:
{
if (lean_obj_tag(v_a_1154_) == 0)
{
lean_object* v___x_1156_; 
v___x_1156_ = lean_array_to_list(v_a_1155_);
return v___x_1156_;
}
else
{
lean_object* v_head_1157_; lean_object* v_tail_1158_; lean_object* v___x_1159_; 
v_head_1157_ = lean_ctor_get(v_a_1154_, 0);
lean_inc(v_head_1157_);
v_tail_1158_ = lean_ctor_get(v_a_1154_, 1);
lean_inc(v_tail_1158_);
lean_dec_ref_known(v_a_1154_, 2);
v___x_1159_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_1155_, v_head_1157_);
v_a_1154_ = v_tail_1158_;
v_a_1155_ = v___x_1159_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_(lean_object* v_x_1163_){
_start:
{
lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1164_ = ((lean_object*)(lp_importGraph_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_));
v___x_1165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1164_);
lean_ctor_set(v___x_1165_, 1, v_x_1163_);
v___x_1166_ = lean_box(0);
v___x_1167_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1165_);
lean_ctor_set(v___x_1167_, 1, v___x_1166_);
v___x_1168_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1168_, 0, v___x_1167_);
lean_ctor_set(v___x_1168_, 1, v___x_1166_);
v___x_1169_ = ((lean_object*)(lp_importGraph_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_));
v___x_1170_ = lp_importGraph___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29__spec__0(v___x_1168_, v___x_1169_);
v___x_1171_ = l_Lean_Json_mkObj(v___x_1170_);
lean_dec(v___x_1170_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object* v_a_1174_, lean_object* v_a_1175_){
_start:
{
uint8_t v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; 
v___x_1176_ = 1;
v___x_1177_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_a_1174_, v___x_1176_);
v___x_1178_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1177_);
v___x_1179_ = lp_importGraph_instToJsonRpcEncodablePacket_toJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_29_(v___x_1178_);
v___x_1180_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1180_, 0, v___x_1179_);
lean_ctor_set(v___x_1180_, 1, v_a_1175_);
return v___x_1180_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec___redArg_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object* v_j_1181_){
_start:
{
lean_object* v___x_1182_; 
v___x_1182_ = lp_importGraph_instFromJsonRpcEncodablePacket_fromJson_00___x40_ImportGraph_Tools_FindHome_927448520____hygCtx___hyg_10_(v_j_1181_);
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
v___x_1188_ = v___x_1185_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1183_);
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
lean_object* v_a_1191_; lean_object* v___x_1192_; 
v_a_1191_ = lean_ctor_get(v___x_1182_, 0);
lean_inc(v_a_1191_);
lean_dec_ref_known(v___x_1182_, 1);
v___x_1192_ = l_Lean_Name_fromJson_x3f(v_a_1191_);
if (lean_obj_tag(v___x_1192_) == 0)
{
lean_object* v_a_1193_; lean_object* v___x_1195_; uint8_t v_isShared_1196_; uint8_t v_isSharedCheck_1200_; 
v_a_1193_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1195_ = v___x_1192_;
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
else
{
lean_inc(v_a_1193_);
lean_dec(v___x_1192_);
v___x_1195_ = lean_box(0);
v_isShared_1196_ = v_isSharedCheck_1200_;
goto v_resetjp_1194_;
}
v_resetjp_1194_:
{
lean_object* v___x_1198_; 
if (v_isShared_1196_ == 0)
{
v___x_1198_ = v___x_1195_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_a_1193_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
}
else
{
lean_object* v_a_1201_; lean_object* v___x_1203_; uint8_t v_isShared_1204_; uint8_t v_isSharedCheck_1208_; 
v_a_1201_ = lean_ctor_get(v___x_1192_, 0);
v_isSharedCheck_1208_ = !lean_is_exclusive(v___x_1192_);
if (v_isSharedCheck_1208_ == 0)
{
v___x_1203_ = v___x_1192_;
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
else
{
lean_inc(v_a_1201_);
lean_dec(v___x_1192_);
v___x_1203_ = lean_box(0);
v_isShared_1204_ = v_isSharedCheck_1208_;
goto v_resetjp_1202_;
}
v_resetjp_1202_:
{
lean_object* v___x_1206_; 
if (v_isShared_1204_ == 0)
{
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
return v___x_1206_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(lean_object* v_j_1209_, lean_object* v_a_1210_){
_start:
{
lean_object* v___x_1211_; 
v___x_1211_ = lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec___redArg_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(v_j_1209_);
return v___x_1211_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1____boxed(lean_object* v_j_1212_, lean_object* v_a_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_importGraph_instRpcEncodableGoToModuleLinkProps_dec_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_(v_j_1212_, v_a_1213_);
lean_dec_ref(v_a_1213_);
return v_res_1214_;
}
}
static uint64_t _init_lp_importGraph_GoToModuleLink___closed__1(void){
_start:
{
lean_object* v___x_1222_; uint64_t v___x_1223_; 
v___x_1222_ = ((lean_object*)(lp_importGraph_GoToModuleLink___closed__0));
v___x_1223_ = lean_string_hash(v___x_1222_);
return v___x_1223_;
}
}
static lean_object* _init_lp_importGraph_GoToModuleLink___closed__2(void){
_start:
{
uint64_t v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; 
v___x_1224_ = lean_uint64_once(&lp_importGraph_GoToModuleLink___closed__1, &lp_importGraph_GoToModuleLink___closed__1_once, _init_lp_importGraph_GoToModuleLink___closed__1);
v___x_1225_ = ((lean_object*)(lp_importGraph_GoToModuleLink___closed__0));
v___x_1226_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_1226_, 0, v___x_1225_);
lean_ctor_set_uint64(v___x_1226_, sizeof(void*)*1, v___x_1224_);
return v___x_1226_;
}
}
static lean_object* _init_lp_importGraph_GoToModuleLink(void){
_start:
{
lean_object* v___x_1227_; 
v___x_1227_ = lean_obj_once(&lp_importGraph_GoToModuleLink___closed__2, &lp_importGraph_GoToModuleLink___closed__2_once, _init_lp_importGraph_GoToModuleLink___closed__2);
return v___x_1227_;
}
}
static lean_object* _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; 
v___x_1264_ = lean_box(0);
v___x_1265_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1266_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1265_);
lean_ctor_set(v___x_1266_, 1, v___x_1264_);
return v___x_1266_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg(){
_start:
{
lean_object* v___x_1268_; lean_object* v___x_1269_; 
v___x_1268_ = lean_obj_once(&lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0, &lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0_once, _init_lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___closed__0);
v___x_1269_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1269_, 0, v___x_1268_);
return v___x_1269_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg___boxed(lean_object* v___y_1270_){
_start:
{
lean_object* v_res_1271_; 
v_res_1271_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg();
return v_res_1271_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0(lean_object* v_00_u03b1_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_){
_start:
{
lean_object* v___x_1276_; 
v___x_1276_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg();
return v___x_1276_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___boxed(lean_object* v_00_u03b1_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_, lean_object* v___y_1280_){
_start:
{
lean_object* v_res_1281_; 
v_res_1281_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0(v_00_u03b1_1277_, v___y_1278_, v___y_1279_);
lean_dec(v___y_1279_);
lean_dec_ref(v___y_1278_);
return v_res_1281_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__1(lean_object* v_a_1282_, lean_object* v_a_1283_){
_start:
{
if (lean_obj_tag(v_a_1282_) == 0)
{
lean_object* v___x_1284_; 
v___x_1284_ = l_List_reverse___redArg(v_a_1283_);
return v___x_1284_;
}
else
{
lean_object* v_head_1285_; lean_object* v_tail_1286_; lean_object* v___x_1288_; uint8_t v_isShared_1289_; uint8_t v_isSharedCheck_1294_; 
v_head_1285_ = lean_ctor_get(v_a_1282_, 0);
v_tail_1286_ = lean_ctor_get(v_a_1282_, 1);
v_isSharedCheck_1294_ = !lean_is_exclusive(v_a_1282_);
if (v_isSharedCheck_1294_ == 0)
{
v___x_1288_ = v_a_1282_;
v_isShared_1289_ = v_isSharedCheck_1294_;
goto v_resetjp_1287_;
}
else
{
lean_inc(v_tail_1286_);
lean_inc(v_head_1285_);
lean_dec(v_a_1282_);
v___x_1288_ = lean_box(0);
v_isShared_1289_ = v_isSharedCheck_1294_;
goto v_resetjp_1287_;
}
v_resetjp_1287_:
{
lean_object* v___x_1291_; 
if (v_isShared_1289_ == 0)
{
lean_ctor_set(v___x_1288_, 1, v_a_1283_);
v___x_1291_ = v___x_1288_;
goto v_reusejp_1290_;
}
else
{
lean_object* v_reuseFailAlloc_1293_; 
v_reuseFailAlloc_1293_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1293_, 0, v_head_1285_);
lean_ctor_set(v_reuseFailAlloc_1293_, 1, v_a_1283_);
v___x_1291_ = v_reuseFailAlloc_1293_;
goto v_reusejp_1290_;
}
v_reusejp_1290_:
{
v_a_1282_ = v_tail_1286_;
v_a_1283_ = v___x_1291_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3(uint8_t v___x_1295_, lean_object* v_init_1296_, lean_object* v_x_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_){
_start:
{
if (lean_obj_tag(v_x_1297_) == 0)
{
lean_object* v_k_1301_; lean_object* v_l_1302_; lean_object* v_r_1303_; lean_object* v___x_1304_; 
v_k_1301_ = lean_ctor_get(v_x_1297_, 1);
lean_inc(v_k_1301_);
v_l_1302_ = lean_ctor_get(v_x_1297_, 3);
lean_inc(v_l_1302_);
v_r_1303_ = lean_ctor_get(v_x_1297_, 4);
lean_inc(v_r_1303_);
lean_dec_ref_known(v_x_1297_, 5);
v___x_1304_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3(v___x_1295_, v_init_1296_, v_l_1302_, v___y_1298_, v___y_1299_);
if (lean_obj_tag(v___x_1304_) == 0)
{
lean_object* v_a_1305_; lean_object* v_a_1306_; lean_object* v___x_1308_; uint8_t v_isShared_1309_; uint8_t v_isSharedCheck_1333_; 
v_a_1305_ = lean_ctor_get(v___x_1304_, 0);
lean_inc(v_a_1305_);
lean_dec_ref_known(v___x_1304_, 1);
v_a_1306_ = lean_ctor_get(v_a_1305_, 0);
v_isSharedCheck_1333_ = !lean_is_exclusive(v_a_1305_);
if (v_isSharedCheck_1333_ == 0)
{
v___x_1308_ = v_a_1305_;
v_isShared_1309_ = v_isSharedCheck_1333_;
goto v_resetjp_1307_;
}
else
{
lean_inc(v_a_1306_);
lean_dec(v_a_1305_);
v___x_1308_ = lean_box(0);
v_isShared_1309_ = v_isSharedCheck_1333_;
goto v_resetjp_1307_;
}
v_resetjp_1307_:
{
lean_object* v___x_1310_; uint64_t v_javascriptHash_1311_; lean_object* v___x_1312_; lean_object* v___x_1313_; lean_object* v___x_1314_; lean_object* v___x_1315_; 
v___x_1310_ = lp_importGraph_GoToModuleLink;
v_javascriptHash_1311_ = lean_ctor_get_uint64(v___x_1310_, sizeof(void*)*1);
lean_inc(v_k_1301_);
v___x_1312_ = lean_alloc_closure((void*)(lp_importGraph_instRpcEncodableGoToModuleLinkProps_enc_00___x40_ImportGraph_Tools_FindHome_1578893111____hygCtx___hyg_1_), 2, 1);
lean_closure_set(v___x_1312_, 0, v_k_1301_);
v___x_1313_ = lean_box_uint64(v_javascriptHash_1311_);
v___x_1314_ = lean_alloc_closure((void*)(l_Lean_Widget_WidgetInstance_ofHash___boxed), 5, 2);
lean_closure_set(v___x_1314_, 0, v___x_1313_);
lean_closure_set(v___x_1314_, 1, v___x_1312_);
v___x_1315_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1314_, v___y_1298_, v___y_1299_);
if (lean_obj_tag(v___x_1315_) == 0)
{
lean_object* v_a_1316_; lean_object* v___x_1317_; lean_object* v___x_1319_; 
v_a_1316_ = lean_ctor_get(v___x_1315_, 0);
lean_inc(v_a_1316_);
lean_dec_ref_known(v___x_1315_, 1);
v___x_1317_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_k_1301_, v___x_1295_);
if (v_isShared_1309_ == 0)
{
lean_ctor_set_tag(v___x_1308_, 3);
lean_ctor_set(v___x_1308_, 0, v___x_1317_);
v___x_1319_ = v___x_1308_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1324_; 
v_reuseFailAlloc_1324_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1324_, 0, v___x_1317_);
v___x_1319_ = v_reuseFailAlloc_1324_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
lean_object* v___x_1320_; lean_object* v___x_1321_; lean_object* v___x_1322_; 
v___x_1320_ = l_Lean_MessageData_ofFormat(v___x_1319_);
v___x_1321_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1321_, 0, v_a_1316_);
lean_ctor_set(v___x_1321_, 1, v___x_1320_);
v___x_1322_ = lean_array_push(v_a_1306_, v___x_1321_);
v_init_1296_ = v___x_1322_;
v_x_1297_ = v_r_1303_;
goto _start;
}
}
else
{
lean_object* v_a_1325_; lean_object* v___x_1327_; uint8_t v_isShared_1328_; uint8_t v_isSharedCheck_1332_; 
lean_del_object(v___x_1308_);
lean_dec(v_a_1306_);
lean_dec(v_r_1303_);
lean_dec(v_k_1301_);
v_a_1325_ = lean_ctor_get(v___x_1315_, 0);
v_isSharedCheck_1332_ = !lean_is_exclusive(v___x_1315_);
if (v_isSharedCheck_1332_ == 0)
{
v___x_1327_ = v___x_1315_;
v_isShared_1328_ = v_isSharedCheck_1332_;
goto v_resetjp_1326_;
}
else
{
lean_inc(v_a_1325_);
lean_dec(v___x_1315_);
v___x_1327_ = lean_box(0);
v_isShared_1328_ = v_isSharedCheck_1332_;
goto v_resetjp_1326_;
}
v_resetjp_1326_:
{
lean_object* v___x_1330_; 
if (v_isShared_1328_ == 0)
{
v___x_1330_ = v___x_1327_;
goto v_reusejp_1329_;
}
else
{
lean_object* v_reuseFailAlloc_1331_; 
v_reuseFailAlloc_1331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1331_, 0, v_a_1325_);
v___x_1330_ = v_reuseFailAlloc_1331_;
goto v_reusejp_1329_;
}
v_reusejp_1329_:
{
return v___x_1330_;
}
}
}
}
}
else
{
lean_dec(v_r_1303_);
lean_dec(v_k_1301_);
return v___x_1304_;
}
}
else
{
lean_object* v___x_1334_; lean_object* v___x_1335_; 
v___x_1334_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1334_, 0, v_init_1296_);
v___x_1335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1335_, 0, v___x_1334_);
return v___x_1335_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3___boxed(lean_object* v___x_1336_, lean_object* v_init_1337_, lean_object* v_x_1338_, lean_object* v___y_1339_, lean_object* v___y_1340_, lean_object* v___y_1341_){
_start:
{
uint8_t v___x_4631__boxed_1342_; lean_object* v_res_1343_; 
v___x_4631__boxed_1342_ = lean_unbox(v___x_1336_);
v_res_1343_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3(v___x_4631__boxed_1342_, v_init_1337_, v_x_1338_, v___y_1339_, v___y_1340_);
lean_dec(v___y_1340_);
lean_dec_ref(v___y_1339_);
return v_res_1343_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0(uint8_t v___y_1345_, uint8_t v_suppressElabErrors_1346_, lean_object* v_x_1347_){
_start:
{
if (lean_obj_tag(v_x_1347_) == 1)
{
lean_object* v_pre_1348_; 
v_pre_1348_ = lean_ctor_get(v_x_1347_, 0);
if (lean_obj_tag(v_pre_1348_) == 0)
{
lean_object* v_str_1349_; lean_object* v___x_1350_; uint8_t v___x_1351_; 
v_str_1349_ = lean_ctor_get(v_x_1347_, 1);
v___x_1350_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___closed__0));
v___x_1351_ = lean_string_dec_eq(v_str_1349_, v___x_1350_);
if (v___x_1351_ == 0)
{
return v___y_1345_;
}
else
{
return v_suppressElabErrors_1346_;
}
}
else
{
return v___y_1345_;
}
}
else
{
return v___y_1345_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___boxed(lean_object* v___y_1352_, lean_object* v_suppressElabErrors_1353_, lean_object* v_x_1354_){
_start:
{
uint8_t v___y_4720__boxed_1355_; uint8_t v_suppressElabErrors_boxed_1356_; uint8_t v_res_1357_; lean_object* v_r_1358_; 
v___y_4720__boxed_1355_ = lean_unbox(v___y_1352_);
v_suppressElabErrors_boxed_1356_ = lean_unbox(v_suppressElabErrors_1353_);
v_res_1357_ = lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0(v___y_4720__boxed_1355_, v_suppressElabErrors_boxed_1356_, v_x_1354_);
lean_dec(v_x_1354_);
v_r_1358_ = lean_box(v_res_1357_);
return v_r_1358_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4(lean_object* v_opts_1359_, lean_object* v_opt_1360_){
_start:
{
lean_object* v_name_1361_; lean_object* v_defValue_1362_; lean_object* v_map_1363_; lean_object* v___x_1364_; 
v_name_1361_ = lean_ctor_get(v_opt_1360_, 0);
v_defValue_1362_ = lean_ctor_get(v_opt_1360_, 1);
v_map_1363_ = lean_ctor_get(v_opts_1359_, 0);
v___x_1364_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1363_, v_name_1361_);
if (lean_obj_tag(v___x_1364_) == 0)
{
uint8_t v___x_1365_; 
v___x_1365_ = lean_unbox(v_defValue_1362_);
return v___x_1365_;
}
else
{
lean_object* v_val_1366_; 
v_val_1366_ = lean_ctor_get(v___x_1364_, 0);
lean_inc(v_val_1366_);
lean_dec_ref_known(v___x_1364_, 1);
if (lean_obj_tag(v_val_1366_) == 1)
{
uint8_t v_v_1367_; 
v_v_1367_ = lean_ctor_get_uint8(v_val_1366_, 0);
lean_dec_ref_known(v_val_1366_, 0);
return v_v_1367_;
}
else
{
uint8_t v___x_1368_; 
lean_dec(v_val_1366_);
v___x_1368_ = lean_unbox(v_defValue_1362_);
return v___x_1368_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4___boxed(lean_object* v_opts_1369_, lean_object* v_opt_1370_){
_start:
{
uint8_t v_res_1371_; lean_object* v_r_1372_; 
v_res_1371_ = lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4(v_opts_1369_, v_opt_1370_);
lean_dec_ref(v_opt_1370_);
lean_dec_ref(v_opts_1369_);
v_r_1372_ = lean_box(v_res_1371_);
return v_r_1372_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_1373_; 
v___x_1373_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1373_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1(void){
_start:
{
lean_object* v___x_1374_; lean_object* v___x_1375_; 
v___x_1374_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__0);
v___x_1375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1375_, 0, v___x_1374_);
return v___x_1375_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2(void){
_start:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1378_; 
v___x_1376_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1);
v___x_1377_ = lean_unsigned_to_nat(0u);
v___x_1378_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1378_, 0, v___x_1377_);
lean_ctor_set(v___x_1378_, 1, v___x_1377_);
lean_ctor_set(v___x_1378_, 2, v___x_1377_);
lean_ctor_set(v___x_1378_, 3, v___x_1377_);
lean_ctor_set(v___x_1378_, 4, v___x_1376_);
lean_ctor_set(v___x_1378_, 5, v___x_1376_);
lean_ctor_set(v___x_1378_, 6, v___x_1376_);
lean_ctor_set(v___x_1378_, 7, v___x_1376_);
lean_ctor_set(v___x_1378_, 8, v___x_1376_);
lean_ctor_set(v___x_1378_, 9, v___x_1376_);
return v___x_1378_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3(void){
_start:
{
lean_object* v___x_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1379_ = lean_unsigned_to_nat(32u);
v___x_1380_ = lean_mk_empty_array_with_capacity(v___x_1379_);
v___x_1381_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1381_, 0, v___x_1380_);
return v___x_1381_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4(void){
_start:
{
size_t v___x_1382_; lean_object* v___x_1383_; lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; 
v___x_1382_ = ((size_t)5ULL);
v___x_1383_ = lean_unsigned_to_nat(0u);
v___x_1384_ = lean_unsigned_to_nat(32u);
v___x_1385_ = lean_mk_empty_array_with_capacity(v___x_1384_);
v___x_1386_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__3);
v___x_1387_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1387_, 0, v___x_1386_);
lean_ctor_set(v___x_1387_, 1, v___x_1385_);
lean_ctor_set(v___x_1387_, 2, v___x_1383_);
lean_ctor_set(v___x_1387_, 3, v___x_1383_);
lean_ctor_set_usize(v___x_1387_, 4, v___x_1382_);
return v___x_1387_;
}
}
static lean_object* _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5(void){
_start:
{
lean_object* v___x_1388_; lean_object* v___x_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; 
v___x_1388_ = lean_box(1);
v___x_1389_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__4);
v___x_1390_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__1);
v___x_1391_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1391_, 0, v___x_1390_);
lean_ctor_set(v___x_1391_, 1, v___x_1389_);
lean_ctor_set(v___x_1391_, 2, v___x_1388_);
return v___x_1391_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg(lean_object* v_msgData_1392_, lean_object* v___y_1393_){
_start:
{
lean_object* v___x_1395_; lean_object* v_env_1396_; lean_object* v___x_1397_; lean_object* v_scopes_1398_; lean_object* v___x_1399_; lean_object* v___x_1400_; lean_object* v_opts_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v___x_1405_; lean_object* v___x_1406_; 
v___x_1395_ = lean_st_ref_get(v___y_1393_);
v_env_1396_ = lean_ctor_get(v___x_1395_, 0);
lean_inc_ref(v_env_1396_);
lean_dec(v___x_1395_);
v___x_1397_ = lean_st_ref_get(v___y_1393_);
v_scopes_1398_ = lean_ctor_get(v___x_1397_, 2);
lean_inc(v_scopes_1398_);
lean_dec(v___x_1397_);
v___x_1399_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1400_ = l_List_head_x21___redArg(v___x_1399_, v_scopes_1398_);
lean_dec(v_scopes_1398_);
v_opts_1401_ = lean_ctor_get(v___x_1400_, 1);
lean_inc_ref(v_opts_1401_);
lean_dec(v___x_1400_);
v___x_1402_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__2);
v___x_1403_ = lean_obj_once(&lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5, &lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5_once, _init_lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___closed__5);
v___x_1404_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1404_, 0, v_env_1396_);
lean_ctor_set(v___x_1404_, 1, v___x_1402_);
lean_ctor_set(v___x_1404_, 2, v___x_1403_);
lean_ctor_set(v___x_1404_, 3, v_opts_1401_);
v___x_1405_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1405_, 0, v___x_1404_);
lean_ctor_set(v___x_1405_, 1, v_msgData_1392_);
v___x_1406_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1406_, 0, v___x_1405_);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg___boxed(lean_object* v_msgData_1407_, lean_object* v___y_1408_, lean_object* v___y_1409_){
_start:
{
lean_object* v_res_1410_; 
v_res_1410_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg(v_msgData_1407_, v___y_1408_);
lean_dec(v___y_1408_);
return v_res_1410_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2(lean_object* v_ref_1412_, lean_object* v_msgData_1413_, uint8_t v_severity_1414_, uint8_t v_isSilent_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_){
_start:
{
lean_object* v___y_1420_; lean_object* v___y_1421_; lean_object* v___y_1422_; uint8_t v___y_1423_; lean_object* v___y_1424_; uint8_t v___y_1425_; lean_object* v___y_1426_; lean_object* v___y_1427_; uint8_t v___y_1484_; uint8_t v___y_1485_; uint8_t v___y_1486_; lean_object* v___y_1487_; lean_object* v___y_1488_; uint8_t v___y_1512_; lean_object* v___y_1513_; uint8_t v___y_1514_; uint8_t v___y_1515_; lean_object* v___y_1516_; uint8_t v___y_1520_; uint8_t v___y_1521_; uint8_t v___y_1522_; uint8_t v___x_1537_; uint8_t v___y_1539_; uint8_t v___y_1540_; uint8_t v___y_1541_; uint8_t v___y_1543_; uint8_t v___x_1555_; 
v___x_1537_ = 2;
v___x_1555_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1414_, v___x_1537_);
if (v___x_1555_ == 0)
{
v___y_1543_ = v___x_1555_;
goto v___jp_1542_;
}
else
{
uint8_t v___x_1556_; 
lean_inc_ref(v_msgData_1413_);
v___x_1556_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_1413_);
v___y_1543_ = v___x_1556_;
goto v___jp_1542_;
}
v___jp_1419_:
{
lean_object* v___x_1428_; 
v___x_1428_ = l_Lean_Elab_Command_getScope___redArg(v___y_1427_);
if (lean_obj_tag(v___x_1428_) == 0)
{
lean_object* v_a_1429_; lean_object* v___x_1430_; 
v_a_1429_ = lean_ctor_get(v___x_1428_, 0);
lean_inc(v_a_1429_);
lean_dec_ref_known(v___x_1428_, 1);
v___x_1430_ = l_Lean_Elab_Command_getScope___redArg(v___y_1427_);
if (lean_obj_tag(v___x_1430_) == 0)
{
lean_object* v_a_1431_; lean_object* v___x_1433_; uint8_t v_isShared_1434_; uint8_t v_isSharedCheck_1466_; 
v_a_1431_ = lean_ctor_get(v___x_1430_, 0);
v_isSharedCheck_1466_ = !lean_is_exclusive(v___x_1430_);
if (v_isSharedCheck_1466_ == 0)
{
v___x_1433_ = v___x_1430_;
v_isShared_1434_ = v_isSharedCheck_1466_;
goto v_resetjp_1432_;
}
else
{
lean_inc(v_a_1431_);
lean_dec(v___x_1430_);
v___x_1433_ = lean_box(0);
v_isShared_1434_ = v_isSharedCheck_1466_;
goto v_resetjp_1432_;
}
v_resetjp_1432_:
{
lean_object* v___x_1435_; lean_object* v_currNamespace_1436_; lean_object* v_openDecls_1437_; lean_object* v_env_1438_; lean_object* v_messages_1439_; lean_object* v_scopes_1440_; lean_object* v_usedQuotCtxts_1441_; lean_object* v_nextMacroScope_1442_; lean_object* v_maxRecDepth_1443_; lean_object* v_ngen_1444_; lean_object* v_auxDeclNGen_1445_; lean_object* v_infoState_1446_; lean_object* v_traceState_1447_; lean_object* v_snapshotTasks_1448_; lean_object* v_prevLinterStates_1449_; lean_object* v___x_1451_; uint8_t v_isShared_1452_; uint8_t v_isSharedCheck_1465_; 
v___x_1435_ = lean_st_ref_take(v___y_1427_);
v_currNamespace_1436_ = lean_ctor_get(v_a_1429_, 2);
lean_inc(v_currNamespace_1436_);
lean_dec(v_a_1429_);
v_openDecls_1437_ = lean_ctor_get(v_a_1431_, 3);
lean_inc(v_openDecls_1437_);
lean_dec(v_a_1431_);
v_env_1438_ = lean_ctor_get(v___x_1435_, 0);
v_messages_1439_ = lean_ctor_get(v___x_1435_, 1);
v_scopes_1440_ = lean_ctor_get(v___x_1435_, 2);
v_usedQuotCtxts_1441_ = lean_ctor_get(v___x_1435_, 3);
v_nextMacroScope_1442_ = lean_ctor_get(v___x_1435_, 4);
v_maxRecDepth_1443_ = lean_ctor_get(v___x_1435_, 5);
v_ngen_1444_ = lean_ctor_get(v___x_1435_, 6);
v_auxDeclNGen_1445_ = lean_ctor_get(v___x_1435_, 7);
v_infoState_1446_ = lean_ctor_get(v___x_1435_, 8);
v_traceState_1447_ = lean_ctor_get(v___x_1435_, 9);
v_snapshotTasks_1448_ = lean_ctor_get(v___x_1435_, 10);
v_prevLinterStates_1449_ = lean_ctor_get(v___x_1435_, 11);
v_isSharedCheck_1465_ = !lean_is_exclusive(v___x_1435_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1451_ = v___x_1435_;
v_isShared_1452_ = v_isSharedCheck_1465_;
goto v_resetjp_1450_;
}
else
{
lean_inc(v_prevLinterStates_1449_);
lean_inc(v_snapshotTasks_1448_);
lean_inc(v_traceState_1447_);
lean_inc(v_infoState_1446_);
lean_inc(v_auxDeclNGen_1445_);
lean_inc(v_ngen_1444_);
lean_inc(v_maxRecDepth_1443_);
lean_inc(v_nextMacroScope_1442_);
lean_inc(v_usedQuotCtxts_1441_);
lean_inc(v_scopes_1440_);
lean_inc(v_messages_1439_);
lean_inc(v_env_1438_);
lean_dec(v___x_1435_);
v___x_1451_ = lean_box(0);
v_isShared_1452_ = v_isSharedCheck_1465_;
goto v_resetjp_1450_;
}
v_resetjp_1450_:
{
lean_object* v___x_1453_; lean_object* v___x_1454_; lean_object* v___x_1455_; lean_object* v___x_1456_; lean_object* v___x_1458_; 
v___x_1453_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1453_, 0, v_currNamespace_1436_);
lean_ctor_set(v___x_1453_, 1, v_openDecls_1437_);
v___x_1454_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1454_, 0, v___x_1453_);
lean_ctor_set(v___x_1454_, 1, v___y_1420_);
lean_inc_ref(v___y_1422_);
lean_inc_ref(v___y_1421_);
v___x_1455_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_1455_, 0, v___y_1421_);
lean_ctor_set(v___x_1455_, 1, v___y_1424_);
lean_ctor_set(v___x_1455_, 2, v___y_1426_);
lean_ctor_set(v___x_1455_, 3, v___y_1422_);
lean_ctor_set(v___x_1455_, 4, v___x_1454_);
lean_ctor_set_uint8(v___x_1455_, sizeof(void*)*5, v___y_1423_);
lean_ctor_set_uint8(v___x_1455_, sizeof(void*)*5 + 1, v___y_1425_);
lean_ctor_set_uint8(v___x_1455_, sizeof(void*)*5 + 2, v_isSilent_1415_);
v___x_1456_ = l_Lean_MessageLog_add(v___x_1455_, v_messages_1439_);
if (v_isShared_1452_ == 0)
{
lean_ctor_set(v___x_1451_, 1, v___x_1456_);
v___x_1458_ = v___x_1451_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v_env_1438_);
lean_ctor_set(v_reuseFailAlloc_1464_, 1, v___x_1456_);
lean_ctor_set(v_reuseFailAlloc_1464_, 2, v_scopes_1440_);
lean_ctor_set(v_reuseFailAlloc_1464_, 3, v_usedQuotCtxts_1441_);
lean_ctor_set(v_reuseFailAlloc_1464_, 4, v_nextMacroScope_1442_);
lean_ctor_set(v_reuseFailAlloc_1464_, 5, v_maxRecDepth_1443_);
lean_ctor_set(v_reuseFailAlloc_1464_, 6, v_ngen_1444_);
lean_ctor_set(v_reuseFailAlloc_1464_, 7, v_auxDeclNGen_1445_);
lean_ctor_set(v_reuseFailAlloc_1464_, 8, v_infoState_1446_);
lean_ctor_set(v_reuseFailAlloc_1464_, 9, v_traceState_1447_);
lean_ctor_set(v_reuseFailAlloc_1464_, 10, v_snapshotTasks_1448_);
lean_ctor_set(v_reuseFailAlloc_1464_, 11, v_prevLinterStates_1449_);
v___x_1458_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1462_; 
v___x_1459_ = lean_st_ref_set(v___y_1427_, v___x_1458_);
v___x_1460_ = lean_box(0);
if (v_isShared_1434_ == 0)
{
lean_ctor_set(v___x_1433_, 0, v___x_1460_);
v___x_1462_ = v___x_1433_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v___x_1460_);
v___x_1462_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
return v___x_1462_;
}
}
}
}
}
else
{
lean_object* v_a_1467_; lean_object* v___x_1469_; uint8_t v_isShared_1470_; uint8_t v_isSharedCheck_1474_; 
lean_dec(v_a_1429_);
lean_dec(v___y_1426_);
lean_dec_ref(v___y_1424_);
lean_dec_ref(v___y_1420_);
v_a_1467_ = lean_ctor_get(v___x_1430_, 0);
v_isSharedCheck_1474_ = !lean_is_exclusive(v___x_1430_);
if (v_isSharedCheck_1474_ == 0)
{
v___x_1469_ = v___x_1430_;
v_isShared_1470_ = v_isSharedCheck_1474_;
goto v_resetjp_1468_;
}
else
{
lean_inc(v_a_1467_);
lean_dec(v___x_1430_);
v___x_1469_ = lean_box(0);
v_isShared_1470_ = v_isSharedCheck_1474_;
goto v_resetjp_1468_;
}
v_resetjp_1468_:
{
lean_object* v___x_1472_; 
if (v_isShared_1470_ == 0)
{
v___x_1472_ = v___x_1469_;
goto v_reusejp_1471_;
}
else
{
lean_object* v_reuseFailAlloc_1473_; 
v_reuseFailAlloc_1473_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1473_, 0, v_a_1467_);
v___x_1472_ = v_reuseFailAlloc_1473_;
goto v_reusejp_1471_;
}
v_reusejp_1471_:
{
return v___x_1472_;
}
}
}
}
else
{
lean_object* v_a_1475_; lean_object* v___x_1477_; uint8_t v_isShared_1478_; uint8_t v_isSharedCheck_1482_; 
lean_dec(v___y_1426_);
lean_dec_ref(v___y_1424_);
lean_dec_ref(v___y_1420_);
v_a_1475_ = lean_ctor_get(v___x_1428_, 0);
v_isSharedCheck_1482_ = !lean_is_exclusive(v___x_1428_);
if (v_isSharedCheck_1482_ == 0)
{
v___x_1477_ = v___x_1428_;
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
else
{
lean_inc(v_a_1475_);
lean_dec(v___x_1428_);
v___x_1477_ = lean_box(0);
v_isShared_1478_ = v_isSharedCheck_1482_;
goto v_resetjp_1476_;
}
v_resetjp_1476_:
{
lean_object* v___x_1480_; 
if (v_isShared_1478_ == 0)
{
v___x_1480_ = v___x_1477_;
goto v_reusejp_1479_;
}
else
{
lean_object* v_reuseFailAlloc_1481_; 
v_reuseFailAlloc_1481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1481_, 0, v_a_1475_);
v___x_1480_ = v_reuseFailAlloc_1481_;
goto v_reusejp_1479_;
}
v_reusejp_1479_:
{
return v___x_1480_;
}
}
}
}
v___jp_1483_:
{
lean_object* v_fileName_1489_; lean_object* v_fileMap_1490_; uint8_t v_suppressElabErrors_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v_a_1494_; lean_object* v___x_1496_; uint8_t v_isShared_1497_; uint8_t v_isSharedCheck_1510_; 
v_fileName_1489_ = lean_ctor_get(v___y_1416_, 0);
v_fileMap_1490_ = lean_ctor_get(v___y_1416_, 1);
v_suppressElabErrors_1491_ = lean_ctor_get_uint8(v___y_1416_, sizeof(void*)*10);
v___x_1492_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_1413_);
v___x_1493_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg(v___x_1492_, v___y_1417_);
v_a_1494_ = lean_ctor_get(v___x_1493_, 0);
v_isSharedCheck_1510_ = !lean_is_exclusive(v___x_1493_);
if (v_isSharedCheck_1510_ == 0)
{
v___x_1496_ = v___x_1493_;
v_isShared_1497_ = v_isSharedCheck_1510_;
goto v_resetjp_1495_;
}
else
{
lean_inc(v_a_1494_);
lean_dec(v___x_1493_);
v___x_1496_ = lean_box(0);
v_isShared_1497_ = v_isSharedCheck_1510_;
goto v_resetjp_1495_;
}
v_resetjp_1495_:
{
lean_object* v___x_1498_; lean_object* v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; 
lean_inc_ref_n(v_fileMap_1490_, 2);
v___x_1498_ = l_Lean_FileMap_toPosition(v_fileMap_1490_, v___y_1487_);
lean_dec(v___y_1487_);
v___x_1499_ = l_Lean_FileMap_toPosition(v_fileMap_1490_, v___y_1488_);
lean_dec(v___y_1488_);
v___x_1500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1500_, 0, v___x_1499_);
v___x_1501_ = ((lean_object*)(lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___closed__0));
if (v_suppressElabErrors_1491_ == 0)
{
lean_del_object(v___x_1496_);
v___y_1420_ = v_a_1494_;
v___y_1421_ = v_fileName_1489_;
v___y_1422_ = v___x_1501_;
v___y_1423_ = v___y_1485_;
v___y_1424_ = v___x_1498_;
v___y_1425_ = v___y_1486_;
v___y_1426_ = v___x_1500_;
v___y_1427_ = v___y_1417_;
goto v___jp_1419_;
}
else
{
lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___f_1504_; uint8_t v___x_1505_; 
v___x_1502_ = lean_box(v___y_1484_);
v___x_1503_ = lean_box(v_suppressElabErrors_1491_);
v___f_1504_ = lean_alloc_closure((void*)(lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1504_, 0, v___x_1502_);
lean_closure_set(v___f_1504_, 1, v___x_1503_);
lean_inc(v_a_1494_);
v___x_1505_ = l_Lean_MessageData_hasTag(v___f_1504_, v_a_1494_);
if (v___x_1505_ == 0)
{
lean_object* v___x_1506_; lean_object* v___x_1508_; 
lean_dec_ref_known(v___x_1500_, 1);
lean_dec_ref(v___x_1498_);
lean_dec(v_a_1494_);
v___x_1506_ = lean_box(0);
if (v_isShared_1497_ == 0)
{
lean_ctor_set(v___x_1496_, 0, v___x_1506_);
v___x_1508_ = v___x_1496_;
goto v_reusejp_1507_;
}
else
{
lean_object* v_reuseFailAlloc_1509_; 
v_reuseFailAlloc_1509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1509_, 0, v___x_1506_);
v___x_1508_ = v_reuseFailAlloc_1509_;
goto v_reusejp_1507_;
}
v_reusejp_1507_:
{
return v___x_1508_;
}
}
else
{
lean_del_object(v___x_1496_);
v___y_1420_ = v_a_1494_;
v___y_1421_ = v_fileName_1489_;
v___y_1422_ = v___x_1501_;
v___y_1423_ = v___y_1485_;
v___y_1424_ = v___x_1498_;
v___y_1425_ = v___y_1486_;
v___y_1426_ = v___x_1500_;
v___y_1427_ = v___y_1417_;
goto v___jp_1419_;
}
}
}
}
v___jp_1511_:
{
lean_object* v___x_1517_; 
v___x_1517_ = l_Lean_Syntax_getTailPos_x3f(v___y_1513_, v___y_1514_);
lean_dec(v___y_1513_);
if (lean_obj_tag(v___x_1517_) == 0)
{
lean_inc(v___y_1516_);
v___y_1484_ = v___y_1512_;
v___y_1485_ = v___y_1514_;
v___y_1486_ = v___y_1515_;
v___y_1487_ = v___y_1516_;
v___y_1488_ = v___y_1516_;
goto v___jp_1483_;
}
else
{
lean_object* v_val_1518_; 
v_val_1518_ = lean_ctor_get(v___x_1517_, 0);
lean_inc(v_val_1518_);
lean_dec_ref_known(v___x_1517_, 1);
v___y_1484_ = v___y_1512_;
v___y_1485_ = v___y_1514_;
v___y_1486_ = v___y_1515_;
v___y_1487_ = v___y_1516_;
v___y_1488_ = v_val_1518_;
goto v___jp_1483_;
}
}
v___jp_1519_:
{
lean_object* v___x_1523_; 
v___x_1523_ = l_Lean_Elab_Command_getRef___redArg(v___y_1416_);
if (lean_obj_tag(v___x_1523_) == 0)
{
lean_object* v_a_1524_; lean_object* v_ref_1525_; lean_object* v___x_1526_; 
v_a_1524_ = lean_ctor_get(v___x_1523_, 0);
lean_inc(v_a_1524_);
lean_dec_ref_known(v___x_1523_, 1);
v_ref_1525_ = l_Lean_replaceRef(v_ref_1412_, v_a_1524_);
lean_dec(v_a_1524_);
v___x_1526_ = l_Lean_Syntax_getPos_x3f(v_ref_1525_, v___y_1521_);
if (lean_obj_tag(v___x_1526_) == 0)
{
lean_object* v___x_1527_; 
v___x_1527_ = lean_unsigned_to_nat(0u);
v___y_1512_ = v___y_1520_;
v___y_1513_ = v_ref_1525_;
v___y_1514_ = v___y_1521_;
v___y_1515_ = v___y_1522_;
v___y_1516_ = v___x_1527_;
goto v___jp_1511_;
}
else
{
lean_object* v_val_1528_; 
v_val_1528_ = lean_ctor_get(v___x_1526_, 0);
lean_inc(v_val_1528_);
lean_dec_ref_known(v___x_1526_, 1);
v___y_1512_ = v___y_1520_;
v___y_1513_ = v_ref_1525_;
v___y_1514_ = v___y_1521_;
v___y_1515_ = v___y_1522_;
v___y_1516_ = v_val_1528_;
goto v___jp_1511_;
}
}
else
{
lean_object* v_a_1529_; lean_object* v___x_1531_; uint8_t v_isShared_1532_; uint8_t v_isSharedCheck_1536_; 
lean_dec_ref(v_msgData_1413_);
v_a_1529_ = lean_ctor_get(v___x_1523_, 0);
v_isSharedCheck_1536_ = !lean_is_exclusive(v___x_1523_);
if (v_isSharedCheck_1536_ == 0)
{
v___x_1531_ = v___x_1523_;
v_isShared_1532_ = v_isSharedCheck_1536_;
goto v_resetjp_1530_;
}
else
{
lean_inc(v_a_1529_);
lean_dec(v___x_1523_);
v___x_1531_ = lean_box(0);
v_isShared_1532_ = v_isSharedCheck_1536_;
goto v_resetjp_1530_;
}
v_resetjp_1530_:
{
lean_object* v___x_1534_; 
if (v_isShared_1532_ == 0)
{
v___x_1534_ = v___x_1531_;
goto v_reusejp_1533_;
}
else
{
lean_object* v_reuseFailAlloc_1535_; 
v_reuseFailAlloc_1535_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1535_, 0, v_a_1529_);
v___x_1534_ = v_reuseFailAlloc_1535_;
goto v_reusejp_1533_;
}
v_reusejp_1533_:
{
return v___x_1534_;
}
}
}
}
v___jp_1538_:
{
if (v___y_1541_ == 0)
{
v___y_1520_ = v___y_1539_;
v___y_1521_ = v___y_1540_;
v___y_1522_ = v_severity_1414_;
goto v___jp_1519_;
}
else
{
v___y_1520_ = v___y_1539_;
v___y_1521_ = v___y_1540_;
v___y_1522_ = v___x_1537_;
goto v___jp_1519_;
}
}
v___jp_1542_:
{
if (v___y_1543_ == 0)
{
lean_object* v___x_1544_; lean_object* v_scopes_1545_; lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v_opts_1548_; uint8_t v___x_1549_; uint8_t v___x_1550_; 
v___x_1544_ = lean_st_ref_get(v___y_1417_);
v_scopes_1545_ = lean_ctor_get(v___x_1544_, 2);
lean_inc(v_scopes_1545_);
lean_dec(v___x_1544_);
v___x_1546_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_1547_ = l_List_head_x21___redArg(v___x_1546_, v_scopes_1545_);
lean_dec(v_scopes_1545_);
v_opts_1548_ = lean_ctor_get(v___x_1547_, 1);
lean_inc_ref(v_opts_1548_);
lean_dec(v___x_1547_);
v___x_1549_ = 1;
v___x_1550_ = l_Lean_instBEqMessageSeverity_beq(v_severity_1414_, v___x_1549_);
if (v___x_1550_ == 0)
{
lean_dec_ref(v_opts_1548_);
v___y_1539_ = v___y_1543_;
v___y_1540_ = v___y_1543_;
v___y_1541_ = v___x_1550_;
goto v___jp_1538_;
}
else
{
lean_object* v___x_1551_; uint8_t v___x_1552_; 
v___x_1551_ = l_Lean_warningAsError;
v___x_1552_ = lp_importGraph_Lean_Option_get___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__4(v_opts_1548_, v___x_1551_);
lean_dec_ref(v_opts_1548_);
v___y_1539_ = v___y_1543_;
v___y_1540_ = v___y_1543_;
v___y_1541_ = v___x_1552_;
goto v___jp_1538_;
}
}
else
{
lean_object* v___x_1553_; lean_object* v___x_1554_; 
lean_dec_ref(v_msgData_1413_);
v___x_1553_ = lean_box(0);
v___x_1554_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1554_, 0, v___x_1553_);
return v___x_1554_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2___boxed(lean_object* v_ref_1557_, lean_object* v_msgData_1558_, lean_object* v_severity_1559_, lean_object* v_isSilent_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_){
_start:
{
uint8_t v_severity_boxed_1564_; uint8_t v_isSilent_boxed_1565_; lean_object* v_res_1566_; 
v_severity_boxed_1564_ = lean_unbox(v_severity_1559_);
v_isSilent_boxed_1565_ = lean_unbox(v_isSilent_1560_);
v_res_1566_ = lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2(v_ref_1557_, v_msgData_1558_, v_severity_boxed_1564_, v_isSilent_boxed_1565_, v___y_1561_, v___y_1562_);
lean_dec(v___y_1562_);
lean_dec_ref(v___y_1561_);
lean_dec(v_ref_1557_);
return v_res_1566_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2(lean_object* v_ref_1567_, lean_object* v_msgData_1568_, lean_object* v___y_1569_, lean_object* v___y_1570_){
_start:
{
uint8_t v___x_1572_; uint8_t v___x_1573_; lean_object* v___x_1574_; 
v___x_1572_ = 0;
v___x_1573_ = 0;
v___x_1574_ = lp_importGraph_Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2(v_ref_1567_, v_msgData_1568_, v___x_1572_, v___x_1573_, v___y_1569_, v___y_1570_);
return v___x_1574_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2___boxed(lean_object* v_ref_1575_, lean_object* v_msgData_1576_, lean_object* v___y_1577_, lean_object* v___y_1578_, lean_object* v___y_1579_){
_start:
{
lean_object* v_res_1580_; 
v_res_1580_ = lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2(v_ref_1575_, v_msgData_1576_, v___y_1577_, v___y_1578_);
lean_dec(v___y_1578_);
lean_dec_ref(v___y_1577_);
lean_dec(v_ref_1575_);
return v_res_1580_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1(lean_object* v_x_1583_, lean_object* v_a_1584_, lean_object* v_a_1585_){
_start:
{
lean_object* v___x_1587_; uint8_t v___x_1588_; 
v___x_1587_ = ((lean_object*)(lp_importGraph_command_x23find__home_x21___00__closed__1));
lean_inc(v_x_1583_);
v___x_1588_ = l_Lean_Syntax_isOfKind(v_x_1583_, v___x_1587_);
if (v___x_1588_ == 0)
{
lean_object* v___x_1589_; 
lean_dec(v_x_1583_);
v___x_1589_ = lp_importGraph_Lean_Elab_throwUnsupportedSyntax___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__0___redArg();
return v___x_1589_;
}
else
{
lean_object* v___x_1590_; lean_object* v___y_1592_; lean_object* v_a_1593_; lean_object* v___y_1601_; lean_object* v___y_1602_; lean_object* v___y_1603_; lean_object* v_a_1604_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___y_1632_; lean_object* v___x_1666_; 
v___x_1590_ = lean_unsigned_to_nat(0u);
v___x_1627_ = lean_unsigned_to_nat(1u);
v___x_1628_ = l_Lean_Syntax_getArg(v_x_1583_, v___x_1627_);
v___x_1629_ = lean_unsigned_to_nat(2u);
v___x_1630_ = l_Lean_Syntax_getArg(v_x_1583_, v___x_1629_);
lean_dec(v_x_1583_);
v___x_1666_ = l_Lean_Syntax_getOptional_x3f(v___x_1628_);
lean_dec(v___x_1628_);
if (lean_obj_tag(v___x_1666_) == 0)
{
lean_object* v___x_1667_; 
v___x_1667_ = lean_box(0);
v___y_1632_ = v___x_1667_;
goto v___jp_1631_;
}
else
{
lean_object* v_val_1668_; lean_object* v___x_1670_; uint8_t v_isShared_1671_; uint8_t v_isSharedCheck_1675_; 
v_val_1668_ = lean_ctor_get(v___x_1666_, 0);
v_isSharedCheck_1675_ = !lean_is_exclusive(v___x_1666_);
if (v_isSharedCheck_1675_ == 0)
{
v___x_1670_ = v___x_1666_;
v_isShared_1671_ = v_isSharedCheck_1675_;
goto v_resetjp_1669_;
}
else
{
lean_inc(v_val_1668_);
lean_dec(v___x_1666_);
v___x_1670_ = lean_box(0);
v_isShared_1671_ = v_isSharedCheck_1675_;
goto v_resetjp_1669_;
}
v_resetjp_1669_:
{
lean_object* v___x_1673_; 
if (v_isShared_1671_ == 0)
{
v___x_1673_ = v___x_1670_;
goto v_reusejp_1672_;
}
else
{
lean_object* v_reuseFailAlloc_1674_; 
v_reuseFailAlloc_1674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1674_, 0, v_val_1668_);
v___x_1673_ = v_reuseFailAlloc_1674_;
goto v_reusejp_1672_;
}
v_reusejp_1672_:
{
v___y_1632_ = v___x_1673_;
goto v___jp_1631_;
}
}
}
v___jp_1591_:
{
lean_object* v___x_1594_; lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; 
v___x_1594_ = l_Lean_Syntax_getArg(v___y_1592_, v___x_1590_);
lean_dec(v___y_1592_);
v___x_1595_ = lean_array_to_list(v_a_1593_);
v___x_1596_ = lean_box(0);
v___x_1597_ = lp_importGraph_List_mapTR_loop___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__1(v___x_1595_, v___x_1596_);
v___x_1598_ = l_Lean_MessageData_ofList(v___x_1597_);
v___x_1599_ = lp_importGraph_Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2(v___x_1594_, v___x_1598_, v_a_1584_, v_a_1585_);
lean_dec(v___x_1594_);
return v___x_1599_;
}
v___jp_1600_:
{
lean_object* v___x_1605_; lean_object* v___x_1606_; 
v___x_1605_ = lean_alloc_closure((void*)(lp_importGraph_Lean_Name_findHome___boxed), 5, 2);
lean_closure_set(v___x_1605_, 0, v___y_1602_);
lean_closure_set(v___x_1605_, 1, v_a_1604_);
v___x_1606_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1605_, v_a_1584_, v_a_1585_);
if (lean_obj_tag(v___x_1606_) == 0)
{
lean_object* v_a_1607_; lean_object* v___x_1608_; 
v_a_1607_ = lean_ctor_get(v___x_1606_, 0);
lean_inc(v_a_1607_);
lean_dec_ref_known(v___x_1606_, 1);
lean_inc_ref(v___y_1603_);
v___x_1608_ = lp_importGraph_Std_DTreeMap_Internal_Impl_forInStep___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__3(v___x_1588_, v___y_1603_, v_a_1607_, v_a_1584_, v_a_1585_);
if (lean_obj_tag(v___x_1608_) == 0)
{
lean_object* v_a_1609_; lean_object* v_a_1610_; 
v_a_1609_ = lean_ctor_get(v___x_1608_, 0);
lean_inc(v_a_1609_);
lean_dec_ref_known(v___x_1608_, 1);
v_a_1610_ = lean_ctor_get(v_a_1609_, 0);
lean_inc(v_a_1610_);
lean_dec(v_a_1609_);
v___y_1592_ = v___y_1601_;
v_a_1593_ = v_a_1610_;
goto v___jp_1591_;
}
else
{
lean_object* v_a_1611_; lean_object* v___x_1613_; uint8_t v_isShared_1614_; uint8_t v_isSharedCheck_1618_; 
lean_dec(v___y_1601_);
v_a_1611_ = lean_ctor_get(v___x_1608_, 0);
v_isSharedCheck_1618_ = !lean_is_exclusive(v___x_1608_);
if (v_isSharedCheck_1618_ == 0)
{
v___x_1613_ = v___x_1608_;
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
else
{
lean_inc(v_a_1611_);
lean_dec(v___x_1608_);
v___x_1613_ = lean_box(0);
v_isShared_1614_ = v_isSharedCheck_1618_;
goto v_resetjp_1612_;
}
v_resetjp_1612_:
{
lean_object* v___x_1616_; 
if (v_isShared_1614_ == 0)
{
v___x_1616_ = v___x_1613_;
goto v_reusejp_1615_;
}
else
{
lean_object* v_reuseFailAlloc_1617_; 
v_reuseFailAlloc_1617_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1617_, 0, v_a_1611_);
v___x_1616_ = v_reuseFailAlloc_1617_;
goto v_reusejp_1615_;
}
v_reusejp_1615_:
{
return v___x_1616_;
}
}
}
}
else
{
lean_object* v_a_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1626_; 
lean_dec(v___y_1601_);
v_a_1619_ = lean_ctor_get(v___x_1606_, 0);
v_isSharedCheck_1626_ = !lean_is_exclusive(v___x_1606_);
if (v_isSharedCheck_1626_ == 0)
{
v___x_1621_ = v___x_1606_;
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_a_1619_);
lean_dec(v___x_1606_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1626_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1624_; 
if (v_isShared_1622_ == 0)
{
v___x_1624_ = v___x_1621_;
goto v_reusejp_1623_;
}
else
{
lean_object* v_reuseFailAlloc_1625_; 
v_reuseFailAlloc_1625_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1625_, 0, v_a_1619_);
v___x_1624_ = v_reuseFailAlloc_1625_;
goto v_reusejp_1623_;
}
v_reusejp_1623_:
{
return v___x_1624_;
}
}
}
}
v___jp_1631_:
{
lean_object* v___x_1633_; 
v___x_1633_ = l_Lean_Elab_Command_getRef___redArg(v_a_1584_);
if (lean_obj_tag(v___x_1633_) == 0)
{
lean_object* v_a_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; 
v_a_1634_ = lean_ctor_get(v___x_1633_, 0);
lean_inc(v_a_1634_);
lean_dec_ref_known(v___x_1633_, 1);
v___x_1635_ = lean_box(0);
v___x_1636_ = lean_alloc_closure((void*)(l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo___boxed), 5, 2);
lean_closure_set(v___x_1636_, 0, v___x_1630_);
lean_closure_set(v___x_1636_, 1, v___x_1635_);
v___x_1637_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_1636_, v_a_1584_, v_a_1585_);
if (lean_obj_tag(v___x_1637_) == 0)
{
lean_object* v_a_1638_; lean_object* v___x_1639_; 
v_a_1638_ = lean_ctor_get(v___x_1637_, 0);
lean_inc(v_a_1638_);
lean_dec_ref_known(v___x_1637_, 1);
v___x_1639_ = ((lean_object*)(lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___closed__0));
if (lean_obj_tag(v___y_1632_) == 0)
{
v___y_1601_ = v_a_1634_;
v___y_1602_ = v_a_1638_;
v___y_1603_ = v___x_1639_;
v_a_1604_ = v___x_1635_;
goto v___jp_1600_;
}
else
{
lean_object* v___x_1641_; uint8_t v_isShared_1642_; uint8_t v_isSharedCheck_1648_; 
v_isSharedCheck_1648_ = !lean_is_exclusive(v___y_1632_);
if (v_isSharedCheck_1648_ == 0)
{
lean_object* v_unused_1649_; 
v_unused_1649_ = lean_ctor_get(v___y_1632_, 0);
lean_dec(v_unused_1649_);
v___x_1641_ = v___y_1632_;
v_isShared_1642_ = v_isSharedCheck_1648_;
goto v_resetjp_1640_;
}
else
{
lean_dec(v___y_1632_);
v___x_1641_ = lean_box(0);
v_isShared_1642_ = v_isSharedCheck_1648_;
goto v_resetjp_1640_;
}
v_resetjp_1640_:
{
lean_object* v___x_1643_; lean_object* v_env_1644_; lean_object* v___x_1646_; 
v___x_1643_ = lean_st_ref_get(v_a_1585_);
v_env_1644_ = lean_ctor_get(v___x_1643_, 0);
lean_inc_ref(v_env_1644_);
lean_dec(v___x_1643_);
if (v_isShared_1642_ == 0)
{
lean_ctor_set(v___x_1641_, 0, v_env_1644_);
v___x_1646_ = v___x_1641_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1647_; 
v_reuseFailAlloc_1647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1647_, 0, v_env_1644_);
v___x_1646_ = v_reuseFailAlloc_1647_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
v___y_1601_ = v_a_1634_;
v___y_1602_ = v_a_1638_;
v___y_1603_ = v___x_1639_;
v_a_1604_ = v___x_1646_;
goto v___jp_1600_;
}
}
}
}
else
{
lean_object* v_a_1650_; lean_object* v___x_1652_; uint8_t v_isShared_1653_; uint8_t v_isSharedCheck_1657_; 
lean_dec(v_a_1634_);
lean_dec(v___y_1632_);
v_a_1650_ = lean_ctor_get(v___x_1637_, 0);
v_isSharedCheck_1657_ = !lean_is_exclusive(v___x_1637_);
if (v_isSharedCheck_1657_ == 0)
{
v___x_1652_ = v___x_1637_;
v_isShared_1653_ = v_isSharedCheck_1657_;
goto v_resetjp_1651_;
}
else
{
lean_inc(v_a_1650_);
lean_dec(v___x_1637_);
v___x_1652_ = lean_box(0);
v_isShared_1653_ = v_isSharedCheck_1657_;
goto v_resetjp_1651_;
}
v_resetjp_1651_:
{
lean_object* v___x_1655_; 
if (v_isShared_1653_ == 0)
{
v___x_1655_ = v___x_1652_;
goto v_reusejp_1654_;
}
else
{
lean_object* v_reuseFailAlloc_1656_; 
v_reuseFailAlloc_1656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1656_, 0, v_a_1650_);
v___x_1655_ = v_reuseFailAlloc_1656_;
goto v_reusejp_1654_;
}
v_reusejp_1654_:
{
return v___x_1655_;
}
}
}
}
else
{
lean_object* v_a_1658_; lean_object* v___x_1660_; uint8_t v_isShared_1661_; uint8_t v_isSharedCheck_1665_; 
lean_dec(v___y_1632_);
lean_dec(v___x_1630_);
v_a_1658_ = lean_ctor_get(v___x_1633_, 0);
v_isSharedCheck_1665_ = !lean_is_exclusive(v___x_1633_);
if (v_isSharedCheck_1665_ == 0)
{
v___x_1660_ = v___x_1633_;
v_isShared_1661_ = v_isSharedCheck_1665_;
goto v_resetjp_1659_;
}
else
{
lean_inc(v_a_1658_);
lean_dec(v___x_1633_);
v___x_1660_ = lean_box(0);
v_isShared_1661_ = v_isSharedCheck_1665_;
goto v_resetjp_1659_;
}
v_resetjp_1659_:
{
lean_object* v___x_1663_; 
if (v_isShared_1661_ == 0)
{
v___x_1663_ = v___x_1660_;
goto v_reusejp_1662_;
}
else
{
lean_object* v_reuseFailAlloc_1664_; 
v_reuseFailAlloc_1664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1664_, 0, v_a_1658_);
v___x_1663_ = v_reuseFailAlloc_1664_;
goto v_reusejp_1662_;
}
v_reusejp_1662_:
{
return v___x_1663_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1___boxed(lean_object* v_x_1676_, lean_object* v_a_1677_, lean_object* v_a_1678_, lean_object* v_a_1679_){
_start:
{
lean_object* v_res_1680_; 
v_res_1680_ = lp_importGraph___aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1(v_x_1676_, v_a_1677_, v_a_1678_);
lean_dec(v_a_1678_);
lean_dec_ref(v_a_1677_);
return v_res_1680_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3(lean_object* v_msgData_1681_, lean_object* v___y_1682_, lean_object* v___y_1683_){
_start:
{
lean_object* v___x_1685_; 
v___x_1685_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___redArg(v_msgData_1681_, v___y_1683_);
return v___x_1685_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3___boxed(lean_object* v_msgData_1686_, lean_object* v___y_1687_, lean_object* v___y_1688_, lean_object* v___y_1689_){
_start:
{
lean_object* v_res_1690_; 
v_res_1690_ = lp_importGraph_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_logInfoAt___at___00__aux__ImportGraph__Tools__FindHome______elabRules__command_x23find__home_x21____1_spec__2_spec__2_spec__3(v_msgData_1686_, v___y_1687_, v___y_1688_);
lean_dec(v___y_1688_);
lean_dec_ref(v___y_1687_);
return v_res_1690_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Tools_FindHome(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_UserWidget(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Tools_FindHome(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_UserWidget(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_importGraph_getModuleUri___rpc__wrapped = _init_lp_importGraph_getModuleUri___rpc__wrapped();
lean_mark_persistent(lp_importGraph_getModuleUri___rpc__wrapped);
lp_importGraph_GoToModuleLink = _init_lp_importGraph_GoToModuleLink();
lean_mark_persistent(lp_importGraph_GoToModuleLink);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Command(uint8_t builtin);
lean_object* initialize_Lean_Widget_UserWidget(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_RequiredModules(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Tools_FindHome(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Widget_UserWidget(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_RequiredModules(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Tools_FindHome(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Tools_FindHome(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Tools_FindHome(builtin);
}
#ifdef __cplusplus
}
#endif
