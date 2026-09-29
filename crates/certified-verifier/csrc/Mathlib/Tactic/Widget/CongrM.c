// Lean compiler output
// Module: Mathlib.Tactic.Widget.CongrM
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Widget.SelectPanelUtils public import ProofWidgets.Component.Basic public import ProofWidgets.Component.OfRpcMethod public meta import ProofWidgets.Component.Basic
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object*);
uint64_t lean_string_hash(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_String_renameMetaVar(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lp_mathlib_getGoalLocations(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_mathlib_insertMetaVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_sanitizeNames(lean_object*, lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink;
lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_lspRangeOfStx_x3f(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Lsp_instToJsonRange_toJson(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Widget_savePanelWidgetInfo(uint64_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_makeCongrMString___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_makeCongrMString___redArg___closed__0;
static const lean_string_object lp_mathlib_makeCongrMString___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "congrm "};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__1 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__1_value;
static const lean_string_object lp_mathlib_makeCongrMString___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "The goal must be an equality or iff."};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__2 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_makeCongrMString___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_makeCongrMString___redArg___closed__3;
static const lean_string_object lp_mathlib_makeCongrMString___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__4 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_makeCongrMString___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__5 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__5_value;
static const lean_string_object lp_mathlib_makeCongrMString___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__6 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__6_value;
static const lean_ctor_object lp_mathlib_makeCongrMString___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__6_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_makeCongrMString___redArg___closed__7 = (const lean_object*)&lp_mathlib_makeCongrMString___redArg___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value;
static const lean_array_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__0_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__1 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__2 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__3 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__3_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__4 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__4_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__4_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__5 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__5_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__5_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__6 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__6_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__6_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__7 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__7_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__8 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__8_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ml1"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__9 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__9_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__9_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__10 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__10_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__10_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__11 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__11_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__11_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__12 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__12_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "There is no goal to solve!"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__13 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__13_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__13_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__14 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__14_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__14_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__15 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__15_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__15_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__16 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__16_value;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " should be "};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__22 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__22_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "You should select only one sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__23 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__23_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__23_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__24 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__24_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__24_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__25 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__25_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0_value),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__25_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__26 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__26_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "in the main goal or its context."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__27 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__27_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "in the main goal."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__28 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__28_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "All selected sub-expressions"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__29 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__29_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "The selected sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__30 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__30_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_CongrMSelectionPanel_rpc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_makeCongrMString___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___closed__0 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___closed__0_value;
static const lean_string_object lp_mathlib_CongrMSelectionPanel_rpc___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 90, .m_capacity = 90, .m_length = 89, .m_data = "Use shift-click to select sub-expressions in the goal that should become holes in congrm."};
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___closed__1 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___closed__1_value;
static const lean_string_object lp_mathlib_CongrMSelectionPanel_rpc___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 9, .m_data = "CongrM 🔍️"};
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___closed__2 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel_rpc(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel_rpc___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1(lean_object*, lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "CongrMSelectionPanel"};
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__0 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__0_value;
static const lean_string_object lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rpc"};
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__1 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__1_value;
static const lean_ctor_object lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(109, 123, 162, 205, 102, 138, 232, 19)}};
static const lean_ctor_object lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2_value_aux_0),((lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(17, 178, 210, 122, 203, 129, 175, 5)}};
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2_value;
static const lean_closure_object lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_CongrMSelectionPanel_rpc___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__3 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__3_value;
static lean_once_cell_t lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_CongrMSelectionPanel___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3841, .m_capacity = 3841, .m_length = 3840, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import{useRpcSession as o,EnvPosContext as a,useAsyncPersistent as i,mapRpcError as f,importWidgetModule as c}from\"@leanprover/infoview\";function u(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,\"default\")\?e.default:e}var s,l;var p=u(function(){if(l)return s;l=1;var e=\"undefined\"!=typeof Element,t=\"function\"==typeof Map,r=\"function\"==typeof Set,n=\"function\"==typeof ArrayBuffer&&!!ArrayBuffer.isView;function o(a,i){if(a===i)return!0;if(a&&i&&\"object\"==typeof a&&\"object\"==typeof i){if(a.constructor!==i.constructor)return!1;var f,c,u,s;if(Array.isArray(a)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(!o(a[c],i[c]))return!1;return!0}if(t&&a instanceof Map&&i instanceof Map){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;for(s=a.entries();!(c=s.next()).done;)if(!o(c.value[1],i.get(c.value[0])))return!1;return!0}if(r&&a instanceof Set&&i instanceof Set){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;return!0}if(n&&ArrayBuffer.isView(a)&&ArrayBuffer.isView(i)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(a[c]!==i[c])return!1;return!0}if(a.constructor===RegExp)return a.source===i.source&&a.flags===i.flags;if(a.valueOf!==Object.prototype.valueOf&&\"function\"==typeof a.valueOf&&\"function\"==typeof i.valueOf)return a.valueOf()===i.valueOf();if(a.toString!==Object.prototype.toString&&\"function\"==typeof a.toString&&\"function\"==typeof i.toString)return a.toString()===i.toString();if((f=(u=Object.keys(a)).length)!==Object.keys(i).length)return!1;for(c=f;0!==c--;)if(!Object.prototype.hasOwnProperty.call(i,u[c]))return!1;if(e&&a instanceof Element)return!1;for(c=f;0!==c--;)if((\"_owner\"!==u[c]&&\"__v\"!==u[c]&&\"__o\"!==u[c]||!a.$$typeof)&&!o(a[u[c]],i[u[c]]))return!1;return!0}return a!=a&&i!=i}return s=function(e,t){try{return o(e,t)}catch(e){if((e.message||\"\").match(/stack|recursion/i))return console.warn(\"react-fast-compare cannot handle circular refs\"),!1;throw e}}}());async function y(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,f]=i.element,c={};for(const[e,t]of r)c[e]=t;const u=await Promise.all(f.map(async e=>await y(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===u.length\?n.createElement(e,c):n.createElement(e,c,u)}if(\"component\"in i){const[e,t,r,f]=i.component,u=await Promise.all(f.map(async e=>await y(o,a,e))),s={...r,pos:a},l=await c(o,a,e);if(!(t in l))throw new Error(`Module '${e}' does not export '${t}'`);return 0===u.length\?n.createElement(l[t],s):n.createElement(l[t],s,u)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function d({html:c}){const u=o(),s=n.useContext(a),l=i(()=>y(u,s,c),[u,s,c]);return\"resolved\"===l.state\?l.value:\"rejected\"===l.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",f(l.error).message]}):t(r,{})}const m=\"CongrMSelectionPanel.rpc\",g='false';var w=n.memo(e=>{const a=o(),c=n.useRef({fn:()=>{}}),u=i(async()=>{if(c.current.fn(),\"true\"===g){const[t,r]=function(e,t,r){const n={fn:()=>{}};return[new Promise(async(o,a)=>{const i=await e.call(t,r),f=window.setInterval(async()=>{try{const t=await e.call(\"ProofWidgets.checkRequest\",i);if(\"running\"===t)return;window.clearInterval(f),o(t.done.result)}catch(e){window.clearInterval(f),a(e)}},100);n.fn=()=>{e.call(\"ProofWidgets.cancelRequest\",i)}}),n]}(a,m,e);return c.current=r,t}{const t=new AbortController,r=a.call(m,e,{abortSignal:t.signal});return c.current={fn:()=>t.abort()},r}},[a,e]);return n.useEffect(()=>()=>{c.current.fn()},[]),\"rejected\"===u.state\?t(\"p\",{style:{color:\"red\"},children:f(u.error).message}):\"loading\"===u.state\?t(r,{children:\"Loading..\"}):t(d,{html:u.value})},p);export{w as default};"};
static const lean_object* lp_mathlib_CongrMSelectionPanel___closed__0 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel___closed__0_value;
static lean_once_cell_t lp_mathlib_CongrMSelectionPanel___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_CongrMSelectionPanel___closed__1;
static lean_once_cell_t lp_mathlib_CongrMSelectionPanel___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CongrMSelectionPanel___closed__2;
static const lean_string_object lp_mathlib_CongrMSelectionPanel___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_mathlib_CongrMSelectionPanel___closed__3 = (const lean_object*)&lp_mathlib_CongrMSelectionPanel___closed__3_value;
static lean_once_cell_t lp_mathlib_CongrMSelectionPanel___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_CongrMSelectionPanel___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel;
static const lean_string_object lp_mathlib_tacticCongrm_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "tacticCongrm\?"};
static const lean_object* lp_mathlib_tacticCongrm_x3f___closed__0 = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__0_value;
static const lean_ctor_object lp_mathlib_tacticCongrm_x3f___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__0_value),LEAN_SCALAR_PTR_LITERAL(38, 217, 224, 200, 45, 107, 40, 43)}};
static const lean_object* lp_mathlib_tacticCongrm_x3f___closed__1 = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__1_value;
static const lean_string_object lp_mathlib_tacticCongrm_x3f___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "congrm\?"};
static const lean_object* lp_mathlib_tacticCongrm_x3f___closed__2 = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__2_value;
static const lean_ctor_object lp_mathlib_tacticCongrm_x3f___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_tacticCongrm_x3f___closed__3 = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__3_value;
static const lean_ctor_object lp_mathlib_tacticCongrm_x3f___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__3_value)}};
static const lean_object* lp_mathlib_tacticCongrm_x3f___closed__4 = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_tacticCongrm_x3f = (const lean_object*)&lp_mathlib_tacticCongrm_x3f___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "replaceRange"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1(lean_object* v_msgData_1_, lean_object* v___y_2_, lean_object* v___y_3_, lean_object* v___y_4_, lean_object* v___y_5_){
_start:
{
lean_object* v___x_7_; lean_object* v_env_8_; lean_object* v___x_9_; lean_object* v_mctx_10_; lean_object* v_lctx_11_; lean_object* v_options_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_7_ = lean_st_ref_get(v___y_5_);
v_env_8_ = lean_ctor_get(v___x_7_, 0);
lean_inc_ref(v_env_8_);
lean_dec(v___x_7_);
v___x_9_ = lean_st_ref_get(v___y_3_);
v_mctx_10_ = lean_ctor_get(v___x_9_, 0);
lean_inc_ref(v_mctx_10_);
lean_dec(v___x_9_);
v_lctx_11_ = lean_ctor_get(v___y_2_, 2);
v_options_12_ = lean_ctor_get(v___y_4_, 2);
lean_inc_ref(v_options_12_);
lean_inc_ref(v_lctx_11_);
v___x_13_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_13_, 0, v_env_8_);
lean_ctor_set(v___x_13_, 1, v_mctx_10_);
lean_ctor_set(v___x_13_, 2, v_lctx_11_);
lean_ctor_set(v___x_13_, 3, v_options_12_);
v___x_14_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v_msgData_1_);
v___x_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_15_, 0, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1___boxed(lean_object* v_msgData_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1(v_msgData_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg(lean_object* v_msg_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v_ref_29_; lean_object* v___x_30_; lean_object* v_a_31_; lean_object* v___x_33_; uint8_t v_isShared_34_; uint8_t v_isSharedCheck_39_; 
v_ref_29_ = lean_ctor_get(v___y_26_, 5);
v___x_30_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00makeCongrMString_spec__1_spec__1(v_msg_23_, v___y_24_, v___y_25_, v___y_26_, v___y_27_);
v_a_31_ = lean_ctor_get(v___x_30_, 0);
v_isSharedCheck_39_ = !lean_is_exclusive(v___x_30_);
if (v_isSharedCheck_39_ == 0)
{
v___x_33_ = v___x_30_;
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
else
{
lean_inc(v_a_31_);
lean_dec(v___x_30_);
v___x_33_ = lean_box(0);
v_isShared_34_ = v_isSharedCheck_39_;
goto v_resetjp_32_;
}
v_resetjp_32_:
{
lean_object* v___x_35_; lean_object* v___x_37_; 
lean_inc(v_ref_29_);
v___x_35_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_35_, 0, v_ref_29_);
lean_ctor_set(v___x_35_, 1, v_a_31_);
if (v_isShared_34_ == 0)
{
lean_ctor_set_tag(v___x_33_, 1);
lean_ctor_set(v___x_33_, 0, v___x_35_);
v___x_37_ = v___x_33_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_38_; 
v_reuseFailAlloc_38_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_38_, 0, v___x_35_);
v___x_37_ = v_reuseFailAlloc_38_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
return v___x_37_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg___boxed(lean_object* v_msg_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg(v_msg_40_, v___y_41_, v___y_42_, v___y_43_, v___y_44_);
lean_dec(v___y_44_);
lean_dec_ref(v___y_43_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0(lean_object* v_as_47_, size_t v_sz_48_, size_t v_i_49_, lean_object* v_b_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lean_usize_dec_lt(v_i_49_, v_sz_48_);
if (v___x_56_ == 0)
{
lean_object* v___x_57_; 
v___x_57_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_57_, 0, v_b_50_);
return v___x_57_;
}
else
{
lean_object* v_a_58_; lean_object* v___x_59_; 
v_a_58_ = lean_array_uget_borrowed(v_as_47_, v_i_49_);
v___x_59_ = lp_mathlib_insertMetaVar(v_b_50_, v_a_58_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
if (lean_obj_tag(v___x_59_) == 0)
{
lean_object* v_a_60_; size_t v___x_61_; size_t v___x_62_; 
v_a_60_ = lean_ctor_get(v___x_59_, 0);
lean_inc(v_a_60_);
lean_dec_ref_known(v___x_59_, 1);
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_49_, v___x_61_);
v_i_49_ = v___x_62_;
v_b_50_ = v_a_60_;
goto _start;
}
else
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0___boxed(lean_object* v_as_64_, lean_object* v_sz_65_, lean_object* v_i_66_, lean_object* v_b_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
size_t v_sz_boxed_73_; size_t v_i_boxed_74_; lean_object* v_res_75_; 
v_sz_boxed_73_ = lean_unbox_usize(v_sz_65_);
lean_dec(v_sz_65_);
v_i_boxed_74_ = lean_unbox_usize(v_i_66_);
lean_dec(v_i_66_);
v_res_75_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0(v_as_64_, v_sz_boxed_73_, v_i_boxed_74_, v_b_67_, v___y_68_, v___y_69_, v___y_70_, v___y_71_);
lean_dec(v___y_71_);
lean_dec_ref(v___y_70_);
lean_dec(v___y_69_);
lean_dec_ref(v___y_68_);
lean_dec_ref(v_as_64_);
return v_res_75_;
}
}
static lean_object* _init_lp_mathlib_makeCongrMString___redArg___closed__0(void){
_start:
{
lean_object* v___x_76_; lean_object* v_dummy_77_; 
v___x_76_ = lean_box(0);
v_dummy_77_ = l_Lean_Expr_sort___override(v___x_76_);
return v_dummy_77_;
}
}
static lean_object* _init_lp_mathlib_makeCongrMString___redArg___closed__3(void){
_start:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = ((lean_object*)(lp_mathlib_makeCongrMString___redArg___closed__2));
v___x_81_ = l_Lean_stringToMessageData(v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___redArg(lean_object* v_pos_88_, lean_object* v_goalType_89_, lean_object* v_a_90_, lean_object* v_a_91_, lean_object* v_a_92_, lean_object* v_a_93_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___y_99_; lean_object* v___y_100_; lean_object* v___y_101_; lean_object* v___y_102_; lean_object* v___y_103_; lean_object* v___y_104_; lean_object* v_subexprPos_136_; lean_object* v___y_138_; lean_object* v___y_139_; lean_object* v___y_140_; lean_object* v___y_141_; uint8_t v___y_160_; lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_95_ = lean_unsigned_to_nat(0u);
v___x_96_ = lean_unsigned_to_nat(1u);
v___x_97_ = l_Lean_instInhabitedExpr;
v_subexprPos_136_ = lp_mathlib_getGoalLocations(v_pos_88_);
v___x_171_ = ((lean_object*)(lp_mathlib_makeCongrMString___redArg___closed__5));
v___x_172_ = l_Lean_Expr_isAppOf(v_goalType_89_, v___x_171_);
if (v___x_172_ == 0)
{
lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_173_ = ((lean_object*)(lp_mathlib_makeCongrMString___redArg___closed__7));
v___x_174_ = l_Lean_Expr_isAppOf(v_goalType_89_, v___x_173_);
v___y_160_ = v___x_174_;
goto v___jp_159_;
}
else
{
v___y_160_ = v___x_172_;
goto v___jp_159_;
}
v___jp_98_:
{
lean_object* v_dummy_105_; lean_object* v_nargs_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v_dummy_105_ = lean_obj_once(&lp_mathlib_makeCongrMString___redArg___closed__0, &lp_mathlib_makeCongrMString___redArg___closed__0_once, _init_lp_mathlib_makeCongrMString___redArg___closed__0);
v_nargs_106_ = l_Lean_Expr_getAppNumArgs(v___y_99_);
lean_inc(v_nargs_106_);
v___x_107_ = lean_mk_array(v_nargs_106_, v_dummy_105_);
v___x_108_ = lean_nat_sub(v_nargs_106_, v___x_96_);
lean_dec(v_nargs_106_);
v___x_109_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v___y_99_, v___x_107_, v___x_108_);
v___x_110_ = lean_array_get(v___x_97_, v___x_109_, v___y_104_);
lean_dec_ref(v___x_109_);
v___x_111_ = l_Lean_Meta_ppExpr(v___x_110_, v___y_103_, v___y_100_, v___y_102_, v___y_101_);
if (lean_obj_tag(v___x_111_) == 0)
{
lean_object* v_a_112_; lean_object* v___x_114_; uint8_t v_isShared_115_; uint8_t v_isSharedCheck_127_; 
v_a_112_ = lean_ctor_get(v___x_111_, 0);
v_isSharedCheck_127_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_127_ == 0)
{
v___x_114_ = v___x_111_;
v_isShared_115_ = v_isSharedCheck_127_;
goto v_resetjp_113_;
}
else
{
lean_inc(v_a_112_);
lean_dec(v___x_111_);
v___x_114_ = lean_box(0);
v_isShared_115_ = v_isSharedCheck_127_;
goto v_resetjp_113_;
}
v_resetjp_113_:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_125_; 
v___x_116_ = ((lean_object*)(lp_mathlib_makeCongrMString___redArg___closed__1));
v___x_117_ = l_Std_Format_defWidth;
v___x_118_ = l_Std_Format_pretty(v_a_112_, v___x_117_, v___x_95_, v___x_95_);
v___x_119_ = lp_mathlib_String_renameMetaVar(v___x_118_);
lean_dec_ref(v___x_118_);
v___x_120_ = lean_string_append(v___x_116_, v___x_119_);
lean_dec_ref(v___x_119_);
v___x_121_ = lean_box(0);
lean_inc_ref(v___x_120_);
v___x_122_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_120_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
v___x_123_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_123_, 0, v___x_120_);
lean_ctor_set(v___x_123_, 1, v___x_122_);
if (v_isShared_115_ == 0)
{
lean_ctor_set(v___x_114_, 0, v___x_123_);
v___x_125_ = v___x_114_;
goto v_reusejp_124_;
}
else
{
lean_object* v_reuseFailAlloc_126_; 
v_reuseFailAlloc_126_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_126_, 0, v___x_123_);
v___x_125_ = v_reuseFailAlloc_126_;
goto v_reusejp_124_;
}
v_reusejp_124_:
{
return v___x_125_;
}
}
}
else
{
lean_object* v_a_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_135_; 
v_a_128_ = lean_ctor_get(v___x_111_, 0);
v_isSharedCheck_135_ = !lean_is_exclusive(v___x_111_);
if (v_isSharedCheck_135_ == 0)
{
v___x_130_ = v___x_111_;
v_isShared_131_ = v_isSharedCheck_135_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_a_128_);
lean_dec(v___x_111_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_135_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_133_; 
if (v_isShared_131_ == 0)
{
v___x_133_ = v___x_130_;
goto v_reusejp_132_;
}
else
{
lean_object* v_reuseFailAlloc_134_; 
v_reuseFailAlloc_134_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_134_, 0, v_a_128_);
v___x_133_ = v_reuseFailAlloc_134_;
goto v_reusejp_132_;
}
v_reusejp_132_:
{
return v___x_133_;
}
}
}
}
v___jp_137_:
{
size_t v_sz_142_; size_t v___x_143_; lean_object* v___x_144_; 
v_sz_142_ = lean_array_size(v_subexprPos_136_);
v___x_143_ = ((size_t)0ULL);
v___x_144_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00makeCongrMString_spec__0(v_subexprPos_136_, v_sz_142_, v___x_143_, v_goalType_89_, v___y_138_, v___y_139_, v___y_140_, v___y_141_);
if (lean_obj_tag(v___x_144_) == 0)
{
lean_object* v_a_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; uint8_t v___x_149_; 
v_a_145_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_a_145_);
lean_dec_ref_known(v___x_144_, 1);
v___x_146_ = lean_array_get(v___x_96_, v_subexprPos_136_, v___x_95_);
lean_dec_ref(v_subexprPos_136_);
v___x_147_ = l_Lean_SubExpr_Pos_toArray(v___x_146_);
lean_dec(v___x_146_);
v___x_148_ = lean_array_get(v___x_95_, v___x_147_, v___x_95_);
lean_dec_ref(v___x_147_);
v___x_149_ = lean_nat_dec_eq(v___x_148_, v___x_95_);
lean_dec(v___x_148_);
if (v___x_149_ == 0)
{
lean_object* v___x_150_; 
v___x_150_ = lean_unsigned_to_nat(2u);
v___y_99_ = v_a_145_;
v___y_100_ = v___y_139_;
v___y_101_ = v___y_141_;
v___y_102_ = v___y_140_;
v___y_103_ = v___y_138_;
v___y_104_ = v___x_150_;
goto v___jp_98_;
}
else
{
v___y_99_ = v_a_145_;
v___y_100_ = v___y_139_;
v___y_101_ = v___y_141_;
v___y_102_ = v___y_140_;
v___y_103_ = v___y_138_;
v___y_104_ = v___x_96_;
goto v___jp_98_;
}
}
else
{
lean_object* v_a_151_; lean_object* v___x_153_; uint8_t v_isShared_154_; uint8_t v_isSharedCheck_158_; 
lean_dec_ref(v_subexprPos_136_);
v_a_151_ = lean_ctor_get(v___x_144_, 0);
v_isSharedCheck_158_ = !lean_is_exclusive(v___x_144_);
if (v_isSharedCheck_158_ == 0)
{
v___x_153_ = v___x_144_;
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
else
{
lean_inc(v_a_151_);
lean_dec(v___x_144_);
v___x_153_ = lean_box(0);
v_isShared_154_ = v_isSharedCheck_158_;
goto v_resetjp_152_;
}
v_resetjp_152_:
{
lean_object* v___x_156_; 
if (v_isShared_154_ == 0)
{
v___x_156_ = v___x_153_;
goto v_reusejp_155_;
}
else
{
lean_object* v_reuseFailAlloc_157_; 
v_reuseFailAlloc_157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_157_, 0, v_a_151_);
v___x_156_ = v_reuseFailAlloc_157_;
goto v_reusejp_155_;
}
v_reusejp_155_:
{
return v___x_156_;
}
}
}
}
v___jp_159_:
{
if (v___y_160_ == 0)
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
lean_dec_ref(v_subexprPos_136_);
lean_dec_ref(v_goalType_89_);
v___x_161_ = lean_obj_once(&lp_mathlib_makeCongrMString___redArg___closed__3, &lp_mathlib_makeCongrMString___redArg___closed__3_once, _init_lp_mathlib_makeCongrMString___redArg___closed__3);
v___x_162_ = lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg(v___x_161_, v_a_90_, v_a_91_, v_a_92_, v_a_93_);
v_a_163_ = lean_ctor_get(v___x_162_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_162_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_162_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_dec(v___x_162_);
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
else
{
v___y_138_ = v_a_90_;
v___y_139_ = v_a_91_;
v___y_140_ = v_a_92_;
v___y_141_ = v_a_93_;
goto v___jp_137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___redArg___boxed(lean_object* v_pos_175_, lean_object* v_goalType_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_makeCongrMString___redArg(v_pos_175_, v_goalType_176_, v_a_177_, v_a_178_, v_a_179_, v_a_180_);
lean_dec(v_a_180_);
lean_dec_ref(v_a_179_);
lean_dec(v_a_178_);
lean_dec_ref(v_a_177_);
lean_dec_ref(v_pos_175_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString(lean_object* v_pos_183_, lean_object* v_goalType_184_, lean_object* v_x_185_, lean_object* v_a_186_, lean_object* v_a_187_, lean_object* v_a_188_, lean_object* v_a_189_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_makeCongrMString___redArg(v_pos_183_, v_goalType_184_, v_a_186_, v_a_187_, v_a_188_, v_a_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_makeCongrMString___boxed(lean_object* v_pos_192_, lean_object* v_goalType_193_, lean_object* v_x_194_, lean_object* v_a_195_, lean_object* v_a_196_, lean_object* v_a_197_, lean_object* v_a_198_, lean_object* v_a_199_){
_start:
{
lean_object* v_res_200_; 
v_res_200_ = lp_mathlib_makeCongrMString(v_pos_192_, v_goalType_193_, v_x_194_, v_a_195_, v_a_196_, v_a_197_, v_a_198_);
lean_dec(v_a_198_);
lean_dec_ref(v_a_197_);
lean_dec(v_a_196_);
lean_dec_ref(v_a_195_);
lean_dec_ref(v_x_194_);
lean_dec_ref(v_pos_192_);
return v_res_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1(lean_object* v_00_u03b1_201_, lean_object* v_msg_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___redArg(v_msg_202_, v___y_203_, v___y_204_, v___y_205_, v___y_206_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1___boxed(lean_object* v_00_u03b1_209_, lean_object* v_msg_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_mathlib_Lean_throwError___at___00makeCongrMString_spec__1(v_00_u03b1_209_, v_msg_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0(lean_object* v___y_217_){
_start:
{
lean_object* v_doc_219_; lean_object* v___x_220_; 
v_doc_219_ = lean_ctor_get(v___y_217_, 1);
lean_inc_ref(v_doc_219_);
v___x_220_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_220_, 0, v_doc_219_);
return v___x_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0___boxed(lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0(v___y_221_);
lean_dec_ref(v___y_221_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg(lean_object* v_mainGoalName_230_, lean_object* v_errorMsg_231_, uint8_t v___y_232_, lean_object* v_as_233_, size_t v_sz_234_, size_t v_i_235_, lean_object* v_b_236_){
_start:
{
lean_object* v_a_239_; uint8_t v___x_243_; 
v___x_243_ = lean_usize_dec_lt(v_i_235_, v_sz_234_);
if (v___x_243_ == 0)
{
lean_object* v___x_244_; 
lean_dec_ref(v_errorMsg_231_);
v___x_244_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_244_, 0, v_b_236_);
return v___x_244_;
}
else
{
lean_object* v_a_245_; lean_object* v_mvarId_246_; lean_object* v_loc_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_278_; 
lean_dec_ref(v_b_236_);
v_a_245_ = lean_array_uget(v_as_233_, v_i_235_);
v_mvarId_246_ = lean_ctor_get(v_a_245_, 0);
v_loc_247_ = lean_ctor_get(v_a_245_, 1);
v_isSharedCheck_278_ = !lean_is_exclusive(v_a_245_);
if (v_isSharedCheck_278_ == 0)
{
v___x_249_ = v_a_245_;
v_isShared_250_ = v_isSharedCheck_278_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_loc_247_);
lean_inc(v_mvarId_246_);
lean_dec(v_a_245_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_278_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_251_; uint8_t v___x_252_; 
v___x_251_ = lean_box(0);
v___x_252_ = lean_name_eq(v_mvarId_246_, v_mainGoalName_230_);
lean_dec(v_mvarId_246_);
if (v___x_252_ == 0)
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_262_; 
lean_dec_ref(v_loc_247_);
v___x_253_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_254_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_255_, 0, v_errorMsg_231_);
v___x_256_ = lean_unsigned_to_nat(1u);
v___x_257_ = lean_mk_empty_array_with_capacity(v___x_256_);
v___x_258_ = lean_array_push(v___x_257_, v___x_255_);
v___x_259_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_259_, 0, v___x_253_);
lean_ctor_set(v___x_259_, 1, v___x_254_);
lean_ctor_set(v___x_259_, 2, v___x_258_);
v___x_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_260_, 0, v___x_259_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 1, v___x_251_);
lean_ctor_set(v___x_249_, 0, v___x_260_);
v___x_262_ = v___x_249_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v___x_260_);
lean_ctor_set(v_reuseFailAlloc_264_, 1, v___x_251_);
v___x_262_ = v_reuseFailAlloc_264_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
lean_object* v___x_263_; 
v___x_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
return v___x_263_;
}
}
else
{
lean_object* v___x_265_; 
v___x_265_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__2));
if (v___y_232_ == 0)
{
lean_del_object(v___x_249_);
lean_dec_ref(v_loc_247_);
v_a_239_ = v___x_265_;
goto v___jp_238_;
}
else
{
if (lean_obj_tag(v_loc_247_) == 3)
{
lean_dec_ref_known(v_loc_247_, 1);
lean_del_object(v___x_249_);
v_a_239_ = v___x_265_;
goto v___jp_238_;
}
else
{
lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_275_; 
lean_dec_ref(v_loc_247_);
v___x_266_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_267_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_268_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_268_, 0, v_errorMsg_231_);
v___x_269_ = lean_unsigned_to_nat(1u);
v___x_270_ = lean_mk_empty_array_with_capacity(v___x_269_);
v___x_271_ = lean_array_push(v___x_270_, v___x_268_);
v___x_272_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_272_, 0, v___x_266_);
lean_ctor_set(v___x_272_, 1, v___x_267_);
lean_ctor_set(v___x_272_, 2, v___x_271_);
v___x_273_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_273_, 0, v___x_272_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 1, v___x_251_);
lean_ctor_set(v___x_249_, 0, v___x_273_);
v___x_275_ = v___x_249_;
goto v_reusejp_274_;
}
else
{
lean_object* v_reuseFailAlloc_277_; 
v_reuseFailAlloc_277_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_277_, 0, v___x_273_);
lean_ctor_set(v_reuseFailAlloc_277_, 1, v___x_251_);
v___x_275_ = v_reuseFailAlloc_277_;
goto v_reusejp_274_;
}
v_reusejp_274_:
{
lean_object* v___x_276_; 
v___x_276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_276_, 0, v___x_275_);
return v___x_276_;
}
}
}
}
}
}
v___jp_238_:
{
size_t v___x_240_; size_t v___x_241_; 
v___x_240_ = ((size_t)1ULL);
v___x_241_ = lean_usize_add(v_i_235_, v___x_240_);
lean_inc_ref(v_a_239_);
v_i_235_ = v___x_241_;
v_b_236_ = v_a_239_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___boxed(lean_object* v_mainGoalName_279_, lean_object* v_errorMsg_280_, lean_object* v___y_281_, lean_object* v_as_282_, lean_object* v_sz_283_, lean_object* v_i_284_, lean_object* v_b_285_, lean_object* v___y_286_){
_start:
{
uint8_t v___y_1963__boxed_287_; size_t v_sz_boxed_288_; size_t v_i_boxed_289_; lean_object* v_res_290_; 
v___y_1963__boxed_287_ = lean_unbox(v___y_281_);
v_sz_boxed_288_ = lean_unbox_usize(v_sz_283_);
lean_dec(v_sz_283_);
v_i_boxed_289_ = lean_unbox_usize(v_i_284_);
lean_dec(v_i_284_);
v_res_290_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg(v_mainGoalName_279_, v_errorMsg_280_, v___y_1963__boxed_287_, v_as_282_, v_sz_boxed_288_, v_i_boxed_289_, v_b_285_);
lean_dec_ref(v_as_282_);
lean_dec(v_mainGoalName_279_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2___lam__0(lean_object* v_props_291_, lean_object* v___y_292_){
_start:
{
lean_object* v___x_293_; lean_object* v___x_294_; 
v___x_293_ = lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(v_props_291_);
v___x_294_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_294_, 0, v___x_293_);
lean_ctor_set(v___x_294_, 1, v___y_292_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2(lean_object* v_c_295_, lean_object* v_props_296_, lean_object* v_children_297_){
_start:
{
lean_object* v_toModule_298_; lean_object* v_export_299_; lean_object* v_javascript_300_; lean_object* v___f_301_; uint64_t v___x_302_; lean_object* v___x_303_; 
v_toModule_298_ = lean_ctor_get(v_c_295_, 0);
v_export_299_ = lean_ctor_get(v_c_295_, 1);
v_javascript_300_ = lean_ctor_get(v_toModule_298_, 0);
v___f_301_ = lean_alloc_closure((void*)(lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2___lam__0), 2, 1);
lean_closure_set(v___f_301_, 0, v_props_296_);
v___x_302_ = lean_string_hash(v_javascript_300_);
lean_inc_ref(v_export_299_);
v___x_303_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_303_, 0, v_export_299_);
lean_ctor_set(v___x_303_, 1, v___f_301_);
lean_ctor_set(v___x_303_, 2, v_children_297_);
lean_ctor_set_uint64(v___x_303_, sizeof(void*)*3, v___x_302_);
return v___x_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2___boxed(lean_object* v_c_304_, lean_object* v_props_305_, lean_object* v_children_306_){
_start:
{
lean_object* v_res_307_; 
v_res_307_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2(v_c_304_, v_props_305_, v_children_306_);
lean_dec_ref(v_c_304_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0(lean_object* v_mkCmdStr_308_, lean_object* v_selectedLocations_309_, lean_object* v___x_310_, lean_object* v_params_311_, lean_object* v_a_312_, lean_object* v_replaceRange_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lean_apply_8(v_mkCmdStr_308_, v_selectedLocations_309_, v___x_310_, v_params_311_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, lean_box(0));
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v_a_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_340_; 
v_a_320_ = lean_ctor_get(v___x_319_, 0);
v_isSharedCheck_340_ = !lean_is_exclusive(v___x_319_);
if (v_isSharedCheck_340_ == 0)
{
v___x_322_ = v___x_319_;
v_isShared_323_ = v_isSharedCheck_340_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_a_320_);
lean_dec(v___x_319_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_340_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v_snd_324_; lean_object* v_toEditableDocumentCore_325_; lean_object* v_fst_326_; lean_object* v_fst_327_; lean_object* v_snd_328_; lean_object* v_meta_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_338_; 
v_snd_324_ = lean_ctor_get(v_a_320_, 1);
lean_inc(v_snd_324_);
v_toEditableDocumentCore_325_ = lean_ctor_get(v_a_312_, 0);
v_fst_326_ = lean_ctor_get(v_a_320_, 0);
lean_inc(v_fst_326_);
lean_dec(v_a_320_);
v_fst_327_ = lean_ctor_get(v_snd_324_, 0);
lean_inc(v_fst_327_);
v_snd_328_ = lean_ctor_get(v_snd_324_, 1);
lean_inc(v_snd_328_);
lean_dec(v_snd_324_);
v_meta_329_ = lean_ctor_get(v_toEditableDocumentCore_325_, 0);
v___x_330_ = lp_proofwidgets_ProofWidgets_MakeEditLink;
v___x_331_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(v_meta_329_, v_replaceRange_313_, v_fst_327_, v_snd_328_);
v___x_332_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_332_, 0, v_fst_326_);
v___x_333_ = lean_unsigned_to_nat(1u);
v___x_334_ = lean_mk_empty_array_with_capacity(v___x_333_);
v___x_335_ = lean_array_push(v___x_334_, v___x_332_);
v___x_336_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__2(v___x_330_, v___x_331_, v___x_335_);
if (v_isShared_323_ == 0)
{
lean_ctor_set(v___x_322_, 0, v___x_336_);
v___x_338_ = v___x_322_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_336_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
else
{
lean_object* v_a_341_; lean_object* v___x_343_; uint8_t v_isShared_344_; uint8_t v_isSharedCheck_348_; 
lean_dec_ref(v_replaceRange_313_);
v_a_341_ = lean_ctor_get(v___x_319_, 0);
v_isSharedCheck_348_ = !lean_is_exclusive(v___x_319_);
if (v_isSharedCheck_348_ == 0)
{
v___x_343_ = v___x_319_;
v_isShared_344_ = v_isSharedCheck_348_;
goto v_resetjp_342_;
}
else
{
lean_inc(v_a_341_);
lean_dec(v___x_319_);
v___x_343_ = lean_box(0);
v_isShared_344_ = v_isSharedCheck_348_;
goto v_resetjp_342_;
}
v_resetjp_342_:
{
lean_object* v___x_346_; 
if (v_isShared_344_ == 0)
{
v___x_346_ = v___x_343_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v_a_341_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0___boxed(lean_object* v_mkCmdStr_349_, lean_object* v_selectedLocations_350_, lean_object* v___x_351_, lean_object* v_params_352_, lean_object* v_a_353_, lean_object* v_replaceRange_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0(v_mkCmdStr_349_, v_selectedLocations_350_, v___x_351_, v_params_352_, v_a_353_, v_replaceRange_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_);
lean_dec_ref(v_a_353_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg(lean_object* v_lctx_361_, lean_object* v_localInsts_362_, lean_object* v_x_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_361_, v_localInsts_362_, v_x_363_, v___y_364_, v___y_365_, v___y_366_, v___y_367_);
if (lean_obj_tag(v___x_369_) == 0)
{
lean_object* v_a_370_; lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_377_; 
v_a_370_ = lean_ctor_get(v___x_369_, 0);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_369_);
if (v_isSharedCheck_377_ == 0)
{
v___x_372_ = v___x_369_;
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
else
{
lean_inc(v_a_370_);
lean_dec(v___x_369_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_375_; 
if (v_isShared_373_ == 0)
{
v___x_375_ = v___x_372_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_a_370_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
else
{
lean_object* v_a_378_; lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_385_; 
v_a_378_ = lean_ctor_get(v___x_369_, 0);
v_isSharedCheck_385_ = !lean_is_exclusive(v___x_369_);
if (v_isSharedCheck_385_ == 0)
{
v___x_380_ = v___x_369_;
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
else
{
lean_inc(v_a_378_);
lean_dec(v___x_369_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_385_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v___x_383_; 
if (v_isShared_381_ == 0)
{
v___x_383_ = v___x_380_;
goto v_reusejp_382_;
}
else
{
lean_object* v_reuseFailAlloc_384_; 
v_reuseFailAlloc_384_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_384_, 0, v_a_378_);
v___x_383_ = v_reuseFailAlloc_384_;
goto v_reusejp_382_;
}
v_reusejp_382_:
{
return v___x_383_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg___boxed(lean_object* v_lctx_386_, lean_object* v_localInsts_387_, lean_object* v_x_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
lean_object* v_res_394_; 
v_res_394_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg(v_lctx_386_, v_localInsts_387_, v_x_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
return v_res_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1(lean_object* v_mvarId_395_, lean_object* v_mkCmdStr_396_, lean_object* v_selectedLocations_397_, lean_object* v_params_398_, lean_object* v_a_399_, lean_object* v_replaceRange_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = l_Lean_MVarId_getDecl(v_mvarId_395_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
if (lean_obj_tag(v___x_406_) == 0)
{
lean_object* v_a_407_; lean_object* v_options_408_; lean_object* v_lctx_409_; lean_object* v_type_410_; lean_object* v_localInstances_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v_fst_415_; lean_object* v___x_416_; lean_object* v___f_417_; lean_object* v___x_418_; 
v_a_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_a_407_);
lean_dec_ref_known(v___x_406_, 1);
v_options_408_ = lean_ctor_get(v___y_403_, 2);
v_lctx_409_ = lean_ctor_get(v_a_407_, 1);
lean_inc_ref(v_lctx_409_);
v_type_410_ = lean_ctor_get(v_a_407_, 2);
lean_inc_ref(v_type_410_);
v_localInstances_411_ = lean_ctor_get(v_a_407_, 4);
lean_inc_ref(v_localInstances_411_);
lean_dec(v_a_407_);
v___x_412_ = lean_box(1);
lean_inc_ref(v_options_408_);
v___x_413_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_413_, 0, v_options_408_);
lean_ctor_set(v___x_413_, 1, v___x_412_);
lean_ctor_set(v___x_413_, 2, v___x_412_);
v___x_414_ = l_Lean_LocalContext_sanitizeNames(v_lctx_409_, v___x_413_);
v_fst_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc(v_fst_415_);
lean_dec_ref(v___x_414_);
v___x_416_ = l_Lean_Expr_consumeMData(v_type_410_);
lean_dec_ref(v_type_410_);
v___f_417_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__0___boxed), 11, 6);
lean_closure_set(v___f_417_, 0, v_mkCmdStr_396_);
lean_closure_set(v___f_417_, 1, v_selectedLocations_397_);
lean_closure_set(v___f_417_, 2, v___x_416_);
lean_closure_set(v___f_417_, 3, v_params_398_);
lean_closure_set(v___f_417_, 4, v_a_399_);
lean_closure_set(v___f_417_, 5, v_replaceRange_400_);
v___x_418_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg(v_fst_415_, v_localInstances_411_, v___f_417_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
return v___x_418_;
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref(v_replaceRange_400_);
lean_dec_ref(v_a_399_);
lean_dec_ref(v_params_398_);
lean_dec_ref(v_selectedLocations_397_);
lean_dec_ref(v_mkCmdStr_396_);
v_a_419_ = lean_ctor_get(v___x_406_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_406_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_406_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_406_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1___boxed(lean_object* v_mvarId_427_, lean_object* v_mkCmdStr_428_, lean_object* v_selectedLocations_429_, lean_object* v_params_430_, lean_object* v_a_431_, lean_object* v_replaceRange_432_, lean_object* v___y_433_, lean_object* v___y_434_, lean_object* v___y_435_, lean_object* v___y_436_, lean_object* v___y_437_){
_start:
{
lean_object* v_res_438_; 
v_res_438_ = lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1(v_mvarId_427_, v_mkCmdStr_428_, v_selectedLocations_429_, v_params_430_, v_a_431_, v_replaceRange_432_, v___y_433_, v___y_434_, v___y_435_, v___y_436_);
lean_dec(v___y_436_);
lean_dec_ref(v___y_435_);
lean_dec(v___y_434_);
lean_dec_ref(v___y_433_);
return v_res_438_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17(void){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_475_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; 
v___x_476_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17, &lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__17);
v___x_477_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
return v___x_477_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19(void){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_478_ = lean_unsigned_to_nat(32u);
v___x_479_ = lean_mk_empty_array_with_capacity(v___x_478_);
v___x_480_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_480_, 0, v___x_479_);
return v___x_480_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20(void){
_start:
{
size_t v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_481_ = ((size_t)5ULL);
v___x_482_ = lean_unsigned_to_nat(0u);
v___x_483_ = lean_unsigned_to_nat(32u);
v___x_484_ = lean_mk_empty_array_with_capacity(v___x_483_);
v___x_485_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19, &lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__19);
v___x_486_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_486_, 0, v___x_485_);
lean_ctor_set(v___x_486_, 1, v___x_484_);
lean_ctor_set(v___x_486_, 2, v___x_482_);
lean_ctor_set(v___x_486_, 3, v___x_482_);
lean_ctor_set_usize(v___x_486_, 4, v___x_481_);
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21(void){
_start:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; 
v___x_487_ = lean_box(1);
v___x_488_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20, &lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__20);
v___x_489_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18, &lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__18);
v___x_490_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
lean_ctor_set(v___x_490_, 1, v___x_488_);
lean_ctor_set(v___x_490_, 2, v___x_487_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2(lean_object* v_params_507_, lean_object* v_title_508_, lean_object* v_mkCmdStr_509_, uint8_t v_onlyGoal_510_, lean_object* v_helpMsg_511_, uint8_t v_onlyOne_512_, lean_object* v___y_513_){
_start:
{
lean_object* v___x_515_; lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_614_; 
v___x_515_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__0(v___y_513_);
v_a_516_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_614_ == 0)
{
v___x_518_ = v___x_515_;
v_isShared_519_ = v_isSharedCheck_614_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_515_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_614_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v_goals_520_; lean_object* v_selectedLocations_521_; lean_object* v_replaceRange_522_; lean_object* v___x_523_; lean_object* v___x_524_; uint8_t v___x_525_; lean_object* v_a_527_; 
v_goals_520_ = lean_ctor_get(v_params_507_, 1);
v_selectedLocations_521_ = lean_ctor_get(v_params_507_, 2);
lean_inc_ref(v_selectedLocations_521_);
v_replaceRange_522_ = lean_ctor_get(v_params_507_, 3);
lean_inc_ref(v_replaceRange_522_);
v___x_523_ = lean_unsigned_to_nat(0u);
v___x_524_ = lean_array_get_size(v_goals_520_);
v___x_525_ = lean_nat_dec_lt(v___x_523_, v___x_524_);
if (v___x_525_ == 0)
{
lean_object* v___x_552_; lean_object* v___x_553_; 
lean_dec_ref(v_replaceRange_522_);
lean_dec_ref(v_selectedLocations_521_);
lean_del_object(v___x_518_);
lean_dec(v_a_516_);
lean_dec_ref(v_helpMsg_511_);
lean_dec_ref(v_mkCmdStr_509_);
lean_dec_ref(v_title_508_);
lean_dec_ref(v_params_507_);
v___x_552_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__16));
v___x_553_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_553_, 0, v___x_552_);
return v___x_553_;
}
else
{
lean_object* v_mainGoal_554_; lean_object* v_toInteractiveGoalCore_555_; lean_object* v_mvarId_556_; lean_object* v___f_557_; lean_object* v___y_559_; lean_object* v___y_599_; lean_object* v___y_600_; lean_object* v___y_609_; 
v_mainGoal_554_ = lean_array_fget_borrowed(v_goals_520_, v___x_523_);
v_toInteractiveGoalCore_555_ = lean_ctor_get(v_mainGoal_554_, 0);
lean_inc_ref(v_toInteractiveGoalCore_555_);
v_mvarId_556_ = lean_ctor_get(v_mainGoal_554_, 3);
lean_inc_n(v_mvarId_556_, 2);
lean_inc_ref(v_selectedLocations_521_);
v___f_557_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__1___boxed), 11, 6);
lean_closure_set(v___f_557_, 0, v_mvarId_556_);
lean_closure_set(v___f_557_, 1, v_mkCmdStr_509_);
lean_closure_set(v___f_557_, 2, v_selectedLocations_521_);
lean_closure_set(v___f_557_, 3, v_params_507_);
lean_closure_set(v___f_557_, 4, v_a_516_);
lean_closure_set(v___f_557_, 5, v_replaceRange_522_);
if (v_onlyOne_512_ == 0)
{
lean_object* v___x_612_; 
v___x_612_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__29));
v___y_609_ = v___x_612_;
goto v___jp_608_;
}
else
{
lean_object* v___x_613_; 
v___x_613_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__30));
v___y_609_ = v___x_613_;
goto v___jp_608_;
}
v___jp_558_:
{
lean_object* v___x_560_; size_t v_sz_561_; size_t v___x_562_; lean_object* v___x_563_; 
v___x_560_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__2));
v_sz_561_ = lean_array_size(v_selectedLocations_521_);
v___x_562_ = ((size_t)0ULL);
v___x_563_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg(v_mvarId_556_, v___y_559_, v_onlyGoal_510_, v_selectedLocations_521_, v_sz_561_, v___x_562_, v___x_560_);
lean_dec(v_mvarId_556_);
if (lean_obj_tag(v___x_563_) == 0)
{
lean_object* v_a_564_; lean_object* v_fst_565_; 
v_a_564_ = lean_ctor_get(v___x_563_, 0);
lean_inc(v_a_564_);
lean_dec_ref_known(v___x_563_, 1);
v_fst_565_ = lean_ctor_get(v_a_564_, 0);
lean_inc(v_fst_565_);
lean_dec(v_a_564_);
if (lean_obj_tag(v_fst_565_) == 0)
{
lean_object* v___x_566_; uint8_t v___x_567_; 
v___x_566_ = lean_array_get_size(v_selectedLocations_521_);
lean_dec_ref(v_selectedLocations_521_);
v___x_567_ = lean_nat_dec_eq(v___x_566_, v___x_523_);
if (v___x_567_ == 0)
{
lean_object* v_ctx_568_; lean_object* v_val_569_; lean_object* v___x_570_; lean_object* v___x_571_; 
lean_dec_ref(v_helpMsg_511_);
v_ctx_568_ = lean_ctor_get(v_toInteractiveGoalCore_555_, 2);
lean_inc_ref(v_ctx_568_);
lean_dec_ref(v_toInteractiveGoalCore_555_);
v_val_569_ = lean_ctor_get(v_ctx_568_, 0);
lean_inc(v_val_569_);
lean_dec_ref(v_ctx_568_);
v___x_570_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21, &lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21_once, _init_lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__21);
v___x_571_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_val_569_, v___x_570_, v___f_557_);
if (lean_obj_tag(v___x_571_) == 0)
{
lean_object* v_a_572_; 
v_a_572_ = lean_ctor_get(v___x_571_, 0);
lean_inc(v_a_572_);
lean_dec_ref_known(v___x_571_, 1);
v_a_527_ = v_a_572_;
goto v___jp_526_;
}
else
{
lean_object* v_a_573_; lean_object* v___x_575_; uint8_t v_isShared_576_; uint8_t v_isSharedCheck_581_; 
lean_del_object(v___x_518_);
lean_dec_ref(v_title_508_);
v_a_573_ = lean_ctor_get(v___x_571_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_571_);
if (v_isSharedCheck_581_ == 0)
{
v___x_575_ = v___x_571_;
v_isShared_576_ = v_isSharedCheck_581_;
goto v_resetjp_574_;
}
else
{
lean_inc(v_a_573_);
lean_dec(v___x_571_);
v___x_575_ = lean_box(0);
v_isShared_576_ = v_isSharedCheck_581_;
goto v_resetjp_574_;
}
v_resetjp_574_:
{
lean_object* v___x_577_; lean_object* v___x_579_; 
v___x_577_ = l_Lean_Server_RequestError_ofIoError(v_a_573_);
if (v_isShared_576_ == 0)
{
lean_ctor_set(v___x_575_, 0, v___x_577_);
v___x_579_ = v___x_575_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v___x_577_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
}
else
{
lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; 
lean_dec_ref(v___f_557_);
lean_dec_ref(v_toInteractiveGoalCore_555_);
v___x_582_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__0));
v___x_583_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg___closed__1));
v___x_584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_584_, 0, v_helpMsg_511_);
v___x_585_ = lean_unsigned_to_nat(1u);
v___x_586_ = lean_mk_empty_array_with_capacity(v___x_585_);
v___x_587_ = lean_array_push(v___x_586_, v___x_584_);
v___x_588_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_588_, 0, v___x_582_);
lean_ctor_set(v___x_588_, 1, v___x_583_);
lean_ctor_set(v___x_588_, 2, v___x_587_);
v_a_527_ = v___x_588_;
goto v___jp_526_;
}
}
else
{
lean_object* v_val_589_; 
lean_dec_ref(v___f_557_);
lean_dec_ref(v_toInteractiveGoalCore_555_);
lean_dec_ref(v_selectedLocations_521_);
lean_dec_ref(v_helpMsg_511_);
v_val_589_ = lean_ctor_get(v_fst_565_, 0);
lean_inc(v_val_589_);
lean_dec_ref_known(v_fst_565_, 1);
v_a_527_ = v_val_589_;
goto v___jp_526_;
}
}
else
{
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
lean_dec_ref(v___f_557_);
lean_dec_ref(v_toInteractiveGoalCore_555_);
lean_dec_ref(v_selectedLocations_521_);
lean_del_object(v___x_518_);
lean_dec_ref(v_helpMsg_511_);
lean_dec_ref(v_title_508_);
v_a_590_ = lean_ctor_get(v___x_563_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_563_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_563_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_563_);
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
v___jp_598_:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v_errorMsg_603_; 
v___x_601_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__22));
lean_inc_ref(v___y_599_);
v___x_602_ = lean_string_append(v___y_599_, v___x_601_);
v_errorMsg_603_ = lean_string_append(v___x_602_, v___y_600_);
if (v_onlyOne_512_ == 0)
{
v___y_559_ = v_errorMsg_603_;
goto v___jp_558_;
}
else
{
lean_object* v___x_604_; lean_object* v___x_605_; uint8_t v___x_606_; 
v___x_604_ = lean_unsigned_to_nat(1u);
v___x_605_ = lean_array_get_size(v_selectedLocations_521_);
v___x_606_ = lean_nat_dec_lt(v___x_604_, v___x_605_);
if (v___x_606_ == 0)
{
v___y_559_ = v_errorMsg_603_;
goto v___jp_558_;
}
else
{
lean_object* v___x_607_; 
lean_dec_ref(v_errorMsg_603_);
lean_dec_ref(v___f_557_);
lean_dec(v_mvarId_556_);
lean_dec_ref(v_toInteractiveGoalCore_555_);
lean_dec_ref(v_selectedLocations_521_);
lean_dec_ref(v_helpMsg_511_);
v___x_607_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__26));
v_a_527_ = v___x_607_;
goto v___jp_526_;
}
}
}
v___jp_608_:
{
if (v_onlyGoal_510_ == 0)
{
lean_object* v___x_610_; 
v___x_610_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__27));
v___y_599_ = v___y_609_;
v___y_600_ = v___x_610_;
goto v___jp_598_;
}
else
{
lean_object* v___x_611_; 
v___x_611_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__28));
v___y_599_ = v___y_609_;
v___y_600_ = v___x_611_;
goto v___jp_598_;
}
}
}
v___jp_526_:
{
lean_object* v___x_528_; lean_object* v___x_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; lean_object* v___x_548_; lean_object* v___x_550_; 
v___x_528_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__0));
v___x_529_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__1));
v___x_530_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_530_, 0, v___x_525_);
v___x_531_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_531_, 0, v___x_529_);
lean_ctor_set(v___x_531_, 1, v___x_530_);
v___x_532_ = lean_unsigned_to_nat(1u);
v___x_533_ = lean_mk_empty_array_with_capacity(v___x_532_);
lean_inc_ref_n(v___x_533_, 2);
v___x_534_ = lean_array_push(v___x_533_, v___x_531_);
v___x_535_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__2));
v___x_536_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__7));
v___x_537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_537_, 0, v_title_508_);
v___x_538_ = lean_array_push(v___x_533_, v___x_537_);
v___x_539_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_539_, 0, v___x_535_);
lean_ctor_set(v___x_539_, 1, v___x_536_);
lean_ctor_set(v___x_539_, 2, v___x_538_);
v___x_540_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__8));
v___x_541_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___closed__12));
v___x_542_ = lean_array_push(v___x_533_, v_a_527_);
v___x_543_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_543_, 0, v___x_540_);
lean_ctor_set(v___x_543_, 1, v___x_541_);
lean_ctor_set(v___x_543_, 2, v___x_542_);
v___x_544_ = lean_unsigned_to_nat(2u);
v___x_545_ = lean_mk_empty_array_with_capacity(v___x_544_);
v___x_546_ = lean_array_push(v___x_545_, v___x_539_);
v___x_547_ = lean_array_push(v___x_546_, v___x_543_);
v___x_548_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_548_, 0, v___x_528_);
lean_ctor_set(v___x_548_, 1, v___x_534_);
lean_ctor_set(v___x_548_, 2, v___x_547_);
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 0, v___x_548_);
v___x_550_ = v___x_518_;
goto v_reusejp_549_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v___x_548_);
v___x_550_ = v_reuseFailAlloc_551_;
goto v_reusejp_549_;
}
v_reusejp_549_:
{
return v___x_550_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___boxed(lean_object* v_params_615_, lean_object* v_title_616_, lean_object* v_mkCmdStr_617_, lean_object* v_onlyGoal_618_, lean_object* v_helpMsg_619_, lean_object* v_onlyOne_620_, lean_object* v___y_621_, lean_object* v___y_622_){
_start:
{
uint8_t v_onlyGoal_boxed_623_; uint8_t v_onlyOne_boxed_624_; lean_object* v_res_625_; 
v_onlyGoal_boxed_623_ = lean_unbox(v_onlyGoal_618_);
v_onlyOne_boxed_624_ = lean_unbox(v_onlyOne_620_);
v_res_625_ = lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2(v_params_615_, v_title_616_, v_mkCmdStr_617_, v_onlyGoal_boxed_623_, v_helpMsg_619_, v_onlyOne_boxed_624_, v___y_621_);
lean_dec_ref(v___y_621_);
return v_res_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0(lean_object* v_mkCmdStr_626_, lean_object* v_helpMsg_627_, lean_object* v_title_628_, uint8_t v_onlyGoal_629_, uint8_t v_onlyOne_630_, lean_object* v_params_631_, lean_object* v_a_632_){
_start:
{
lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___f_636_; lean_object* v___x_637_; 
v___x_634_ = lean_box(v_onlyGoal_629_);
v___x_635_ = lean_box(v_onlyOne_630_);
v___f_636_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___lam__2___boxed), 8, 6);
lean_closure_set(v___f_636_, 0, v_params_631_);
lean_closure_set(v___f_636_, 1, v_title_628_);
lean_closure_set(v___f_636_, 2, v_mkCmdStr_626_);
lean_closure_set(v___f_636_, 3, v___x_634_);
lean_closure_set(v___f_636_, 4, v_helpMsg_627_);
lean_closure_set(v___f_636_, 5, v___x_635_);
v___x_637_ = l_Lean_Server_RequestM_asTask___redArg(v___f_636_, v_a_632_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0___boxed(lean_object* v_mkCmdStr_638_, lean_object* v_helpMsg_639_, lean_object* v_title_640_, lean_object* v_onlyGoal_641_, lean_object* v_onlyOne_642_, lean_object* v_params_643_, lean_object* v_a_644_, lean_object* v_a_645_){
_start:
{
uint8_t v_onlyGoal_boxed_646_; uint8_t v_onlyOne_boxed_647_; lean_object* v_res_648_; 
v_onlyGoal_boxed_646_ = lean_unbox(v_onlyGoal_641_);
v_onlyOne_boxed_647_ = lean_unbox(v_onlyOne_642_);
v_res_648_ = lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0(v_mkCmdStr_638_, v_helpMsg_639_, v_title_640_, v_onlyGoal_boxed_646_, v_onlyOne_boxed_647_, v_params_643_, v_a_644_);
lean_dec_ref(v_a_644_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel_rpc(lean_object* v_params_652_, lean_object* v_a_653_){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; uint8_t v___x_658_; uint8_t v___x_659_; lean_object* v___x_660_; 
v___x_655_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel_rpc___closed__0));
v___x_656_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel_rpc___closed__1));
v___x_657_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel_rpc___closed__2));
v___x_658_ = 1;
v___x_659_ = 0;
v___x_660_ = lp_mathlib_mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0(v___x_655_, v___x_656_, v___x_657_, v___x_658_, v___x_659_, v_params_652_, v_a_653_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CongrMSelectionPanel_rpc___boxed(lean_object* v_params_661_, lean_object* v_a_662_, lean_object* v_a_663_){
_start:
{
lean_object* v_res_664_; 
v_res_664_ = lp_mathlib_CongrMSelectionPanel_rpc(v_params_661_, v_a_662_);
lean_dec_ref(v_a_662_);
return v_res_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3(lean_object* v_00_u03b1_665_, lean_object* v_lctx_666_, lean_object* v_localInsts_667_, lean_object* v_x_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_, lean_object* v___y_672_){
_start:
{
lean_object* v___x_674_; 
v___x_674_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___redArg(v_lctx_666_, v_localInsts_667_, v_x_668_, v___y_669_, v___y_670_, v___y_671_, v___y_672_);
return v___x_674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3___boxed(lean_object* v_00_u03b1_675_, lean_object* v_lctx_676_, lean_object* v_localInsts_677_, lean_object* v_x_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_, lean_object* v___y_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_Lean_Meta_withLCtx___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__3(v_00_u03b1_675_, v_lctx_676_, v_localInsts_677_, v_x_678_, v___y_679_, v___y_680_, v___y_681_, v___y_682_);
lean_dec(v___y_682_);
lean_dec_ref(v___y_681_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1(lean_object* v_mainGoalName_685_, lean_object* v_errorMsg_686_, uint8_t v___y_687_, lean_object* v_as_688_, size_t v_sz_689_, size_t v_i_690_, lean_object* v_b_691_, lean_object* v___y_692_){
_start:
{
lean_object* v___x_694_; 
v___x_694_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___redArg(v_mainGoalName_685_, v_errorMsg_686_, v___y_687_, v_as_688_, v_sz_689_, v_i_690_, v_b_691_);
return v___x_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1___boxed(lean_object* v_mainGoalName_695_, lean_object* v_errorMsg_696_, lean_object* v___y_697_, lean_object* v_as_698_, lean_object* v_sz_699_, lean_object* v_i_700_, lean_object* v_b_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
uint8_t v___y_2695__boxed_704_; size_t v_sz_boxed_705_; size_t v_i_boxed_706_; lean_object* v_res_707_; 
v___y_2695__boxed_704_ = lean_unbox(v___y_697_);
v_sz_boxed_705_ = lean_unbox_usize(v_sz_699_);
lean_dec(v_sz_699_);
v_i_boxed_706_ = lean_unbox_usize(v_i_700_);
lean_dec(v_i_700_);
v_res_707_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00mkSelectionPanelRPC___at___00CongrMSelectionPanel_rpc_spec__0_spec__1(v_mainGoalName_695_, v_errorMsg_696_, v___y_2695__boxed_704_, v_as_698_, v_sz_boxed_705_, v_i_boxed_706_, v_b_701_, v___y_702_);
lean_dec_ref(v___y_702_);
lean_dec_ref(v_as_698_);
lean_dec(v_mainGoalName_695_);
return v_res_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_708_, uint64_t v_k_709_){
_start:
{
if (lean_obj_tag(v_t_708_) == 0)
{
lean_object* v_k_710_; lean_object* v_v_711_; lean_object* v_l_712_; lean_object* v_r_713_; uint64_t v___x_714_; uint8_t v___x_715_; 
v_k_710_ = lean_ctor_get(v_t_708_, 1);
v_v_711_ = lean_ctor_get(v_t_708_, 2);
v_l_712_ = lean_ctor_get(v_t_708_, 3);
v_r_713_ = lean_ctor_get(v_t_708_, 4);
v___x_714_ = lean_unbox_uint64(v_k_710_);
v___x_715_ = lean_uint64_dec_lt(v_k_709_, v___x_714_);
if (v___x_715_ == 0)
{
uint64_t v___x_716_; uint8_t v___x_717_; 
v___x_716_ = lean_unbox_uint64(v_k_710_);
v___x_717_ = lean_uint64_dec_eq(v_k_709_, v___x_716_);
if (v___x_717_ == 0)
{
v_t_708_ = v_r_713_;
goto _start;
}
else
{
lean_object* v___x_719_; 
lean_inc(v_v_711_);
v___x_719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_719_, 0, v_v_711_);
return v___x_719_;
}
}
else
{
v_t_708_ = v_l_712_;
goto _start;
}
}
else
{
lean_object* v___x_721_; 
v___x_721_ = lean_box(0);
return v___x_721_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_722_, lean_object* v_k_723_){
_start:
{
uint64_t v_k_boxed_724_; lean_object* v_res_725_; 
v_k_boxed_724_ = lean_unbox_uint64(v_k_723_);
lean_dec_ref(v_k_723_);
v_res_725_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_722_, v_k_boxed_724_);
lean_dec(v_t_722_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_726_, lean_object* v_x_727_){
_start:
{
lean_object* v___x_728_; 
v___x_728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_728_, 0, v_x_727_);
lean_ctor_set(v___x_728_, 1, v_expireTime_726_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(lean_object* v_val_729_, lean_object* v___f_730_, lean_object* v_x_731_, lean_object* v___y_732_){
_start:
{
if (lean_obj_tag(v_x_731_) == 0)
{
lean_object* v_a_734_; lean_object* v___x_736_; uint8_t v_isShared_737_; uint8_t v_isSharedCheck_741_; 
lean_dec_ref(v___f_730_);
v_a_734_ = lean_ctor_get(v_x_731_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v_x_731_);
if (v_isSharedCheck_741_ == 0)
{
v___x_736_ = v_x_731_;
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
else
{
lean_inc(v_a_734_);
lean_dec(v_x_731_);
v___x_736_ = lean_box(0);
v_isShared_737_ = v_isSharedCheck_741_;
goto v_resetjp_735_;
}
v_resetjp_735_:
{
lean_object* v___x_739_; 
if (v_isShared_737_ == 0)
{
lean_ctor_set_tag(v___x_736_, 1);
v___x_739_ = v___x_736_;
goto v_reusejp_738_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_a_734_);
v___x_739_ = v_reuseFailAlloc_740_;
goto v_reusejp_738_;
}
v_reusejp_738_:
{
return v___x_739_;
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_758_; 
v_a_742_ = lean_ctor_get(v_x_731_, 0);
v_isSharedCheck_758_ = !lean_is_exclusive(v_x_731_);
if (v_isSharedCheck_758_ == 0)
{
v___x_744_ = v_x_731_;
v_isShared_745_ = v_isSharedCheck_758_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v_x_731_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_758_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_746_; lean_object* v_objects_747_; lean_object* v_expireTime_748_; lean_object* v___f_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v_fst_752_; lean_object* v_snd_753_; lean_object* v___x_754_; lean_object* v___x_756_; 
v___x_746_ = lean_st_ref_take(v_val_729_);
v_objects_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc_ref(v_objects_747_);
v_expireTime_748_ = lean_ctor_get(v___x_746_, 1);
lean_inc(v_expireTime_748_);
lean_dec(v___x_746_);
v___f_749_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_749_, 0, v_expireTime_748_);
v___x_750_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_a_742_, v_objects_747_);
v___x_751_ = l_Prod_map___redArg(v___f_730_, v___f_749_, v___x_750_);
v_fst_752_ = lean_ctor_get(v___x_751_, 0);
lean_inc(v_fst_752_);
v_snd_753_ = lean_ctor_get(v___x_751_, 1);
lean_inc(v_snd_753_);
lean_dec_ref(v___x_751_);
v___x_754_ = lean_st_ref_set(v_val_729_, v_snd_753_);
if (v_isShared_745_ == 0)
{
lean_ctor_set_tag(v___x_744_, 0);
lean_ctor_set(v___x_744_, 0, v_fst_752_);
v___x_756_ = v___x_744_;
goto v_reusejp_755_;
}
else
{
lean_object* v_reuseFailAlloc_757_; 
v_reuseFailAlloc_757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_757_, 0, v_fst_752_);
v___x_756_ = v_reuseFailAlloc_757_;
goto v_reusejp_755_;
}
v_reusejp_755_:
{
return v___x_756_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_759_, lean_object* v___f_760_, lean_object* v_x_761_, lean_object* v___y_762_, lean_object* v___y_763_){
_start:
{
lean_object* v_res_764_; 
v_res_764_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2(v_val_759_, v___f_760_, v_x_761_, v___y_762_);
lean_dec_ref(v___y_762_);
lean_dec(v_val_759_);
return v_res_764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(lean_object* v_method_772_, lean_object* v_handler_773_, lean_object* v___f_774_, uint64_t v_seshId_775_, lean_object* v_j_776_, lean_object* v___y_777_){
_start:
{
lean_object* v_rpcSessions_779_; lean_object* v___x_780_; 
v_rpcSessions_779_ = lean_ctor_get(v___y_777_, 0);
v___x_780_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_779_, v_seshId_775_);
if (lean_obj_tag(v___x_780_) == 1)
{
lean_object* v_val_781_; lean_object* v___x_782_; lean_object* v_objects_783_; lean_object* v___x_784_; 
v_val_781_ = lean_ctor_get(v___x_780_, 0);
lean_inc(v_val_781_);
lean_dec_ref_known(v___x_780_, 1);
v___x_782_ = lean_st_ref_get(v_val_781_);
v_objects_783_ = lean_ctor_get(v___x_782_, 0);
lean_inc_ref(v_objects_783_);
lean_dec(v___x_782_);
lean_inc(v_j_776_);
v___x_784_ = lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(v_j_776_, v_objects_783_);
lean_dec_ref(v_objects_783_);
if (lean_obj_tag(v___x_784_) == 0)
{
lean_object* v_a_785_; lean_object* v___x_787_; uint8_t v_isShared_788_; uint8_t v_isSharedCheck_805_; 
lean_dec(v_val_781_);
lean_dec_ref(v___f_774_);
lean_dec_ref(v_handler_773_);
v_a_785_ = lean_ctor_get(v___x_784_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_784_);
if (v_isSharedCheck_805_ == 0)
{
v___x_787_ = v___x_784_;
v_isShared_788_ = v_isSharedCheck_805_;
goto v_resetjp_786_;
}
else
{
lean_inc(v_a_785_);
lean_dec(v___x_784_);
v___x_787_ = lean_box(0);
v_isShared_788_ = v_isSharedCheck_805_;
goto v_resetjp_786_;
}
v_resetjp_786_:
{
uint8_t v___x_789_; lean_object* v___x_790_; uint8_t v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v___x_801_; lean_object* v___x_803_; 
v___x_789_ = 3;
v___x_790_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_791_ = 1;
v___x_792_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_772_, v___x_791_);
v___x_793_ = lean_string_append(v___x_790_, v___x_792_);
lean_dec_ref(v___x_792_);
v___x_794_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_795_ = lean_string_append(v___x_793_, v___x_794_);
v___x_796_ = l_Lean_Json_compress(v_j_776_);
v___x_797_ = lean_string_append(v___x_795_, v___x_796_);
lean_dec_ref(v___x_796_);
v___x_798_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_799_ = lean_string_append(v___x_797_, v___x_798_);
v___x_800_ = lean_string_append(v___x_799_, v_a_785_);
lean_dec(v_a_785_);
v___x_801_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_801_, 0, v___x_800_);
lean_ctor_set_uint8(v___x_801_, sizeof(void*)*1, v___x_789_);
if (v_isShared_788_ == 0)
{
lean_ctor_set_tag(v___x_787_, 1);
lean_ctor_set(v___x_787_, 0, v___x_801_);
v___x_803_ = v___x_787_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v___x_801_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
else
{
lean_object* v_a_806_; lean_object* v___x_807_; 
lean_dec(v_j_776_);
lean_dec(v_method_772_);
v_a_806_ = lean_ctor_get(v___x_784_, 0);
lean_inc(v_a_806_);
lean_dec_ref_known(v___x_784_, 1);
lean_inc_ref(v___y_777_);
v___x_807_ = lean_apply_3(v_handler_773_, v_a_806_, v___y_777_, lean_box(0));
if (lean_obj_tag(v___x_807_) == 0)
{
lean_object* v_a_808_; lean_object* v___f_809_; lean_object* v___x_810_; 
v_a_808_ = lean_ctor_get(v___x_807_, 0);
lean_inc(v_a_808_);
lean_dec_ref_known(v___x_807_, 1);
v___f_809_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_809_, 0, v_val_781_);
lean_closure_set(v___f_809_, 1, v___f_774_);
v___x_810_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_808_, v___f_809_, v___y_777_);
return v___x_810_;
}
else
{
lean_object* v_a_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_818_; 
lean_dec(v_val_781_);
lean_dec_ref(v___f_774_);
v_a_811_ = lean_ctor_get(v___x_807_, 0);
v_isSharedCheck_818_ = !lean_is_exclusive(v___x_807_);
if (v_isSharedCheck_818_ == 0)
{
v___x_813_ = v___x_807_;
v_isShared_814_ = v_isSharedCheck_818_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_a_811_);
lean_dec(v___x_807_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_818_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v___x_816_; 
if (v_isShared_814_ == 0)
{
v___x_816_ = v___x_813_;
goto v_reusejp_815_;
}
else
{
lean_object* v_reuseFailAlloc_817_; 
v_reuseFailAlloc_817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_817_, 0, v_a_811_);
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
}
else
{
lean_object* v___x_819_; lean_object* v___x_820_; 
lean_dec(v___x_780_);
lean_dec(v_j_776_);
lean_dec_ref(v___f_774_);
lean_dec_ref(v_handler_773_);
lean_dec(v_method_772_);
v___x_819_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
return v___x_820_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_821_, lean_object* v_handler_822_, lean_object* v___f_823_, lean_object* v_seshId_824_, lean_object* v_j_825_, lean_object* v___y_826_, lean_object* v___y_827_){
_start:
{
uint64_t v_seshId_boxed_828_; lean_object* v_res_829_; 
v_seshId_boxed_828_ = lean_unbox_uint64(v_seshId_824_);
lean_dec_ref(v_seshId_824_);
v_res_829_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3(v_method_821_, v_handler_822_, v___f_823_, v_seshId_boxed_828_, v_j_825_, v___y_826_);
lean_dec_ref(v___y_826_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(lean_object* v___y_830_){
_start:
{
lean_inc(v___y_830_);
return v___y_830_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_831_){
_start:
{
lean_object* v_res_832_; 
v_res_832_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__0(v___y_831_);
lean_dec(v___y_831_);
return v_res_832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0(lean_object* v_method_834_, lean_object* v_handler_835_){
_start:
{
lean_object* v___f_836_; lean_object* v___f_837_; 
v___f_836_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___closed__0));
v___f_837_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_837_, 0, v_method_834_);
lean_closure_set(v___f_837_, 1, v_handler_835_);
lean_closure_set(v___f_837_, 2, v___f_836_);
return v___f_837_;
}
}
static lean_object* _init_lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4(void){
_start:
{
lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_844_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__3));
v___x_845_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__2));
v___x_846_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0(v___x_845_, v___x_844_);
return v___x_846_;
}
}
static lean_object* _init_lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped(void){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = lean_obj_once(&lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4, &lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4_once, _init_lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped___closed__4);
return v___x_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_848_, lean_object* v_t_849_, uint64_t v_k_850_){
_start:
{
lean_object* v___x_851_; 
v___x_851_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_849_, v_k_850_);
return v___x_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_852_, lean_object* v_t_853_, lean_object* v_k_854_){
_start:
{
uint64_t v_k_boxed_855_; lean_object* v_res_856_; 
v_k_boxed_855_ = lean_unbox_uint64(v_k_854_);
lean_dec_ref(v_k_854_);
v_res_856_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00CongrMSelectionPanel_rpc___rpc__wrapped_spec__0_spec__0(v_00_u03b4_852_, v_t_853_, v_k_boxed_855_);
lean_dec(v_t_853_);
return v_res_856_;
}
}
static uint64_t _init_lp_mathlib_CongrMSelectionPanel___closed__1(void){
_start:
{
lean_object* v___x_858_; uint64_t v___x_859_; 
v___x_858_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel___closed__0));
v___x_859_ = lean_string_hash(v___x_858_);
return v___x_859_;
}
}
static lean_object* _init_lp_mathlib_CongrMSelectionPanel___closed__2(void){
_start:
{
uint64_t v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_860_ = lean_uint64_once(&lp_mathlib_CongrMSelectionPanel___closed__1, &lp_mathlib_CongrMSelectionPanel___closed__1_once, _init_lp_mathlib_CongrMSelectionPanel___closed__1);
v___x_861_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel___closed__0));
v___x_862_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_862_, 0, v___x_861_);
lean_ctor_set_uint64(v___x_862_, sizeof(void*)*1, v___x_860_);
return v___x_862_;
}
}
static lean_object* _init_lp_mathlib_CongrMSelectionPanel___closed__4(void){
_start:
{
lean_object* v___x_864_; lean_object* v___x_865_; lean_object* v___x_866_; 
v___x_864_ = ((lean_object*)(lp_mathlib_CongrMSelectionPanel___closed__3));
v___x_865_ = lean_obj_once(&lp_mathlib_CongrMSelectionPanel___closed__2, &lp_mathlib_CongrMSelectionPanel___closed__2_once, _init_lp_mathlib_CongrMSelectionPanel___closed__2);
v___x_866_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_866_, 0, v___x_865_);
lean_ctor_set(v___x_866_, 1, v___x_864_);
return v___x_866_;
}
}
static lean_object* _init_lp_mathlib_CongrMSelectionPanel(void){
_start:
{
lean_object* v___x_867_; 
v___x_867_ = lean_obj_once(&lp_mathlib_CongrMSelectionPanel___closed__4, &lp_mathlib_CongrMSelectionPanel___closed__4_once, _init_lp_mathlib_CongrMSelectionPanel___closed__4);
return v___x_867_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; 
v___x_880_ = lean_box(0);
v___x_881_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_882_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_882_, 0, v___x_881_);
lean_ctor_set(v___x_882_, 1, v___x_880_);
return v___x_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg(){
_start:
{
lean_object* v___x_884_; lean_object* v___x_885_; 
v___x_884_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___closed__0);
v___x_885_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_885_, 0, v___x_884_);
return v___x_885_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg___boxed(lean_object* v___y_886_){
_start:
{
lean_object* v_res_887_; 
v_res_887_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg();
return v_res_887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0(lean_object* v_00_u03b1_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_){
_start:
{
lean_object* v___x_898_; 
v___x_898_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg();
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___boxed(lean_object* v_00_u03b1_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_){
_start:
{
lean_object* v_res_909_; 
v_res_909_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0(v_00_u03b1_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_);
lean_dec(v___y_907_);
lean_dec_ref(v___y_906_);
lean_dec(v___y_905_);
lean_dec_ref(v___y_904_);
lean_dec(v___y_903_);
lean_dec_ref(v___y_902_);
lean_dec(v___y_901_);
lean_dec_ref(v___y_900_);
return v_res_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___lam__0(lean_object* v___x_910_, lean_object* v___y_911_){
_start:
{
lean_object* v___x_912_; 
v___x_912_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_912_, 0, v___x_910_);
lean_ctor_set(v___x_912_, 1, v___y_911_);
return v___x_912_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1(lean_object* v_x_914_, lean_object* v_a_915_, lean_object* v_a_916_, lean_object* v_a_917_, lean_object* v_a_918_, lean_object* v_a_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_){
_start:
{
lean_object* v___x_924_; uint8_t v___x_925_; 
v___x_924_ = ((lean_object*)(lp_mathlib_tacticCongrm_x3f___closed__1));
lean_inc(v_x_914_);
v___x_925_ = l_Lean_Syntax_isOfKind(v_x_914_, v___x_924_);
if (v___x_925_ == 0)
{
lean_object* v___x_926_; 
lean_dec(v_x_914_);
v___x_926_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1_spec__0___redArg();
return v___x_926_;
}
else
{
lean_object* v_fileMap_927_; lean_object* v___x_928_; lean_object* v_stx_929_; uint8_t v___x_930_; lean_object* v___x_931_; 
v_fileMap_927_ = lean_ctor_get(v_a_921_, 1);
v___x_928_ = lean_unsigned_to_nat(0u);
v_stx_929_ = l_Lean_Syntax_getArg(v_x_914_, v___x_928_);
lean_dec(v_x_914_);
v___x_930_ = 0;
lean_inc_ref(v_fileMap_927_);
v___x_931_ = l_Lean_FileMap_lspRangeOfStx_x3f(v_fileMap_927_, v_stx_929_, v___x_930_);
if (lean_obj_tag(v___x_931_) == 1)
{
lean_object* v_val_932_; lean_object* v___x_933_; lean_object* v_toModule_934_; uint64_t v_javascriptHash_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___f_942_; lean_object* v___x_943_; 
v_val_932_ = lean_ctor_get(v___x_931_, 0);
lean_inc(v_val_932_);
lean_dec_ref_known(v___x_931_, 1);
v___x_933_ = lp_mathlib_CongrMSelectionPanel;
v_toModule_934_ = lean_ctor_get(v___x_933_, 0);
v_javascriptHash_935_ = lean_ctor_get_uint64(v_toModule_934_, sizeof(void*)*1);
v___x_936_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___closed__0));
v___x_937_ = l_Lean_Lsp_instToJsonRange_toJson(v_val_932_);
v___x_938_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_938_, 0, v___x_936_);
lean_ctor_set(v___x_938_, 1, v___x_937_);
v___x_939_ = lean_box(0);
v___x_940_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_940_, 0, v___x_938_);
lean_ctor_set(v___x_940_, 1, v___x_939_);
v___x_941_ = l_Lean_Json_mkObj(v___x_940_);
lean_dec_ref_known(v___x_940_, 2);
v___f_942_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___lam__0), 2, 1);
lean_closure_set(v___f_942_, 0, v___x_941_);
v___x_943_ = l_Lean_Widget_savePanelWidgetInfo(v_javascriptHash_935_, v___f_942_, v_stx_929_, v_a_921_, v_a_922_);
return v___x_943_;
}
else
{
lean_object* v___x_944_; lean_object* v___x_945_; 
lean_dec(v___x_931_);
lean_dec(v_stx_929_);
v___x_944_ = lean_box(0);
v___x_945_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_945_, 0, v___x_944_);
return v___x_945_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1___boxed(lean_object* v_x_946_, lean_object* v_a_947_, lean_object* v_a_948_, lean_object* v_a_949_, lean_object* v_a_950_, lean_object* v_a_951_, lean_object* v_a_952_, lean_object* v_a_953_, lean_object* v_a_954_, lean_object* v_a_955_){
_start:
{
lean_object* v_res_956_; 
v_res_956_ = lp_mathlib___aux__Mathlib__Tactic__Widget__CongrM______elabRules__tacticCongrm_x3f__1(v_x_946_, v_a_947_, v_a_948_, v_a_949_, v_a_950_, v_a_951_, v_a_952_, v_a_953_, v_a_954_);
lean_dec(v_a_954_);
lean_dec_ref(v_a_953_);
lean_dec(v_a_952_);
lean_dec_ref(v_a_951_);
lean_dec(v_a_950_);
lean_dec_ref(v_a_949_);
lean_dec(v_a_948_);
lean_dec_ref(v_a_947_);
return v_res_956_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped = _init_lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped();
lean_mark_persistent(lp_mathlib_CongrMSelectionPanel_rpc___rpc__wrapped);
lp_mathlib_CongrMSelectionPanel = _init_lp_mathlib_CongrMSelectionPanel();
lean_mark_persistent(lp_mathlib_CongrMSelectionPanel);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Widget_CongrM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Widget_CongrM(builtin);
}
#ifdef __cplusplus
}
#endif
