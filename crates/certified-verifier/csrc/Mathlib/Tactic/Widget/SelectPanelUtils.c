// Lean compiler output
// Module: Mathlib.Tactic.Widget.SelectPanelUtils
// Imports: public import Init public meta import Init public meta import Lean.Meta.ExprLens public meta import Mathlib.Tactic.Widget.SelectInsertParamsClass public import Mathlib.Tactic.Widget.SelectInsertParamsClass public import ProofWidgets.Component.MakeEditLink public import ProofWidgets.Data.Html
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
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_SubExpr_instToJsonGoalsLocation_toJson(lean_object*);
lean_object* l_Lean_Lsp_instToJsonPosition_toJson(lean_object*);
lean_object* l_Lean_Lsp_instToJsonRange_toJson(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Json_getObjValD(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instFromJsonPosition_fromJson(lean_object*);
lean_object* l_Lean_Json_pretty(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Lsp_instFromJsonRange_fromJson(lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_le(uint32_t, uint32_t);
lean_object* l_String_Slice_intercalate(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_sanitizeNames(lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson(lean_object*);
lean_object* l_Lean_Server_instRpcEncodableOfFromJsonOfToJson___redArg(lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_MakeEditLink;
lean_object* lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_withLCtx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_ReaderT_read___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_readDoc___redArg(lean_object*, lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_getGoalLocations___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_getGoalLocations___closed__0 = (const lean_object*)&lp_mathlib_getGoalLocations___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_getGoalLocations(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_getGoalLocations___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Lean.Expr"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__0_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "_private.Lean.Expr.0.Lean.Expr.updateMData!Impl"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__1 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mdata expected"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Invalid coordinate "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " for "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__6 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__6_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Lensing on types is not supported"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__8 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__8_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_insertMetaVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_insertMetaVar___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_insertMetaVar___closed__0 = (const lean_object*)&lp_mathlib_insertMetaVar___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00String_renameMetaVar_spec__1(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_renameMetaVar___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\?m."};
static const lean_object* lp_mathlib_String_renameMetaVar___closed__0 = (const lean_object*)&lp_mathlib_String_renameMetaVar___closed__0_value;
static const lean_string_object lp_mathlib_String_renameMetaVar___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_String_renameMetaVar___closed__1 = (const lean_object*)&lp_mathlib_String_renameMetaVar___closed__1_value;
static const lean_string_object lp_mathlib_String_renameMetaVar___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\?_"};
static const lean_object* lp_mathlib_String_renameMetaVar___closed__2 = (const lean_object*)&lp_mathlib_String_renameMetaVar___closed__2_value;
static lean_once_cell_t lp_mathlib_String_renameMetaVar___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_renameMetaVar___closed__3;
static lean_once_cell_t lp_mathlib_String_renameMetaVar___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_String_renameMetaVar___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_String_renameMetaVar(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_renameMetaVar___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3___boxed(lean_object*);
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__0 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__0_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__1 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__1_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__2 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__2_value;
static const lean_closure_object lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__3 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__3_value;
static const lean_ctor_object lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__0_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__1_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__2_value),((lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__3_value)}};
static const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__4 = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams = (const lean_object*)&lp_mathlib_instSelectInsertParamsClassSelectInsertParams___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "pos"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "goals"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "selectedLocations"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
static const lean_string_object lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "replaceRange"};
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
LEAN_EXPORT lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_(lean_object*);
static const lean_closure_object lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
LEAN_EXPORT const lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_ = (const lean_object*)&lp_mathlib_instFromJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__spec__0(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value;
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35____boxed(lean_object*);
static const lean_closure_object lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value;
LEAN_EXPORT const lean_object* lp_mathlib_instToJsonRpcEncodablePacket_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_ = (const lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "expected JSON array, got '"};
static const lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "'"};
static const lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__1 = (const lean_object*)&lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3(size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instRpcEncodableSelectInsertParams___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instRpcEncodableSelectInsertParams___closed__0 = (const lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__0_value;
static const lean_closure_object lp_mathlib_instRpcEncodableSelectInsertParams___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instRpcEncodableSelectInsertParams___closed__1 = (const lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__1_value;
static const lean_ctor_object lp_mathlib_instRpcEncodableSelectInsertParams___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__0_value),((lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__1_value)}};
static const lean_object* lp_mathlib_instRpcEncodableSelectInsertParams___closed__2 = (const lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_instRpcEncodableSelectInsertParams = (const lean_object*)&lp_mathlib_instRpcEncodableSelectInsertParams___closed__2_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instFromJsonMakeEditLinkProps_fromJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__0_value;
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_ProofWidgets_instToJsonMakeEditLinkProps_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__1 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1;
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__2 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__2_value;
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__3 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__0 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__1 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__2 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__3 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__4 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__4_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__5 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__5_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__5_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__6 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__6_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__6_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__7 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__7_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__8 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__8_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "ml1"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__9 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__9_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__10 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__10_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__3_value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__10_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__11 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__11_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__11_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__12 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__12_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "There is no goal to solve!"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__13 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__13_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__13_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__14 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__14_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__14_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__15 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__15_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0_value),((lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__15_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__16 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__16_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__17 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__17_value;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22;
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__23 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__23_value;
static const lean_closure_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__24 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__24_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " should be "};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__25 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__25_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "You should select only one sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__26 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__26_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__26_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__27 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__27_value;
static const lean_array_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__27_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__28 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__28_value;
static const lean_ctor_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0_value),((lean_object*)&lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__value),((lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__28_value)}};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__29 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__29_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "in the main goal or its context."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__30 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__30_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "in the main goal."};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__31 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__31_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "All selected sub-expressions"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__32 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__32_value;
static const lean_string_object lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "The selected sub-expression"};
static const lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__33 = (const lean_object*)&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__33_value;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___closed__0;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___closed__1;
static lean_once_cell_t lp_mathlib_mkSelectionPanelRPC___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0(lean_object* v_as_1_, size_t v_sz_2_, size_t v_i_3_, lean_object* v_b_4_){
_start:
{
lean_object* v_a_6_; uint8_t v___x_10_; 
v___x_10_ = lean_usize_dec_lt(v_i_3_, v_sz_2_);
if (v___x_10_ == 0)
{
return v_b_4_;
}
else
{
lean_object* v_a_11_; lean_object* v_loc_12_; 
v_a_11_ = lean_array_uget_borrowed(v_as_1_, v_i_3_);
v_loc_12_ = lean_ctor_get(v_a_11_, 1);
if (lean_obj_tag(v_loc_12_) == 3)
{
lean_object* v_a_13_; lean_object* v___x_14_; 
v_a_13_ = lean_ctor_get(v_loc_12_, 0);
lean_inc(v_a_13_);
v___x_14_ = lean_array_push(v_b_4_, v_a_13_);
v_a_6_ = v___x_14_;
goto v___jp_5_;
}
else
{
v_a_6_ = v_b_4_;
goto v___jp_5_;
}
}
v___jp_5_:
{
size_t v___x_7_; size_t v___x_8_; 
v___x_7_ = ((size_t)1ULL);
v___x_8_ = lean_usize_add(v_i_3_, v___x_7_);
v_i_3_ = v___x_8_;
v_b_4_ = v_a_6_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0___boxed(lean_object* v_as_15_, lean_object* v_sz_16_, lean_object* v_i_17_, lean_object* v_b_18_){
_start:
{
size_t v_sz_boxed_19_; size_t v_i_boxed_20_; lean_object* v_res_21_; 
v_sz_boxed_19_ = lean_unbox_usize(v_sz_16_);
lean_dec(v_sz_16_);
v_i_boxed_20_ = lean_unbox_usize(v_i_17_);
lean_dec(v_i_17_);
v_res_21_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0(v_as_15_, v_sz_boxed_19_, v_i_boxed_20_, v_b_18_);
lean_dec_ref(v_as_15_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_getGoalLocations(lean_object* v_locations_24_){
_start:
{
lean_object* v_res_25_; size_t v_sz_26_; size_t v___x_27_; lean_object* v___x_28_; 
v_res_25_ = ((lean_object*)(lp_mathlib_getGoalLocations___closed__0));
v_sz_26_ = lean_array_size(v_locations_24_);
v___x_27_ = ((size_t)0ULL);
v___x_28_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00getGoalLocations_spec__0(v_locations_24_, v_sz_26_, v___x_27_, v_res_25_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_getGoalLocations___boxed(lean_object* v_locations_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_getGoalLocations(v_locations_29_);
lean_dec_ref(v_locations_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___lam__0(lean_object* v_x_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_){
_start:
{
lean_object* v___x_37_; uint8_t v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_37_ = lean_box(0);
v___x_38_ = 1;
v___x_39_ = lean_box(0);
v___x_40_ = l_Lean_Meta_mkFreshExprMVar(v___x_37_, v___x_38_, v___x_39_, v___y_32_, v___y_33_, v___y_34_, v___y_35_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___lam__0___boxed(lean_object* v_x_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_insertMetaVar___lam__0(v_x_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_);
lean_dec(v___y_45_);
lean_dec_ref(v___y_44_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec_ref(v_x_41_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0(lean_object* v_body_48_, lean_object* v_g_49_, lean_object* v_x_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_expr_instantiate1(v_body_48_, v_x_50_);
lean_inc(v___y_54_);
lean_inc_ref(v___y_53_);
lean_inc(v___y_52_);
lean_inc_ref(v___y_51_);
v___x_57_ = lean_apply_6(v_g_49_, v___x_56_, v___y_51_, v___y_52_, v___y_53_, v___y_54_, lean_box(0));
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0___boxed(lean_object* v_body_58_, lean_object* v_g_59_, lean_object* v_x_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0(v_body_58_, v_g_59_, v_x_60_, v___y_61_, v___y_62_, v___y_63_, v___y_64_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
lean_dec(v___y_62_);
lean_dec_ref(v___y_61_);
lean_dec_ref(v_x_60_);
lean_dec_ref(v_body_58_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__3(lean_object* v_msg_67_){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = l_Lean_instInhabitedExpr;
v___x_69_ = lean_panic_fn_borrowed(v___x_68_, v_msg_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2(lean_object* v___x_70_, lean_object* v_body_71_, lean_object* v_g_72_, uint8_t v___x_73_, uint8_t v___x_74_, lean_object* v_x_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_81_ = lean_mk_empty_array_with_capacity(v___x_70_);
v___x_82_ = lean_array_push(v___x_81_, v_x_75_);
v___x_83_ = lean_expr_instantiate_rev(v_body_71_, v___x_82_);
lean_inc(v___y_79_);
lean_inc_ref(v___y_78_);
lean_inc(v___y_77_);
lean_inc_ref(v___y_76_);
v___x_84_ = lean_apply_6(v_g_72_, v___x_83_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, lean_box(0));
if (lean_obj_tag(v___x_84_) == 0)
{
lean_object* v_a_85_; uint8_t v___x_86_; lean_object* v___x_87_; 
v_a_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_a_85_);
lean_dec_ref_known(v___x_84_, 1);
v___x_86_ = 1;
v___x_87_ = l_Lean_Meta_mkForallFVars(v___x_82_, v_a_85_, v___x_73_, v___x_74_, v___x_74_, v___x_86_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec_ref(v___x_82_);
return v___x_87_;
}
else
{
lean_dec_ref(v___x_82_);
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2___boxed(lean_object* v___x_88_, lean_object* v_body_89_, lean_object* v_g_90_, lean_object* v___x_91_, lean_object* v___x_92_, lean_object* v_x_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
uint8_t v___x_4212__boxed_99_; uint8_t v___x_4213__boxed_100_; lean_object* v_res_101_; 
v___x_4212__boxed_99_ = lean_unbox(v___x_91_);
v___x_4213__boxed_100_ = lean_unbox(v___x_92_);
v_res_101_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2(v___x_88_, v_body_89_, v_g_90_, v___x_4212__boxed_99_, v___x_4213__boxed_100_, v_x_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
lean_dec_ref(v_body_89_);
lean_dec(v___x_88_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0(lean_object* v_k_102_, lean_object* v_b_103_, lean_object* v___y_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_){
_start:
{
lean_object* v___x_109_; 
lean_inc(v___y_107_);
lean_inc_ref(v___y_106_);
lean_inc(v___y_105_);
lean_inc_ref(v___y_104_);
v___x_109_ = lean_apply_6(v_k_102_, v_b_103_, v___y_104_, v___y_105_, v___y_106_, v___y_107_, lean_box(0));
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0___boxed(lean_object* v_k_110_, lean_object* v_b_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0(v_k_110_, v_b_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(lean_object* v_name_118_, uint8_t v_bi_119_, lean_object* v_type_120_, lean_object* v_k_121_, uint8_t v_kind_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_){
_start:
{
lean_object* v___f_128_; lean_object* v___x_129_; 
v___f_128_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_128_, 0, v_k_121_);
v___x_129_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_118_, v_bi_119_, v_type_120_, v___f_128_, v_kind_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_);
if (lean_obj_tag(v___x_129_) == 0)
{
lean_object* v_a_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_137_; 
v_a_130_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_137_ == 0)
{
v___x_132_ = v___x_129_;
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_a_130_);
lean_dec(v___x_129_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_135_; 
if (v_isShared_133_ == 0)
{
v___x_135_ = v___x_132_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_a_130_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_a_138_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_129_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_129_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___boxed(lean_object* v_name_146_, lean_object* v_bi_147_, lean_object* v_type_148_, lean_object* v_k_149_, lean_object* v_kind_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_){
_start:
{
uint8_t v_bi_boxed_156_; uint8_t v_kind_boxed_157_; lean_object* v_res_158_; 
v_bi_boxed_156_ = lean_unbox(v_bi_147_);
v_kind_boxed_157_ = lean_unbox(v_kind_150_);
v_res_158_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(v_name_146_, v_bi_boxed_156_, v_type_148_, v_k_149_, v_kind_boxed_157_, v___y_151_, v___y_152_, v___y_153_, v___y_154_);
lean_dec(v___y_154_);
lean_dec_ref(v___y_153_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1(lean_object* v___x_159_, lean_object* v_body_160_, lean_object* v_g_161_, uint8_t v___x_162_, uint8_t v___x_163_, lean_object* v_x_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v___x_170_ = lean_mk_empty_array_with_capacity(v___x_159_);
v___x_171_ = lean_array_push(v___x_170_, v_x_164_);
v___x_172_ = lean_expr_instantiate_rev(v_body_160_, v___x_171_);
lean_inc(v___y_168_);
lean_inc_ref(v___y_167_);
lean_inc(v___y_166_);
lean_inc_ref(v___y_165_);
v___x_173_ = lean_apply_6(v_g_161_, v___x_172_, v___y_165_, v___y_166_, v___y_167_, v___y_168_, lean_box(0));
if (lean_obj_tag(v___x_173_) == 0)
{
lean_object* v_a_174_; uint8_t v___x_175_; lean_object* v___x_176_; 
v_a_174_ = lean_ctor_get(v___x_173_, 0);
lean_inc(v_a_174_);
lean_dec_ref_known(v___x_173_, 1);
v___x_175_ = 1;
v___x_176_ = l_Lean_Meta_mkLambdaFVars(v___x_171_, v_a_174_, v___x_162_, v___x_163_, v___x_162_, v___x_163_, v___x_175_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
lean_dec_ref(v___x_171_);
return v___x_176_;
}
else
{
lean_dec_ref(v___x_171_);
return v___x_173_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1___boxed(lean_object* v___x_177_, lean_object* v_body_178_, lean_object* v_g_179_, lean_object* v___x_180_, lean_object* v___x_181_, lean_object* v_x_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_){
_start:
{
uint8_t v___x_4322__boxed_188_; uint8_t v___x_4323__boxed_189_; lean_object* v_res_190_; 
v___x_4322__boxed_188_ = lean_unbox(v___x_180_);
v___x_4323__boxed_189_ = lean_unbox(v___x_181_);
v_res_190_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1(v___x_177_, v_body_178_, v_g_179_, v___x_4322__boxed_188_, v___x_4323__boxed_189_, v_x_182_, v___y_183_, v___y_184_, v___y_185_, v___y_186_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec_ref(v_body_178_);
lean_dec(v___x_177_);
return v_res_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object* v_msgData_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_){
_start:
{
lean_object* v___x_197_; lean_object* v_env_198_; lean_object* v___x_199_; lean_object* v_mctx_200_; lean_object* v_lctx_201_; lean_object* v_options_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; 
v___x_197_ = lean_st_ref_get(v___y_195_);
v_env_198_ = lean_ctor_get(v___x_197_, 0);
lean_inc_ref(v_env_198_);
lean_dec(v___x_197_);
v___x_199_ = lean_st_ref_get(v___y_193_);
v_mctx_200_ = lean_ctor_get(v___x_199_, 0);
lean_inc_ref(v_mctx_200_);
lean_dec(v___x_199_);
v_lctx_201_ = lean_ctor_get(v___y_192_, 2);
v_options_202_ = lean_ctor_get(v___y_194_, 2);
lean_inc_ref(v_options_202_);
lean_inc_ref(v_lctx_201_);
v___x_203_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_203_, 0, v_env_198_);
lean_ctor_set(v___x_203_, 1, v_mctx_200_);
lean_ctor_set(v___x_203_, 2, v_lctx_201_);
lean_ctor_set(v___x_203_, 3, v_options_202_);
v___x_204_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_203_);
lean_ctor_set(v___x_204_, 1, v_msgData_191_);
v___x_205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_205_, 0, v___x_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3___boxed(lean_object* v_msgData_206_, lean_object* v___y_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v_res_212_; 
v_res_212_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3(v_msgData_206_, v___y_207_, v___y_208_, v___y_209_, v___y_210_);
lean_dec(v___y_210_);
lean_dec_ref(v___y_209_);
lean_dec(v___y_208_);
lean_dec_ref(v___y_207_);
return v_res_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_msg_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_){
_start:
{
lean_object* v_ref_219_; lean_object* v___x_220_; lean_object* v_a_221_; lean_object* v___x_223_; uint8_t v_isShared_224_; uint8_t v_isSharedCheck_229_; 
v_ref_219_ = lean_ctor_get(v___y_216_, 5);
v___x_220_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2_spec__3(v_msg_213_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
v_a_221_ = lean_ctor_get(v___x_220_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_220_);
if (v_isSharedCheck_229_ == 0)
{
v___x_223_ = v___x_220_;
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
else
{
lean_inc(v_a_221_);
lean_dec(v___x_220_);
v___x_223_ = lean_box(0);
v_isShared_224_ = v_isSharedCheck_229_;
goto v_resetjp_222_;
}
v_resetjp_222_:
{
lean_object* v___x_225_; lean_object* v___x_227_; 
lean_inc(v_ref_219_);
v___x_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_225_, 0, v_ref_219_);
lean_ctor_set(v___x_225_, 1, v_a_221_);
if (v_isShared_224_ == 0)
{
lean_ctor_set_tag(v___x_223_, 1);
lean_ctor_set(v___x_223_, 0, v___x_225_);
v___x_227_ = v___x_223_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v___x_225_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msg_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(v_msg_230_, v___y_231_, v___y_232_, v___y_233_, v___y_234_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
lean_dec(v___y_232_);
lean_dec_ref(v___y_231_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(lean_object* v_name_237_, lean_object* v_type_238_, lean_object* v_val_239_, lean_object* v_k_240_, uint8_t v_nondep_241_, uint8_t v_kind_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v___f_248_; lean_object* v___x_249_; 
v___f_248_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_248_, 0, v_k_240_);
v___x_249_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_237_, v_type_238_, v_val_239_, v___f_248_, v_nondep_241_, v_kind_242_, v___y_243_, v___y_244_, v___y_245_, v___y_246_);
if (lean_obj_tag(v___x_249_) == 0)
{
lean_object* v_a_250_; lean_object* v___x_252_; uint8_t v_isShared_253_; uint8_t v_isSharedCheck_257_; 
v_a_250_ = lean_ctor_get(v___x_249_, 0);
v_isSharedCheck_257_ = !lean_is_exclusive(v___x_249_);
if (v_isSharedCheck_257_ == 0)
{
v___x_252_ = v___x_249_;
v_isShared_253_ = v_isSharedCheck_257_;
goto v_resetjp_251_;
}
else
{
lean_inc(v_a_250_);
lean_dec(v___x_249_);
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
v_reuseFailAlloc_256_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_258_; lean_object* v___x_260_; uint8_t v_isShared_261_; uint8_t v_isSharedCheck_265_; 
v_a_258_ = lean_ctor_get(v___x_249_, 0);
v_isSharedCheck_265_ = !lean_is_exclusive(v___x_249_);
if (v_isSharedCheck_265_ == 0)
{
v___x_260_ = v___x_249_;
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
else
{
lean_inc(v_a_258_);
lean_dec(v___x_249_);
v___x_260_ = lean_box(0);
v_isShared_261_ = v_isSharedCheck_265_;
goto v_resetjp_259_;
}
v_resetjp_259_:
{
lean_object* v___x_263_; 
if (v_isShared_261_ == 0)
{
v___x_263_ = v___x_260_;
goto v_reusejp_262_;
}
else
{
lean_object* v_reuseFailAlloc_264_; 
v_reuseFailAlloc_264_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_264_, 0, v_a_258_);
v___x_263_ = v_reuseFailAlloc_264_;
goto v_reusejp_262_;
}
v_reusejp_262_:
{
return v___x_263_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg___boxed(lean_object* v_name_266_, lean_object* v_type_267_, lean_object* v_val_268_, lean_object* v_k_269_, lean_object* v_nondep_270_, lean_object* v_kind_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
uint8_t v_nondep_boxed_277_; uint8_t v_kind_boxed_278_; lean_object* v_res_279_; 
v_nondep_boxed_277_ = lean_unbox(v_nondep_270_);
v_kind_boxed_278_ = lean_unbox(v_kind_271_);
v_res_279_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_name_266_, v_type_267_, v_val_268_, v_k_269_, v_nondep_boxed_277_, v_kind_boxed_278_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
lean_dec(v___y_275_);
lean_dec_ref(v___y_274_);
lean_dec(v___y_273_);
lean_dec_ref(v___y_272_);
return v_res_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0(lean_object* v_k_280_, uint8_t v_usedLetOnly_281_, lean_object* v_x_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; 
lean_inc(v___y_286_);
lean_inc_ref(v___y_285_);
lean_inc(v___y_284_);
lean_inc_ref(v___y_283_);
lean_inc_ref(v_x_282_);
v___x_288_ = lean_apply_6(v_k_280_, v_x_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, lean_box(0));
if (lean_obj_tag(v___x_288_) == 0)
{
lean_object* v_a_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; uint8_t v___x_293_; uint8_t v___x_294_; lean_object* v___x_295_; 
v_a_289_ = lean_ctor_get(v___x_288_, 0);
lean_inc(v_a_289_);
lean_dec_ref_known(v___x_288_, 1);
v___x_290_ = lean_unsigned_to_nat(1u);
v___x_291_ = lean_mk_empty_array_with_capacity(v___x_290_);
v___x_292_ = lean_array_push(v___x_291_, v_x_282_);
v___x_293_ = 0;
v___x_294_ = 1;
v___x_295_ = l_Lean_Meta_mkLetFVars(v___x_292_, v_a_289_, v_usedLetOnly_281_, v___x_293_, v___x_294_, v___y_283_, v___y_284_, v___y_285_, v___y_286_);
lean_dec_ref(v___x_292_);
return v___x_295_;
}
else
{
lean_dec_ref(v_x_282_);
return v___x_288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0___boxed(lean_object* v_k_296_, lean_object* v_usedLetOnly_297_, lean_object* v_x_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
uint8_t v_usedLetOnly_boxed_304_; lean_object* v_res_305_; 
v_usedLetOnly_boxed_304_ = lean_unbox(v_usedLetOnly_297_);
v_res_305_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0(v_k_296_, v_usedLetOnly_boxed_304_, v_x_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
return v_res_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4(lean_object* v_name_306_, lean_object* v_type_307_, lean_object* v_val_308_, lean_object* v_k_309_, uint8_t v_nondep_310_, uint8_t v_kind_311_, uint8_t v_usedLetOnly_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v___x_318_; lean_object* v___f_319_; lean_object* v___x_320_; 
v___x_318_ = lean_box(v_usedLetOnly_312_);
v___f_319_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___lam__0___boxed), 8, 2);
lean_closure_set(v___f_319_, 0, v_k_309_);
lean_closure_set(v___f_319_, 1, v___x_318_);
v___x_320_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_name_306_, v_type_307_, v_val_308_, v___f_319_, v_nondep_310_, v_kind_311_, v___y_313_, v___y_314_, v___y_315_, v___y_316_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_name_321_, lean_object* v_type_322_, lean_object* v_val_323_, lean_object* v_k_324_, lean_object* v_nondep_325_, lean_object* v_kind_326_, lean_object* v_usedLetOnly_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_){
_start:
{
uint8_t v_nondep_boxed_333_; uint8_t v_kind_boxed_334_; uint8_t v_usedLetOnly_boxed_335_; lean_object* v_res_336_; 
v_nondep_boxed_333_ = lean_unbox(v_nondep_325_);
v_kind_boxed_334_ = lean_unbox(v_kind_326_);
v_usedLetOnly_boxed_335_ = lean_unbox(v_usedLetOnly_327_);
v_res_336_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4(v_name_321_, v_type_322_, v_val_323_, v_k_324_, v_nondep_boxed_333_, v_kind_boxed_334_, v_usedLetOnly_boxed_335_, v___y_328_, v___y_329_, v___y_330_, v___y_331_);
lean_dec(v___y_331_);
lean_dec_ref(v___y_330_);
lean_dec(v___y_329_);
lean_dec_ref(v___y_328_);
return v_res_336_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_340_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__2));
v___x_341_ = lean_unsigned_to_nat(17u);
v___x_342_ = lean_unsigned_to_nat(1885u);
v___x_343_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__1));
v___x_344_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__0));
v___x_345_ = l_mkPanicMessageWithDecl(v___x_344_, v___x_343_, v___x_342_, v___x_341_, v___x_340_);
return v___x_345_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_347_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__4));
v___x_348_ = l_Lean_stringToMessageData(v___x_347_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; 
v___x_350_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__6));
v___x_351_ = l_Lean_stringToMessageData(v___x_350_);
return v___x_351_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9(void){
_start:
{
lean_object* v___x_353_; lean_object* v___x_354_; 
v___x_353_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__8));
v___x_354_ = l_Lean_stringToMessageData(v___x_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1(lean_object* v_g_355_, lean_object* v_n_356_, lean_object* v_e_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_){
_start:
{
lean_object* v_n_364_; lean_object* v_a_365_; lean_object* v_c_395_; lean_object* v_e_396_; lean_object* v___x_407_; uint8_t v___x_408_; 
v___x_407_ = lean_unsigned_to_nat(0u);
v___x_408_ = lean_nat_dec_eq(v_n_356_, v___x_407_);
if (v___x_408_ == 0)
{
lean_object* v___x_409_; uint8_t v___x_410_; 
v___x_409_ = lean_unsigned_to_nat(1u);
v___x_410_ = lean_nat_dec_eq(v_n_356_, v___x_409_);
if (v___x_410_ == 0)
{
lean_object* v___x_411_; uint8_t v___x_412_; 
v___x_411_ = lean_unsigned_to_nat(2u);
v___x_412_ = lean_nat_dec_eq(v_n_356_, v___x_411_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; uint8_t v___x_414_; 
v___x_413_ = lean_unsigned_to_nat(3u);
v___x_414_ = lean_nat_dec_eq(v_n_356_, v___x_413_);
if (v___x_414_ == 0)
{
if (lean_obj_tag(v_e_357_) == 10)
{
lean_object* v_expr_415_; 
v_expr_415_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_expr_415_);
v_n_364_ = v_n_356_;
v_a_365_ = v_expr_415_;
goto v___jp_363_;
}
else
{
lean_dec_ref(v_g_355_);
v_c_395_ = v_n_356_;
v_e_396_ = v_e_357_;
goto v___jp_394_;
}
}
else
{
lean_dec(v_n_356_);
if (lean_obj_tag(v_e_357_) == 10)
{
lean_object* v_expr_416_; 
v_expr_416_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_expr_416_);
v_n_364_ = v___x_413_;
v_a_365_ = v_expr_416_;
goto v___jp_363_;
}
else
{
lean_object* v___x_417_; lean_object* v___x_418_; 
lean_dec_ref(v_e_357_);
lean_dec_ref(v_g_355_);
v___x_417_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__9);
v___x_418_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(v___x_417_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
return v___x_418_;
}
}
}
else
{
lean_dec(v_n_356_);
switch(lean_obj_tag(v_e_357_))
{
case 8:
{
lean_object* v_declName_419_; lean_object* v_type_420_; lean_object* v_value_421_; lean_object* v_body_422_; uint8_t v_nondep_423_; lean_object* v___f_424_; uint8_t v___x_425_; lean_object* v___x_426_; 
v_declName_419_ = lean_ctor_get(v_e_357_, 0);
lean_inc(v_declName_419_);
v_type_420_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_type_420_);
v_value_421_ = lean_ctor_get(v_e_357_, 2);
lean_inc_ref(v_value_421_);
v_body_422_ = lean_ctor_get(v_e_357_, 3);
lean_inc_ref(v_body_422_);
v_nondep_423_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_357_, 4);
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__0___boxed), 8, 2);
lean_closure_set(v___f_424_, 0, v_body_422_);
lean_closure_set(v___f_424_, 1, v_g_355_);
v___x_425_ = 0;
v___x_426_ = lp_mathlib_Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4(v_declName_419_, v_type_420_, v_value_421_, v___f_424_, v_nondep_423_, v___x_425_, v___x_410_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
return v___x_426_;
}
case 10:
{
lean_object* v_expr_427_; 
v_expr_427_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_expr_427_);
v_n_364_ = v___x_411_;
v_a_365_ = v_expr_427_;
goto v___jp_363_;
}
default: 
{
lean_dec_ref(v_g_355_);
v_c_395_ = v___x_411_;
v_e_396_ = v_e_357_;
goto v___jp_394_;
}
}
}
}
else
{
lean_dec(v_n_356_);
switch(lean_obj_tag(v_e_357_))
{
case 5:
{
lean_object* v_fn_428_; lean_object* v_arg_429_; lean_object* v___x_430_; 
v_fn_428_ = lean_ctor_get(v_e_357_, 0);
v_arg_429_ = lean_ctor_get(v_e_357_, 1);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_arg_429_);
v___x_430_ = lean_apply_6(v_g_355_, v_arg_429_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_449_; 
v_a_431_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_449_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_449_ == 0)
{
v___x_433_ = v___x_430_;
v_isShared_434_ = v_isSharedCheck_449_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_430_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_449_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
uint8_t v___y_436_; size_t v___x_444_; uint8_t v___x_445_; 
v___x_444_ = lean_ptr_addr(v_fn_428_);
v___x_445_ = lean_usize_dec_eq(v___x_444_, v___x_444_);
if (v___x_445_ == 0)
{
v___y_436_ = v___x_445_;
goto v___jp_435_;
}
else
{
size_t v___x_446_; size_t v___x_447_; uint8_t v___x_448_; 
v___x_446_ = lean_ptr_addr(v_arg_429_);
v___x_447_ = lean_ptr_addr(v_a_431_);
v___x_448_ = lean_usize_dec_eq(v___x_446_, v___x_447_);
v___y_436_ = v___x_448_;
goto v___jp_435_;
}
v___jp_435_:
{
if (v___y_436_ == 0)
{
lean_object* v___x_437_; lean_object* v___x_439_; 
lean_inc_ref(v_fn_428_);
lean_dec_ref_known(v_e_357_, 2);
v___x_437_ = l_Lean_Expr_app___override(v_fn_428_, v_a_431_);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 0, v___x_437_);
v___x_439_ = v___x_433_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v___x_437_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
else
{
lean_object* v___x_442_; 
lean_dec(v_a_431_);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 0, v_e_357_);
v___x_442_ = v___x_433_;
goto v_reusejp_441_;
}
else
{
lean_object* v_reuseFailAlloc_443_; 
v_reuseFailAlloc_443_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_443_, 0, v_e_357_);
v___x_442_ = v_reuseFailAlloc_443_;
goto v_reusejp_441_;
}
v_reusejp_441_:
{
return v___x_442_;
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_357_, 2);
return v___x_430_;
}
}
case 6:
{
lean_object* v_binderName_450_; lean_object* v_binderType_451_; lean_object* v_body_452_; uint8_t v_binderInfo_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___f_456_; uint8_t v___x_457_; lean_object* v___x_458_; 
v_binderName_450_ = lean_ctor_get(v_e_357_, 0);
lean_inc(v_binderName_450_);
v_binderType_451_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_binderType_451_);
v_body_452_ = lean_ctor_get(v_e_357_, 2);
lean_inc_ref(v_body_452_);
v_binderInfo_453_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_357_, 3);
v___x_454_ = lean_box(v___x_408_);
v___x_455_ = lean_box(v___x_410_);
v___f_456_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__1___boxed), 11, 5);
lean_closure_set(v___f_456_, 0, v___x_409_);
lean_closure_set(v___f_456_, 1, v_body_452_);
lean_closure_set(v___f_456_, 2, v_g_355_);
lean_closure_set(v___f_456_, 3, v___x_454_);
lean_closure_set(v___f_456_, 4, v___x_455_);
v___x_457_ = 0;
v___x_458_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(v_binderName_450_, v_binderInfo_453_, v_binderType_451_, v___f_456_, v___x_457_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
return v___x_458_;
}
case 7:
{
lean_object* v_binderName_459_; lean_object* v_binderType_460_; lean_object* v_body_461_; uint8_t v_binderInfo_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___f_465_; uint8_t v___x_466_; lean_object* v___x_467_; 
v_binderName_459_ = lean_ctor_get(v_e_357_, 0);
lean_inc(v_binderName_459_);
v_binderType_460_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_binderType_460_);
v_body_461_ = lean_ctor_get(v_e_357_, 2);
lean_inc_ref(v_body_461_);
v_binderInfo_462_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_357_, 3);
v___x_463_ = lean_box(v___x_408_);
v___x_464_ = lean_box(v___x_410_);
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___lam__2___boxed), 11, 5);
lean_closure_set(v___f_465_, 0, v___x_409_);
lean_closure_set(v___f_465_, 1, v_body_461_);
lean_closure_set(v___f_465_, 2, v_g_355_);
lean_closure_set(v___f_465_, 3, v___x_463_);
lean_closure_set(v___f_465_, 4, v___x_464_);
v___x_466_ = 0;
v___x_467_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(v_binderName_459_, v_binderInfo_462_, v_binderType_460_, v___f_465_, v___x_466_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
return v___x_467_;
}
case 8:
{
lean_object* v_declName_468_; lean_object* v_type_469_; lean_object* v_value_470_; lean_object* v_body_471_; uint8_t v_nondep_472_; lean_object* v___x_473_; 
v_declName_468_ = lean_ctor_get(v_e_357_, 0);
v_type_469_ = lean_ctor_get(v_e_357_, 1);
v_value_470_ = lean_ctor_get(v_e_357_, 2);
v_body_471_ = lean_ctor_get(v_e_357_, 3);
v_nondep_472_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*4 + 8);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_value_470_);
v___x_473_ = lean_apply_6(v_g_355_, v_value_470_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_473_) == 0)
{
lean_object* v_a_474_; lean_object* v___x_476_; uint8_t v_isShared_477_; uint8_t v_isSharedCheck_498_; 
v_a_474_ = lean_ctor_get(v___x_473_, 0);
v_isSharedCheck_498_ = !lean_is_exclusive(v___x_473_);
if (v_isSharedCheck_498_ == 0)
{
v___x_476_ = v___x_473_;
v_isShared_477_ = v_isSharedCheck_498_;
goto v_resetjp_475_;
}
else
{
lean_inc(v_a_474_);
lean_dec(v___x_473_);
v___x_476_ = lean_box(0);
v_isShared_477_ = v_isSharedCheck_498_;
goto v_resetjp_475_;
}
v_resetjp_475_:
{
uint8_t v___y_479_; size_t v___x_493_; uint8_t v___x_494_; 
v___x_493_ = lean_ptr_addr(v_type_469_);
v___x_494_ = lean_usize_dec_eq(v___x_493_, v___x_493_);
if (v___x_494_ == 0)
{
v___y_479_ = v___x_494_;
goto v___jp_478_;
}
else
{
size_t v___x_495_; size_t v___x_496_; uint8_t v___x_497_; 
v___x_495_ = lean_ptr_addr(v_value_470_);
v___x_496_ = lean_ptr_addr(v_a_474_);
v___x_497_ = lean_usize_dec_eq(v___x_495_, v___x_496_);
v___y_479_ = v___x_497_;
goto v___jp_478_;
}
v___jp_478_:
{
if (v___y_479_ == 0)
{
lean_object* v___x_480_; lean_object* v___x_482_; 
lean_inc_ref(v_body_471_);
lean_inc_ref(v_type_469_);
lean_inc(v_declName_468_);
lean_dec_ref_known(v_e_357_, 4);
v___x_480_ = l_Lean_Expr_letE___override(v_declName_468_, v_type_469_, v_a_474_, v_body_471_, v_nondep_472_);
if (v_isShared_477_ == 0)
{
lean_ctor_set(v___x_476_, 0, v___x_480_);
v___x_482_ = v___x_476_;
goto v_reusejp_481_;
}
else
{
lean_object* v_reuseFailAlloc_483_; 
v_reuseFailAlloc_483_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_483_, 0, v___x_480_);
v___x_482_ = v_reuseFailAlloc_483_;
goto v_reusejp_481_;
}
v_reusejp_481_:
{
return v___x_482_;
}
}
else
{
size_t v___x_484_; uint8_t v___x_485_; 
v___x_484_ = lean_ptr_addr(v_body_471_);
v___x_485_ = lean_usize_dec_eq(v___x_484_, v___x_484_);
if (v___x_485_ == 0)
{
lean_object* v___x_486_; lean_object* v___x_488_; 
lean_inc_ref(v_body_471_);
lean_inc_ref(v_type_469_);
lean_inc(v_declName_468_);
lean_dec_ref_known(v_e_357_, 4);
v___x_486_ = l_Lean_Expr_letE___override(v_declName_468_, v_type_469_, v_a_474_, v_body_471_, v_nondep_472_);
if (v_isShared_477_ == 0)
{
lean_ctor_set(v___x_476_, 0, v___x_486_);
v___x_488_ = v___x_476_;
goto v_reusejp_487_;
}
else
{
lean_object* v_reuseFailAlloc_489_; 
v_reuseFailAlloc_489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_489_, 0, v___x_486_);
v___x_488_ = v_reuseFailAlloc_489_;
goto v_reusejp_487_;
}
v_reusejp_487_:
{
return v___x_488_;
}
}
else
{
lean_object* v___x_491_; 
lean_dec(v_a_474_);
if (v_isShared_477_ == 0)
{
lean_ctor_set(v___x_476_, 0, v_e_357_);
v___x_491_ = v___x_476_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_492_; 
v_reuseFailAlloc_492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_492_, 0, v_e_357_);
v___x_491_ = v_reuseFailAlloc_492_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
return v___x_491_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_357_, 4);
return v___x_473_;
}
}
case 10:
{
lean_object* v_expr_499_; 
v_expr_499_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_expr_499_);
v_n_364_ = v___x_409_;
v_a_365_ = v_expr_499_;
goto v___jp_363_;
}
default: 
{
lean_dec_ref(v_g_355_);
v_c_395_ = v___x_409_;
v_e_396_ = v_e_357_;
goto v___jp_394_;
}
}
}
}
else
{
lean_dec(v_n_356_);
switch(lean_obj_tag(v_e_357_))
{
case 5:
{
lean_object* v_fn_500_; lean_object* v_arg_501_; lean_object* v___x_502_; 
v_fn_500_ = lean_ctor_get(v_e_357_, 0);
v_arg_501_ = lean_ctor_get(v_e_357_, 1);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_fn_500_);
v___x_502_ = lean_apply_6(v_g_355_, v_fn_500_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_502_) == 0)
{
lean_object* v_a_503_; lean_object* v___x_505_; uint8_t v_isShared_506_; uint8_t v_isSharedCheck_521_; 
v_a_503_ = lean_ctor_get(v___x_502_, 0);
v_isSharedCheck_521_ = !lean_is_exclusive(v___x_502_);
if (v_isSharedCheck_521_ == 0)
{
v___x_505_ = v___x_502_;
v_isShared_506_ = v_isSharedCheck_521_;
goto v_resetjp_504_;
}
else
{
lean_inc(v_a_503_);
lean_dec(v___x_502_);
v___x_505_ = lean_box(0);
v_isShared_506_ = v_isSharedCheck_521_;
goto v_resetjp_504_;
}
v_resetjp_504_:
{
uint8_t v___y_508_; size_t v___x_516_; size_t v___x_517_; uint8_t v___x_518_; 
v___x_516_ = lean_ptr_addr(v_fn_500_);
v___x_517_ = lean_ptr_addr(v_a_503_);
v___x_518_ = lean_usize_dec_eq(v___x_516_, v___x_517_);
if (v___x_518_ == 0)
{
v___y_508_ = v___x_518_;
goto v___jp_507_;
}
else
{
size_t v___x_519_; uint8_t v___x_520_; 
v___x_519_ = lean_ptr_addr(v_arg_501_);
v___x_520_ = lean_usize_dec_eq(v___x_519_, v___x_519_);
v___y_508_ = v___x_520_;
goto v___jp_507_;
}
v___jp_507_:
{
if (v___y_508_ == 0)
{
lean_object* v___x_509_; lean_object* v___x_511_; 
lean_inc_ref(v_arg_501_);
lean_dec_ref_known(v_e_357_, 2);
v___x_509_ = l_Lean_Expr_app___override(v_a_503_, v_arg_501_);
if (v_isShared_506_ == 0)
{
lean_ctor_set(v___x_505_, 0, v___x_509_);
v___x_511_ = v___x_505_;
goto v_reusejp_510_;
}
else
{
lean_object* v_reuseFailAlloc_512_; 
v_reuseFailAlloc_512_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_512_, 0, v___x_509_);
v___x_511_ = v_reuseFailAlloc_512_;
goto v_reusejp_510_;
}
v_reusejp_510_:
{
return v___x_511_;
}
}
else
{
lean_object* v___x_514_; 
lean_dec(v_a_503_);
if (v_isShared_506_ == 0)
{
lean_ctor_set(v___x_505_, 0, v_e_357_);
v___x_514_ = v___x_505_;
goto v_reusejp_513_;
}
else
{
lean_object* v_reuseFailAlloc_515_; 
v_reuseFailAlloc_515_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_515_, 0, v_e_357_);
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
lean_dec_ref_known(v_e_357_, 2);
return v___x_502_;
}
}
case 6:
{
lean_object* v_binderName_522_; lean_object* v_binderType_523_; lean_object* v_body_524_; uint8_t v_binderInfo_525_; lean_object* v___x_526_; 
v_binderName_522_ = lean_ctor_get(v_e_357_, 0);
v_binderType_523_ = lean_ctor_get(v_e_357_, 1);
v_body_524_ = lean_ctor_get(v_e_357_, 2);
v_binderInfo_525_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*3 + 8);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_binderType_523_);
v___x_526_ = lean_apply_6(v_g_355_, v_binderType_523_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_526_) == 0)
{
lean_object* v_a_527_; lean_object* v___x_529_; uint8_t v_isShared_530_; uint8_t v_isSharedCheck_550_; 
v_a_527_ = lean_ctor_get(v___x_526_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_526_);
if (v_isSharedCheck_550_ == 0)
{
v___x_529_ = v___x_526_;
v_isShared_530_ = v_isSharedCheck_550_;
goto v_resetjp_528_;
}
else
{
lean_inc(v_a_527_);
lean_dec(v___x_526_);
v___x_529_ = lean_box(0);
v_isShared_530_ = v_isSharedCheck_550_;
goto v_resetjp_528_;
}
v_resetjp_528_:
{
uint8_t v___y_532_; size_t v___x_545_; size_t v___x_546_; uint8_t v___x_547_; 
v___x_545_ = lean_ptr_addr(v_binderType_523_);
v___x_546_ = lean_ptr_addr(v_a_527_);
v___x_547_ = lean_usize_dec_eq(v___x_545_, v___x_546_);
if (v___x_547_ == 0)
{
v___y_532_ = v___x_547_;
goto v___jp_531_;
}
else
{
size_t v___x_548_; uint8_t v___x_549_; 
v___x_548_ = lean_ptr_addr(v_body_524_);
v___x_549_ = lean_usize_dec_eq(v___x_548_, v___x_548_);
v___y_532_ = v___x_549_;
goto v___jp_531_;
}
v___jp_531_:
{
if (v___y_532_ == 0)
{
lean_object* v___x_533_; lean_object* v___x_535_; 
lean_inc_ref(v_body_524_);
lean_inc(v_binderName_522_);
lean_dec_ref_known(v_e_357_, 3);
v___x_533_ = l_Lean_Expr_lam___override(v_binderName_522_, v_a_527_, v_body_524_, v_binderInfo_525_);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v___x_533_);
v___x_535_ = v___x_529_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_536_; 
v_reuseFailAlloc_536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_536_, 0, v___x_533_);
v___x_535_ = v_reuseFailAlloc_536_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
return v___x_535_;
}
}
else
{
uint8_t v___x_537_; 
v___x_537_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_525_, v_binderInfo_525_);
if (v___x_537_ == 0)
{
lean_object* v___x_538_; lean_object* v___x_540_; 
lean_inc_ref(v_body_524_);
lean_inc(v_binderName_522_);
lean_dec_ref_known(v_e_357_, 3);
v___x_538_ = l_Lean_Expr_lam___override(v_binderName_522_, v_a_527_, v_body_524_, v_binderInfo_525_);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v___x_538_);
v___x_540_ = v___x_529_;
goto v_reusejp_539_;
}
else
{
lean_object* v_reuseFailAlloc_541_; 
v_reuseFailAlloc_541_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_541_, 0, v___x_538_);
v___x_540_ = v_reuseFailAlloc_541_;
goto v_reusejp_539_;
}
v_reusejp_539_:
{
return v___x_540_;
}
}
else
{
lean_object* v___x_543_; 
lean_dec(v_a_527_);
if (v_isShared_530_ == 0)
{
lean_ctor_set(v___x_529_, 0, v_e_357_);
v___x_543_ = v___x_529_;
goto v_reusejp_542_;
}
else
{
lean_object* v_reuseFailAlloc_544_; 
v_reuseFailAlloc_544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_544_, 0, v_e_357_);
v___x_543_ = v_reuseFailAlloc_544_;
goto v_reusejp_542_;
}
v_reusejp_542_:
{
return v___x_543_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_357_, 3);
return v___x_526_;
}
}
case 7:
{
lean_object* v_binderName_551_; lean_object* v_binderType_552_; lean_object* v_body_553_; uint8_t v_binderInfo_554_; lean_object* v___x_555_; 
v_binderName_551_ = lean_ctor_get(v_e_357_, 0);
v_binderType_552_ = lean_ctor_get(v_e_357_, 1);
v_body_553_ = lean_ctor_get(v_e_357_, 2);
v_binderInfo_554_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*3 + 8);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_binderType_552_);
v___x_555_ = lean_apply_6(v_g_355_, v_binderType_552_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_555_) == 0)
{
lean_object* v_a_556_; lean_object* v___x_558_; uint8_t v_isShared_559_; uint8_t v_isSharedCheck_579_; 
v_a_556_ = lean_ctor_get(v___x_555_, 0);
v_isSharedCheck_579_ = !lean_is_exclusive(v___x_555_);
if (v_isSharedCheck_579_ == 0)
{
v___x_558_ = v___x_555_;
v_isShared_559_ = v_isSharedCheck_579_;
goto v_resetjp_557_;
}
else
{
lean_inc(v_a_556_);
lean_dec(v___x_555_);
v___x_558_ = lean_box(0);
v_isShared_559_ = v_isSharedCheck_579_;
goto v_resetjp_557_;
}
v_resetjp_557_:
{
uint8_t v___y_561_; size_t v___x_574_; size_t v___x_575_; uint8_t v___x_576_; 
v___x_574_ = lean_ptr_addr(v_binderType_552_);
v___x_575_ = lean_ptr_addr(v_a_556_);
v___x_576_ = lean_usize_dec_eq(v___x_574_, v___x_575_);
if (v___x_576_ == 0)
{
v___y_561_ = v___x_576_;
goto v___jp_560_;
}
else
{
size_t v___x_577_; uint8_t v___x_578_; 
v___x_577_ = lean_ptr_addr(v_body_553_);
v___x_578_ = lean_usize_dec_eq(v___x_577_, v___x_577_);
v___y_561_ = v___x_578_;
goto v___jp_560_;
}
v___jp_560_:
{
if (v___y_561_ == 0)
{
lean_object* v___x_562_; lean_object* v___x_564_; 
lean_inc_ref(v_body_553_);
lean_inc(v_binderName_551_);
lean_dec_ref_known(v_e_357_, 3);
v___x_562_ = l_Lean_Expr_forallE___override(v_binderName_551_, v_a_556_, v_body_553_, v_binderInfo_554_);
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 0, v___x_562_);
v___x_564_ = v___x_558_;
goto v_reusejp_563_;
}
else
{
lean_object* v_reuseFailAlloc_565_; 
v_reuseFailAlloc_565_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_565_, 0, v___x_562_);
v___x_564_ = v_reuseFailAlloc_565_;
goto v_reusejp_563_;
}
v_reusejp_563_:
{
return v___x_564_;
}
}
else
{
uint8_t v___x_566_; 
v___x_566_ = l_Lean_instBEqBinderInfo_beq(v_binderInfo_554_, v_binderInfo_554_);
if (v___x_566_ == 0)
{
lean_object* v___x_567_; lean_object* v___x_569_; 
lean_inc_ref(v_body_553_);
lean_inc(v_binderName_551_);
lean_dec_ref_known(v_e_357_, 3);
v___x_567_ = l_Lean_Expr_forallE___override(v_binderName_551_, v_a_556_, v_body_553_, v_binderInfo_554_);
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 0, v___x_567_);
v___x_569_ = v___x_558_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
else
{
lean_object* v___x_572_; 
lean_dec(v_a_556_);
if (v_isShared_559_ == 0)
{
lean_ctor_set(v___x_558_, 0, v_e_357_);
v___x_572_ = v___x_558_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_573_; 
v_reuseFailAlloc_573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_573_, 0, v_e_357_);
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
}
}
else
{
lean_dec_ref_known(v_e_357_, 3);
return v___x_555_;
}
}
case 8:
{
lean_object* v_declName_580_; lean_object* v_type_581_; lean_object* v_value_582_; lean_object* v_body_583_; uint8_t v_nondep_584_; lean_object* v___x_585_; 
v_declName_580_ = lean_ctor_get(v_e_357_, 0);
v_type_581_ = lean_ctor_get(v_e_357_, 1);
v_value_582_ = lean_ctor_get(v_e_357_, 2);
v_body_583_ = lean_ctor_get(v_e_357_, 3);
v_nondep_584_ = lean_ctor_get_uint8(v_e_357_, sizeof(void*)*4 + 8);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_type_581_);
v___x_585_ = lean_apply_6(v_g_355_, v_type_581_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_585_) == 0)
{
lean_object* v_a_586_; lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_610_; 
v_a_586_ = lean_ctor_get(v___x_585_, 0);
v_isSharedCheck_610_ = !lean_is_exclusive(v___x_585_);
if (v_isSharedCheck_610_ == 0)
{
v___x_588_ = v___x_585_;
v_isShared_589_ = v_isSharedCheck_610_;
goto v_resetjp_587_;
}
else
{
lean_inc(v_a_586_);
lean_dec(v___x_585_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_610_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
uint8_t v___y_591_; size_t v___x_605_; size_t v___x_606_; uint8_t v___x_607_; 
v___x_605_ = lean_ptr_addr(v_type_581_);
v___x_606_ = lean_ptr_addr(v_a_586_);
v___x_607_ = lean_usize_dec_eq(v___x_605_, v___x_606_);
if (v___x_607_ == 0)
{
v___y_591_ = v___x_607_;
goto v___jp_590_;
}
else
{
size_t v___x_608_; uint8_t v___x_609_; 
v___x_608_ = lean_ptr_addr(v_value_582_);
v___x_609_ = lean_usize_dec_eq(v___x_608_, v___x_608_);
v___y_591_ = v___x_609_;
goto v___jp_590_;
}
v___jp_590_:
{
if (v___y_591_ == 0)
{
lean_object* v___x_592_; lean_object* v___x_594_; 
lean_inc_ref(v_body_583_);
lean_inc_ref(v_value_582_);
lean_inc(v_declName_580_);
lean_dec_ref_known(v_e_357_, 4);
v___x_592_ = l_Lean_Expr_letE___override(v_declName_580_, v_a_586_, v_value_582_, v_body_583_, v_nondep_584_);
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v___x_592_);
v___x_594_ = v___x_588_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v___x_592_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
else
{
size_t v___x_596_; uint8_t v___x_597_; 
v___x_596_ = lean_ptr_addr(v_body_583_);
v___x_597_ = lean_usize_dec_eq(v___x_596_, v___x_596_);
if (v___x_597_ == 0)
{
lean_object* v___x_598_; lean_object* v___x_600_; 
lean_inc_ref(v_body_583_);
lean_inc_ref(v_value_582_);
lean_inc(v_declName_580_);
lean_dec_ref_known(v_e_357_, 4);
v___x_598_ = l_Lean_Expr_letE___override(v_declName_580_, v_a_586_, v_value_582_, v_body_583_, v_nondep_584_);
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v___x_598_);
v___x_600_ = v___x_588_;
goto v_reusejp_599_;
}
else
{
lean_object* v_reuseFailAlloc_601_; 
v_reuseFailAlloc_601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_601_, 0, v___x_598_);
v___x_600_ = v_reuseFailAlloc_601_;
goto v_reusejp_599_;
}
v_reusejp_599_:
{
return v___x_600_;
}
}
else
{
lean_object* v___x_603_; 
lean_dec(v_a_586_);
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 0, v_e_357_);
v___x_603_ = v___x_588_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v_e_357_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_e_357_, 4);
return v___x_585_;
}
}
case 11:
{
lean_object* v_typeName_611_; lean_object* v_idx_612_; lean_object* v_struct_613_; lean_object* v___x_614_; 
v_typeName_611_ = lean_ctor_get(v_e_357_, 0);
v_idx_612_ = lean_ctor_get(v_e_357_, 1);
v_struct_613_ = lean_ctor_get(v_e_357_, 2);
lean_inc(v___y_361_);
lean_inc_ref(v___y_360_);
lean_inc(v___y_359_);
lean_inc_ref(v___y_358_);
lean_inc_ref(v_struct_613_);
v___x_614_ = lean_apply_6(v_g_355_, v_struct_613_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, lean_box(0));
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_629_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_629_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_629_ == 0)
{
v___x_617_ = v___x_614_;
v_isShared_618_ = v_isSharedCheck_629_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_614_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_629_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
size_t v___x_619_; size_t v___x_620_; uint8_t v___x_621_; 
v___x_619_ = lean_ptr_addr(v_struct_613_);
v___x_620_ = lean_ptr_addr(v_a_615_);
v___x_621_ = lean_usize_dec_eq(v___x_619_, v___x_620_);
if (v___x_621_ == 0)
{
lean_object* v___x_622_; lean_object* v___x_624_; 
lean_inc(v_idx_612_);
lean_inc(v_typeName_611_);
lean_dec_ref_known(v_e_357_, 3);
v___x_622_ = l_Lean_Expr_proj___override(v_typeName_611_, v_idx_612_, v_a_615_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v___x_622_);
v___x_624_ = v___x_617_;
goto v_reusejp_623_;
}
else
{
lean_object* v_reuseFailAlloc_625_; 
v_reuseFailAlloc_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_625_, 0, v___x_622_);
v___x_624_ = v_reuseFailAlloc_625_;
goto v_reusejp_623_;
}
v_reusejp_623_:
{
return v___x_624_;
}
}
else
{
lean_object* v___x_627_; 
lean_dec(v_a_615_);
if (v_isShared_618_ == 0)
{
lean_ctor_set(v___x_617_, 0, v_e_357_);
v___x_627_ = v___x_617_;
goto v_reusejp_626_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v_e_357_);
v___x_627_ = v_reuseFailAlloc_628_;
goto v_reusejp_626_;
}
v_reusejp_626_:
{
return v___x_627_;
}
}
}
}
else
{
lean_dec_ref_known(v_e_357_, 3);
return v___x_614_;
}
}
case 10:
{
lean_object* v_expr_630_; 
v_expr_630_ = lean_ctor_get(v_e_357_, 1);
lean_inc_ref(v_expr_630_);
v_n_364_ = v___x_407_;
v_a_365_ = v_expr_630_;
goto v___jp_363_;
}
default: 
{
lean_dec_ref(v_g_355_);
v_c_395_ = v___x_407_;
v_e_396_ = v_e_357_;
goto v___jp_394_;
}
}
}
v___jp_363_:
{
lean_object* v___x_366_; 
v___x_366_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1(v_g_355_, v_n_364_, v_a_365_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
if (lean_obj_tag(v___x_366_) == 0)
{
if (lean_obj_tag(v_e_357_) == 10)
{
lean_object* v_a_367_; lean_object* v___x_369_; uint8_t v_isShared_370_; uint8_t v_isSharedCheck_383_; 
v_a_367_ = lean_ctor_get(v___x_366_, 0);
v_isSharedCheck_383_ = !lean_is_exclusive(v___x_366_);
if (v_isSharedCheck_383_ == 0)
{
v___x_369_ = v___x_366_;
v_isShared_370_ = v_isSharedCheck_383_;
goto v_resetjp_368_;
}
else
{
lean_inc(v_a_367_);
lean_dec(v___x_366_);
v___x_369_ = lean_box(0);
v_isShared_370_ = v_isSharedCheck_383_;
goto v_resetjp_368_;
}
v_resetjp_368_:
{
lean_object* v_data_371_; lean_object* v_expr_372_; size_t v___x_373_; size_t v___x_374_; uint8_t v___x_375_; 
v_data_371_ = lean_ctor_get(v_e_357_, 0);
v_expr_372_ = lean_ctor_get(v_e_357_, 1);
v___x_373_ = lean_ptr_addr(v_expr_372_);
v___x_374_ = lean_ptr_addr(v_a_367_);
v___x_375_ = lean_usize_dec_eq(v___x_373_, v___x_374_);
if (v___x_375_ == 0)
{
lean_object* v___x_376_; lean_object* v___x_378_; 
lean_inc(v_data_371_);
lean_dec_ref_known(v_e_357_, 2);
v___x_376_ = l_Lean_Expr_mdata___override(v_data_371_, v_a_367_);
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 0, v___x_376_);
v___x_378_ = v___x_369_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_379_; 
v_reuseFailAlloc_379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_379_, 0, v___x_376_);
v___x_378_ = v_reuseFailAlloc_379_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
return v___x_378_;
}
}
else
{
lean_object* v___x_381_; 
lean_dec(v_a_367_);
if (v_isShared_370_ == 0)
{
lean_ctor_set(v___x_369_, 0, v_e_357_);
v___x_381_ = v___x_369_;
goto v_reusejp_380_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v_e_357_);
v___x_381_ = v_reuseFailAlloc_382_;
goto v_reusejp_380_;
}
v_reusejp_380_:
{
return v___x_381_;
}
}
}
}
else
{
lean_object* v___x_385_; uint8_t v_isShared_386_; uint8_t v_isSharedCheck_392_; 
lean_dec_ref(v_e_357_);
v_isSharedCheck_392_ = !lean_is_exclusive(v___x_366_);
if (v_isSharedCheck_392_ == 0)
{
lean_object* v_unused_393_; 
v_unused_393_ = lean_ctor_get(v___x_366_, 0);
lean_dec(v_unused_393_);
v___x_385_ = v___x_366_;
v_isShared_386_ = v_isSharedCheck_392_;
goto v_resetjp_384_;
}
else
{
lean_dec(v___x_366_);
v___x_385_ = lean_box(0);
v_isShared_386_ = v_isSharedCheck_392_;
goto v_resetjp_384_;
}
v_resetjp_384_:
{
lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_390_; 
v___x_387_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__3);
v___x_388_ = lp_mathlib_panic___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__3(v___x_387_);
if (v_isShared_386_ == 0)
{
lean_ctor_set(v___x_385_, 0, v___x_388_);
v___x_390_ = v___x_385_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v___x_388_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
}
}
else
{
lean_dec_ref(v_e_357_);
return v___x_366_;
}
}
v___jp_394_:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v___x_397_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__5);
v___x_398_ = l_Nat_reprFast(v_c_395_);
v___x_399_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
v___x_400_ = l_Lean_MessageData_ofFormat(v___x_399_);
v___x_401_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_401_, 0, v___x_397_);
lean_ctor_set(v___x_401_, 1, v___x_400_);
v___x_402_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___closed__7);
v___x_403_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_401_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
v___x_404_ = l_Lean_MessageData_ofExpr(v_e_396_);
v___x_405_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_403_);
lean_ctor_set(v___x_405_, 1, v___x_404_);
v___x_406_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(v___x_405_, v___y_358_, v___y_359_, v___y_360_, v___y_361_);
return v___x_406_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1___boxed(lean_object* v_g_631_, lean_object* v_n_632_, lean_object* v_e_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1(v_g_631_, v_n_632_, v_e_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0___boxed(lean_object* v_g_640_, lean_object* v_x_641_, lean_object* v_x_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_){
_start:
{
lean_object* v_res_648_; 
v_res_648_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0(v_g_640_, v_x_641_, v_x_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
return v_res_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0(lean_object* v_g_649_, lean_object* v_x_650_, lean_object* v_x_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_){
_start:
{
if (lean_obj_tag(v_x_650_) == 0)
{
lean_object* v___x_657_; 
lean_inc(v___y_655_);
lean_inc_ref(v___y_654_);
lean_inc(v___y_653_);
lean_inc_ref(v___y_652_);
v___x_657_ = lean_apply_6(v_g_649_, v_x_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_, lean_box(0));
return v___x_657_;
}
else
{
lean_object* v_head_658_; lean_object* v_tail_659_; lean_object* v___x_660_; lean_object* v___x_661_; 
v_head_658_ = lean_ctor_get(v_x_650_, 0);
lean_inc(v_head_658_);
v_tail_659_ = lean_ctor_get(v_x_650_, 1);
lean_inc(v_tail_659_);
lean_dec_ref_known(v_x_650_, 2);
v___x_660_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0___boxed), 8, 2);
lean_closure_set(v___x_660_, 0, v_g_649_);
lean_closure_set(v___x_660_, 1, v_tail_659_);
v___x_661_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1(v___x_660_, v_head_658_, v_x_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
return v___x_661_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0(lean_object* v_replace_662_, lean_object* v_p_663_, lean_object* v_root_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_){
_start:
{
lean_object* v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v___x_670_ = l_Lean_SubExpr_Pos_toArray(v_p_663_);
v___x_671_ = lean_array_to_list(v___x_670_);
v___x_672_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0(v_replace_662_, v___x_671_, v_root_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0___boxed(lean_object* v_replace_673_, lean_object* v_p_674_, lean_object* v_root_675_, lean_object* v___y_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0(v_replace_673_, v_p_674_, v_root_675_, v___y_676_, v___y_677_, v___y_678_, v___y_679_);
lean_dec(v___y_679_);
lean_dec_ref(v___y_678_);
lean_dec(v___y_677_);
lean_dec_ref(v___y_676_);
lean_dec(v_p_674_);
return v_res_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar(lean_object* v_e_683_, lean_object* v_pos_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_){
_start:
{
lean_object* v___f_690_; lean_object* v___x_691_; 
v___f_690_ = ((lean_object*)(lp_mathlib_insertMetaVar___closed__0));
v___x_691_ = lp_mathlib_Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0(v___f_690_, v_pos_684_, v_e_683_, v_a_685_, v_a_686_, v_a_687_, v_a_688_);
return v___x_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_insertMetaVar___boxed(lean_object* v_e_692_, lean_object* v_pos_693_, lean_object* v_a_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_mathlib_insertMetaVar(v_e_692_, v_pos_693_, v_a_694_, v_a_695_, v_a_696_, v_a_697_);
lean_dec(v_a_697_);
lean_dec_ref(v_a_696_);
lean_dec(v_a_695_);
lean_dec_ref(v_a_694_);
lean_dec(v_pos_693_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5(lean_object* v_00_u03b1_700_, lean_object* v_name_701_, uint8_t v_bi_702_, lean_object* v_type_703_, lean_object* v_k_704_, uint8_t v_kind_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_){
_start:
{
lean_object* v___x_711_; 
v___x_711_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___redArg(v_name_701_, v_bi_702_, v_type_703_, v_k_704_, v_kind_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
return v___x_711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5___boxed(lean_object* v_00_u03b1_712_, lean_object* v_name_713_, lean_object* v_bi_714_, lean_object* v_type_715_, lean_object* v_k_716_, lean_object* v_kind_717_, lean_object* v___y_718_, lean_object* v___y_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_){
_start:
{
uint8_t v_bi_boxed_723_; uint8_t v_kind_boxed_724_; lean_object* v_res_725_; 
v_bi_boxed_723_ = lean_unbox(v_bi_714_);
v_kind_boxed_724_ = lean_unbox(v_kind_717_);
v_res_725_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__5(v_00_u03b1_712_, v_name_713_, v_bi_boxed_723_, v_type_715_, v_k_716_, v_kind_boxed_724_, v___y_718_, v___y_719_, v___y_720_, v___y_721_);
lean_dec(v___y_721_);
lean_dec_ref(v___y_720_);
lean_dec(v___y_719_);
lean_dec_ref(v___y_718_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b1_726_, lean_object* v_msg_727_, lean_object* v___y_728_, lean_object* v___y_729_, lean_object* v___y_730_, lean_object* v___y_731_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___redArg(v_msg_727_, v___y_728_, v___y_729_, v___y_730_, v___y_731_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2___boxed(lean_object* v_00_u03b1_734_, lean_object* v_msg_735_, lean_object* v___y_736_, lean_object* v___y_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_){
_start:
{
lean_object* v_res_741_; 
v_res_741_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__2(v_00_u03b1_734_, v_msg_735_, v___y_736_, v___y_737_, v___y_738_, v___y_739_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec(v___y_737_);
lean_dec_ref(v___y_736_);
return v_res_741_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6(lean_object* v_00_u03b1_742_, lean_object* v_name_743_, lean_object* v_type_744_, lean_object* v_val_745_, lean_object* v_k_746_, uint8_t v_nondep_747_, uint8_t v_kind_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_){
_start:
{
lean_object* v___x_754_; 
v___x_754_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___redArg(v_name_743_, v_type_744_, v_val_745_, v_k_746_, v_nondep_747_, v_kind_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_);
return v___x_754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6___boxed(lean_object* v_00_u03b1_755_, lean_object* v_name_756_, lean_object* v_type_757_, lean_object* v_val_758_, lean_object* v_k_759_, lean_object* v_nondep_760_, lean_object* v_kind_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_, lean_object* v___y_765_, lean_object* v___y_766_){
_start:
{
uint8_t v_nondep_boxed_767_; uint8_t v_kind_boxed_768_; lean_object* v_res_769_; 
v_nondep_boxed_767_ = lean_unbox(v_nondep_760_);
v_kind_boxed_768_ = lean_unbox(v_kind_761_);
v_res_769_ = lp_mathlib_Lean_Meta_withLetDecl___at___00Lean_Meta_mapLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensCoord___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_lensAux___at___00Lean_Meta_replaceSubexpr___at___00insertMetaVar_spec__0_spec__0_spec__1_spec__4_spec__6(v_00_u03b1_755_, v_name_756_, v_type_757_, v_val_758_, v_k_759_, v_nondep_boxed_767_, v_kind_boxed_768_, v___y_762_, v___y_763_, v___y_764_, v___y_765_);
lean_dec(v___y_765_);
lean_dec_ref(v___y_764_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0(lean_object* v_s_770_, lean_object* v_pos_771_){
_start:
{
lean_object* v_str_772_; lean_object* v_startInclusive_773_; lean_object* v_endExclusive_774_; lean_object* v___x_775_; uint8_t v___y_777_; lean_object* v___x_783_; lean_object* v___x_784_; uint8_t v___x_785_; 
v_str_772_ = lean_ctor_get(v_s_770_, 0);
v_startInclusive_773_ = lean_ctor_get(v_s_770_, 1);
v_endExclusive_774_ = lean_ctor_get(v_s_770_, 2);
v___x_775_ = lean_nat_add(v_startInclusive_773_, v_pos_771_);
v___x_783_ = lean_unsigned_to_nat(0u);
v___x_784_ = lean_nat_sub(v_endExclusive_774_, v___x_775_);
v___x_785_ = lean_nat_dec_eq(v___x_783_, v___x_784_);
lean_dec(v___x_784_);
if (v___x_785_ == 0)
{
uint32_t v___x_786_; uint32_t v___x_787_; uint8_t v___x_788_; 
v___x_786_ = lean_string_utf8_get_fast(v_str_772_, v___x_775_);
v___x_787_ = 48;
v___x_788_ = lean_uint32_dec_le(v___x_787_, v___x_786_);
if (v___x_788_ == 0)
{
v___y_777_ = v___x_788_;
goto v___jp_776_;
}
else
{
uint32_t v___x_789_; uint8_t v___x_790_; 
v___x_789_ = 57;
v___x_790_ = lean_uint32_dec_le(v___x_786_, v___x_789_);
v___y_777_ = v___x_790_;
goto v___jp_776_;
}
}
else
{
lean_dec(v___x_775_);
return v_pos_771_;
}
v___jp_776_:
{
if (v___y_777_ == 0)
{
lean_dec(v___x_775_);
return v_pos_771_;
}
else
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v___x_780_; uint8_t v___x_781_; 
v___x_778_ = lean_string_utf8_next_fast(v_str_772_, v___x_775_);
v___x_779_ = lean_nat_sub(v___x_778_, v___x_775_);
lean_dec(v___x_775_);
v___x_780_ = lean_nat_add(v_pos_771_, v___x_779_);
lean_dec(v___x_779_);
v___x_781_ = lean_nat_dec_lt(v_pos_771_, v___x_780_);
if (v___x_781_ == 0)
{
lean_dec(v___x_780_);
return v_pos_771_;
}
else
{
lean_dec(v_pos_771_);
v_pos_771_ = v___x_780_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0___boxed(lean_object* v_s_791_, lean_object* v_pos_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0(v_s_791_, v_pos_792_);
lean_dec_ref(v_s_791_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00String_renameMetaVar_spec__1(lean_object* v_a_794_, lean_object* v_a_795_){
_start:
{
if (lean_obj_tag(v_a_794_) == 0)
{
lean_object* v___x_796_; 
v___x_796_ = l_List_reverse___redArg(v_a_795_);
return v___x_796_;
}
else
{
lean_object* v_head_797_; lean_object* v_tail_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_811_; 
v_head_797_ = lean_ctor_get(v_a_794_, 0);
v_tail_798_ = lean_ctor_get(v_a_794_, 1);
v_isSharedCheck_811_ = !lean_is_exclusive(v_a_794_);
if (v_isSharedCheck_811_ == 0)
{
v___x_800_ = v_a_794_;
v_isShared_801_ = v_isSharedCheck_811_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_tail_798_);
lean_inc(v_head_797_);
lean_dec(v_a_794_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_811_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_802_; lean_object* v___x_803_; lean_object* v___x_804_; lean_object* v___x_805_; lean_object* v___x_806_; lean_object* v___x_808_; 
v___x_802_ = lean_unsigned_to_nat(0u);
v___x_803_ = lean_string_utf8_byte_size(v_head_797_);
lean_inc(v_head_797_);
v___x_804_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_804_, 0, v_head_797_);
lean_ctor_set(v___x_804_, 1, v___x_802_);
lean_ctor_set(v___x_804_, 2, v___x_803_);
v___x_805_ = lp_mathlib_String_Slice_Pos_skipWhile___at___00String_renameMetaVar_spec__0(v___x_804_, v___x_802_);
lean_dec_ref_known(v___x_804_, 3);
v___x_806_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_806_, 0, v_head_797_);
lean_ctor_set(v___x_806_, 1, v___x_805_);
lean_ctor_set(v___x_806_, 2, v___x_803_);
if (v_isShared_801_ == 0)
{
lean_ctor_set(v___x_800_, 1, v_a_795_);
lean_ctor_set(v___x_800_, 0, v___x_806_);
v___x_808_ = v___x_800_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_810_; 
v_reuseFailAlloc_810_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_810_, 0, v___x_806_);
lean_ctor_set(v_reuseFailAlloc_810_, 1, v_a_795_);
v___x_808_ = v_reuseFailAlloc_810_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
v_a_794_ = v_tail_798_;
v_a_795_ = v___x_808_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_String_renameMetaVar___closed__3(void){
_start:
{
lean_object* v___x_815_; lean_object* v___x_816_; 
v___x_815_ = ((lean_object*)(lp_mathlib_String_renameMetaVar___closed__2));
v___x_816_ = lean_string_utf8_byte_size(v___x_815_);
return v___x_816_;
}
}
static lean_object* _init_lp_mathlib_String_renameMetaVar___closed__4(void){
_start:
{
lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; 
v___x_817_ = lean_obj_once(&lp_mathlib_String_renameMetaVar___closed__3, &lp_mathlib_String_renameMetaVar___closed__3_once, _init_lp_mathlib_String_renameMetaVar___closed__3);
v___x_818_ = lean_unsigned_to_nat(0u);
v___x_819_ = ((lean_object*)(lp_mathlib_String_renameMetaVar___closed__2));
v___x_820_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_820_, 0, v___x_819_);
lean_ctor_set(v___x_820_, 1, v___x_818_);
lean_ctor_set(v___x_820_, 2, v___x_817_);
return v___x_820_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_renameMetaVar(lean_object* v_s_821_){
_start:
{
lean_object* v___x_822_; lean_object* v___x_823_; lean_object* v___x_824_; lean_object* v___x_825_; 
v___x_822_ = ((lean_object*)(lp_mathlib_String_renameMetaVar___closed__0));
v___x_823_ = lean_unsigned_to_nat(0u);
v___x_824_ = lean_box(0);
v___x_825_ = l_String_splitOnAux(v_s_821_, v___x_822_, v___x_823_, v___x_823_, v___x_823_, v___x_824_);
if (lean_obj_tag(v___x_825_) == 0)
{
lean_object* v___x_826_; 
v___x_826_ = ((lean_object*)(lp_mathlib_String_renameMetaVar___closed__1));
return v___x_826_;
}
else
{
lean_object* v_tail_827_; 
v_tail_827_ = lean_ctor_get(v___x_825_, 1);
lean_inc(v_tail_827_);
if (lean_obj_tag(v_tail_827_) == 0)
{
lean_object* v_head_828_; 
v_head_828_ = lean_ctor_get(v___x_825_, 0);
lean_inc(v_head_828_);
lean_dec_ref_known(v___x_825_, 2);
return v_head_828_;
}
else
{
lean_object* v_head_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v_head_829_ = lean_ctor_get(v___x_825_, 0);
lean_inc(v_head_829_);
lean_dec_ref_known(v___x_825_, 2);
v___x_830_ = ((lean_object*)(lp_mathlib_String_renameMetaVar___closed__2));
v___x_831_ = lean_string_append(v_head_829_, v___x_830_);
v___x_832_ = lean_obj_once(&lp_mathlib_String_renameMetaVar___closed__4, &lp_mathlib_String_renameMetaVar___closed__4_once, _init_lp_mathlib_String_renameMetaVar___closed__4);
v___x_833_ = lp_mathlib_List_mapTR_loop___at___00String_renameMetaVar_spec__1(v_tail_827_, v___x_824_);
v___x_834_ = l_String_Slice_intercalate(v___x_832_, v___x_833_);
lean_dec(v___x_833_);
v___x_835_ = lean_string_append(v___x_831_, v___x_834_);
lean_dec_ref(v___x_834_);
return v___x_835_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_renameMetaVar___boxed(lean_object* v_s_836_){
_start:
{
lean_object* v_res_837_; 
v_res_837_ = lp_mathlib_String_renameMetaVar(v_s_836_);
lean_dec_ref(v_s_836_);
return v_res_837_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0(lean_object* v_prop_838_){
_start:
{
lean_object* v_pos_839_; 
v_pos_839_ = lean_ctor_get(v_prop_838_, 0);
lean_inc_ref(v_pos_839_);
return v_pos_839_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0___boxed(lean_object* v_prop_840_){
_start:
{
lean_object* v_res_841_; 
v_res_841_ = lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__0(v_prop_840_);
lean_dec_ref(v_prop_840_);
return v_res_841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1(lean_object* v_prop_842_){
_start:
{
lean_object* v_goals_843_; 
v_goals_843_ = lean_ctor_get(v_prop_842_, 1);
lean_inc_ref(v_goals_843_);
return v_goals_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1___boxed(lean_object* v_prop_844_){
_start:
{
lean_object* v_res_845_; 
v_res_845_ = lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__1(v_prop_844_);
lean_dec_ref(v_prop_844_);
return v_res_845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2(lean_object* v_prop_846_){
_start:
{
lean_object* v_selectedLocations_847_; 
v_selectedLocations_847_ = lean_ctor_get(v_prop_846_, 2);
lean_inc_ref(v_selectedLocations_847_);
return v_selectedLocations_847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2___boxed(lean_object* v_prop_848_){
_start:
{
lean_object* v_res_849_; 
v_res_849_ = lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__2(v_prop_848_);
lean_dec_ref(v_prop_848_);
return v_res_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3(lean_object* v_prop_850_){
_start:
{
lean_object* v_replaceRange_851_; 
v_replaceRange_851_ = lean_ctor_get(v_prop_850_, 3);
lean_inc_ref(v_replaceRange_851_);
return v_replaceRange_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3___boxed(lean_object* v_prop_852_){
_start:
{
lean_object* v_res_853_; 
v_res_853_ = lp_mathlib_instSelectInsertParamsClassSelectInsertParams___lam__3(v_prop_852_);
lean_dec_ref(v_prop_852_);
return v_res_853_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(lean_object* v_j_864_, lean_object* v_k_865_){
_start:
{
lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_866_ = l_Lean_Json_getObjValD(v_j_864_, v_k_865_);
v___x_867_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_867_, 0, v___x_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0___boxed(lean_object* v_j_868_, lean_object* v_k_869_){
_start:
{
lean_object* v_res_870_; 
v_res_870_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(v_j_868_, v_k_869_);
lean_dec_ref(v_k_869_);
return v_res_870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_(lean_object* v_json_875_){
_start:
{
lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v_a_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v_a_881_; lean_object* v___x_882_; lean_object* v___x_883_; lean_object* v_a_884_; lean_object* v___x_885_; lean_object* v___x_886_; lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_895_; 
v___x_876_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
lean_inc_n(v_json_875_, 3);
v___x_877_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(v_json_875_, v___x_876_);
v_a_878_ = lean_ctor_get(v___x_877_, 0);
lean_inc(v_a_878_);
lean_dec_ref(v___x_877_);
v___x_879_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
v___x_880_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(v_json_875_, v___x_879_);
v_a_881_ = lean_ctor_get(v___x_880_, 0);
lean_inc(v_a_881_);
lean_dec_ref(v___x_880_);
v___x_882_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
v___x_883_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(v_json_875_, v___x_882_);
v_a_884_ = lean_ctor_get(v___x_883_, 0);
lean_inc(v_a_884_);
lean_dec_ref(v___x_883_);
v___x_885_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
v___x_886_ = lp_mathlib_Lean_Json_getObjValAs_x3f___at___00instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16__spec__0(v_json_875_, v___x_885_);
v_a_887_ = lean_ctor_get(v___x_886_, 0);
v_isSharedCheck_895_ = !lean_is_exclusive(v___x_886_);
if (v_isSharedCheck_895_ == 0)
{
v___x_889_ = v___x_886_;
v_isShared_890_ = v_isSharedCheck_895_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_886_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_895_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_891_; lean_object* v___x_893_; 
v___x_891_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_891_, 0, v_a_878_);
lean_ctor_set(v___x_891_, 1, v_a_881_);
lean_ctor_set(v___x_891_, 2, v_a_884_);
lean_ctor_set(v___x_891_, 3, v_a_887_);
if (v_isShared_890_ == 0)
{
lean_ctor_set(v___x_889_, 0, v___x_891_);
v___x_893_ = v___x_889_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v___x_891_);
v___x_893_ = v_reuseFailAlloc_894_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
return v___x_893_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__spec__0(lean_object* v_a_898_, lean_object* v_a_899_){
_start:
{
if (lean_obj_tag(v_a_898_) == 0)
{
lean_object* v___x_900_; 
v___x_900_ = lean_array_to_list(v_a_899_);
return v___x_900_;
}
else
{
lean_object* v_head_901_; lean_object* v_tail_902_; lean_object* v___x_903_; 
v_head_901_ = lean_ctor_get(v_a_898_, 0);
lean_inc(v_head_901_);
v_tail_902_ = lean_ctor_get(v_a_898_, 1);
lean_inc(v_tail_902_);
lean_dec_ref_known(v_a_898_, 2);
v___x_903_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_899_, v_head_901_);
v_a_898_ = v_tail_902_;
v_a_899_ = v___x_903_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_(lean_object* v_x_907_){
_start:
{
lean_object* v_pos_908_; lean_object* v_goals_909_; lean_object* v_selectedLocations_910_; lean_object* v_replaceRange_911_; lean_object* v___x_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; 
v_pos_908_ = lean_ctor_get(v_x_907_, 0);
v_goals_909_ = lean_ctor_get(v_x_907_, 1);
v_selectedLocations_910_ = lean_ctor_get(v_x_907_, 2);
v_replaceRange_911_ = lean_ctor_get(v_x_907_, 3);
v___x_912_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
lean_inc(v_pos_908_);
v___x_913_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_913_, 0, v___x_912_);
lean_ctor_set(v___x_913_, 1, v_pos_908_);
v___x_914_ = lean_box(0);
v___x_915_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_915_, 0, v___x_913_);
lean_ctor_set(v___x_915_, 1, v___x_914_);
v___x_916_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__1_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
lean_inc(v_goals_909_);
v___x_917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_917_, 0, v___x_916_);
lean_ctor_set(v___x_917_, 1, v_goals_909_);
v___x_918_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_918_, 0, v___x_917_);
lean_ctor_set(v___x_918_, 1, v___x_914_);
v___x_919_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__2_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
lean_inc(v_selectedLocations_910_);
v___x_920_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_920_, 0, v___x_919_);
lean_ctor_set(v___x_920_, 1, v_selectedLocations_910_);
v___x_921_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_920_);
lean_ctor_set(v___x_921_, 1, v___x_914_);
v___x_922_ = ((lean_object*)(lp_mathlib_instFromJsonRpcEncodablePacket_fromJson___closed__3_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_));
lean_inc(v_replaceRange_911_);
v___x_923_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_923_, 0, v___x_922_);
lean_ctor_set(v___x_923_, 1, v_replaceRange_911_);
v___x_924_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_924_, 0, v___x_923_);
lean_ctor_set(v___x_924_, 1, v___x_914_);
v___x_925_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_925_, 0, v___x_924_);
lean_ctor_set(v___x_925_, 1, v___x_914_);
v___x_926_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_926_, 0, v___x_921_);
lean_ctor_set(v___x_926_, 1, v___x_925_);
v___x_927_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_927_, 0, v___x_918_);
lean_ctor_set(v___x_927_, 1, v___x_926_);
v___x_928_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_928_, 0, v___x_915_);
lean_ctor_set(v___x_928_, 1, v___x_927_);
v___x_929_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_));
v___x_930_ = lp_mathlib___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35__spec__0(v___x_928_, v___x_929_);
v___x_931_ = l_Lean_Json_mkObj(v___x_930_);
lean_dec(v___x_930_);
return v___x_931_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35____boxed(lean_object* v_x_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_(v_x_932_);
lean_dec_ref(v_x_932_);
return v_res_933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(size_t v_sz_936_, size_t v_i_937_, lean_object* v_bs_938_, lean_object* v___y_939_){
_start:
{
uint8_t v___x_940_; 
v___x_940_ = lean_usize_dec_lt(v_i_937_, v_sz_936_);
if (v___x_940_ == 0)
{
lean_object* v___x_941_; 
v___x_941_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_941_, 0, v_bs_938_);
lean_ctor_set(v___x_941_, 1, v___y_939_);
return v___x_941_;
}
else
{
lean_object* v_v_942_; lean_object* v___x_943_; lean_object* v_fst_944_; lean_object* v_snd_945_; lean_object* v___x_946_; lean_object* v_bs_x27_947_; size_t v___x_948_; size_t v___x_949_; lean_object* v___x_950_; 
v_v_942_ = lean_array_uget_borrowed(v_bs_938_, v_i_937_);
lean_inc(v_v_942_);
v___x_943_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_enc_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_942_, v___y_939_);
v_fst_944_ = lean_ctor_get(v___x_943_, 0);
lean_inc(v_fst_944_);
v_snd_945_ = lean_ctor_get(v___x_943_, 1);
lean_inc(v_snd_945_);
lean_dec_ref(v___x_943_);
v___x_946_ = lean_unsigned_to_nat(0u);
v_bs_x27_947_ = lean_array_uset(v_bs_938_, v_i_937_, v___x_946_);
v___x_948_ = ((size_t)1ULL);
v___x_949_ = lean_usize_add(v_i_937_, v___x_948_);
v___x_950_ = lean_array_uset(v_bs_x27_947_, v_i_937_, v_fst_944_);
v_i_937_ = v___x_949_;
v_bs_938_ = v___x_950_;
v___y_939_ = v_snd_945_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___boxed(lean_object* v_sz_952_, lean_object* v_i_953_, lean_object* v_bs_954_, lean_object* v___y_955_){
_start:
{
size_t v_sz_boxed_956_; size_t v_i_boxed_957_; lean_object* v_res_958_; 
v_sz_boxed_956_ = lean_unbox_usize(v_sz_952_);
lean_dec(v_sz_952_);
v_i_boxed_957_ = lean_unbox_usize(v_i_953_);
lean_dec(v_i_953_);
v_res_958_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(v_sz_boxed_956_, v_i_boxed_957_, v_bs_954_, v___y_955_);
return v_res_958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2(size_t v_sz_959_, size_t v_i_960_, lean_object* v_bs_961_){
_start:
{
uint8_t v___x_962_; 
v___x_962_ = lean_usize_dec_lt(v_i_960_, v_sz_959_);
if (v___x_962_ == 0)
{
return v_bs_961_;
}
else
{
lean_object* v_v_963_; lean_object* v___x_964_; lean_object* v_bs_x27_965_; size_t v___x_966_; size_t v___x_967_; lean_object* v___x_968_; 
v_v_963_ = lean_array_uget(v_bs_961_, v_i_960_);
v___x_964_ = lean_unsigned_to_nat(0u);
v_bs_x27_965_ = lean_array_uset(v_bs_961_, v_i_960_, v___x_964_);
v___x_966_ = ((size_t)1ULL);
v___x_967_ = lean_usize_add(v_i_960_, v___x_966_);
v___x_968_ = lean_array_uset(v_bs_x27_965_, v_i_960_, v_v_963_);
v_i_960_ = v___x_967_;
v_bs_961_ = v___x_968_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2___boxed(lean_object* v_sz_970_, lean_object* v_i_971_, lean_object* v_bs_972_){
_start:
{
size_t v_sz_boxed_973_; size_t v_i_boxed_974_; lean_object* v_res_975_; 
v_sz_boxed_973_ = lean_unbox_usize(v_sz_970_);
lean_dec(v_sz_970_);
v_i_boxed_974_ = lean_unbox_usize(v_i_971_);
lean_dec(v_i_971_);
v_res_975_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2(v_sz_boxed_973_, v_i_boxed_974_, v_bs_972_);
return v_res_975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(lean_object* v_a_976_){
_start:
{
size_t v_sz_977_; size_t v___x_978_; lean_object* v___x_979_; lean_object* v___x_980_; 
v_sz_977_ = lean_array_size(v_a_976_);
v___x_978_ = ((size_t)0ULL);
v___x_979_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2_spec__2(v_sz_977_, v___x_978_, v_a_976_);
v___x_980_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_980_, 0, v___x_979_);
return v___x_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(size_t v_sz_981_, size_t v_i_982_, lean_object* v_bs_983_, lean_object* v___y_984_){
_start:
{
uint8_t v___x_985_; 
v___x_985_ = lean_usize_dec_lt(v_i_982_, v_sz_981_);
if (v___x_985_ == 0)
{
lean_object* v___x_986_; 
v___x_986_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_986_, 0, v_bs_983_);
lean_ctor_set(v___x_986_, 1, v___y_984_);
return v___x_986_;
}
else
{
lean_object* v_v_987_; lean_object* v___x_988_; lean_object* v_bs_x27_989_; lean_object* v___x_990_; size_t v___x_991_; size_t v___x_992_; lean_object* v___x_993_; 
v_v_987_ = lean_array_uget(v_bs_983_, v_i_982_);
v___x_988_ = lean_unsigned_to_nat(0u);
v_bs_x27_989_ = lean_array_uset(v_bs_983_, v_i_982_, v___x_988_);
v___x_990_ = l_Lean_SubExpr_instToJsonGoalsLocation_toJson(v_v_987_);
v___x_991_ = ((size_t)1ULL);
v___x_992_ = lean_usize_add(v_i_982_, v___x_991_);
v___x_993_ = lean_array_uset(v_bs_x27_989_, v_i_982_, v___x_990_);
v_i_982_ = v___x_992_;
v_bs_983_ = v___x_993_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___boxed(lean_object* v_sz_995_, lean_object* v_i_996_, lean_object* v_bs_997_, lean_object* v___y_998_){
_start:
{
size_t v_sz_boxed_999_; size_t v_i_boxed_1000_; lean_object* v_res_1001_; 
v_sz_boxed_999_ = lean_unbox_usize(v_sz_995_);
lean_dec(v_sz_995_);
v_i_boxed_1000_ = lean_unbox_usize(v_i_996_);
lean_dec(v_i_996_);
v_res_1001_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(v_sz_boxed_999_, v_i_boxed_1000_, v_bs_997_, v___y_998_);
return v_res_1001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object* v_a_1002_, lean_object* v_a_1003_){
_start:
{
lean_object* v_pos_1004_; lean_object* v_goals_1005_; lean_object* v_selectedLocations_1006_; lean_object* v_replaceRange_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1035_; 
v_pos_1004_ = lean_ctor_get(v_a_1002_, 0);
v_goals_1005_ = lean_ctor_get(v_a_1002_, 1);
v_selectedLocations_1006_ = lean_ctor_get(v_a_1002_, 2);
v_replaceRange_1007_ = lean_ctor_get(v_a_1002_, 3);
v_isSharedCheck_1035_ = !lean_is_exclusive(v_a_1002_);
if (v_isSharedCheck_1035_ == 0)
{
v___x_1009_ = v_a_1002_;
v_isShared_1010_ = v_isSharedCheck_1035_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_replaceRange_1007_);
lean_inc(v_selectedLocations_1006_);
lean_inc(v_goals_1005_);
lean_inc(v_pos_1004_);
lean_dec(v_a_1002_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1035_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
size_t v_sz_1011_; size_t v___x_1012_; lean_object* v___x_1013_; lean_object* v_fst_1014_; lean_object* v_snd_1015_; size_t v_sz_1016_; lean_object* v___x_1017_; lean_object* v_fst_1018_; lean_object* v_snd_1019_; lean_object* v___x_1021_; uint8_t v_isShared_1022_; uint8_t v_isSharedCheck_1034_; 
v_sz_1011_ = lean_array_size(v_goals_1005_);
v___x_1012_ = ((size_t)0ULL);
v___x_1013_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(v_sz_1011_, v___x_1012_, v_goals_1005_, v_a_1003_);
v_fst_1014_ = lean_ctor_get(v___x_1013_, 0);
lean_inc(v_fst_1014_);
v_snd_1015_ = lean_ctor_get(v___x_1013_, 1);
lean_inc(v_snd_1015_);
lean_dec_ref(v___x_1013_);
v_sz_1016_ = lean_array_size(v_selectedLocations_1006_);
v___x_1017_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(v_sz_1016_, v___x_1012_, v_selectedLocations_1006_, v_snd_1015_);
v_fst_1018_ = lean_ctor_get(v___x_1017_, 0);
v_snd_1019_ = lean_ctor_get(v___x_1017_, 1);
v_isSharedCheck_1034_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1034_ == 0)
{
v___x_1021_ = v___x_1017_;
v_isShared_1022_ = v_isSharedCheck_1034_;
goto v_resetjp_1020_;
}
else
{
lean_inc(v_snd_1019_);
lean_inc(v_fst_1018_);
lean_dec(v___x_1017_);
v___x_1021_ = lean_box(0);
v_isShared_1022_ = v_isSharedCheck_1034_;
goto v_resetjp_1020_;
}
v_resetjp_1020_:
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1028_; 
v___x_1023_ = l_Lean_Lsp_instToJsonPosition_toJson(v_pos_1004_);
v___x_1024_ = lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(v_fst_1014_);
v___x_1025_ = lp_mathlib_Lean_Array_toJson___at___00instRpcEncodableSelectInsertParams_enc_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(v_fst_1018_);
v___x_1026_ = l_Lean_Lsp_instToJsonRange_toJson(v_replaceRange_1007_);
if (v_isShared_1010_ == 0)
{
lean_ctor_set(v___x_1009_, 3, v___x_1026_);
lean_ctor_set(v___x_1009_, 2, v___x_1025_);
lean_ctor_set(v___x_1009_, 1, v___x_1024_);
lean_ctor_set(v___x_1009_, 0, v___x_1023_);
v___x_1028_ = v___x_1009_;
goto v_reusejp_1027_;
}
else
{
lean_object* v_reuseFailAlloc_1033_; 
v_reuseFailAlloc_1033_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1033_, 0, v___x_1023_);
lean_ctor_set(v_reuseFailAlloc_1033_, 1, v___x_1024_);
lean_ctor_set(v_reuseFailAlloc_1033_, 2, v___x_1025_);
lean_ctor_set(v_reuseFailAlloc_1033_, 3, v___x_1026_);
v___x_1028_ = v_reuseFailAlloc_1033_;
goto v_reusejp_1027_;
}
v_reusejp_1027_:
{
lean_object* v___x_1029_; lean_object* v___x_1031_; 
v___x_1029_ = lp_mathlib_instToJsonRpcEncodablePacket_toJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_(v___x_1028_);
lean_dec_ref(v___x_1028_);
if (v_isShared_1022_ == 0)
{
lean_ctor_set(v___x_1021_, 0, v___x_1029_);
v___x_1031_ = v___x_1021_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1032_; 
v_reuseFailAlloc_1032_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1032_, 0, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1032_, 1, v_snd_1019_);
v___x_1031_ = v_reuseFailAlloc_1032_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
return v___x_1031_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___redArg(lean_object* v_x_1036_){
_start:
{
lean_inc_ref(v_x_1036_);
return v_x_1036_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___redArg___boxed(lean_object* v_x_1037_){
_start:
{
lean_object* v_res_1038_; 
v_res_1038_ = lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___redArg(v_x_1037_);
lean_dec_ref(v_x_1037_);
return v_res_1038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(lean_object* v_00_u03b1_1039_, lean_object* v_x_1040_, lean_object* v___y_1041_){
_start:
{
lean_inc_ref(v_x_1040_);
return v_x_1040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0___boxed(lean_object* v_00_u03b1_1042_, lean_object* v_x_1043_, lean_object* v___y_1044_){
_start:
{
lean_object* v_res_1045_; 
v_res_1045_ = lp_mathlib_MonadExcept_ofExcept___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__0(v_00_u03b1_1042_, v_x_1043_, v___y_1044_);
lean_dec_ref(v___y_1044_);
lean_dec_ref(v_x_1043_);
return v_res_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1(size_t v_sz_1046_, size_t v_i_1047_, lean_object* v_bs_1048_){
_start:
{
uint8_t v___x_1049_; 
v___x_1049_ = lean_usize_dec_lt(v_i_1047_, v_sz_1046_);
if (v___x_1049_ == 0)
{
lean_object* v___x_1050_; 
v___x_1050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1050_, 0, v_bs_1048_);
return v___x_1050_;
}
else
{
lean_object* v_v_1051_; lean_object* v___x_1052_; lean_object* v_bs_x27_1053_; size_t v___x_1054_; size_t v___x_1055_; lean_object* v___x_1056_; 
v_v_1051_ = lean_array_uget(v_bs_1048_, v_i_1047_);
v___x_1052_ = lean_unsigned_to_nat(0u);
v_bs_x27_1053_ = lean_array_uset(v_bs_1048_, v_i_1047_, v___x_1052_);
v___x_1054_ = ((size_t)1ULL);
v___x_1055_ = lean_usize_add(v_i_1047_, v___x_1054_);
v___x_1056_ = lean_array_uset(v_bs_x27_1053_, v_i_1047_, v_v_1051_);
v_i_1047_ = v___x_1055_;
v_bs_1048_ = v___x_1056_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1___boxed(lean_object* v_sz_1058_, lean_object* v_i_1059_, lean_object* v_bs_1060_){
_start:
{
size_t v_sz_boxed_1061_; size_t v_i_boxed_1062_; lean_object* v_res_1063_; 
v_sz_boxed_1061_ = lean_unbox_usize(v_sz_1058_);
lean_dec(v_sz_1058_);
v_i_boxed_1062_ = lean_unbox_usize(v_i_1059_);
lean_dec(v_i_1059_);
v_res_1063_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1(v_sz_boxed_1061_, v_i_boxed_1062_, v_bs_1060_);
return v_res_1063_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(lean_object* v_x_1066_){
_start:
{
if (lean_obj_tag(v_x_1066_) == 4)
{
lean_object* v_elems_1067_; size_t v_sz_1068_; size_t v___x_1069_; lean_object* v___x_1070_; 
v_elems_1067_ = lean_ctor_get(v_x_1066_, 0);
lean_inc_ref(v_elems_1067_);
lean_dec_ref_known(v_x_1066_, 1);
v_sz_1068_ = lean_array_size(v_elems_1067_);
v___x_1069_ = ((size_t)0ULL);
v___x_1070_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1_spec__1(v_sz_1068_, v___x_1069_, v_elems_1067_);
return v___x_1070_;
}
else
{
lean_object* v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; lean_object* v___x_1074_; lean_object* v___x_1075_; lean_object* v___x_1076_; lean_object* v___x_1077_; 
v___x_1071_ = ((lean_object*)(lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__0));
v___x_1072_ = lean_unsigned_to_nat(80u);
v___x_1073_ = l_Lean_Json_pretty(v_x_1066_, v___x_1072_);
v___x_1074_ = lean_string_append(v___x_1071_, v___x_1073_);
lean_dec_ref(v___x_1073_);
v___x_1075_ = ((lean_object*)(lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1___closed__1));
v___x_1076_ = lean_string_append(v___x_1074_, v___x_1075_);
v___x_1077_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1077_, 0, v___x_1076_);
return v___x_1077_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(size_t v_sz_1078_, size_t v_i_1079_, lean_object* v_bs_1080_, lean_object* v___y_1081_){
_start:
{
uint8_t v___x_1082_; 
v___x_1082_ = lean_usize_dec_lt(v_i_1079_, v_sz_1078_);
if (v___x_1082_ == 0)
{
lean_object* v___x_1083_; 
v___x_1083_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1083_, 0, v_bs_1080_);
return v___x_1083_;
}
else
{
lean_object* v_v_1084_; lean_object* v___x_1085_; 
v_v_1084_ = lean_array_uget_borrowed(v_bs_1080_, v_i_1079_);
lean_inc(v_v_1084_);
v___x_1085_ = l_Lean_Widget_instRpcEncodableInteractiveGoal_dec_00___x40_Lean_Widget_InteractiveGoal_3114798910____hygCtx___hyg_1_(v_v_1084_, v___y_1081_);
if (lean_obj_tag(v___x_1085_) == 0)
{
lean_object* v_a_1086_; lean_object* v___x_1088_; uint8_t v_isShared_1089_; uint8_t v_isSharedCheck_1093_; 
lean_dec_ref(v_bs_1080_);
v_a_1086_ = lean_ctor_get(v___x_1085_, 0);
v_isSharedCheck_1093_ = !lean_is_exclusive(v___x_1085_);
if (v_isSharedCheck_1093_ == 0)
{
v___x_1088_ = v___x_1085_;
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
else
{
lean_inc(v_a_1086_);
lean_dec(v___x_1085_);
v___x_1088_ = lean_box(0);
v_isShared_1089_ = v_isSharedCheck_1093_;
goto v_resetjp_1087_;
}
v_resetjp_1087_:
{
lean_object* v___x_1091_; 
if (v_isShared_1089_ == 0)
{
v___x_1091_ = v___x_1088_;
goto v_reusejp_1090_;
}
else
{
lean_object* v_reuseFailAlloc_1092_; 
v_reuseFailAlloc_1092_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1092_, 0, v_a_1086_);
v___x_1091_ = v_reuseFailAlloc_1092_;
goto v_reusejp_1090_;
}
v_reusejp_1090_:
{
return v___x_1091_;
}
}
}
else
{
lean_object* v_a_1094_; lean_object* v___x_1095_; lean_object* v_bs_x27_1096_; size_t v___x_1097_; size_t v___x_1098_; lean_object* v___x_1099_; 
v_a_1094_ = lean_ctor_get(v___x_1085_, 0);
lean_inc(v_a_1094_);
lean_dec_ref_known(v___x_1085_, 1);
v___x_1095_ = lean_unsigned_to_nat(0u);
v_bs_x27_1096_ = lean_array_uset(v_bs_1080_, v_i_1079_, v___x_1095_);
v___x_1097_ = ((size_t)1ULL);
v___x_1098_ = lean_usize_add(v_i_1079_, v___x_1097_);
v___x_1099_ = lean_array_uset(v_bs_x27_1096_, v_i_1079_, v_a_1094_);
v_i_1079_ = v___x_1098_;
v_bs_1080_ = v___x_1099_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2___boxed(lean_object* v_sz_1101_, lean_object* v_i_1102_, lean_object* v_bs_1103_, lean_object* v___y_1104_){
_start:
{
size_t v_sz_boxed_1105_; size_t v_i_boxed_1106_; lean_object* v_res_1107_; 
v_sz_boxed_1105_ = lean_unbox_usize(v_sz_1101_);
lean_dec(v_sz_1101_);
v_i_boxed_1106_ = lean_unbox_usize(v_i_1102_);
lean_dec(v_i_1102_);
v_res_1107_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(v_sz_boxed_1105_, v_i_boxed_1106_, v_bs_1103_, v___y_1104_);
lean_dec_ref(v___y_1104_);
return v_res_1107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg(size_t v_sz_1108_, size_t v_i_1109_, lean_object* v_bs_1110_){
_start:
{
uint8_t v___x_1111_; 
v___x_1111_ = lean_usize_dec_lt(v_i_1109_, v_sz_1108_);
if (v___x_1111_ == 0)
{
lean_object* v___x_1112_; 
v___x_1112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1112_, 0, v_bs_1110_);
return v___x_1112_;
}
else
{
lean_object* v_v_1113_; lean_object* v___x_1114_; 
v_v_1113_ = lean_array_uget_borrowed(v_bs_1110_, v_i_1109_);
lean_inc(v_v_1113_);
v___x_1114_ = l_Lean_SubExpr_instFromJsonGoalsLocation_fromJson(v_v_1113_);
if (lean_obj_tag(v___x_1114_) == 0)
{
lean_object* v_a_1115_; lean_object* v___x_1117_; uint8_t v_isShared_1118_; uint8_t v_isSharedCheck_1122_; 
lean_dec_ref(v_bs_1110_);
v_a_1115_ = lean_ctor_get(v___x_1114_, 0);
v_isSharedCheck_1122_ = !lean_is_exclusive(v___x_1114_);
if (v_isSharedCheck_1122_ == 0)
{
v___x_1117_ = v___x_1114_;
v_isShared_1118_ = v_isSharedCheck_1122_;
goto v_resetjp_1116_;
}
else
{
lean_inc(v_a_1115_);
lean_dec(v___x_1114_);
v___x_1117_ = lean_box(0);
v_isShared_1118_ = v_isSharedCheck_1122_;
goto v_resetjp_1116_;
}
v_resetjp_1116_:
{
lean_object* v___x_1120_; 
if (v_isShared_1118_ == 0)
{
v___x_1120_ = v___x_1117_;
goto v_reusejp_1119_;
}
else
{
lean_object* v_reuseFailAlloc_1121_; 
v_reuseFailAlloc_1121_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1121_, 0, v_a_1115_);
v___x_1120_ = v_reuseFailAlloc_1121_;
goto v_reusejp_1119_;
}
v_reusejp_1119_:
{
return v___x_1120_;
}
}
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1124_; lean_object* v_bs_x27_1125_; size_t v___x_1126_; size_t v___x_1127_; lean_object* v___x_1128_; 
v_a_1123_ = lean_ctor_get(v___x_1114_, 0);
lean_inc(v_a_1123_);
lean_dec_ref_known(v___x_1114_, 1);
v___x_1124_ = lean_unsigned_to_nat(0u);
v_bs_x27_1125_ = lean_array_uset(v_bs_1110_, v_i_1109_, v___x_1124_);
v___x_1126_ = ((size_t)1ULL);
v___x_1127_ = lean_usize_add(v_i_1109_, v___x_1126_);
v___x_1128_ = lean_array_uset(v_bs_x27_1125_, v_i_1109_, v_a_1123_);
v_i_1109_ = v___x_1127_;
v_bs_1110_ = v___x_1128_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg___boxed(lean_object* v_sz_1130_, lean_object* v_i_1131_, lean_object* v_bs_1132_){
_start:
{
size_t v_sz_boxed_1133_; size_t v_i_boxed_1134_; lean_object* v_res_1135_; 
v_sz_boxed_1133_ = lean_unbox_usize(v_sz_1130_);
lean_dec(v_sz_1130_);
v_i_boxed_1134_ = lean_unbox_usize(v_i_1131_);
lean_dec(v_i_1131_);
v_res_1135_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg(v_sz_boxed_1133_, v_i_boxed_1134_, v_bs_1132_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(lean_object* v_j_1136_, lean_object* v_a_1137_){
_start:
{
lean_object* v___x_1138_; 
v___x_1138_ = lp_mathlib_instFromJsonRpcEncodablePacket_fromJson_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_16_(v_j_1136_);
if (lean_obj_tag(v___x_1138_) == 0)
{
lean_object* v_a_1139_; lean_object* v___x_1141_; uint8_t v_isShared_1142_; uint8_t v_isSharedCheck_1146_; 
v_a_1139_ = lean_ctor_get(v___x_1138_, 0);
v_isSharedCheck_1146_ = !lean_is_exclusive(v___x_1138_);
if (v_isSharedCheck_1146_ == 0)
{
v___x_1141_ = v___x_1138_;
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
else
{
lean_inc(v_a_1139_);
lean_dec(v___x_1138_);
v___x_1141_ = lean_box(0);
v_isShared_1142_ = v_isSharedCheck_1146_;
goto v_resetjp_1140_;
}
v_resetjp_1140_:
{
lean_object* v___x_1144_; 
if (v_isShared_1142_ == 0)
{
v___x_1144_ = v___x_1141_;
goto v_reusejp_1143_;
}
else
{
lean_object* v_reuseFailAlloc_1145_; 
v_reuseFailAlloc_1145_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1145_, 0, v_a_1139_);
v___x_1144_ = v_reuseFailAlloc_1145_;
goto v_reusejp_1143_;
}
v_reusejp_1143_:
{
return v___x_1144_;
}
}
}
else
{
lean_object* v_a_1147_; lean_object* v_pos_1148_; lean_object* v_goals_1149_; lean_object* v_selectedLocations_1150_; lean_object* v_replaceRange_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1228_; 
v_a_1147_ = lean_ctor_get(v___x_1138_, 0);
lean_inc(v_a_1147_);
lean_dec_ref_known(v___x_1138_, 1);
v_pos_1148_ = lean_ctor_get(v_a_1147_, 0);
v_goals_1149_ = lean_ctor_get(v_a_1147_, 1);
v_selectedLocations_1150_ = lean_ctor_get(v_a_1147_, 2);
v_replaceRange_1151_ = lean_ctor_get(v_a_1147_, 3);
v_isSharedCheck_1228_ = !lean_is_exclusive(v_a_1147_);
if (v_isSharedCheck_1228_ == 0)
{
v___x_1153_ = v_a_1147_;
v_isShared_1154_ = v_isSharedCheck_1228_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_replaceRange_1151_);
lean_inc(v_selectedLocations_1150_);
lean_inc(v_goals_1149_);
lean_inc(v_pos_1148_);
lean_dec(v_a_1147_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1228_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1155_; 
v___x_1155_ = l_Lean_Lsp_instFromJsonPosition_fromJson(v_pos_1148_);
if (lean_obj_tag(v___x_1155_) == 0)
{
lean_object* v_a_1156_; lean_object* v___x_1158_; uint8_t v_isShared_1159_; uint8_t v_isSharedCheck_1163_; 
lean_del_object(v___x_1153_);
lean_dec(v_replaceRange_1151_);
lean_dec(v_selectedLocations_1150_);
lean_dec(v_goals_1149_);
v_a_1156_ = lean_ctor_get(v___x_1155_, 0);
v_isSharedCheck_1163_ = !lean_is_exclusive(v___x_1155_);
if (v_isSharedCheck_1163_ == 0)
{
v___x_1158_ = v___x_1155_;
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
else
{
lean_inc(v_a_1156_);
lean_dec(v___x_1155_);
v___x_1158_ = lean_box(0);
v_isShared_1159_ = v_isSharedCheck_1163_;
goto v_resetjp_1157_;
}
v_resetjp_1157_:
{
lean_object* v___x_1161_; 
if (v_isShared_1159_ == 0)
{
v___x_1161_ = v___x_1158_;
goto v_reusejp_1160_;
}
else
{
lean_object* v_reuseFailAlloc_1162_; 
v_reuseFailAlloc_1162_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1162_, 0, v_a_1156_);
v___x_1161_ = v_reuseFailAlloc_1162_;
goto v_reusejp_1160_;
}
v_reusejp_1160_:
{
return v___x_1161_;
}
}
}
else
{
lean_object* v_a_1164_; lean_object* v___x_1165_; 
v_a_1164_ = lean_ctor_get(v___x_1155_, 0);
lean_inc(v_a_1164_);
lean_dec_ref_known(v___x_1155_, 1);
v___x_1165_ = lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(v_goals_1149_);
if (lean_obj_tag(v___x_1165_) == 0)
{
lean_object* v_a_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1173_; 
lean_dec(v_a_1164_);
lean_del_object(v___x_1153_);
lean_dec(v_replaceRange_1151_);
lean_dec(v_selectedLocations_1150_);
v_a_1166_ = lean_ctor_get(v___x_1165_, 0);
v_isSharedCheck_1173_ = !lean_is_exclusive(v___x_1165_);
if (v_isSharedCheck_1173_ == 0)
{
v___x_1168_ = v___x_1165_;
v_isShared_1169_ = v_isSharedCheck_1173_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_a_1166_);
lean_dec(v___x_1165_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1173_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
lean_object* v___x_1171_; 
if (v_isShared_1169_ == 0)
{
v___x_1171_ = v___x_1168_;
goto v_reusejp_1170_;
}
else
{
lean_object* v_reuseFailAlloc_1172_; 
v_reuseFailAlloc_1172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1172_, 0, v_a_1166_);
v___x_1171_ = v_reuseFailAlloc_1172_;
goto v_reusejp_1170_;
}
v_reusejp_1170_:
{
return v___x_1171_;
}
}
}
else
{
lean_object* v_a_1174_; size_t v_sz_1175_; size_t v___x_1176_; lean_object* v___x_1177_; 
v_a_1174_ = lean_ctor_get(v___x_1165_, 0);
lean_inc(v_a_1174_);
lean_dec_ref_known(v___x_1165_, 1);
v_sz_1175_ = lean_array_size(v_a_1174_);
v___x_1176_ = ((size_t)0ULL);
v___x_1177_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__2(v_sz_1175_, v___x_1176_, v_a_1174_, v_a_1137_);
if (lean_obj_tag(v___x_1177_) == 0)
{
lean_object* v_a_1178_; lean_object* v___x_1180_; uint8_t v_isShared_1181_; uint8_t v_isSharedCheck_1185_; 
lean_dec(v_a_1164_);
lean_del_object(v___x_1153_);
lean_dec(v_replaceRange_1151_);
lean_dec(v_selectedLocations_1150_);
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1185_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1180_ = v___x_1177_;
v_isShared_1181_ = v_isSharedCheck_1185_;
goto v_resetjp_1179_;
}
else
{
lean_inc(v_a_1178_);
lean_dec(v___x_1177_);
v___x_1180_ = lean_box(0);
v_isShared_1181_ = v_isSharedCheck_1185_;
goto v_resetjp_1179_;
}
v_resetjp_1179_:
{
lean_object* v___x_1183_; 
if (v_isShared_1181_ == 0)
{
v___x_1183_ = v___x_1180_;
goto v_reusejp_1182_;
}
else
{
lean_object* v_reuseFailAlloc_1184_; 
v_reuseFailAlloc_1184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1184_, 0, v_a_1178_);
v___x_1183_ = v_reuseFailAlloc_1184_;
goto v_reusejp_1182_;
}
v_reusejp_1182_:
{
return v___x_1183_;
}
}
}
else
{
lean_object* v_a_1186_; lean_object* v___x_1187_; 
v_a_1186_ = lean_ctor_get(v___x_1177_, 0);
lean_inc(v_a_1186_);
lean_dec_ref_known(v___x_1177_, 1);
v___x_1187_ = lp_mathlib_Lean_Array_fromJson_x3f___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__1(v_selectedLocations_1150_);
if (lean_obj_tag(v___x_1187_) == 0)
{
lean_object* v_a_1188_; lean_object* v___x_1190_; uint8_t v_isShared_1191_; uint8_t v_isSharedCheck_1195_; 
lean_dec(v_a_1186_);
lean_dec(v_a_1164_);
lean_del_object(v___x_1153_);
lean_dec(v_replaceRange_1151_);
v_a_1188_ = lean_ctor_get(v___x_1187_, 0);
v_isSharedCheck_1195_ = !lean_is_exclusive(v___x_1187_);
if (v_isSharedCheck_1195_ == 0)
{
v___x_1190_ = v___x_1187_;
v_isShared_1191_ = v_isSharedCheck_1195_;
goto v_resetjp_1189_;
}
else
{
lean_inc(v_a_1188_);
lean_dec(v___x_1187_);
v___x_1190_ = lean_box(0);
v_isShared_1191_ = v_isSharedCheck_1195_;
goto v_resetjp_1189_;
}
v_resetjp_1189_:
{
lean_object* v___x_1193_; 
if (v_isShared_1191_ == 0)
{
v___x_1193_ = v___x_1190_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1194_; 
v_reuseFailAlloc_1194_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1194_, 0, v_a_1188_);
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
lean_object* v_a_1196_; size_t v_sz_1197_; lean_object* v___x_1198_; 
v_a_1196_ = lean_ctor_get(v___x_1187_, 0);
lean_inc(v_a_1196_);
lean_dec_ref_known(v___x_1187_, 1);
v_sz_1197_ = lean_array_size(v_a_1196_);
v___x_1198_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg(v_sz_1197_, v___x_1176_, v_a_1196_);
if (lean_obj_tag(v___x_1198_) == 0)
{
lean_object* v_a_1199_; lean_object* v___x_1201_; uint8_t v_isShared_1202_; uint8_t v_isSharedCheck_1206_; 
lean_dec(v_a_1186_);
lean_dec(v_a_1164_);
lean_del_object(v___x_1153_);
lean_dec(v_replaceRange_1151_);
v_a_1199_ = lean_ctor_get(v___x_1198_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v___x_1198_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1201_ = v___x_1198_;
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
else
{
lean_inc(v_a_1199_);
lean_dec(v___x_1198_);
v___x_1201_ = lean_box(0);
v_isShared_1202_ = v_isSharedCheck_1206_;
goto v_resetjp_1200_;
}
v_resetjp_1200_:
{
lean_object* v___x_1204_; 
if (v_isShared_1202_ == 0)
{
v___x_1204_ = v___x_1201_;
goto v_reusejp_1203_;
}
else
{
lean_object* v_reuseFailAlloc_1205_; 
v_reuseFailAlloc_1205_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1205_, 0, v_a_1199_);
v___x_1204_ = v_reuseFailAlloc_1205_;
goto v_reusejp_1203_;
}
v_reusejp_1203_:
{
return v___x_1204_;
}
}
}
else
{
lean_object* v_a_1207_; lean_object* v___x_1208_; 
v_a_1207_ = lean_ctor_get(v___x_1198_, 0);
lean_inc(v_a_1207_);
lean_dec_ref_known(v___x_1198_, 1);
v___x_1208_ = l_Lean_Lsp_instFromJsonRange_fromJson(v_replaceRange_1151_);
if (lean_obj_tag(v___x_1208_) == 0)
{
lean_object* v_a_1209_; lean_object* v___x_1211_; uint8_t v_isShared_1212_; uint8_t v_isSharedCheck_1216_; 
lean_dec(v_a_1207_);
lean_dec(v_a_1186_);
lean_dec(v_a_1164_);
lean_del_object(v___x_1153_);
v_a_1209_ = lean_ctor_get(v___x_1208_, 0);
v_isSharedCheck_1216_ = !lean_is_exclusive(v___x_1208_);
if (v_isSharedCheck_1216_ == 0)
{
v___x_1211_ = v___x_1208_;
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
else
{
lean_inc(v_a_1209_);
lean_dec(v___x_1208_);
v___x_1211_ = lean_box(0);
v_isShared_1212_ = v_isSharedCheck_1216_;
goto v_resetjp_1210_;
}
v_resetjp_1210_:
{
lean_object* v___x_1214_; 
if (v_isShared_1212_ == 0)
{
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
return v___x_1214_;
}
}
}
else
{
lean_object* v_a_1217_; lean_object* v___x_1219_; uint8_t v_isShared_1220_; uint8_t v_isSharedCheck_1227_; 
v_a_1217_ = lean_ctor_get(v___x_1208_, 0);
v_isSharedCheck_1227_ = !lean_is_exclusive(v___x_1208_);
if (v_isSharedCheck_1227_ == 0)
{
v___x_1219_ = v___x_1208_;
v_isShared_1220_ = v_isSharedCheck_1227_;
goto v_resetjp_1218_;
}
else
{
lean_inc(v_a_1217_);
lean_dec(v___x_1208_);
v___x_1219_ = lean_box(0);
v_isShared_1220_ = v_isSharedCheck_1227_;
goto v_resetjp_1218_;
}
v_resetjp_1218_:
{
lean_object* v___x_1222_; 
if (v_isShared_1154_ == 0)
{
lean_ctor_set(v___x_1153_, 3, v_a_1217_);
lean_ctor_set(v___x_1153_, 2, v_a_1207_);
lean_ctor_set(v___x_1153_, 1, v_a_1186_);
lean_ctor_set(v___x_1153_, 0, v_a_1164_);
v___x_1222_ = v___x_1153_;
goto v_reusejp_1221_;
}
else
{
lean_object* v_reuseFailAlloc_1226_; 
v_reuseFailAlloc_1226_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_1226_, 0, v_a_1164_);
lean_ctor_set(v_reuseFailAlloc_1226_, 1, v_a_1186_);
lean_ctor_set(v_reuseFailAlloc_1226_, 2, v_a_1207_);
lean_ctor_set(v_reuseFailAlloc_1226_, 3, v_a_1217_);
v___x_1222_ = v_reuseFailAlloc_1226_;
goto v_reusejp_1221_;
}
v_reusejp_1221_:
{
lean_object* v___x_1224_; 
if (v_isShared_1220_ == 0)
{
lean_ctor_set(v___x_1219_, 0, v___x_1222_);
v___x_1224_ = v___x_1219_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1225_; 
v_reuseFailAlloc_1225_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1225_, 0, v___x_1222_);
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
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1____boxed(lean_object* v_j_1229_, lean_object* v_a_1230_){
_start:
{
lean_object* v_res_1231_; 
v_res_1231_ = lp_mathlib_instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1_(v_j_1229_, v_a_1230_);
lean_dec_ref(v_a_1230_);
return v_res_1231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3(size_t v_sz_1232_, size_t v_i_1233_, lean_object* v_bs_1234_, lean_object* v___y_1235_){
_start:
{
lean_object* v___x_1236_; 
v___x_1236_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___redArg(v_sz_1232_, v_i_1233_, v_bs_1234_);
return v___x_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3___boxed(lean_object* v_sz_1237_, lean_object* v_i_1238_, lean_object* v_bs_1239_, lean_object* v___y_1240_){
_start:
{
size_t v_sz_boxed_1241_; size_t v_i_boxed_1242_; lean_object* v_res_1243_; 
v_sz_boxed_1241_ = lean_unbox_usize(v_sz_1237_);
lean_dec(v_sz_1237_);
v_i_boxed_1242_ = lean_unbox_usize(v_i_1238_);
lean_dec(v_i_1238_);
v_res_1243_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00instRpcEncodableSelectInsertParams_dec_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2749655504____hygCtx___hyg_1__spec__3(v_sz_boxed_1241_, v_i_boxed_1242_, v_bs_1239_, v___y_1240_);
lean_dec_ref(v___y_1240_);
return v_res_1243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__0(lean_object* v_mvarId_1251_, lean_object* v___x_1252_, lean_object* v_errorMsg_1253_, lean_object* v___x_1254_, uint8_t v_onlyGoal_1255_, lean_object* v___x_1256_, lean_object* v_a_1257_, lean_object* v_x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_){
_start:
{
lean_object* v_mvarId_1262_; lean_object* v_loc_1263_; lean_object* v___x_1265_; uint8_t v_isShared_1266_; uint8_t v_isSharedCheck_1305_; 
v_mvarId_1262_ = lean_ctor_get(v_a_1257_, 0);
v_loc_1263_ = lean_ctor_get(v_a_1257_, 1);
v_isSharedCheck_1305_ = !lean_is_exclusive(v_a_1257_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1265_ = v_a_1257_;
v_isShared_1266_ = v_isSharedCheck_1305_;
goto v_resetjp_1264_;
}
else
{
lean_inc(v_loc_1263_);
lean_inc(v_mvarId_1262_);
lean_dec(v_a_1257_);
v___x_1265_ = lean_box(0);
v_isShared_1266_ = v_isSharedCheck_1305_;
goto v_resetjp_1264_;
}
v_resetjp_1264_:
{
uint8_t v___x_1267_; 
v___x_1267_ = lean_name_eq(v_mvarId_1262_, v_mvarId_1251_);
lean_dec(v_mvarId_1262_);
if (v___x_1267_ == 0)
{
lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1277_; 
lean_dec_ref(v_loc_1263_);
lean_dec_ref(v___x_1256_);
v___x_1268_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0));
v___x_1269_ = lean_mk_empty_array_with_capacity(v___x_1252_);
v___x_1270_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1270_, 0, v_errorMsg_1253_);
v___x_1271_ = lean_unsigned_to_nat(1u);
v___x_1272_ = lean_mk_empty_array_with_capacity(v___x_1271_);
v___x_1273_ = lean_array_push(v___x_1272_, v___x_1270_);
v___x_1274_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1268_);
lean_ctor_set(v___x_1274_, 1, v___x_1269_);
lean_ctor_set(v___x_1274_, 2, v___x_1273_);
v___x_1275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1274_);
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 1, v___x_1254_);
lean_ctor_set(v___x_1265_, 0, v___x_1275_);
v___x_1277_ = v___x_1265_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v___x_1275_);
lean_ctor_set(v_reuseFailAlloc_1280_, 1, v___x_1254_);
v___x_1277_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
lean_object* v___x_1278_; lean_object* v___x_1279_; 
v___x_1278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
v___x_1279_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1278_);
return v___x_1279_;
}
}
else
{
if (v_onlyGoal_1255_ == 0)
{
lean_object* v___x_1281_; lean_object* v___x_1282_; 
lean_del_object(v___x_1265_);
lean_dec_ref(v_loc_1263_);
lean_dec_ref(v_errorMsg_1253_);
v___x_1281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1281_, 0, v___x_1256_);
v___x_1282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1282_, 0, v___x_1281_);
return v___x_1282_;
}
else
{
if (lean_obj_tag(v_loc_1263_) == 3)
{
lean_object* v___x_1284_; uint8_t v_isShared_1285_; uint8_t v_isSharedCheck_1290_; 
lean_del_object(v___x_1265_);
lean_dec_ref(v_errorMsg_1253_);
v_isSharedCheck_1290_ = !lean_is_exclusive(v_loc_1263_);
if (v_isSharedCheck_1290_ == 0)
{
lean_object* v_unused_1291_; 
v_unused_1291_ = lean_ctor_get(v_loc_1263_, 0);
lean_dec(v_unused_1291_);
v___x_1284_ = v_loc_1263_;
v_isShared_1285_ = v_isSharedCheck_1290_;
goto v_resetjp_1283_;
}
else
{
lean_dec(v_loc_1263_);
v___x_1284_ = lean_box(0);
v_isShared_1285_ = v_isSharedCheck_1290_;
goto v_resetjp_1283_;
}
v_resetjp_1283_:
{
lean_object* v___x_1287_; 
if (v_isShared_1285_ == 0)
{
lean_ctor_set_tag(v___x_1284_, 1);
lean_ctor_set(v___x_1284_, 0, v___x_1256_);
v___x_1287_ = v___x_1284_;
goto v_reusejp_1286_;
}
else
{
lean_object* v_reuseFailAlloc_1289_; 
v_reuseFailAlloc_1289_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1289_, 0, v___x_1256_);
v___x_1287_ = v_reuseFailAlloc_1289_;
goto v_reusejp_1286_;
}
v_reusejp_1286_:
{
lean_object* v___x_1288_; 
v___x_1288_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1288_, 0, v___x_1287_);
return v___x_1288_;
}
}
}
else
{
lean_object* v___x_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; lean_object* v___x_1296_; lean_object* v___x_1297_; lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1301_; 
lean_dec_ref(v_loc_1263_);
lean_dec_ref(v___x_1256_);
v___x_1292_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0));
v___x_1293_ = lean_mk_empty_array_with_capacity(v___x_1252_);
v___x_1294_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1294_, 0, v_errorMsg_1253_);
v___x_1295_ = lean_unsigned_to_nat(1u);
v___x_1296_ = lean_mk_empty_array_with_capacity(v___x_1295_);
v___x_1297_ = lean_array_push(v___x_1296_, v___x_1294_);
v___x_1298_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1298_, 0, v___x_1292_);
lean_ctor_set(v___x_1298_, 1, v___x_1293_);
lean_ctor_set(v___x_1298_, 2, v___x_1297_);
v___x_1299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1299_, 0, v___x_1298_);
if (v_isShared_1266_ == 0)
{
lean_ctor_set(v___x_1265_, 1, v___x_1254_);
lean_ctor_set(v___x_1265_, 0, v___x_1299_);
v___x_1301_ = v___x_1265_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v___x_1299_);
lean_ctor_set(v_reuseFailAlloc_1304_, 1, v___x_1254_);
v___x_1301_ = v_reuseFailAlloc_1304_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
lean_object* v___x_1302_; lean_object* v___x_1303_; 
v___x_1302_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1302_, 0, v___x_1301_);
v___x_1303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1303_, 0, v___x_1302_);
return v___x_1303_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___boxed(lean_object* v_mvarId_1306_, lean_object* v___x_1307_, lean_object* v_errorMsg_1308_, lean_object* v___x_1309_, lean_object* v_onlyGoal_1310_, lean_object* v___x_1311_, lean_object* v_a_1312_, lean_object* v_x_1313_, lean_object* v___y_1314_, lean_object* v___y_1315_, lean_object* v___y_1316_){
_start:
{
uint8_t v_onlyGoal_boxed_1317_; lean_object* v_res_1318_; 
v_onlyGoal_boxed_1317_ = lean_unbox(v_onlyGoal_1310_);
v_res_1318_ = lp_mathlib_mkSelectionPanelRPC___redArg___lam__0(v_mvarId_1306_, v___x_1307_, v_errorMsg_1308_, v___x_1309_, v_onlyGoal_boxed_1317_, v___x_1311_, v_a_1312_, v_x_1313_, v___y_1314_, v___y_1315_);
lean_dec_ref(v___y_1315_);
lean_dec_ref(v___y_1314_);
lean_dec(v___x_1307_);
lean_dec(v_mvarId_1306_);
return v_res_1318_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1321_; lean_object* v___x_1322_; lean_object* v___x_1323_; 
v___x_1321_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__1));
v___x_1322_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__0));
v___x_1323_ = l_Lean_Server_instRpcEncodableOfFromJsonOfToJson___redArg(v___x_1322_, v___x_1321_);
return v___x_1323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1(lean_object* v_mkCmdStr_1324_, lean_object* v___x_1325_, lean_object* v___x_1326_, lean_object* v_params_1327_, lean_object* v_doc_1328_, lean_object* v_replaceRange_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_){
_start:
{
lean_object* v___x_1335_; 
lean_inc(v_params_1327_);
v___x_1335_ = lean_apply_8(v_mkCmdStr_1324_, v___x_1325_, v___x_1326_, v_params_1327_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, lean_box(0));
if (lean_obj_tag(v___x_1335_) == 0)
{
lean_object* v_a_1336_; lean_object* v___x_1338_; uint8_t v_isShared_1339_; uint8_t v_isSharedCheck_1358_; 
v_a_1336_ = lean_ctor_get(v___x_1335_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1335_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1338_ = v___x_1335_;
v_isShared_1339_ = v_isSharedCheck_1358_;
goto v_resetjp_1337_;
}
else
{
lean_inc(v_a_1336_);
lean_dec(v___x_1335_);
v___x_1338_ = lean_box(0);
v_isShared_1339_ = v_isSharedCheck_1358_;
goto v_resetjp_1337_;
}
v_resetjp_1337_:
{
lean_object* v_snd_1340_; lean_object* v_fst_1341_; lean_object* v_fst_1342_; lean_object* v_snd_1343_; lean_object* v___x_1344_; lean_object* v_toEditableDocumentCore_1345_; lean_object* v_meta_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; lean_object* v___x_1352_; lean_object* v___x_1353_; lean_object* v___x_1354_; lean_object* v___x_1356_; 
v_snd_1340_ = lean_ctor_get(v_a_1336_, 1);
lean_inc(v_snd_1340_);
v_fst_1341_ = lean_ctor_get(v_a_1336_, 0);
lean_inc(v_fst_1341_);
lean_dec(v_a_1336_);
v_fst_1342_ = lean_ctor_get(v_snd_1340_, 0);
lean_inc(v_fst_1342_);
v_snd_1343_ = lean_ctor_get(v_snd_1340_, 1);
lean_inc(v_snd_1343_);
lean_dec(v_snd_1340_);
v___x_1344_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___closed__2);
v_toEditableDocumentCore_1345_ = lean_ctor_get(v_doc_1328_, 0);
v_meta_1346_ = lean_ctor_get(v_toEditableDocumentCore_1345_, 0);
v___x_1347_ = lp_proofwidgets_ProofWidgets_MakeEditLink;
v___x_1348_ = lean_apply_1(v_replaceRange_1329_, v_params_1327_);
v___x_1349_ = lp_proofwidgets_ProofWidgets_MakeEditLinkProps_ofReplaceRange(v_meta_1346_, v___x_1348_, v_fst_1342_, v_snd_1343_);
v___x_1350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1350_, 0, v_fst_1341_);
v___x_1351_ = lean_unsigned_to_nat(1u);
v___x_1352_ = lean_mk_empty_array_with_capacity(v___x_1351_);
v___x_1353_ = lean_array_push(v___x_1352_, v___x_1350_);
v___x_1354_ = lp_proofwidgets_ProofWidgets_Html_ofComponent___redArg(v___x_1344_, v___x_1347_, v___x_1349_, v___x_1353_);
if (v_isShared_1339_ == 0)
{
lean_ctor_set(v___x_1338_, 0, v___x_1354_);
v___x_1356_ = v___x_1338_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v___x_1354_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
else
{
lean_object* v_a_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1366_; 
lean_dec_ref(v_replaceRange_1329_);
lean_dec(v_params_1327_);
v_a_1359_ = lean_ctor_get(v___x_1335_, 0);
v_isSharedCheck_1366_ = !lean_is_exclusive(v___x_1335_);
if (v_isSharedCheck_1366_ == 0)
{
v___x_1361_ = v___x_1335_;
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_a_1359_);
lean_dec(v___x_1335_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1366_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
lean_object* v___x_1364_; 
if (v_isShared_1362_ == 0)
{
v___x_1364_ = v___x_1361_;
goto v_reusejp_1363_;
}
else
{
lean_object* v_reuseFailAlloc_1365_; 
v_reuseFailAlloc_1365_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1365_, 0, v_a_1359_);
v___x_1364_ = v_reuseFailAlloc_1365_;
goto v_reusejp_1363_;
}
v_reusejp_1363_:
{
return v___x_1364_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___boxed(lean_object* v_mkCmdStr_1367_, lean_object* v___x_1368_, lean_object* v___x_1369_, lean_object* v_params_1370_, lean_object* v_doc_1371_, lean_object* v_replaceRange_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_){
_start:
{
lean_object* v_res_1378_; 
v_res_1378_ = lp_mathlib_mkSelectionPanelRPC___redArg___lam__1(v_mkCmdStr_1367_, v___x_1368_, v___x_1369_, v_params_1370_, v_doc_1371_, v_replaceRange_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_);
lean_dec_ref(v_doc_1371_);
return v_res_1378_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0(void){
_start:
{
lean_object* v___x_1379_; 
v___x_1379_ = l_instMonadEIO(lean_box(0));
return v___x_1379_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1(void){
_start:
{
lean_object* v___x_1380_; lean_object* v___x_1381_; 
v___x_1380_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0);
v___x_1381_ = l_StateRefT_x27_instMonad___redArg(v___x_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2(lean_object* v_mvarId_1384_, lean_object* v_mkCmdStr_1385_, lean_object* v___x_1386_, lean_object* v_params_1387_, lean_object* v_doc_1388_, lean_object* v_replaceRange_1389_, lean_object* v___x_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_, lean_object* v___y_1393_, lean_object* v___y_1394_){
_start:
{
lean_object* v___x_1396_; 
v___x_1396_ = l_Lean_MVarId_getDecl(v_mvarId_1384_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_);
if (lean_obj_tag(v___x_1396_) == 0)
{
lean_object* v_a_1397_; lean_object* v_options_1398_; lean_object* v_lctx_1399_; lean_object* v_type_1400_; lean_object* v_localInstances_1401_; lean_object* v___x_1402_; lean_object* v___x_1403_; lean_object* v___x_1404_; lean_object* v_fst_1405_; lean_object* v___x_1407_; uint8_t v_isShared_1408_; uint8_t v_isSharedCheck_1434_; 
v_a_1397_ = lean_ctor_get(v___x_1396_, 0);
lean_inc(v_a_1397_);
lean_dec_ref_known(v___x_1396_, 1);
v_options_1398_ = lean_ctor_get(v___y_1393_, 2);
v_lctx_1399_ = lean_ctor_get(v_a_1397_, 1);
lean_inc_ref(v_lctx_1399_);
v_type_1400_ = lean_ctor_get(v_a_1397_, 2);
lean_inc_ref(v_type_1400_);
v_localInstances_1401_ = lean_ctor_get(v_a_1397_, 4);
lean_inc_ref(v_localInstances_1401_);
lean_dec(v_a_1397_);
v___x_1402_ = lean_box(1);
lean_inc_ref(v_options_1398_);
v___x_1403_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1403_, 0, v_options_1398_);
lean_ctor_set(v___x_1403_, 1, v___x_1402_);
lean_ctor_set(v___x_1403_, 2, v___x_1402_);
v___x_1404_ = l_Lean_LocalContext_sanitizeNames(v_lctx_1399_, v___x_1403_);
v_fst_1405_ = lean_ctor_get(v___x_1404_, 0);
v_isSharedCheck_1434_ = !lean_is_exclusive(v___x_1404_);
if (v_isSharedCheck_1434_ == 0)
{
lean_object* v_unused_1435_; 
v_unused_1435_ = lean_ctor_get(v___x_1404_, 1);
lean_dec(v_unused_1435_);
v___x_1407_ = v___x_1404_;
v_isShared_1408_ = v_isSharedCheck_1434_;
goto v_resetjp_1406_;
}
else
{
lean_inc(v_fst_1405_);
lean_dec(v___x_1404_);
v___x_1407_ = lean_box(0);
v_isShared_1408_ = v_isSharedCheck_1434_;
goto v_resetjp_1406_;
}
v_resetjp_1406_:
{
lean_object* v___x_1409_; lean_object* v_toApplicative_1410_; lean_object* v_toFunctor_1411_; lean_object* v_toSeq_1412_; lean_object* v_toSeqLeft_1413_; lean_object* v_toSeqRight_1414_; lean_object* v___f_1415_; lean_object* v___f_1416_; lean_object* v___f_1417_; lean_object* v___f_1418_; lean_object* v___x_1420_; 
v___x_1409_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1);
v_toApplicative_1410_ = lean_ctor_get(v___x_1409_, 0);
v_toFunctor_1411_ = lean_ctor_get(v_toApplicative_1410_, 0);
v_toSeq_1412_ = lean_ctor_get(v_toApplicative_1410_, 2);
v_toSeqLeft_1413_ = lean_ctor_get(v_toApplicative_1410_, 3);
v_toSeqRight_1414_ = lean_ctor_get(v_toApplicative_1410_, 4);
v___f_1415_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__2));
v___f_1416_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__3));
lean_inc_ref_n(v_toFunctor_1411_, 2);
v___f_1417_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1417_, 0, v_toFunctor_1411_);
v___f_1418_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1418_, 0, v_toFunctor_1411_);
if (v_isShared_1408_ == 0)
{
lean_ctor_set(v___x_1407_, 1, v___f_1418_);
lean_ctor_set(v___x_1407_, 0, v___f_1417_);
v___x_1420_ = v___x_1407_;
goto v_reusejp_1419_;
}
else
{
lean_object* v_reuseFailAlloc_1433_; 
v_reuseFailAlloc_1433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1433_, 0, v___f_1417_);
lean_ctor_set(v_reuseFailAlloc_1433_, 1, v___f_1418_);
v___x_1420_ = v_reuseFailAlloc_1433_;
goto v_reusejp_1419_;
}
v_reusejp_1419_:
{
lean_object* v___f_1421_; lean_object* v___f_1422_; lean_object* v___f_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; lean_object* v___f_1430_; lean_object* v___x_4525__overap_1431_; lean_object* v___x_1432_; 
lean_inc(v_toSeqRight_1414_);
v___f_1421_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1421_, 0, v_toSeqRight_1414_);
lean_inc(v_toSeqLeft_1413_);
v___f_1422_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1422_, 0, v_toSeqLeft_1413_);
lean_inc(v_toSeq_1412_);
v___f_1423_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1423_, 0, v_toSeq_1412_);
v___x_1424_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1424_, 0, v___x_1420_);
lean_ctor_set(v___x_1424_, 1, v___f_1415_);
lean_ctor_set(v___x_1424_, 2, v___f_1423_);
lean_ctor_set(v___x_1424_, 3, v___f_1422_);
lean_ctor_set(v___x_1424_, 4, v___f_1421_);
v___x_1425_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1424_);
lean_ctor_set(v___x_1425_, 1, v___f_1416_);
v___x_1426_ = l_StateRefT_x27_instMonad___redArg(v___x_1425_);
v___x_1427_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_1427_, 0, lean_box(0));
lean_closure_set(v___x_1427_, 1, lean_box(0));
lean_closure_set(v___x_1427_, 2, v___x_1426_);
v___x_1428_ = l_instMonadControlTOfPure___redArg(v___x_1427_);
v___x_1429_ = l_Lean_Expr_consumeMData(v_type_1400_);
lean_dec_ref(v_type_1400_);
v___f_1430_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__1___boxed), 11, 6);
lean_closure_set(v___f_1430_, 0, v_mkCmdStr_1385_);
lean_closure_set(v___f_1430_, 1, v___x_1386_);
lean_closure_set(v___f_1430_, 2, v___x_1429_);
lean_closure_set(v___f_1430_, 3, v_params_1387_);
lean_closure_set(v___f_1430_, 4, v_doc_1388_);
lean_closure_set(v___f_1430_, 5, v_replaceRange_1389_);
v___x_4525__overap_1431_ = l_Lean_Meta_withLCtx___redArg(v___x_1428_, v___x_1390_, v_fst_1405_, v_localInstances_1401_, v___f_1430_);
v___x_1432_ = lean_apply_5(v___x_4525__overap_1431_, v___y_1391_, v___y_1392_, v___y_1393_, v___y_1394_, lean_box(0));
return v___x_1432_;
}
}
}
else
{
lean_object* v_a_1436_; lean_object* v___x_1438_; uint8_t v_isShared_1439_; uint8_t v_isSharedCheck_1443_; 
lean_dec(v___y_1394_);
lean_dec_ref(v___y_1393_);
lean_dec(v___y_1392_);
lean_dec_ref(v___y_1391_);
lean_dec_ref(v___x_1390_);
lean_dec_ref(v_replaceRange_1389_);
lean_dec_ref(v_doc_1388_);
lean_dec(v_params_1387_);
lean_dec_ref(v___x_1386_);
lean_dec_ref(v_mkCmdStr_1385_);
v_a_1436_ = lean_ctor_get(v___x_1396_, 0);
v_isSharedCheck_1443_ = !lean_is_exclusive(v___x_1396_);
if (v_isSharedCheck_1443_ == 0)
{
v___x_1438_ = v___x_1396_;
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
else
{
lean_inc(v_a_1436_);
lean_dec(v___x_1396_);
v___x_1438_ = lean_box(0);
v_isShared_1439_ = v_isSharedCheck_1443_;
goto v_resetjp_1437_;
}
v_resetjp_1437_:
{
lean_object* v___x_1441_; 
if (v_isShared_1439_ == 0)
{
v___x_1441_ = v___x_1438_;
goto v_reusejp_1440_;
}
else
{
lean_object* v_reuseFailAlloc_1442_; 
v_reuseFailAlloc_1442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1442_, 0, v_a_1436_);
v___x_1441_ = v_reuseFailAlloc_1442_;
goto v_reusejp_1440_;
}
v_reusejp_1440_:
{
return v___x_1441_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___boxed(lean_object* v_mvarId_1444_, lean_object* v_mkCmdStr_1445_, lean_object* v___x_1446_, lean_object* v_params_1447_, lean_object* v_doc_1448_, lean_object* v_replaceRange_1449_, lean_object* v___x_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_){
_start:
{
lean_object* v_res_1456_; 
v_res_1456_ = lp_mathlib_mkSelectionPanelRPC___redArg___lam__2(v_mvarId_1444_, v_mkCmdStr_1445_, v___x_1446_, v_params_1447_, v_doc_1448_, v_replaceRange_1449_, v___x_1450_, v___y_1451_, v___y_1452_, v___y_1453_, v___y_1454_);
return v_res_1456_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18(void){
_start:
{
lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___x_1496_ = lean_unsigned_to_nat(32u);
v___x_1497_ = lean_mk_empty_array_with_capacity(v___x_1496_);
v___x_1498_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1498_, 0, v___x_1497_);
return v___x_1498_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19(void){
_start:
{
size_t v___x_1499_; lean_object* v___x_1500_; lean_object* v___x_1501_; lean_object* v___x_1502_; lean_object* v___x_1503_; lean_object* v___x_1504_; 
v___x_1499_ = ((size_t)5ULL);
v___x_1500_ = lean_unsigned_to_nat(0u);
v___x_1501_ = lean_unsigned_to_nat(32u);
v___x_1502_ = lean_mk_empty_array_with_capacity(v___x_1501_);
v___x_1503_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__18);
v___x_1504_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1504_, 0, v___x_1503_);
lean_ctor_set(v___x_1504_, 1, v___x_1502_);
lean_ctor_set(v___x_1504_, 2, v___x_1500_);
lean_ctor_set(v___x_1504_, 3, v___x_1500_);
lean_ctor_set_usize(v___x_1504_, 4, v___x_1499_);
return v___x_1504_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20(void){
_start:
{
lean_object* v___x_1505_; 
v___x_1505_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1505_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21(void){
_start:
{
lean_object* v___x_1506_; lean_object* v___x_1507_; 
v___x_1506_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__20);
v___x_1507_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1507_, 0, v___x_1506_);
return v___x_1507_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22(void){
_start:
{
lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; 
v___x_1508_ = lean_box(1);
v___x_1509_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__19);
v___x_1510_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__21);
v___x_1511_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1511_, 0, v___x_1510_);
lean_ctor_set(v___x_1511_, 1, v___x_1509_);
lean_ctor_set(v___x_1511_, 2, v___x_1508_);
return v___x_1511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3(lean_object* v_inst_1530_, lean_object* v_params_1531_, lean_object* v_title_1532_, uint8_t v_onlyGoal_1533_, lean_object* v___x_1534_, lean_object* v_mkCmdStr_1535_, lean_object* v_helpMsg_1536_, uint8_t v_onlyOne_1537_, lean_object* v_doc_1538_, lean_object* v___y_1539_){
_start:
{
lean_object* v_goals_1541_; lean_object* v_selectedLocations_1542_; lean_object* v_replaceRange_1543_; lean_object* v___x_1544_; lean_object* v___x_1545_; lean_object* v___x_1546_; uint8_t v___x_1547_; lean_object* v_a_1549_; 
v_goals_1541_ = lean_ctor_get(v_inst_1530_, 1);
lean_inc_ref(v_goals_1541_);
v_selectedLocations_1542_ = lean_ctor_get(v_inst_1530_, 2);
lean_inc_ref(v_selectedLocations_1542_);
v_replaceRange_1543_ = lean_ctor_get(v_inst_1530_, 3);
lean_inc_ref(v_replaceRange_1543_);
lean_dec_ref(v_inst_1530_);
v___x_1544_ = lean_unsigned_to_nat(0u);
lean_inc(v_params_1531_);
v___x_1545_ = lean_apply_1(v_goals_1541_, v_params_1531_);
v___x_1546_ = lean_array_get_size(v___x_1545_);
v___x_1547_ = lean_nat_dec_lt(v___x_1544_, v___x_1546_);
if (v___x_1547_ == 0)
{
lean_object* v___x_1572_; lean_object* v___x_1573_; 
lean_dec_ref(v___x_1545_);
lean_dec_ref(v_replaceRange_1543_);
lean_dec_ref(v_selectedLocations_1542_);
lean_dec_ref(v_doc_1538_);
lean_dec_ref(v_helpMsg_1536_);
lean_dec_ref(v_mkCmdStr_1535_);
lean_dec_ref(v___x_1534_);
lean_dec_ref(v_title_1532_);
lean_dec(v_params_1531_);
v___x_1572_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__16));
v___x_1573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1573_, 0, v___x_1572_);
return v___x_1573_;
}
else
{
lean_object* v_mainGoal_1574_; lean_object* v_toInteractiveGoalCore_1575_; lean_object* v_mvarId_1576_; lean_object* v___y_1578_; lean_object* v___y_1677_; lean_object* v___y_1678_; lean_object* v___y_1688_; 
v_mainGoal_1574_ = lean_array_fget(v___x_1545_, v___x_1544_);
lean_dec_ref(v___x_1545_);
v_toInteractiveGoalCore_1575_ = lean_ctor_get(v_mainGoal_1574_, 0);
lean_inc_ref(v_toInteractiveGoalCore_1575_);
v_mvarId_1576_ = lean_ctor_get(v_mainGoal_1574_, 3);
lean_inc(v_mvarId_1576_);
lean_dec(v_mainGoal_1574_);
if (v_onlyOne_1537_ == 0)
{
lean_object* v___x_1691_; 
v___x_1691_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__32));
v___y_1688_ = v___x_1691_;
goto v___jp_1687_;
}
else
{
lean_object* v___x_1692_; 
v___x_1692_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__33));
v___y_1688_ = v___x_1692_;
goto v___jp_1687_;
}
v___jp_1577_:
{
lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; lean_object* v___f_1583_; size_t v_sz_1584_; size_t v___x_1585_; lean_object* v___x_4591__overap_1586_; lean_object* v___x_1587_; 
lean_inc(v_params_1531_);
v___x_1579_ = lean_apply_1(v_selectedLocations_1542_, v_params_1531_);
v___x_1580_ = lean_box(0);
v___x_1581_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__17));
v___x_1582_ = lean_box(v_onlyGoal_1533_);
lean_inc(v_mvarId_1576_);
v___f_1583_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___boxed), 11, 6);
lean_closure_set(v___f_1583_, 0, v_mvarId_1576_);
lean_closure_set(v___f_1583_, 1, v___x_1544_);
lean_closure_set(v___f_1583_, 2, v___y_1578_);
lean_closure_set(v___f_1583_, 3, v___x_1580_);
lean_closure_set(v___f_1583_, 4, v___x_1582_);
lean_closure_set(v___f_1583_, 5, v___x_1581_);
v_sz_1584_ = lean_array_size(v___x_1579_);
v___x_1585_ = ((size_t)0ULL);
lean_inc_ref(v___x_1579_);
v___x_4591__overap_1586_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_1534_, v___x_1579_, v___f_1583_, v_sz_1584_, v___x_1585_, v___x_1581_);
lean_inc_ref(v___y_1539_);
v___x_1587_ = lean_apply_2(v___x_4591__overap_1586_, v___y_1539_, lean_box(0));
if (lean_obj_tag(v___x_1587_) == 0)
{
lean_object* v_a_1588_; lean_object* v_fst_1589_; lean_object* v___x_1591_; uint8_t v_isShared_1592_; uint8_t v_isSharedCheck_1666_; 
v_a_1588_ = lean_ctor_get(v___x_1587_, 0);
lean_inc(v_a_1588_);
lean_dec_ref_known(v___x_1587_, 1);
v_fst_1589_ = lean_ctor_get(v_a_1588_, 0);
v_isSharedCheck_1666_ = !lean_is_exclusive(v_a_1588_);
if (v_isSharedCheck_1666_ == 0)
{
lean_object* v_unused_1667_; 
v_unused_1667_ = lean_ctor_get(v_a_1588_, 1);
lean_dec(v_unused_1667_);
v___x_1591_ = v_a_1588_;
v_isShared_1592_ = v_isSharedCheck_1666_;
goto v_resetjp_1590_;
}
else
{
lean_inc(v_fst_1589_);
lean_dec(v_a_1588_);
v___x_1591_ = lean_box(0);
v_isShared_1592_ = v_isSharedCheck_1666_;
goto v_resetjp_1590_;
}
v_resetjp_1590_:
{
if (lean_obj_tag(v_fst_1589_) == 0)
{
lean_object* v___x_1593_; uint8_t v___x_1594_; 
v___x_1593_ = lean_array_get_size(v___x_1579_);
v___x_1594_ = lean_nat_dec_eq(v___x_1593_, v___x_1544_);
if (v___x_1594_ == 0)
{
lean_object* v_ctx_1595_; lean_object* v_val_1596_; lean_object* v___x_1597_; lean_object* v_toApplicative_1598_; lean_object* v_toFunctor_1599_; lean_object* v_toSeq_1600_; lean_object* v_toSeqLeft_1601_; lean_object* v_toSeqRight_1602_; lean_object* v___f_1603_; lean_object* v___f_1604_; lean_object* v___f_1605_; lean_object* v___f_1606_; lean_object* v___x_1608_; 
lean_dec_ref(v_helpMsg_1536_);
v_ctx_1595_ = lean_ctor_get(v_toInteractiveGoalCore_1575_, 2);
lean_inc_ref(v_ctx_1595_);
lean_dec_ref(v_toInteractiveGoalCore_1575_);
v_val_1596_ = lean_ctor_get(v_ctx_1595_, 0);
lean_inc(v_val_1596_);
lean_dec_ref(v_ctx_1595_);
v___x_1597_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__1);
v_toApplicative_1598_ = lean_ctor_get(v___x_1597_, 0);
v_toFunctor_1599_ = lean_ctor_get(v_toApplicative_1598_, 0);
v_toSeq_1600_ = lean_ctor_get(v_toApplicative_1598_, 2);
v_toSeqLeft_1601_ = lean_ctor_get(v_toApplicative_1598_, 3);
v_toSeqRight_1602_ = lean_ctor_get(v_toApplicative_1598_, 4);
v___f_1603_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__2));
v___f_1604_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__3));
lean_inc_ref_n(v_toFunctor_1599_, 2);
v___f_1605_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1605_, 0, v_toFunctor_1599_);
v___f_1606_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1606_, 0, v_toFunctor_1599_);
if (v_isShared_1592_ == 0)
{
lean_ctor_set(v___x_1591_, 1, v___f_1606_);
lean_ctor_set(v___x_1591_, 0, v___f_1605_);
v___x_1608_ = v___x_1591_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1657_; 
v_reuseFailAlloc_1657_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1657_, 0, v___f_1605_);
lean_ctor_set(v_reuseFailAlloc_1657_, 1, v___f_1606_);
v___x_1608_ = v_reuseFailAlloc_1657_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
lean_object* v___f_1609_; lean_object* v___f_1610_; lean_object* v___f_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v_toApplicative_1615_; lean_object* v___x_1617_; uint8_t v_isShared_1618_; uint8_t v_isSharedCheck_1655_; 
lean_inc(v_toSeqRight_1602_);
v___f_1609_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1609_, 0, v_toSeqRight_1602_);
lean_inc(v_toSeqLeft_1601_);
v___f_1610_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1610_, 0, v_toSeqLeft_1601_);
lean_inc(v_toSeq_1600_);
v___f_1611_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1611_, 0, v_toSeq_1600_);
v___x_1612_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1612_, 0, v___x_1608_);
lean_ctor_set(v___x_1612_, 1, v___f_1603_);
lean_ctor_set(v___x_1612_, 2, v___f_1611_);
lean_ctor_set(v___x_1612_, 3, v___f_1610_);
lean_ctor_set(v___x_1612_, 4, v___f_1609_);
v___x_1613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1613_, 0, v___x_1612_);
lean_ctor_set(v___x_1613_, 1, v___f_1604_);
v___x_1614_ = l_StateRefT_x27_instMonad___redArg(v___x_1613_);
v_toApplicative_1615_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1655_ == 0)
{
lean_object* v_unused_1656_; 
v_unused_1656_ = lean_ctor_get(v___x_1614_, 1);
lean_dec(v_unused_1656_);
v___x_1617_ = v___x_1614_;
v_isShared_1618_ = v_isSharedCheck_1655_;
goto v_resetjp_1616_;
}
else
{
lean_inc(v_toApplicative_1615_);
lean_dec(v___x_1614_);
v___x_1617_ = lean_box(0);
v_isShared_1618_ = v_isSharedCheck_1655_;
goto v_resetjp_1616_;
}
v_resetjp_1616_:
{
lean_object* v_toFunctor_1619_; lean_object* v_toSeq_1620_; lean_object* v_toSeqLeft_1621_; lean_object* v_toSeqRight_1622_; lean_object* v___x_1624_; uint8_t v_isShared_1625_; uint8_t v_isSharedCheck_1653_; 
v_toFunctor_1619_ = lean_ctor_get(v_toApplicative_1615_, 0);
v_toSeq_1620_ = lean_ctor_get(v_toApplicative_1615_, 2);
v_toSeqLeft_1621_ = lean_ctor_get(v_toApplicative_1615_, 3);
v_toSeqRight_1622_ = lean_ctor_get(v_toApplicative_1615_, 4);
v_isSharedCheck_1653_ = !lean_is_exclusive(v_toApplicative_1615_);
if (v_isSharedCheck_1653_ == 0)
{
lean_object* v_unused_1654_; 
v_unused_1654_ = lean_ctor_get(v_toApplicative_1615_, 1);
lean_dec(v_unused_1654_);
v___x_1624_ = v_toApplicative_1615_;
v_isShared_1625_ = v_isSharedCheck_1653_;
goto v_resetjp_1623_;
}
else
{
lean_inc(v_toSeqRight_1622_);
lean_inc(v_toSeqLeft_1621_);
lean_inc(v_toSeq_1620_);
lean_inc(v_toFunctor_1619_);
lean_dec(v_toApplicative_1615_);
v___x_1624_ = lean_box(0);
v_isShared_1625_ = v_isSharedCheck_1653_;
goto v_resetjp_1623_;
}
v_resetjp_1623_:
{
lean_object* v___x_1626_; lean_object* v___f_1627_; lean_object* v___f_1628_; lean_object* v___f_1629_; lean_object* v___f_1630_; lean_object* v___x_1631_; lean_object* v___f_1632_; lean_object* v___f_1633_; lean_object* v___f_1634_; lean_object* v___x_1636_; 
v___x_1626_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__22);
v___f_1627_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__23));
v___f_1628_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__24));
lean_inc_ref(v_toFunctor_1619_);
v___f_1629_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_1629_, 0, v_toFunctor_1619_);
v___f_1630_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1630_, 0, v_toFunctor_1619_);
v___x_1631_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___f_1629_);
lean_ctor_set(v___x_1631_, 1, v___f_1630_);
v___f_1632_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1632_, 0, v_toSeqRight_1622_);
v___f_1633_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_1633_, 0, v_toSeqLeft_1621_);
v___f_1634_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1634_, 0, v_toSeq_1620_);
if (v_isShared_1625_ == 0)
{
lean_ctor_set(v___x_1624_, 4, v___f_1632_);
lean_ctor_set(v___x_1624_, 3, v___f_1633_);
lean_ctor_set(v___x_1624_, 2, v___f_1634_);
lean_ctor_set(v___x_1624_, 1, v___f_1627_);
lean_ctor_set(v___x_1624_, 0, v___x_1631_);
v___x_1636_ = v___x_1624_;
goto v_reusejp_1635_;
}
else
{
lean_object* v_reuseFailAlloc_1652_; 
v_reuseFailAlloc_1652_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1652_, 0, v___x_1631_);
lean_ctor_set(v_reuseFailAlloc_1652_, 1, v___f_1627_);
lean_ctor_set(v_reuseFailAlloc_1652_, 2, v___f_1634_);
lean_ctor_set(v_reuseFailAlloc_1652_, 3, v___f_1633_);
lean_ctor_set(v_reuseFailAlloc_1652_, 4, v___f_1632_);
v___x_1636_ = v_reuseFailAlloc_1652_;
goto v_reusejp_1635_;
}
v_reusejp_1635_:
{
lean_object* v___x_1638_; 
if (v_isShared_1618_ == 0)
{
lean_ctor_set(v___x_1617_, 1, v___f_1628_);
lean_ctor_set(v___x_1617_, 0, v___x_1636_);
v___x_1638_ = v___x_1617_;
goto v_reusejp_1637_;
}
else
{
lean_object* v_reuseFailAlloc_1651_; 
v_reuseFailAlloc_1651_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1651_, 0, v___x_1636_);
lean_ctor_set(v_reuseFailAlloc_1651_, 1, v___f_1628_);
v___x_1638_ = v_reuseFailAlloc_1651_;
goto v_reusejp_1637_;
}
v_reusejp_1637_:
{
lean_object* v___f_1639_; lean_object* v___x_1640_; 
v___f_1639_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___boxed), 12, 7);
lean_closure_set(v___f_1639_, 0, v_mvarId_1576_);
lean_closure_set(v___f_1639_, 1, v_mkCmdStr_1535_);
lean_closure_set(v___f_1639_, 2, v___x_1579_);
lean_closure_set(v___f_1639_, 3, v_params_1531_);
lean_closure_set(v___f_1639_, 4, v_doc_1538_);
lean_closure_set(v___f_1639_, 5, v_replaceRange_1543_);
lean_closure_set(v___f_1639_, 6, v___x_1638_);
v___x_1640_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_val_1596_, v___x_1626_, v___f_1639_);
if (lean_obj_tag(v___x_1640_) == 0)
{
lean_object* v_a_1641_; 
v_a_1641_ = lean_ctor_get(v___x_1640_, 0);
lean_inc(v_a_1641_);
lean_dec_ref_known(v___x_1640_, 1);
v_a_1549_ = v_a_1641_;
goto v___jp_1548_;
}
else
{
lean_object* v_a_1642_; lean_object* v___x_1644_; uint8_t v_isShared_1645_; uint8_t v_isSharedCheck_1650_; 
lean_dec_ref(v_title_1532_);
v_a_1642_ = lean_ctor_get(v___x_1640_, 0);
v_isSharedCheck_1650_ = !lean_is_exclusive(v___x_1640_);
if (v_isSharedCheck_1650_ == 0)
{
v___x_1644_ = v___x_1640_;
v_isShared_1645_ = v_isSharedCheck_1650_;
goto v_resetjp_1643_;
}
else
{
lean_inc(v_a_1642_);
lean_dec(v___x_1640_);
v___x_1644_ = lean_box(0);
v_isShared_1645_ = v_isSharedCheck_1650_;
goto v_resetjp_1643_;
}
v_resetjp_1643_:
{
lean_object* v___x_1646_; lean_object* v___x_1648_; 
v___x_1646_ = l_Lean_Server_RequestError_ofIoError(v_a_1642_);
if (v_isShared_1645_ == 0)
{
lean_ctor_set(v___x_1644_, 0, v___x_1646_);
v___x_1648_ = v___x_1644_;
goto v_reusejp_1647_;
}
else
{
lean_object* v_reuseFailAlloc_1649_; 
v_reuseFailAlloc_1649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1649_, 0, v___x_1646_);
v___x_1648_ = v_reuseFailAlloc_1649_;
goto v_reusejp_1647_;
}
v_reusejp_1647_:
{
return v___x_1648_;
}
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
lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1661_; lean_object* v___x_1662_; lean_object* v___x_1663_; lean_object* v___x_1664_; 
lean_del_object(v___x_1591_);
lean_dec_ref(v___x_1579_);
lean_dec(v_mvarId_1576_);
lean_dec_ref(v_toInteractiveGoalCore_1575_);
lean_dec_ref(v_replaceRange_1543_);
lean_dec_ref(v_doc_1538_);
lean_dec_ref(v_mkCmdStr_1535_);
lean_dec(v_params_1531_);
v___x_1658_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__0___closed__0));
v___x_1659_ = ((lean_object*)(lp_mathlib_instToJsonRpcEncodablePacket_toJson___closed__0_00___x40_Mathlib_Tactic_Widget_SelectPanelUtils_2516298745____hygCtx___hyg_35_));
v___x_1660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1660_, 0, v_helpMsg_1536_);
v___x_1661_ = lean_unsigned_to_nat(1u);
v___x_1662_ = lean_mk_empty_array_with_capacity(v___x_1661_);
v___x_1663_ = lean_array_push(v___x_1662_, v___x_1660_);
v___x_1664_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1664_, 0, v___x_1658_);
lean_ctor_set(v___x_1664_, 1, v___x_1659_);
lean_ctor_set(v___x_1664_, 2, v___x_1663_);
v_a_1549_ = v___x_1664_;
goto v___jp_1548_;
}
}
else
{
lean_object* v_val_1665_; 
lean_del_object(v___x_1591_);
lean_dec_ref(v___x_1579_);
lean_dec(v_mvarId_1576_);
lean_dec_ref(v_toInteractiveGoalCore_1575_);
lean_dec_ref(v_replaceRange_1543_);
lean_dec_ref(v_doc_1538_);
lean_dec_ref(v_helpMsg_1536_);
lean_dec_ref(v_mkCmdStr_1535_);
lean_dec(v_params_1531_);
v_val_1665_ = lean_ctor_get(v_fst_1589_, 0);
lean_inc(v_val_1665_);
lean_dec_ref_known(v_fst_1589_, 1);
v_a_1549_ = v_val_1665_;
goto v___jp_1548_;
}
}
}
else
{
lean_object* v_a_1668_; lean_object* v___x_1670_; uint8_t v_isShared_1671_; uint8_t v_isSharedCheck_1675_; 
lean_dec_ref(v___x_1579_);
lean_dec(v_mvarId_1576_);
lean_dec_ref(v_toInteractiveGoalCore_1575_);
lean_dec_ref(v_replaceRange_1543_);
lean_dec_ref(v_doc_1538_);
lean_dec_ref(v_helpMsg_1536_);
lean_dec_ref(v_mkCmdStr_1535_);
lean_dec_ref(v_title_1532_);
lean_dec(v_params_1531_);
v_a_1668_ = lean_ctor_get(v___x_1587_, 0);
v_isSharedCheck_1675_ = !lean_is_exclusive(v___x_1587_);
if (v_isSharedCheck_1675_ == 0)
{
v___x_1670_ = v___x_1587_;
v_isShared_1671_ = v_isSharedCheck_1675_;
goto v_resetjp_1669_;
}
else
{
lean_inc(v_a_1668_);
lean_dec(v___x_1587_);
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
lean_ctor_set(v_reuseFailAlloc_1674_, 0, v_a_1668_);
v___x_1673_ = v_reuseFailAlloc_1674_;
goto v_reusejp_1672_;
}
v_reusejp_1672_:
{
return v___x_1673_;
}
}
}
}
v___jp_1676_:
{
lean_object* v___x_1679_; lean_object* v___x_1680_; lean_object* v_errorMsg_1681_; 
v___x_1679_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__25));
lean_inc_ref(v___y_1677_);
v___x_1680_ = lean_string_append(v___y_1677_, v___x_1679_);
v_errorMsg_1681_ = lean_string_append(v___x_1680_, v___y_1678_);
if (v_onlyOne_1537_ == 0)
{
v___y_1578_ = v_errorMsg_1681_;
goto v___jp_1577_;
}
else
{
lean_object* v___x_1682_; lean_object* v___x_1683_; lean_object* v___x_1684_; uint8_t v___x_1685_; 
v___x_1682_ = lean_unsigned_to_nat(1u);
lean_inc_ref(v_selectedLocations_1542_);
lean_inc(v_params_1531_);
v___x_1683_ = lean_apply_1(v_selectedLocations_1542_, v_params_1531_);
v___x_1684_ = lean_array_get_size(v___x_1683_);
lean_dec_ref(v___x_1683_);
v___x_1685_ = lean_nat_dec_lt(v___x_1682_, v___x_1684_);
if (v___x_1685_ == 0)
{
v___y_1578_ = v_errorMsg_1681_;
goto v___jp_1577_;
}
else
{
lean_object* v___x_1686_; 
lean_dec_ref(v_errorMsg_1681_);
lean_dec(v_mvarId_1576_);
lean_dec_ref(v_toInteractiveGoalCore_1575_);
lean_dec_ref(v_replaceRange_1543_);
lean_dec_ref(v_selectedLocations_1542_);
lean_dec_ref(v_doc_1538_);
lean_dec_ref(v_helpMsg_1536_);
lean_dec_ref(v_mkCmdStr_1535_);
lean_dec_ref(v___x_1534_);
lean_dec(v_params_1531_);
v___x_1686_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__29));
v_a_1549_ = v___x_1686_;
goto v___jp_1548_;
}
}
}
v___jp_1687_:
{
if (v_onlyGoal_1533_ == 0)
{
lean_object* v___x_1689_; 
v___x_1689_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__30));
v___y_1677_ = v___y_1688_;
v___y_1678_ = v___x_1689_;
goto v___jp_1676_;
}
else
{
lean_object* v___x_1690_; 
v___x_1690_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__31));
v___y_1677_ = v___y_1688_;
v___y_1678_ = v___x_1690_;
goto v___jp_1676_;
}
}
}
v___jp_1548_:
{
lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; lean_object* v___x_1560_; lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1567_; lean_object* v___x_1568_; lean_object* v___x_1569_; lean_object* v___x_1570_; lean_object* v___x_1571_; 
v___x_1550_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__0));
v___x_1551_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__1));
v___x_1552_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_1552_, 0, v___x_1547_);
v___x_1553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1553_, 0, v___x_1551_);
lean_ctor_set(v___x_1553_, 1, v___x_1552_);
v___x_1554_ = lean_unsigned_to_nat(1u);
v___x_1555_ = lean_mk_empty_array_with_capacity(v___x_1554_);
lean_inc_ref_n(v___x_1555_, 2);
v___x_1556_ = lean_array_push(v___x_1555_, v___x_1553_);
v___x_1557_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__2));
v___x_1558_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__7));
v___x_1559_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1559_, 0, v_title_1532_);
v___x_1560_ = lean_array_push(v___x_1555_, v___x_1559_);
v___x_1561_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1561_, 0, v___x_1557_);
lean_ctor_set(v___x_1561_, 1, v___x_1558_);
lean_ctor_set(v___x_1561_, 2, v___x_1560_);
v___x_1562_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__8));
v___x_1563_ = ((lean_object*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___closed__12));
v___x_1564_ = lean_array_push(v___x_1555_, v_a_1549_);
v___x_1565_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1565_, 0, v___x_1562_);
lean_ctor_set(v___x_1565_, 1, v___x_1563_);
lean_ctor_set(v___x_1565_, 2, v___x_1564_);
v___x_1566_ = lean_unsigned_to_nat(2u);
v___x_1567_ = lean_mk_empty_array_with_capacity(v___x_1566_);
v___x_1568_ = lean_array_push(v___x_1567_, v___x_1561_);
v___x_1569_ = lean_array_push(v___x_1568_, v___x_1565_);
v___x_1570_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1570_, 0, v___x_1550_);
lean_ctor_set(v___x_1570_, 1, v___x_1556_);
lean_ctor_set(v___x_1570_, 2, v___x_1569_);
v___x_1571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1571_, 0, v___x_1570_);
return v___x_1571_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___boxed(lean_object* v_inst_1693_, lean_object* v_params_1694_, lean_object* v_title_1695_, lean_object* v_onlyGoal_1696_, lean_object* v___x_1697_, lean_object* v_mkCmdStr_1698_, lean_object* v_helpMsg_1699_, lean_object* v_onlyOne_1700_, lean_object* v_doc_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
uint8_t v_onlyGoal_boxed_1704_; uint8_t v_onlyOne_boxed_1705_; lean_object* v_res_1706_; 
v_onlyGoal_boxed_1704_ = lean_unbox(v_onlyGoal_1696_);
v_onlyOne_boxed_1705_ = lean_unbox(v_onlyOne_1700_);
v_res_1706_ = lp_mathlib_mkSelectionPanelRPC___redArg___lam__3(v_inst_1693_, v_params_1694_, v_title_1695_, v_onlyGoal_boxed_1704_, v___x_1697_, v_mkCmdStr_1698_, v_helpMsg_1699_, v_onlyOne_boxed_1705_, v_doc_1701_, v___y_1702_);
lean_dec_ref(v___y_1702_);
return v_res_1706_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__0(void){
_start:
{
lean_object* v___x_1707_; lean_object* v___x_1708_; 
v___x_1707_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0);
v___x_1708_ = l_ReaderT_instMonad___redArg(v___x_1707_);
return v___x_1708_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__1(void){
_start:
{
lean_object* v___x_1709_; lean_object* v___x_1710_; 
v___x_1709_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0);
v___x_1710_ = lean_alloc_closure((void*)(l_ReaderT_read___boxed), 4, 3);
lean_closure_set(v___x_1710_, 0, lean_box(0));
lean_closure_set(v___x_1710_, 1, lean_box(0));
lean_closure_set(v___x_1710_, 2, v___x_1709_);
return v___x_1710_;
}
}
static lean_object* _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__2(void){
_start:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___x_1713_; 
v___x_1711_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___closed__1, &lp_mathlib_mkSelectionPanelRPC___redArg___closed__1_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__1);
v___x_1712_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__0);
v___x_1713_ = l_Lean_Server_RequestM_readDoc___redArg(v___x_1712_, v___x_1711_);
return v___x_1713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg(lean_object* v_inst_1714_, lean_object* v_mkCmdStr_1715_, lean_object* v_helpMsg_1716_, lean_object* v_title_1717_, uint8_t v_onlyGoal_1718_, uint8_t v_onlyOne_1719_, lean_object* v_params_1720_, lean_object* v_a_1721_){
_start:
{
lean_object* v___x_1723_; lean_object* v___x_1724_; lean_object* v___x_1725_; lean_object* v___x_1726_; lean_object* v___f_1727_; lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___x_1723_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___lam__2___closed__0);
v___x_1724_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___closed__0, &lp_mathlib_mkSelectionPanelRPC___redArg___closed__0_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__0);
v___x_1725_ = lean_box(v_onlyGoal_1718_);
v___x_1726_ = lean_box(v_onlyOne_1719_);
v___f_1727_ = lean_alloc_closure((void*)(lp_mathlib_mkSelectionPanelRPC___redArg___lam__3___boxed), 11, 8);
lean_closure_set(v___f_1727_, 0, v_inst_1714_);
lean_closure_set(v___f_1727_, 1, v_params_1720_);
lean_closure_set(v___f_1727_, 2, v_title_1717_);
lean_closure_set(v___f_1727_, 3, v___x_1725_);
lean_closure_set(v___f_1727_, 4, v___x_1724_);
lean_closure_set(v___f_1727_, 5, v_mkCmdStr_1715_);
lean_closure_set(v___f_1727_, 6, v_helpMsg_1716_);
lean_closure_set(v___f_1727_, 7, v___x_1726_);
v___x_1728_ = lean_obj_once(&lp_mathlib_mkSelectionPanelRPC___redArg___closed__2, &lp_mathlib_mkSelectionPanelRPC___redArg___closed__2_once, _init_lp_mathlib_mkSelectionPanelRPC___redArg___closed__2);
v___x_1729_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_1729_, 0, lean_box(0));
lean_closure_set(v___x_1729_, 1, lean_box(0));
lean_closure_set(v___x_1729_, 2, v___x_1723_);
lean_closure_set(v___x_1729_, 3, lean_box(0));
lean_closure_set(v___x_1729_, 4, lean_box(0));
lean_closure_set(v___x_1729_, 5, v___x_1728_);
lean_closure_set(v___x_1729_, 6, v___f_1727_);
v___x_1730_ = l_Lean_Server_RequestM_asTask___redArg(v___x_1729_, v_a_1721_);
return v___x_1730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___redArg___boxed(lean_object* v_inst_1731_, lean_object* v_mkCmdStr_1732_, lean_object* v_helpMsg_1733_, lean_object* v_title_1734_, lean_object* v_onlyGoal_1735_, lean_object* v_onlyOne_1736_, lean_object* v_params_1737_, lean_object* v_a_1738_, lean_object* v_a_1739_){
_start:
{
uint8_t v_onlyGoal_boxed_1740_; uint8_t v_onlyOne_boxed_1741_; lean_object* v_res_1742_; 
v_onlyGoal_boxed_1740_ = lean_unbox(v_onlyGoal_1735_);
v_onlyOne_boxed_1741_ = lean_unbox(v_onlyOne_1736_);
v_res_1742_ = lp_mathlib_mkSelectionPanelRPC___redArg(v_inst_1731_, v_mkCmdStr_1732_, v_helpMsg_1733_, v_title_1734_, v_onlyGoal_boxed_1740_, v_onlyOne_boxed_1741_, v_params_1737_, v_a_1738_);
lean_dec_ref(v_a_1738_);
return v_res_1742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC(lean_object* v_Params_1743_, lean_object* v_inst_1744_, lean_object* v_mkCmdStr_1745_, lean_object* v_helpMsg_1746_, lean_object* v_title_1747_, uint8_t v_onlyGoal_1748_, uint8_t v_onlyOne_1749_, lean_object* v_params_1750_, lean_object* v_a_1751_){
_start:
{
lean_object* v___x_1753_; 
v___x_1753_ = lp_mathlib_mkSelectionPanelRPC___redArg(v_inst_1744_, v_mkCmdStr_1745_, v_helpMsg_1746_, v_title_1747_, v_onlyGoal_1748_, v_onlyOne_1749_, v_params_1750_, v_a_1751_);
return v___x_1753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_mkSelectionPanelRPC___boxed(lean_object* v_Params_1754_, lean_object* v_inst_1755_, lean_object* v_mkCmdStr_1756_, lean_object* v_helpMsg_1757_, lean_object* v_title_1758_, lean_object* v_onlyGoal_1759_, lean_object* v_onlyOne_1760_, lean_object* v_params_1761_, lean_object* v_a_1762_, lean_object* v_a_1763_){
_start:
{
uint8_t v_onlyGoal_boxed_1764_; uint8_t v_onlyOne_boxed_1765_; lean_object* v_res_1766_; 
v_onlyGoal_boxed_1764_ = lean_unbox(v_onlyGoal_1759_);
v_onlyOne_boxed_1765_ = lean_unbox(v_onlyOne_1760_);
v_res_1766_ = lp_mathlib_mkSelectionPanelRPC(v_Params_1754_, v_inst_1755_, v_mkCmdStr_1756_, v_helpMsg_1757_, v_title_1758_, v_onlyGoal_boxed_1764_, v_onlyOne_boxed_1765_, v_params_1761_, v_a_1762_);
lean_dec_ref(v_a_1762_);
return v_res_1766_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_ExprLens(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Data_Html(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_ExprLens(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Widget_SelectInsertParamsClass(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_MakeEditLink(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Data_Html(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Widget_SelectPanelUtils(builtin);
}
#ifdef __cplusplus
}
#endif
