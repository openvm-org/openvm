// Lean compiler output
// Module: Mathlib.Tactic.ClickSuggestions
// Imports: public import Init public meta import Init public import Mathlib.Tactic.ClickSuggestions.TryPremises public import Mathlib.Tactic.ClickSuggestions.Unfold public meta import Mathlib.Lean.Meta.KAbstractPositions public meta import Lean.Server.FileWorker.RequestHandling public import Lean.Widget.InteractiveGoal public meta import Mathlib.Lean.GoalsLocation public import ProofWidgets.Component.OfRpcMethod
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
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_mathlib_Lean_SubExpr_GoalsLocation_fvarId_x3f(lean_object*);
lean_object* lp_mathlib_Lean_SubExpr_GoalsLocation_pos(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_MetavarContext_getDecl(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_instInhabitedPersistentArrayNode_default(lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_left(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkAuxDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
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
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
extern lean_object* l_Lean_instInhabitedLocalContext_default;
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_LocalContext_mkLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* l_Lean_LocalContext_mkLetDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_sharecommon_quick(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_LocalContext_sanitizeNames(lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_mkRefreshComponent(lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_RefreshToken_update(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_librarySearchSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_markProgress___redArg(lean_object*);
lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_toArray(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getType___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_trackingComputation___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Prod_map___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_uint64_dec_lt(uint64_t, uint64_t);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_findFinIdx_x3f_loop(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_FileMap_lspPosToUtf8Pos(lean_object*, lean_object*);
lean_object* l_Lean_Server_FileWorker_findGoalsAt_x3f(lean_object*, lean_object*);
lean_object* lean_task_get_own(lean_object*);
extern lean_object* l_Lean_interruptExceptionId;
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Exception_toMessageData(lean_object*);
lean_object* l_Lean_Server_WithRpcRef_mk___redArg(lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_InteractiveMessage;
uint64_t lean_string_hash(lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps_enc_00___x40_ProofWidgets_Component_Basic_2277670097____hygCtx___hyg_1____boxed(lean_object*, lean_object*);
lean_object* lean_io_as_task(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* lp_mathlib_Lean_SubExpr_GoalsLocation_rootExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Context_config(lean_object*);
lean_object* l_Lean_Elab_Term_setElabConfig(lean_object*);
uint64_t l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(lean_object*);
lean_object* l_Lean_Elab_ContextInfo_runMetaM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Server_RequestError_ofIoError(lean_object*);
lean_object* l_Lean_Server_RequestM_asTask___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
extern lean_object* l___private_Lean_Widget_UserWidget_0__Lean_Widget_panelWidgetsExt;
lean_object* l_Lean_ScopedEnvExtension_modifyState___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(lean_object*, lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_Lean_Server_RequestM_mapTaskCheap___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Widget_WidgetInstance_ofHash___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_liftCoreM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_viewSubexpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Meta_viewKAbstractSubExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "div"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "style"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "marginLeft"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "4px"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__12_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Invalid coordinate "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " for "};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3;
static const lean_string_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = "Internal: Types should be handled by viewAux"};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "internal click_suggestions error: selected location is a `.hypValue`"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__2_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__3 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__3_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__4 = (const lean_object*)&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Lean.MetavarContext"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Lean.instantiateLCtxMVars"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Invalid auxiliary declaration found in local context: "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__2_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 40, .m_capacity = 40, .m_length = 39, .m_data = " does not have an associated full name."};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3;
static lean_once_cell_t lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "This component was cancelled"};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__0 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__1 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "span"};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__2 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__2_value;
static const lean_array_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__3 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "\n            An error occurred in the mkRefreshComponentM thread:\n            "};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__4 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__4_value)}};
static const lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__5 = (const lean_object*)&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "#click_suggestions has started searching."};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "details"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "open"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "summary"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "className"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "mv2 pointer"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__7_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__8_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 246}, .m_size = 1, .m_capacity = 1, .m_data = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "\n        Suggestions for "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__10_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ": "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__12_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " hypothesis "};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__15_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = "#click_suggestions cannot suggest anything about the value of a let variable."};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__0_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "#click_suggestions: Please reload the tactic state"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 82, .m_capacity = 82, .m_length = 81, .m_data = "Internal #click_suggestions error: could not find any goal at the cursor position"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "#click_suggestions: please reload the tactic state"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 66, .m_capacity = 66, .m_length = 65, .m_data = "Shift-click an expression in the tactic state to get suggestions."};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__8_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = "Cannot decode params in RPC call '"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__0_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = ")'\n"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__2_value;
static const lean_string_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "Outdated RPC session"};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value;
static const lean_ctor_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(9, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3(lean_object*, lean_object*, lean_object*, uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "ClickSuggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rpc"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__2_value),LEAN_SCALAR_PTR_LITERAL(231, 102, 210, 246, 204, 42, 207, 225)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__3_value),LEAN_SCALAR_PTR_LITERAL(19, 141, 204, 63, 158, 71, 90, 118)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0(lean_object*, lean_object*, uint64_t);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3852, .m_capacity = 3852, .m_length = 3851, .m_data = "window;import{jsxs as e,jsx as t,Fragment as r}from\"react/jsx-runtime\";import*as n from\"react\";import{useRpcSession as o,EnvPosContext as a,useAsyncPersistent as i,mapRpcError as f,importWidgetModule as c}from\"@leanprover/infoview\";function u(e){return e&&e.__esModule&&Object.prototype.hasOwnProperty.call(e,\"default\")\?e.default:e}var s,l;var p=u(function(){if(l)return s;l=1;var e=\"undefined\"!=typeof Element,t=\"function\"==typeof Map,r=\"function\"==typeof Set,n=\"function\"==typeof ArrayBuffer&&!!ArrayBuffer.isView;function o(a,i){if(a===i)return!0;if(a&&i&&\"object\"==typeof a&&\"object\"==typeof i){if(a.constructor!==i.constructor)return!1;var f,c,u,s;if(Array.isArray(a)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(!o(a[c],i[c]))return!1;return!0}if(t&&a instanceof Map&&i instanceof Map){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;for(s=a.entries();!(c=s.next()).done;)if(!o(c.value[1],i.get(c.value[0])))return!1;return!0}if(r&&a instanceof Set&&i instanceof Set){if(a.size!==i.size)return!1;for(s=a.entries();!(c=s.next()).done;)if(!i.has(c.value[0]))return!1;return!0}if(n&&ArrayBuffer.isView(a)&&ArrayBuffer.isView(i)){if((f=a.length)!=i.length)return!1;for(c=f;0!==c--;)if(a[c]!==i[c])return!1;return!0}if(a.constructor===RegExp)return a.source===i.source&&a.flags===i.flags;if(a.valueOf!==Object.prototype.valueOf&&\"function\"==typeof a.valueOf&&\"function\"==typeof i.valueOf)return a.valueOf()===i.valueOf();if(a.toString!==Object.prototype.toString&&\"function\"==typeof a.toString&&\"function\"==typeof i.toString)return a.toString()===i.toString();if((f=(u=Object.keys(a)).length)!==Object.keys(i).length)return!1;for(c=f;0!==c--;)if(!Object.prototype.hasOwnProperty.call(i,u[c]))return!1;if(e&&a instanceof Element)return!1;for(c=f;0!==c--;)if((\"_owner\"!==u[c]&&\"__v\"!==u[c]&&\"__o\"!==u[c]||!a.$$typeof)&&!o(a[u[c]],i[u[c]]))return!1;return!0}return a!=a&&i!=i}return s=function(e,t){try{return o(e,t)}catch(e){if((e.message||\"\").match(/stack|recursion/i))return console.warn(\"react-fast-compare cannot handle circular refs\"),!1;throw e}}}());async function y(o,a,i){if(\"text\"in i)return t(r,{children:i.text});if(\"element\"in i){const[e,r,f]=i.element,c={};for(const[e,t]of r)c[e]=t;const u=await Promise.all(f.map(async e=>await y(o,a,e)));return\"hr\"===e\?t(\"hr\",{}):0===u.length\?n.createElement(e,c):n.createElement(e,c,u)}if(\"component\"in i){const[e,t,r,f]=i.component,u=await Promise.all(f.map(async e=>await y(o,a,e))),s={...r,pos:a},l=await c(o,a,e);if(!(t in l))throw new Error(`Module '${e}' does not export '${t}'`);return 0===u.length\?n.createElement(l[t],s):n.createElement(l[t],s,u)}return e(\"span\",{className:\"red\",children:[\"Unknown HTML variant: \",JSON.stringify(i)]})}function d({html:c}){const u=o(),s=n.useContext(a),l=i(()=>y(u,s,c),[u,s,c]);return\"resolved\"===l.state\?l.value:\"rejected\"===l.state\?e(\"span\",{className:\"red\",children:[\"Error rendering HTML: \",f(l.error).message]}):t(r,{})}const m=\"Mathlib.Tactic.ClickSuggestions.rpc\",g='false';var w=n.memo(e=>{const a=o(),c=n.useRef({fn:()=>{}}),u=i(async()=>{if(c.current.fn(),\"true\"===g){const[t,r]=function(e,t,r){const n={fn:()=>{}};return[new Promise(async(o,a)=>{const i=await e.call(t,r),f=window.setInterval(async()=>{try{const t=await e.call(\"ProofWidgets.checkRequest\",i);if(\"running\"===t)return;window.clearInterval(f),o(t.done.result)}catch(e){window.clearInterval(f),a(e)}},100);n.fn=()=>{e.call(\"ProofWidgets.cancelRequest\",i)}}),n]}(a,m,e);return c.current=r,t}{const t=new AbortController,r=a.call(m,e,{abortSignal:t.signal});return c.current={fn:()=>t.abort()},r}},[a,e]);return n.useEffect(()=>()=>{c.current.fn()},[]),\"rejected\"===u.state\?t(\"p\",{style:{color:\"red\"},children:f(u.error).message}):\"loading\"===u.state\?t(r,{children:\"Loading..\"}):t(d,{html:u.value})},p);export{w as default};"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__3_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "command#click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__2_value),LEAN_SCALAR_PTR_LITERAL(231, 102, 210, 246, 204, 42, 207, 225)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__0_value),LEAN_SCALAR_PTR_LITERAL(234, 2, 67, 171, 111, 202, 107, 71)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "#click_suggestions"};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__3_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions = (const lean_object*)&lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg(lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(uint64_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1(lean_object*, lean_object*, uint64_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2(lean_object*, uint64_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0(lean_object* v_k_1_, lean_object* v_x_2_, lean_object* v_e_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lean_box(0);
v___x_5_ = lean_apply_2(v_k_1_, v_e_3_, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0___boxed(lean_object* v_k_6_, lean_object* v_x_7_, lean_object* v_e_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0(v_k_6_, v_x_7_, v_e_8_);
lean_dec_ref(v_x_7_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1(lean_object* v_snd_10_, lean_object* v_k_11_, lean_object* v_fst_12_, uint8_t v_tpCorrect_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_14_, 0, v_snd_10_);
lean_ctor_set_uint8(v___x_14_, sizeof(void*)*1, v_tpCorrect_13_);
v___x_15_ = lean_apply_2(v_k_11_, v_fst_12_, v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1___boxed(lean_object* v_snd_16_, lean_object* v_k_17_, lean_object* v_fst_18_, lean_object* v_tpCorrect_19_){
_start:
{
uint8_t v_tpCorrect_boxed_20_; lean_object* v_res_21_; 
v_tpCorrect_boxed_20_ = lean_unbox(v_tpCorrect_19_);
v_res_21_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1(v_snd_16_, v_k_17_, v_fst_18_, v_tpCorrect_boxed_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__2(lean_object* v_k_22_, lean_object* v_e_23_, lean_object* v_pos_24_, lean_object* v_inst_25_, lean_object* v_toBind_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v___f_30_, lean_object* v_____do__lift_31_){
_start:
{
if (lean_obj_tag(v_____do__lift_31_) == 1)
{
lean_object* v_val_32_; lean_object* v_fst_33_; lean_object* v_snd_34_; lean_object* v___f_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
lean_dec(v___f_30_);
lean_dec_ref(v_inst_29_);
lean_dec_ref(v_inst_28_);
lean_dec_ref(v_inst_27_);
v_val_32_ = lean_ctor_get(v_____do__lift_31_, 0);
lean_inc(v_val_32_);
lean_dec_ref_known(v_____do__lift_31_, 1);
v_fst_33_ = lean_ctor_get(v_val_32_, 0);
lean_inc_n(v_fst_33_, 2);
v_snd_34_ = lean_ctor_get(v_val_32_, 1);
lean_inc(v_snd_34_);
lean_dec(v_val_32_);
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_35_, 0, v_snd_34_);
lean_closure_set(v___f_35_, 1, v_k_22_);
lean_closure_set(v___f_35_, 2, v_fst_33_);
v___x_36_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_kabstractIsTypeCorrect___boxed), 8, 3);
lean_closure_set(v___x_36_, 0, v_e_23_);
lean_closure_set(v___x_36_, 1, v_fst_33_);
lean_closure_set(v___x_36_, 2, v_pos_24_);
v___x_37_ = lean_apply_2(v_inst_25_, lean_box(0), v___x_36_);
v___x_38_ = lean_apply_4(v_toBind_26_, lean_box(0), lean_box(0), v___x_37_, v___f_35_);
return v___x_38_;
}
else
{
lean_object* v___x_39_; 
lean_dec(v_____do__lift_31_);
lean_dec(v_toBind_26_);
lean_dec(v_k_22_);
v___x_39_ = l_Lean_Meta_viewSubexpr___redArg(v_inst_27_, v_inst_25_, v_inst_28_, v_inst_29_, v___f_30_, v_pos_24_, v_e_23_);
lean_dec(v_pos_24_);
return v___x_39_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg(lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_e_44_, lean_object* v_pos_45_, lean_object* v_k_46_){
_start:
{
lean_object* v_toBind_47_; lean_object* v___f_48_; lean_object* v___f_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v_toBind_47_ = lean_ctor_get(v_inst_40_, 1);
lean_inc_n(v_toBind_47_, 2);
lean_inc(v_k_46_);
v___f_48_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_48_, 0, v_k_46_);
lean_inc(v_inst_41_);
lean_inc(v_pos_45_);
lean_inc_ref(v_e_44_);
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg___lam__2), 10, 9);
lean_closure_set(v___f_49_, 0, v_k_46_);
lean_closure_set(v___f_49_, 1, v_e_44_);
lean_closure_set(v___f_49_, 2, v_pos_45_);
lean_closure_set(v___f_49_, 3, v_inst_41_);
lean_closure_set(v___f_49_, 4, v_toBind_47_);
lean_closure_set(v___f_49_, 5, v_inst_40_);
lean_closure_set(v___f_49_, 6, v_inst_42_);
lean_closure_set(v___f_49_, 7, v_inst_43_);
lean_closure_set(v___f_49_, 8, v___f_48_);
v___x_50_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_viewKAbstractSubExpr___boxed), 7, 2);
lean_closure_set(v___x_50_, 0, v_e_44_);
lean_closure_set(v___x_50_, 1, v_pos_45_);
v___x_51_ = lean_apply_2(v_inst_41_, lean_box(0), v___x_50_);
v___x_52_ = lean_apply_4(v_toBind_47_, lean_box(0), lean_box(0), v___x_51_, v___f_49_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27(lean_object* v_m_53_, lean_object* v_00_u03b1_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_e_59_, lean_object* v_pos_60_, lean_object* v_k_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___redArg(v_inst_55_, v_inst_56_, v_inst_57_, v_inst_58_, v_e_59_, v_pos_60_, v_k_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg(lean_object* v_lctx_63_, lean_object* v_x_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_){
_start:
{
lean_object* v_keyedConfig_72_; uint8_t v_trackZetaDelta_73_; lean_object* v_zetaDeltaSet_74_; lean_object* v_localInstances_75_; lean_object* v_defEqCtx_x3f_76_; lean_object* v_synthPendingDepth_77_; lean_object* v_customCanUnfoldPredicate_x3f_78_; uint8_t v_univApprox_79_; uint8_t v_inTypeClassResolution_80_; uint8_t v_cacheInferType_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
v_keyedConfig_72_ = lean_ctor_get(v___y_67_, 0);
v_trackZetaDelta_73_ = lean_ctor_get_uint8(v___y_67_, sizeof(void*)*7);
v_zetaDeltaSet_74_ = lean_ctor_get(v___y_67_, 1);
v_localInstances_75_ = lean_ctor_get(v___y_67_, 3);
v_defEqCtx_x3f_76_ = lean_ctor_get(v___y_67_, 4);
v_synthPendingDepth_77_ = lean_ctor_get(v___y_67_, 5);
v_customCanUnfoldPredicate_x3f_78_ = lean_ctor_get(v___y_67_, 6);
v_univApprox_79_ = lean_ctor_get_uint8(v___y_67_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_80_ = lean_ctor_get_uint8(v___y_67_, sizeof(void*)*7 + 2);
v_cacheInferType_81_ = lean_ctor_get_uint8(v___y_67_, sizeof(void*)*7 + 3);
lean_inc(v_customCanUnfoldPredicate_x3f_78_);
lean_inc(v_synthPendingDepth_77_);
lean_inc(v_defEqCtx_x3f_76_);
lean_inc_ref(v_localInstances_75_);
lean_inc(v_zetaDeltaSet_74_);
lean_inc_ref(v_keyedConfig_72_);
v___x_82_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_82_, 0, v_keyedConfig_72_);
lean_ctor_set(v___x_82_, 1, v_zetaDeltaSet_74_);
lean_ctor_set(v___x_82_, 2, v_lctx_63_);
lean_ctor_set(v___x_82_, 3, v_localInstances_75_);
lean_ctor_set(v___x_82_, 4, v_defEqCtx_x3f_76_);
lean_ctor_set(v___x_82_, 5, v_synthPendingDepth_77_);
lean_ctor_set(v___x_82_, 6, v_customCanUnfoldPredicate_x3f_78_);
lean_ctor_set_uint8(v___x_82_, sizeof(void*)*7, v_trackZetaDelta_73_);
lean_ctor_set_uint8(v___x_82_, sizeof(void*)*7 + 1, v_univApprox_79_);
lean_ctor_set_uint8(v___x_82_, sizeof(void*)*7 + 2, v_inTypeClassResolution_80_);
lean_ctor_set_uint8(v___x_82_, sizeof(void*)*7 + 3, v_cacheInferType_81_);
lean_inc(v___y_70_);
lean_inc_ref(v___y_69_);
lean_inc(v___y_68_);
lean_inc(v___y_66_);
lean_inc_ref(v___y_65_);
v___x_83_ = lean_apply_7(v_x_64_, v___y_65_, v___y_66_, v___x_82_, v___y_68_, v___y_69_, v___y_70_, lean_box(0));
if (lean_obj_tag(v___x_83_) == 0)
{
lean_object* v_a_84_; lean_object* v___x_86_; uint8_t v_isShared_87_; uint8_t v_isSharedCheck_91_; 
v_a_84_ = lean_ctor_get(v___x_83_, 0);
v_isSharedCheck_91_ = !lean_is_exclusive(v___x_83_);
if (v_isSharedCheck_91_ == 0)
{
v___x_86_ = v___x_83_;
v_isShared_87_ = v_isSharedCheck_91_;
goto v_resetjp_85_;
}
else
{
lean_inc(v_a_84_);
lean_dec(v___x_83_);
v___x_86_ = lean_box(0);
v_isShared_87_ = v_isSharedCheck_91_;
goto v_resetjp_85_;
}
v_resetjp_85_:
{
lean_object* v___x_89_; 
if (v_isShared_87_ == 0)
{
v___x_89_ = v___x_86_;
goto v_reusejp_88_;
}
else
{
lean_object* v_reuseFailAlloc_90_; 
v_reuseFailAlloc_90_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_90_, 0, v_a_84_);
v___x_89_ = v_reuseFailAlloc_90_;
goto v_reusejp_88_;
}
v_reusejp_88_:
{
return v___x_89_;
}
}
}
else
{
return v___x_83_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg___boxed(lean_object* v_lctx_92_, lean_object* v_x_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg(v_lctx_92_, v_x_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
lean_dec(v___y_99_);
lean_dec_ref(v___y_98_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2(lean_object* v_00_u03b1_102_, lean_object* v_lctx_103_, lean_object* v_x_104_, lean_object* v___y_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg(v_lctx_103_, v_x_104_, v___y_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___boxed(lean_object* v_00_u03b1_113_, lean_object* v_lctx_114_, lean_object* v_x_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2(v_00_u03b1_113_, v_lctx_114_, v_x_115_, v___y_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
lean_dec(v___y_121_);
lean_dec_ref(v___y_120_);
lean_dec(v___y_119_);
lean_dec_ref(v___y_118_);
lean_dec(v___y_117_);
lean_dec_ref(v___y_116_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0(lean_object* v_x_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v___x_132_; 
lean_inc(v___y_126_);
lean_inc_ref(v___y_125_);
v___x_132_ = lean_apply_7(v_x_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_, v___y_130_, lean_box(0));
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0___boxed(lean_object* v_x_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_){
_start:
{
lean_object* v_res_141_; 
v_res_141_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0(v_x_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_, v___y_138_, v___y_139_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
return v_res_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg(lean_object* v_mvarId_142_, lean_object* v_x_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; 
lean_inc(v___y_145_);
lean_inc_ref(v___y_144_);
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_151_, 0, v_x_143_);
lean_closure_set(v___f_151_, 1, v___y_144_);
lean_closure_set(v___f_151_, 2, v___y_145_);
v___x_152_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_142_, v___f_151_, v___y_146_, v___y_147_, v___y_148_, v___y_149_);
if (lean_obj_tag(v___x_152_) == 0)
{
return v___x_152_;
}
else
{
lean_object* v_a_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_160_; 
v_a_153_ = lean_ctor_get(v___x_152_, 0);
v_isSharedCheck_160_ = !lean_is_exclusive(v___x_152_);
if (v_isSharedCheck_160_ == 0)
{
v___x_155_ = v___x_152_;
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_a_153_);
lean_dec(v___x_152_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_160_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_158_; 
if (v_isShared_156_ == 0)
{
v___x_158_ = v___x_155_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v_a_153_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg___boxed(lean_object* v_mvarId_161_, lean_object* v_x_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg(v_mvarId_161_, v_x_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_, v___y_167_, v___y_168_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
lean_dec(v___y_166_);
lean_dec_ref(v___y_165_);
lean_dec(v___y_164_);
lean_dec_ref(v___y_163_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3(lean_object* v_00_u03b1_171_, lean_object* v_mvarId_172_, lean_object* v_x_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg(v_mvarId_172_, v_x_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___boxed(lean_object* v_00_u03b1_182_, lean_object* v_mvarId_183_, lean_object* v_x_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3(v_00_u03b1_182_, v_mvarId_183_, v_x_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_);
lean_dec(v___y_190_);
lean_dec_ref(v___y_189_);
lean_dec(v___y_188_);
lean_dec_ref(v___y_187_);
lean_dec(v___y_186_);
lean_dec_ref(v___y_185_);
return v_res_192_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9(void){
_start:
{
lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_208_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__8));
v___x_209_ = l_Lean_Json_mkObj(v___x_208_);
return v___x_209_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10(void){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_210_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__9);
v___x_211_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__3));
v___x_212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_212_, 0, v___x_211_);
lean_ctor_set(v___x_212_, 1, v___x_210_);
return v___x_212_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_213_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__10);
v___x_214_ = lean_unsigned_to_nat(1u);
v___x_215_ = lean_mk_empty_array_with_capacity(v___x_214_);
v___x_216_ = lean_array_push(v___x_215_, v___x_213_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0(lean_object* v_token_219_, lean_object* v_rootExpr_220_, lean_object* v_fst_221_, lean_object* v_parentDecl_x3f_222_, lean_object* v_subExpr_223_, lean_object* v_rwKind_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_, lean_object* v___y_229_, lean_object* v___y_230_){
_start:
{
lean_object* v_htmls_233_; lean_object* v___y_234_; lean_object* v___y_235_; lean_object* v___y_236_; lean_object* v___y_237_; lean_object* v___y_238_; lean_object* v___y_239_; lean_object* v___x_250_; 
lean_inc(v_rwKind_224_);
lean_inc_ref(v_subExpr_223_);
v___x_250_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_suggestUnfold(v_subExpr_223_, v_rwKind_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_, v___y_229_, v___y_230_);
if (lean_obj_tag(v___x_250_) == 0)
{
lean_object* v_a_251_; lean_object* v___x_252_; 
v_a_251_ = lean_ctor_get(v___x_250_, 0);
lean_inc(v_a_251_);
lean_dec_ref_known(v___x_250_, 1);
v___x_252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__12));
if (lean_obj_tag(v_a_251_) == 1)
{
lean_object* v_val_253_; lean_object* v___x_254_; 
v_val_253_ = lean_ctor_get(v_a_251_, 0);
lean_inc(v_val_253_);
lean_dec_ref_known(v_a_251_, 1);
v___x_254_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_markProgress___redArg(v___y_226_);
if (lean_obj_tag(v___x_254_) == 0)
{
lean_object* v___x_255_; 
lean_dec_ref_known(v___x_254_, 1);
v___x_255_ = lean_array_push(v___x_252_, v_val_253_);
v_htmls_233_ = v___x_255_;
v___y_234_ = v___y_225_;
v___y_235_ = v___y_226_;
v___y_236_ = v___y_227_;
v___y_237_ = v___y_228_;
v___y_238_ = v___y_229_;
v___y_239_ = v___y_230_;
goto v___jp_232_;
}
else
{
lean_dec(v_val_253_);
lean_dec(v_rwKind_224_);
lean_dec_ref(v_subExpr_223_);
lean_dec(v_parentDecl_x3f_222_);
lean_dec(v_fst_221_);
lean_dec_ref(v_rootExpr_220_);
lean_dec_ref(v_token_219_);
return v___x_254_;
}
}
else
{
lean_dec(v_a_251_);
v_htmls_233_ = v___x_252_;
v___y_234_ = v___y_225_;
v___y_235_ = v___y_226_;
v___y_236_ = v___y_227_;
v___y_237_ = v___y_228_;
v___y_238_ = v___y_229_;
v___y_239_ = v___y_230_;
goto v___jp_232_;
}
}
else
{
lean_object* v_a_256_; lean_object* v___x_258_; uint8_t v_isShared_259_; uint8_t v_isSharedCheck_263_; 
lean_dec(v_rwKind_224_);
lean_dec_ref(v_subExpr_223_);
lean_dec(v_parentDecl_x3f_222_);
lean_dec(v_fst_221_);
lean_dec_ref(v_rootExpr_220_);
lean_dec_ref(v_token_219_);
v_a_256_ = lean_ctor_get(v___x_250_, 0);
v_isSharedCheck_263_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_263_ == 0)
{
v___x_258_ = v___x_250_;
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
else
{
lean_inc(v_a_256_);
lean_dec(v___x_250_);
v___x_258_ = lean_box(0);
v_isShared_259_ = v_isSharedCheck_263_;
goto v_resetjp_257_;
}
v_resetjp_257_:
{
lean_object* v___x_261_; 
if (v_isShared_259_ == 0)
{
v___x_261_ = v___x_258_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_262_; 
v_reuseFailAlloc_262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_262_, 0, v_a_256_);
v___x_261_ = v_reuseFailAlloc_262_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
return v___x_261_;
}
}
}
v___jp_232_:
{
lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v_fst_242_; lean_object* v_snd_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_240_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__1));
v___x_241_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent(v___x_240_);
v_fst_242_ = lean_ctor_get(v___x_241_, 0);
lean_inc(v_fst_242_);
v_snd_243_ = lean_ctor_get(v___x_241_, 1);
lean_inc(v_snd_243_);
lean_dec_ref(v___x_241_);
v___x_244_ = lean_array_push(v_htmls_233_, v_fst_242_);
v___x_245_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__2));
v___x_246_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__11);
v___x_247_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_247_, 0, v___x_245_);
lean_ctor_set(v___x_247_, 1, v___x_246_);
lean_ctor_set(v___x_247_, 2, v___x_244_);
v___x_248_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_token_219_, v___x_247_);
v___x_249_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_librarySearchSuggestions(v_rootExpr_220_, v_subExpr_223_, v_fst_221_, v_rwKind_224_, v_parentDecl_x3f_222_, v_snd_243_, v___y_234_, v___y_235_, v___y_236_, v___y_237_, v___y_238_, v___y_239_);
return v___x_249_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___boxed(lean_object* v_token_264_, lean_object* v_rootExpr_265_, lean_object* v_fst_266_, lean_object* v_parentDecl_x3f_267_, lean_object* v_subExpr_268_, lean_object* v_rwKind_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_, lean_object* v___y_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0(v_token_264_, v_rootExpr_265_, v_fst_266_, v_parentDecl_x3f_267_, v_subExpr_268_, v_rwKind_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
lean_dec(v___y_275_);
lean_dec_ref(v___y_274_);
lean_dec(v___y_273_);
lean_dec_ref(v___y_272_);
lean_dec(v___y_271_);
lean_dec_ref(v___y_270_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0(lean_object* v_k_278_, lean_object* v_x_279_, lean_object* v_e_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_, lean_object* v___y_286_){
_start:
{
lean_object* v___x_288_; lean_object* v___x_289_; 
v___x_288_ = lean_box(0);
lean_inc(v___y_286_);
lean_inc_ref(v___y_285_);
lean_inc(v___y_284_);
lean_inc_ref(v___y_283_);
lean_inc(v___y_282_);
lean_inc_ref(v___y_281_);
v___x_289_ = lean_apply_9(v_k_278_, v_e_280_, v___x_288_, v___y_281_, v___y_282_, v___y_283_, v___y_284_, v___y_285_, v___y_286_, lean_box(0));
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0___boxed(lean_object* v_k_290_, lean_object* v_x_291_, lean_object* v_e_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0(v_k_290_, v_x_291_, v_e_292_, v___y_293_, v___y_294_, v___y_295_, v___y_296_, v___y_297_, v___y_298_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec(v___y_296_);
lean_dec_ref(v___y_295_);
lean_dec(v___y_294_);
lean_dec_ref(v___y_293_);
lean_dec_ref(v_x_291_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1(lean_object* v_fvars_301_, lean_object* v_k_302_, lean_object* v_b_303_, lean_object* v_x_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; 
v___x_312_ = lean_array_push(v_fvars_301_, v_x_304_);
lean_inc(v___y_310_);
lean_inc_ref(v___y_309_);
lean_inc(v___y_308_);
lean_inc_ref(v___y_307_);
lean_inc(v___y_306_);
lean_inc_ref(v___y_305_);
v___x_313_ = lean_apply_9(v_k_302_, v___x_312_, v_b_303_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_, lean_box(0));
return v___x_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1___boxed(lean_object* v_fvars_314_, lean_object* v_k_315_, lean_object* v_b_316_, lean_object* v_x_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_){
_start:
{
lean_object* v_res_325_; 
v_res_325_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1(v_fvars_314_, v_k_315_, v_b_316_, v_x_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_, v___y_322_, v___y_323_);
lean_dec(v___y_323_);
lean_dec_ref(v___y_322_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
return v_res_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0(lean_object* v_k_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v_b_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_){
_start:
{
lean_object* v___x_335_; 
lean_inc(v___y_333_);
lean_inc_ref(v___y_332_);
lean_inc(v___y_331_);
lean_inc_ref(v___y_330_);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
v___x_335_ = lean_apply_8(v_k_326_, v_b_329_, v___y_327_, v___y_328_, v___y_330_, v___y_331_, v___y_332_, v___y_333_, lean_box(0));
return v___x_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0___boxed(lean_object* v_k_336_, lean_object* v___y_337_, lean_object* v___y_338_, lean_object* v_b_339_, lean_object* v___y_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0(v_k_336_, v___y_337_, v___y_338_, v_b_339_, v___y_340_, v___y_341_, v___y_342_, v___y_343_);
lean_dec(v___y_343_);
lean_dec_ref(v___y_342_);
lean_dec(v___y_341_);
lean_dec_ref(v___y_340_);
lean_dec(v___y_338_);
lean_dec_ref(v___y_337_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg(lean_object* v_name_346_, lean_object* v_type_347_, lean_object* v_val_348_, lean_object* v_k_349_, uint8_t v_nondep_350_, uint8_t v_kind_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_){
_start:
{
lean_object* v___f_359_; lean_object* v___x_360_; 
lean_inc(v___y_353_);
lean_inc_ref(v___y_352_);
v___f_359_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_359_, 0, v_k_349_);
lean_closure_set(v___f_359_, 1, v___y_352_);
lean_closure_set(v___f_359_, 2, v___y_353_);
v___x_360_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_346_, v_type_347_, v_val_348_, v___f_359_, v_nondep_350_, v_kind_351_, v___y_354_, v___y_355_, v___y_356_, v___y_357_);
if (lean_obj_tag(v___x_360_) == 0)
{
return v___x_360_;
}
else
{
lean_object* v_a_361_; lean_object* v___x_363_; uint8_t v_isShared_364_; uint8_t v_isSharedCheck_368_; 
v_a_361_ = lean_ctor_get(v___x_360_, 0);
v_isSharedCheck_368_ = !lean_is_exclusive(v___x_360_);
if (v_isSharedCheck_368_ == 0)
{
v___x_363_ = v___x_360_;
v_isShared_364_ = v_isSharedCheck_368_;
goto v_resetjp_362_;
}
else
{
lean_inc(v_a_361_);
lean_dec(v___x_360_);
v___x_363_ = lean_box(0);
v_isShared_364_ = v_isSharedCheck_368_;
goto v_resetjp_362_;
}
v_resetjp_362_:
{
lean_object* v___x_366_; 
if (v_isShared_364_ == 0)
{
v___x_366_ = v___x_363_;
goto v_reusejp_365_;
}
else
{
lean_object* v_reuseFailAlloc_367_; 
v_reuseFailAlloc_367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_367_, 0, v_a_361_);
v___x_366_ = v_reuseFailAlloc_367_;
goto v_reusejp_365_;
}
v_reusejp_365_:
{
return v___x_366_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg___boxed(lean_object* v_name_369_, lean_object* v_type_370_, lean_object* v_val_371_, lean_object* v_k_372_, lean_object* v_nondep_373_, lean_object* v_kind_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
uint8_t v_nondep_boxed_382_; uint8_t v_kind_boxed_383_; lean_object* v_res_384_; 
v_nondep_boxed_382_ = lean_unbox(v_nondep_373_);
v_kind_boxed_383_ = lean_unbox(v_kind_374_);
v_res_384_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg(v_name_369_, v_type_370_, v_val_371_, v_k_372_, v_nondep_boxed_382_, v_kind_boxed_383_, v___y_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
lean_dec(v___y_378_);
lean_dec_ref(v___y_377_);
lean_dec(v___y_376_);
lean_dec_ref(v___y_375_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22(lean_object* v_msgData_385_, lean_object* v___y_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_){
_start:
{
lean_object* v___x_391_; lean_object* v_env_392_; lean_object* v___x_393_; lean_object* v_mctx_394_; lean_object* v_lctx_395_; lean_object* v_options_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_391_ = lean_st_ref_get(v___y_389_);
v_env_392_ = lean_ctor_get(v___x_391_, 0);
lean_inc_ref(v_env_392_);
lean_dec(v___x_391_);
v___x_393_ = lean_st_ref_get(v___y_387_);
v_mctx_394_ = lean_ctor_get(v___x_393_, 0);
lean_inc_ref(v_mctx_394_);
lean_dec(v___x_393_);
v_lctx_395_ = lean_ctor_get(v___y_386_, 2);
v_options_396_ = lean_ctor_get(v___y_388_, 2);
lean_inc_ref(v_options_396_);
lean_inc_ref(v_lctx_395_);
v___x_397_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_397_, 0, v_env_392_);
lean_ctor_set(v___x_397_, 1, v_mctx_394_);
lean_ctor_set(v___x_397_, 2, v_lctx_395_);
lean_ctor_set(v___x_397_, 3, v_options_396_);
v___x_398_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
lean_ctor_set(v___x_398_, 1, v_msgData_385_);
v___x_399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_399_, 0, v___x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22___boxed(lean_object* v_msgData_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_){
_start:
{
lean_object* v_res_406_; 
v_res_406_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22(v_msgData_400_, v___y_401_, v___y_402_, v___y_403_, v___y_404_);
lean_dec(v___y_404_);
lean_dec_ref(v___y_403_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(lean_object* v_msg_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_, lean_object* v___y_411_){
_start:
{
lean_object* v_ref_413_; lean_object* v___x_414_; lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_423_; 
v_ref_413_ = lean_ctor_get(v___y_410_, 5);
v___x_414_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22(v_msg_407_, v___y_408_, v___y_409_, v___y_410_, v___y_411_);
v_a_415_ = lean_ctor_get(v___x_414_, 0);
v_isSharedCheck_423_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_423_ == 0)
{
v___x_417_ = v___x_414_;
v_isShared_418_ = v_isSharedCheck_423_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_414_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_423_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_419_; lean_object* v___x_421_; 
lean_inc(v_ref_413_);
v___x_419_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_419_, 0, v_ref_413_);
lean_ctor_set(v___x_419_, 1, v_a_415_);
if (v_isShared_418_ == 0)
{
lean_ctor_set_tag(v___x_417_, 1);
lean_ctor_set(v___x_417_, 0, v___x_419_);
v___x_421_ = v___x_417_;
goto v_reusejp_420_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_419_);
v___x_421_ = v_reuseFailAlloc_422_;
goto v_reusejp_420_;
}
v_reusejp_420_:
{
return v___x_421_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg___boxed(lean_object* v_msg_424_, lean_object* v___y_425_, lean_object* v___y_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_){
_start:
{
lean_object* v_res_430_; 
v_res_430_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(v_msg_424_, v___y_425_, v___y_426_, v___y_427_, v___y_428_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v___y_426_);
lean_dec_ref(v___y_425_);
return v_res_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg(lean_object* v_name_431_, uint8_t v_bi_432_, lean_object* v_type_433_, lean_object* v_k_434_, uint8_t v_kind_435_, lean_object* v___y_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v___f_443_; lean_object* v___x_444_; 
lean_inc(v___y_437_);
lean_inc_ref(v___y_436_);
v___f_443_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_443_, 0, v_k_434_);
lean_closure_set(v___f_443_, 1, v___y_436_);
lean_closure_set(v___f_443_, 2, v___y_437_);
v___x_444_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_431_, v_bi_432_, v_type_433_, v___f_443_, v_kind_435_, v___y_438_, v___y_439_, v___y_440_, v___y_441_);
if (lean_obj_tag(v___x_444_) == 0)
{
return v___x_444_;
}
else
{
lean_object* v_a_445_; lean_object* v___x_447_; uint8_t v_isShared_448_; uint8_t v_isSharedCheck_452_; 
v_a_445_ = lean_ctor_get(v___x_444_, 0);
v_isSharedCheck_452_ = !lean_is_exclusive(v___x_444_);
if (v_isSharedCheck_452_ == 0)
{
v___x_447_ = v___x_444_;
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
else
{
lean_inc(v_a_445_);
lean_dec(v___x_444_);
v___x_447_ = lean_box(0);
v_isShared_448_ = v_isSharedCheck_452_;
goto v_resetjp_446_;
}
v_resetjp_446_:
{
lean_object* v___x_450_; 
if (v_isShared_448_ == 0)
{
v___x_450_ = v___x_447_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_451_; 
v_reuseFailAlloc_451_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_451_, 0, v_a_445_);
v___x_450_ = v_reuseFailAlloc_451_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
return v___x_450_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg___boxed(lean_object* v_name_453_, lean_object* v_bi_454_, lean_object* v_type_455_, lean_object* v_k_456_, lean_object* v_kind_457_, lean_object* v___y_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
uint8_t v_bi_boxed_465_; uint8_t v_kind_boxed_466_; lean_object* v_res_467_; 
v_bi_boxed_465_ = lean_unbox(v_bi_454_);
v_kind_boxed_466_ = lean_unbox(v_kind_457_);
v_res_467_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg(v_name_453_, v_bi_boxed_465_, v_type_455_, v_k_456_, v_kind_boxed_466_, v___y_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_, v___y_463_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
lean_dec(v___y_459_);
lean_dec_ref(v___y_458_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0(lean_object* v_fvars_468_, lean_object* v_k_469_, lean_object* v_body_470_, lean_object* v_x_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_479_ = lean_array_push(v_fvars_468_, v_x_471_);
lean_inc(v___y_477_);
lean_inc_ref(v___y_476_);
lean_inc(v___y_475_);
lean_inc_ref(v___y_474_);
lean_inc(v___y_473_);
lean_inc_ref(v___y_472_);
v___x_480_ = lean_apply_9(v_k_469_, v___x_479_, v_body_470_, v___y_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, lean_box(0));
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0___boxed(lean_object* v_fvars_481_, lean_object* v_k_482_, lean_object* v_body_483_, lean_object* v_x_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0(v_fvars_481_, v_k_482_, v_body_483_, v_x_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_, v___y_490_);
lean_dec(v___y_490_);
lean_dec_ref(v___y_489_);
lean_dec(v___y_488_);
lean_dec_ref(v___y_487_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
return v_res_492_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1(void){
_start:
{
lean_object* v___x_494_; lean_object* v___x_495_; 
v___x_494_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__0));
v___x_495_ = l_Lean_stringToMessageData(v___x_494_);
return v___x_495_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3(void){
_start:
{
lean_object* v___x_497_; lean_object* v___x_498_; 
v___x_497_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__2));
v___x_498_ = l_Lean_stringToMessageData(v___x_497_);
return v___x_498_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5(void){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__4));
v___x_501_ = l_Lean_stringToMessageData(v___x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg(lean_object* v_k_502_, lean_object* v_fvars_503_, lean_object* v_n_504_, lean_object* v_e_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v_c_514_; lean_object* v_e_515_; lean_object* v_n_527_; lean_object* v_y_528_; lean_object* v_b_529_; uint8_t v_c_530_; lean_object* v___x_535_; uint8_t v___x_536_; 
v___x_535_ = lean_unsigned_to_nat(3u);
v___x_536_ = lean_nat_dec_eq(v_n_504_, v___x_535_);
if (v___x_536_ == 0)
{
lean_object* v___x_537_; uint8_t v___x_538_; 
v___x_537_ = lean_unsigned_to_nat(0u);
v___x_538_ = lean_nat_dec_eq(v_n_504_, v___x_537_);
if (v___x_538_ == 0)
{
lean_object* v___x_539_; uint8_t v___x_540_; 
v___x_539_ = lean_unsigned_to_nat(1u);
v___x_540_ = lean_nat_dec_eq(v_n_504_, v___x_539_);
if (v___x_540_ == 0)
{
lean_object* v___x_541_; uint8_t v___x_542_; 
v___x_541_ = lean_unsigned_to_nat(2u);
v___x_542_ = lean_nat_dec_eq(v_n_504_, v___x_541_);
if (v___x_542_ == 0)
{
if (lean_obj_tag(v_e_505_) == 10)
{
lean_object* v_expr_543_; 
v_expr_543_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_expr_543_);
lean_dec_ref_known(v_e_505_, 2);
v_e_505_ = v_expr_543_;
goto _start;
}
else
{
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_k_502_);
v_c_514_ = v_n_504_;
v_e_515_ = v_e_505_;
goto v___jp_513_;
}
}
else
{
lean_dec(v_n_504_);
switch(lean_obj_tag(v_e_505_))
{
case 8:
{
lean_object* v_declName_545_; lean_object* v_type_546_; lean_object* v_value_547_; lean_object* v_body_548_; lean_object* v___f_549_; lean_object* v___x_550_; lean_object* v___x_551_; uint8_t v___x_552_; lean_object* v___x_553_; 
v_declName_545_ = lean_ctor_get(v_e_505_, 0);
lean_inc(v_declName_545_);
v_type_546_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_type_546_);
v_value_547_ = lean_ctor_get(v_e_505_, 2);
lean_inc_ref(v_value_547_);
v_body_548_ = lean_ctor_get(v_e_505_, 3);
lean_inc_ref(v_body_548_);
lean_dec_ref_known(v_e_505_, 4);
lean_inc_ref(v_fvars_503_);
v___f_549_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__0___boxed), 11, 3);
lean_closure_set(v___f_549_, 0, v_fvars_503_);
lean_closure_set(v___f_549_, 1, v_k_502_);
lean_closure_set(v___f_549_, 2, v_body_548_);
v___x_550_ = lean_expr_instantiate_rev(v_type_546_, v_fvars_503_);
lean_dec_ref(v_type_546_);
v___x_551_ = lean_expr_instantiate_rev(v_value_547_, v_fvars_503_);
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_value_547_);
v___x_552_ = 0;
v___x_553_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg(v_declName_545_, v___x_550_, v___x_551_, v___f_549_, v___x_540_, v___x_552_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
return v___x_553_;
}
case 10:
{
lean_object* v_expr_554_; 
v_expr_554_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_expr_554_);
lean_dec_ref_known(v_e_505_, 2);
v_n_504_ = v___x_541_;
v_e_505_ = v_expr_554_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_k_502_);
v_c_514_ = v___x_541_;
v_e_515_ = v_e_505_;
goto v___jp_513_;
}
}
}
}
else
{
lean_dec(v_n_504_);
switch(lean_obj_tag(v_e_505_))
{
case 5:
{
lean_object* v_arg_556_; lean_object* v___x_557_; 
v_arg_556_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_arg_556_);
lean_dec_ref_known(v_e_505_, 2);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_557_ = lean_apply_9(v_k_502_, v_fvars_503_, v_arg_556_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_557_;
}
case 6:
{
lean_object* v_binderName_558_; lean_object* v_binderType_559_; lean_object* v_body_560_; uint8_t v_binderInfo_561_; 
v_binderName_558_ = lean_ctor_get(v_e_505_, 0);
lean_inc(v_binderName_558_);
v_binderType_559_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_binderType_559_);
v_body_560_ = lean_ctor_get(v_e_505_, 2);
lean_inc_ref(v_body_560_);
v_binderInfo_561_ = lean_ctor_get_uint8(v_e_505_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_505_, 3);
v_n_527_ = v_binderName_558_;
v_y_528_ = v_binderType_559_;
v_b_529_ = v_body_560_;
v_c_530_ = v_binderInfo_561_;
goto v___jp_526_;
}
case 7:
{
lean_object* v_binderName_562_; lean_object* v_binderType_563_; lean_object* v_body_564_; uint8_t v_binderInfo_565_; 
v_binderName_562_ = lean_ctor_get(v_e_505_, 0);
lean_inc(v_binderName_562_);
v_binderType_563_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_binderType_563_);
v_body_564_ = lean_ctor_get(v_e_505_, 2);
lean_inc_ref(v_body_564_);
v_binderInfo_565_ = lean_ctor_get_uint8(v_e_505_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_505_, 3);
v_n_527_ = v_binderName_562_;
v_y_528_ = v_binderType_563_;
v_b_529_ = v_body_564_;
v_c_530_ = v_binderInfo_565_;
goto v___jp_526_;
}
case 8:
{
lean_object* v_value_566_; lean_object* v___x_567_; 
v_value_566_ = lean_ctor_get(v_e_505_, 2);
lean_inc_ref(v_value_566_);
lean_dec_ref_known(v_e_505_, 4);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_567_ = lean_apply_9(v_k_502_, v_fvars_503_, v_value_566_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_567_;
}
case 10:
{
lean_object* v_expr_568_; 
v_expr_568_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_expr_568_);
lean_dec_ref_known(v_e_505_, 2);
v_n_504_ = v___x_539_;
v_e_505_ = v_expr_568_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_k_502_);
v_c_514_ = v___x_539_;
v_e_515_ = v_e_505_;
goto v___jp_513_;
}
}
}
}
else
{
lean_dec(v_n_504_);
switch(lean_obj_tag(v_e_505_))
{
case 5:
{
lean_object* v_fn_570_; lean_object* v___x_571_; 
v_fn_570_ = lean_ctor_get(v_e_505_, 0);
lean_inc_ref(v_fn_570_);
lean_dec_ref_known(v_e_505_, 2);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_571_ = lean_apply_9(v_k_502_, v_fvars_503_, v_fn_570_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_571_;
}
case 6:
{
lean_object* v_binderType_572_; lean_object* v___x_573_; 
v_binderType_572_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_binderType_572_);
lean_dec_ref_known(v_e_505_, 3);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_573_ = lean_apply_9(v_k_502_, v_fvars_503_, v_binderType_572_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_573_;
}
case 7:
{
lean_object* v_binderType_574_; lean_object* v___x_575_; 
v_binderType_574_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_binderType_574_);
lean_dec_ref_known(v_e_505_, 3);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_575_ = lean_apply_9(v_k_502_, v_fvars_503_, v_binderType_574_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_575_;
}
case 8:
{
lean_object* v_type_576_; lean_object* v___x_577_; 
v_type_576_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_type_576_);
lean_dec_ref_known(v_e_505_, 4);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_577_ = lean_apply_9(v_k_502_, v_fvars_503_, v_type_576_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_577_;
}
case 11:
{
lean_object* v_struct_578_; lean_object* v___x_579_; 
v_struct_578_ = lean_ctor_get(v_e_505_, 2);
lean_inc_ref(v_struct_578_);
lean_dec_ref_known(v_e_505_, 3);
lean_inc(v___y_511_);
lean_inc_ref(v___y_510_);
lean_inc(v___y_509_);
lean_inc_ref(v___y_508_);
lean_inc(v___y_507_);
lean_inc_ref(v___y_506_);
v___x_579_ = lean_apply_9(v_k_502_, v_fvars_503_, v_struct_578_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, lean_box(0));
return v___x_579_;
}
case 10:
{
lean_object* v_expr_580_; 
v_expr_580_ = lean_ctor_get(v_e_505_, 1);
lean_inc_ref(v_expr_580_);
lean_dec_ref_known(v_e_505_, 2);
v_n_504_ = v___x_537_;
v_e_505_ = v_expr_580_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_k_502_);
v_c_514_ = v___x_537_;
v_e_515_ = v_e_505_;
goto v___jp_513_;
}
}
}
}
else
{
lean_object* v___x_582_; lean_object* v___x_583_; 
lean_dec_ref(v_e_505_);
lean_dec(v_n_504_);
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_k_502_);
v___x_582_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5);
v___x_583_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(v___x_582_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
return v___x_583_;
}
v___jp_513_:
{
lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; lean_object* v___x_521_; lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v___x_525_; 
v___x_516_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1);
v___x_517_ = l_Nat_reprFast(v_c_514_);
v___x_518_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_518_, 0, v___x_517_);
v___x_519_ = l_Lean_MessageData_ofFormat(v___x_518_);
v___x_520_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_516_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
v___x_521_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3);
v___x_522_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_522_, 0, v___x_520_);
lean_ctor_set(v___x_522_, 1, v___x_521_);
v___x_523_ = l_Lean_MessageData_ofExpr(v_e_515_);
v___x_524_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_524_, 0, v___x_522_);
lean_ctor_set(v___x_524_, 1, v___x_523_);
v___x_525_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(v___x_524_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
return v___x_525_;
}
v___jp_526_:
{
lean_object* v___f_531_; lean_object* v___x_532_; uint8_t v___x_533_; lean_object* v___x_534_; 
lean_inc_ref(v_fvars_503_);
v___f_531_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___lam__1___boxed), 11, 3);
lean_closure_set(v___f_531_, 0, v_fvars_503_);
lean_closure_set(v___f_531_, 1, v_k_502_);
lean_closure_set(v___f_531_, 2, v_b_529_);
v___x_532_ = lean_expr_instantiate_rev(v_y_528_, v_fvars_503_);
lean_dec_ref(v_fvars_503_);
lean_dec_ref(v_y_528_);
v___x_533_ = 0;
v___x_534_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg(v_n_527_, v_c_530_, v___x_532_, v___f_531_, v___x_533_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_);
return v___x_534_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___boxed(lean_object* v_k_584_, lean_object* v_fvars_585_, lean_object* v_n_586_, lean_object* v_e_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_){
_start:
{
lean_object* v_res_595_; 
v_res_595_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg(v_k_584_, v_fvars_585_, v_n_586_, v_e_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_);
lean_dec(v___y_593_);
lean_dec_ref(v___y_592_);
lean_dec(v___y_591_);
lean_dec_ref(v___y_590_);
lean_dec(v___y_589_);
lean_dec_ref(v___y_588_);
return v_res_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1(lean_object* v_fvars_596_, lean_object* v_k_597_, lean_object* v_otherFvars_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_, lean_object* v___y_605_){
_start:
{
lean_object* v___x_607_; lean_object* v___x_608_; 
v___x_607_ = l_Array_append___redArg(v_fvars_596_, v_otherFvars_598_);
lean_inc(v___y_605_);
lean_inc_ref(v___y_604_);
lean_inc(v___y_603_);
lean_inc_ref(v___y_602_);
lean_inc(v___y_601_);
lean_inc_ref(v___y_600_);
v___x_608_ = lean_apply_9(v_k_597_, v___x_607_, v___y_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_, v___y_604_, v___y_605_, lean_box(0));
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1___boxed(lean_object* v_fvars_609_, lean_object* v_k_610_, lean_object* v_otherFvars_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1(v_fvars_609_, v_k_610_, v_otherFvars_611_, v___y_612_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
lean_dec(v___y_616_);
lean_dec_ref(v___y_615_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
lean_dec_ref(v_otherFvars_611_);
return v_res_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0___boxed(lean_object* v_k_621_, lean_object* v_tail_622_, lean_object* v_fvars_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_){
_start:
{
lean_object* v_res_632_; 
v_res_632_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0(v_k_621_, v_tail_622_, v_fvars_623_, v___y_624_, v___y_625_, v___y_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_);
lean_dec(v___y_630_);
lean_dec_ref(v___y_629_);
lean_dec(v___y_628_);
lean_dec_ref(v___y_627_);
lean_dec(v___y_626_);
lean_dec_ref(v___y_625_);
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(lean_object* v_k_635_, lean_object* v_fvars_636_, lean_object* v_x_637_, lean_object* v_x_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_){
_start:
{
if (lean_obj_tag(v_x_637_) == 0)
{
lean_object* v___x_646_; lean_object* v___x_647_; 
v___x_646_ = lean_expr_instantiate_rev(v_x_638_, v_fvars_636_);
lean_dec_ref(v_x_638_);
lean_inc(v___y_644_);
lean_inc_ref(v___y_643_);
lean_inc(v___y_642_);
lean_inc_ref(v___y_641_);
lean_inc(v___y_640_);
lean_inc_ref(v___y_639_);
v___x_647_ = lean_apply_9(v_k_635_, v_fvars_636_, v___x_646_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_, lean_box(0));
return v___x_647_;
}
else
{
lean_object* v_head_648_; lean_object* v_tail_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v_head_648_ = lean_ctor_get(v_x_637_, 0);
lean_inc(v_head_648_);
v_tail_649_ = lean_ctor_get(v_x_637_, 1);
lean_inc(v_tail_649_);
lean_dec_ref_known(v_x_637_, 2);
v___x_650_ = lean_unsigned_to_nat(3u);
v___x_651_ = lean_nat_dec_eq(v_head_648_, v___x_650_);
if (v___x_651_ == 0)
{
lean_object* v___f_652_; lean_object* v___x_653_; 
v___f_652_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0___boxed), 11, 2);
lean_closure_set(v___f_652_, 0, v_k_635_);
lean_closure_set(v___f_652_, 1, v_tail_649_);
v___x_653_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg(v___f_652_, v_fvars_636_, v_head_648_, v_x_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
return v___x_653_;
}
else
{
lean_object* v___x_654_; lean_object* v___x_655_; 
lean_dec(v_head_648_);
v___x_654_ = lean_expr_instantiate_rev(v_x_638_, v_fvars_636_);
lean_dec_ref(v_x_638_);
lean_inc(v___y_644_);
lean_inc_ref(v___y_643_);
lean_inc(v___y_642_);
lean_inc_ref(v___y_641_);
v___x_655_ = lean_infer_type(v___x_654_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
if (lean_obj_tag(v___x_655_) == 0)
{
lean_object* v_a_656_; lean_object* v___f_657_; lean_object* v___x_658_; 
v_a_656_ = lean_ctor_get(v___x_655_, 0);
lean_inc(v_a_656_);
lean_dec_ref_known(v___x_655_, 1);
v___f_657_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__1___boxed), 11, 2);
lean_closure_set(v___f_657_, 0, v_fvars_636_);
lean_closure_set(v___f_657_, 1, v_k_635_);
v___x_658_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0));
v_k_635_ = v___f_657_;
v_fvars_636_ = v___x_658_;
v_x_637_ = v_tail_649_;
v_x_638_ = v_a_656_;
goto _start;
}
else
{
lean_object* v_a_660_; lean_object* v___x_662_; uint8_t v_isShared_663_; uint8_t v_isSharedCheck_667_; 
lean_dec(v_tail_649_);
lean_dec_ref(v_fvars_636_);
lean_dec_ref(v_k_635_);
v_a_660_ = lean_ctor_get(v___x_655_, 0);
v_isSharedCheck_667_ = !lean_is_exclusive(v___x_655_);
if (v_isSharedCheck_667_ == 0)
{
v___x_662_ = v___x_655_;
v_isShared_663_ = v_isSharedCheck_667_;
goto v_resetjp_661_;
}
else
{
lean_inc(v_a_660_);
lean_dec(v___x_655_);
v___x_662_ = lean_box(0);
v_isShared_663_ = v_isSharedCheck_667_;
goto v_resetjp_661_;
}
v_resetjp_661_:
{
lean_object* v___x_665_; 
if (v_isShared_663_ == 0)
{
v___x_665_ = v___x_662_;
goto v_reusejp_664_;
}
else
{
lean_object* v_reuseFailAlloc_666_; 
v_reuseFailAlloc_666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_666_, 0, v_a_660_);
v___x_665_ = v_reuseFailAlloc_666_;
goto v_reusejp_664_;
}
v_reusejp_664_:
{
return v___x_665_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___lam__0(lean_object* v_k_668_, lean_object* v_tail_669_, lean_object* v_fvars_670_, lean_object* v___y_671_, lean_object* v___y_672_, lean_object* v___y_673_, lean_object* v___y_674_, lean_object* v___y_675_, lean_object* v___y_676_, lean_object* v___y_677_){
_start:
{
lean_object* v___x_679_; 
v___x_679_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(v_k_668_, v_fvars_670_, v_tail_669_, v___y_671_, v___y_672_, v___y_673_, v___y_674_, v___y_675_, v___y_676_, v___y_677_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___boxed(lean_object* v_k_680_, lean_object* v_fvars_681_, lean_object* v_x_682_, lean_object* v_x_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_, lean_object* v___y_688_, lean_object* v___y_689_, lean_object* v___y_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(v_k_680_, v_fvars_681_, v_x_682_, v_x_683_, v___y_684_, v___y_685_, v___y_686_, v___y_687_, v___y_688_, v___y_689_);
lean_dec(v___y_689_);
lean_dec_ref(v___y_688_);
lean_dec(v___y_687_);
lean_dec_ref(v___y_686_);
lean_dec(v___y_685_);
lean_dec_ref(v___y_684_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg(lean_object* v_visit_692_, lean_object* v_p_693_, lean_object* v_root_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_){
_start:
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_702_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0));
v___x_703_ = l_Lean_SubExpr_Pos_toArray(v_p_693_);
v___x_704_ = lean_array_to_list(v___x_703_);
v___x_705_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(v_visit_692_, v___x_702_, v___x_704_, v_root_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg___boxed(lean_object* v_visit_706_, lean_object* v_p_707_, lean_object* v_root_708_, lean_object* v___y_709_, lean_object* v___y_710_, lean_object* v___y_711_, lean_object* v___y_712_, lean_object* v___y_713_, lean_object* v___y_714_, lean_object* v___y_715_){
_start:
{
lean_object* v_res_716_; 
v_res_716_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg(v_visit_706_, v_p_707_, v_root_708_, v___y_709_, v___y_710_, v___y_711_, v___y_712_, v___y_713_, v___y_714_);
lean_dec(v___y_714_);
lean_dec_ref(v___y_713_);
lean_dec(v___y_712_);
lean_dec_ref(v___y_711_);
lean_dec(v___y_710_);
lean_dec_ref(v___y_709_);
lean_dec(v_p_707_);
return v_res_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg(lean_object* v_e_717_, lean_object* v_pos_718_, lean_object* v_k_719_, lean_object* v___y_720_, lean_object* v___y_721_, lean_object* v___y_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_){
_start:
{
lean_object* v___x_727_; 
lean_inc_ref(v_e_717_);
v___x_727_ = lp_mathlib_Lean_Meta_viewKAbstractSubExpr(v_e_717_, v_pos_718_, v___y_722_, v___y_723_, v___y_724_, v___y_725_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_object* v_a_728_; 
v_a_728_ = lean_ctor_get(v___x_727_, 0);
lean_inc(v_a_728_);
lean_dec_ref_known(v___x_727_, 1);
if (lean_obj_tag(v_a_728_) == 1)
{
lean_object* v_val_729_; lean_object* v_fst_730_; lean_object* v_snd_731_; lean_object* v___x_732_; 
v_val_729_ = lean_ctor_get(v_a_728_, 0);
lean_inc(v_val_729_);
lean_dec_ref_known(v_a_728_, 1);
v_fst_730_ = lean_ctor_get(v_val_729_, 0);
lean_inc_n(v_fst_730_, 2);
v_snd_731_ = lean_ctor_get(v_val_729_, 1);
lean_inc(v_snd_731_);
lean_dec(v_val_729_);
v___x_732_ = lp_mathlib_Lean_Meta_kabstractIsTypeCorrect(v_e_717_, v_fst_730_, v_pos_718_, v___y_722_, v___y_723_, v___y_724_, v___y_725_);
if (lean_obj_tag(v___x_732_) == 0)
{
lean_object* v_a_733_; lean_object* v___x_734_; uint8_t v___x_735_; lean_object* v___x_736_; 
v_a_733_ = lean_ctor_get(v___x_732_, 0);
lean_inc(v_a_733_);
lean_dec_ref_known(v___x_732_, 1);
v___x_734_ = lean_alloc_ctor(1, 1, 1);
lean_ctor_set(v___x_734_, 0, v_snd_731_);
v___x_735_ = lean_unbox(v_a_733_);
lean_dec(v_a_733_);
lean_ctor_set_uint8(v___x_734_, sizeof(void*)*1, v___x_735_);
lean_inc(v___y_725_);
lean_inc_ref(v___y_724_);
lean_inc(v___y_723_);
lean_inc_ref(v___y_722_);
lean_inc(v___y_721_);
lean_inc_ref(v___y_720_);
v___x_736_ = lean_apply_9(v_k_719_, v_fst_730_, v___x_734_, v___y_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_, lean_box(0));
return v___x_736_;
}
else
{
lean_object* v_a_737_; lean_object* v___x_739_; uint8_t v_isShared_740_; uint8_t v_isSharedCheck_744_; 
lean_dec(v_snd_731_);
lean_dec(v_fst_730_);
lean_dec_ref(v_k_719_);
v_a_737_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_744_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_744_ == 0)
{
v___x_739_ = v___x_732_;
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
else
{
lean_inc(v_a_737_);
lean_dec(v___x_732_);
v___x_739_ = lean_box(0);
v_isShared_740_ = v_isSharedCheck_744_;
goto v_resetjp_738_;
}
v_resetjp_738_:
{
lean_object* v___x_742_; 
if (v_isShared_740_ == 0)
{
v___x_742_ = v___x_739_;
goto v_reusejp_741_;
}
else
{
lean_object* v_reuseFailAlloc_743_; 
v_reuseFailAlloc_743_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_743_, 0, v_a_737_);
v___x_742_ = v_reuseFailAlloc_743_;
goto v_reusejp_741_;
}
v_reusejp_741_:
{
return v___x_742_;
}
}
}
}
else
{
lean_object* v___f_745_; lean_object* v___x_746_; 
lean_dec(v_a_728_);
v___f_745_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___lam__0___boxed), 10, 1);
lean_closure_set(v___f_745_, 0, v_k_719_);
v___x_746_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg(v___f_745_, v_pos_718_, v_e_717_, v___y_720_, v___y_721_, v___y_722_, v___y_723_, v___y_724_, v___y_725_);
lean_dec(v_pos_718_);
return v___x_746_;
}
}
else
{
lean_object* v_a_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_754_; 
lean_dec_ref(v_k_719_);
lean_dec(v_pos_718_);
lean_dec_ref(v_e_717_);
v_a_747_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_754_ == 0)
{
v___x_749_ = v___x_727_;
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_a_747_);
lean_dec(v___x_727_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_752_; 
if (v_isShared_750_ == 0)
{
v___x_752_ = v___x_749_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v_a_747_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg___boxed(lean_object* v_e_755_, lean_object* v_pos_756_, lean_object* v_k_757_, lean_object* v___y_758_, lean_object* v___y_759_, lean_object* v___y_760_, lean_object* v___y_761_, lean_object* v___y_762_, lean_object* v___y_763_, lean_object* v___y_764_){
_start:
{
lean_object* v_res_765_; 
v_res_765_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg(v_e_755_, v_pos_756_, v_k_757_, v___y_758_, v___y_759_, v___y_760_, v___y_761_, v___y_762_, v___y_763_);
lean_dec(v___y_763_);
lean_dec_ref(v___y_762_);
lean_dec(v___y_761_);
lean_dec_ref(v___y_760_);
lean_dec(v___y_759_);
lean_dec_ref(v___y_758_);
return v_res_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1(lean_object* v_token_766_, lean_object* v_fst_767_, lean_object* v_parentDecl_x3f_768_, lean_object* v_mvarId_769_, lean_object* v_____x_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_){
_start:
{
lean_object* v_fst_778_; lean_object* v_snd_779_; lean_object* v_rootExpr_781_; lean_object* v___y_782_; lean_object* v___y_783_; lean_object* v___y_784_; lean_object* v___y_785_; lean_object* v___y_786_; lean_object* v___y_787_; 
v_fst_778_ = lean_ctor_get(v_____x_770_, 0);
lean_inc(v_fst_778_);
v_snd_779_ = lean_ctor_get(v_____x_770_, 1);
lean_inc(v_snd_779_);
lean_dec_ref(v_____x_770_);
if (lean_obj_tag(v_fst_778_) == 0)
{
lean_object* v___x_790_; 
v___x_790_ = l_Lean_MVarId_getType(v_mvarId_769_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
if (lean_obj_tag(v___x_790_) == 0)
{
lean_object* v_a_791_; 
v_a_791_ = lean_ctor_get(v___x_790_, 0);
lean_inc(v_a_791_);
lean_dec_ref_known(v___x_790_, 1);
v_rootExpr_781_ = v_a_791_;
v___y_782_ = v___y_771_;
v___y_783_ = v___y_772_;
v___y_784_ = v___y_773_;
v___y_785_ = v___y_774_;
v___y_786_ = v___y_775_;
v___y_787_ = v___y_776_;
goto v___jp_780_;
}
else
{
lean_object* v_a_792_; lean_object* v___x_794_; uint8_t v_isShared_795_; uint8_t v_isSharedCheck_799_; 
lean_dec(v_snd_779_);
lean_dec(v_parentDecl_x3f_768_);
lean_dec(v_fst_767_);
lean_dec_ref(v_token_766_);
v_a_792_ = lean_ctor_get(v___x_790_, 0);
v_isSharedCheck_799_ = !lean_is_exclusive(v___x_790_);
if (v_isSharedCheck_799_ == 0)
{
v___x_794_ = v___x_790_;
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
else
{
lean_inc(v_a_792_);
lean_dec(v___x_790_);
v___x_794_ = lean_box(0);
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
v_resetjp_793_:
{
lean_object* v___x_797_; 
if (v_isShared_795_ == 0)
{
v___x_797_ = v___x_794_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_a_792_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
}
}
else
{
lean_object* v_val_800_; lean_object* v___x_801_; 
lean_dec(v_mvarId_769_);
v_val_800_ = lean_ctor_get(v_fst_778_, 0);
lean_inc(v_val_800_);
lean_dec_ref_known(v_fst_778_, 1);
v___x_801_ = l_Lean_FVarId_getType___redArg(v_val_800_, v___y_773_, v___y_775_, v___y_776_);
if (lean_obj_tag(v___x_801_) == 0)
{
lean_object* v_a_802_; 
v_a_802_ = lean_ctor_get(v___x_801_, 0);
lean_inc(v_a_802_);
lean_dec_ref_known(v___x_801_, 1);
v_rootExpr_781_ = v_a_802_;
v___y_782_ = v___y_771_;
v___y_783_ = v___y_772_;
v___y_784_ = v___y_773_;
v___y_785_ = v___y_774_;
v___y_786_ = v___y_775_;
v___y_787_ = v___y_776_;
goto v___jp_780_;
}
else
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_810_; 
lean_dec(v_snd_779_);
lean_dec(v_parentDecl_x3f_768_);
lean_dec(v_fst_767_);
lean_dec_ref(v_token_766_);
v_a_803_ = lean_ctor_get(v___x_801_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_801_);
if (v_isSharedCheck_810_ == 0)
{
v___x_805_ = v___x_801_;
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___x_801_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v___x_808_; 
if (v_isShared_806_ == 0)
{
v___x_808_ = v___x_805_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_a_803_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
}
v___jp_780_:
{
lean_object* v___f_788_; lean_object* v___x_789_; 
lean_inc_ref(v_rootExpr_781_);
v___f_788_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___boxed), 13, 4);
lean_closure_set(v___f_788_, 0, v_token_766_);
lean_closure_set(v___f_788_, 1, v_rootExpr_781_);
lean_closure_set(v___f_788_, 2, v_fst_767_);
lean_closure_set(v___f_788_, 3, v_parentDecl_x3f_768_);
v___x_789_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg(v_rootExpr_781_, v_snd_779_, v___f_788_, v___y_782_, v___y_783_, v___y_784_, v___y_785_, v___y_786_, v___y_787_);
return v___x_789_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1___boxed(lean_object* v_token_811_, lean_object* v_fst_812_, lean_object* v_parentDecl_x3f_813_, lean_object* v_mvarId_814_, lean_object* v_____x_815_, lean_object* v___y_816_, lean_object* v___y_817_, lean_object* v___y_818_, lean_object* v___y_819_, lean_object* v___y_820_, lean_object* v___y_821_, lean_object* v___y_822_){
_start:
{
lean_object* v_res_823_; 
v_res_823_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1(v_token_811_, v_fst_812_, v_parentDecl_x3f_813_, v_mvarId_814_, v_____x_815_, v___y_816_, v___y_817_, v___y_818_, v___y_819_, v___y_820_, v___y_821_);
lean_dec(v___y_821_);
lean_dec_ref(v___y_820_);
lean_dec(v___y_819_);
lean_dec_ref(v___y_818_);
lean_dec(v___y_817_);
lean_dec_ref(v___y_816_);
return v_res_823_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2(lean_object* v_loc_827_, lean_object* v___f_828_, lean_object* v_token_829_, lean_object* v___y_830_, lean_object* v___y_831_, lean_object* v___y_832_, lean_object* v___y_833_, lean_object* v___y_834_, lean_object* v___y_835_){
_start:
{
switch(lean_obj_tag(v_loc_827_))
{
case 0:
{
lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_844_; 
lean_dec_ref(v_token_829_);
lean_dec_ref(v___f_828_);
v_isSharedCheck_844_ = !lean_is_exclusive(v_loc_827_);
if (v_isSharedCheck_844_ == 0)
{
lean_object* v_unused_845_; 
v_unused_845_ = lean_ctor_get(v_loc_827_, 0);
lean_dec(v_unused_845_);
v___x_838_ = v_loc_827_;
v_isShared_839_ = v_isSharedCheck_844_;
goto v_resetjp_837_;
}
else
{
lean_dec(v_loc_827_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_844_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___x_840_; lean_object* v___x_842_; 
v___x_840_ = lean_box(0);
if (v_isShared_839_ == 0)
{
lean_ctor_set(v___x_838_, 0, v___x_840_);
v___x_842_ = v___x_838_;
goto v_reusejp_841_;
}
else
{
lean_object* v_reuseFailAlloc_843_; 
v_reuseFailAlloc_843_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_843_, 0, v___x_840_);
v___x_842_ = v_reuseFailAlloc_843_;
goto v_reusejp_841_;
}
v_reusejp_841_:
{
return v___x_842_;
}
}
}
case 1:
{
lean_object* v_a_846_; lean_object* v_a_847_; lean_object* v___x_849_; uint8_t v_isShared_850_; uint8_t v_isSharedCheck_856_; 
lean_dec_ref(v_token_829_);
v_a_846_ = lean_ctor_get(v_loc_827_, 0);
v_a_847_ = lean_ctor_get(v_loc_827_, 1);
v_isSharedCheck_856_ = !lean_is_exclusive(v_loc_827_);
if (v_isSharedCheck_856_ == 0)
{
v___x_849_ = v_loc_827_;
v_isShared_850_ = v_isSharedCheck_856_;
goto v_resetjp_848_;
}
else
{
lean_inc(v_a_847_);
lean_inc(v_a_846_);
lean_dec(v_loc_827_);
v___x_849_ = lean_box(0);
v_isShared_850_ = v_isSharedCheck_856_;
goto v_resetjp_848_;
}
v_resetjp_848_:
{
lean_object* v___x_851_; lean_object* v___x_853_; 
v___x_851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_851_, 0, v_a_846_);
if (v_isShared_850_ == 0)
{
lean_ctor_set_tag(v___x_849_, 0);
lean_ctor_set(v___x_849_, 0, v___x_851_);
v___x_853_ = v___x_849_;
goto v_reusejp_852_;
}
else
{
lean_object* v_reuseFailAlloc_855_; 
v_reuseFailAlloc_855_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_855_, 0, v___x_851_);
lean_ctor_set(v_reuseFailAlloc_855_, 1, v_a_847_);
v___x_853_ = v_reuseFailAlloc_855_;
goto v_reusejp_852_;
}
v_reusejp_852_:
{
lean_object* v___x_854_; 
lean_inc(v___y_835_);
lean_inc_ref(v___y_834_);
lean_inc(v___y_833_);
lean_inc_ref(v___y_832_);
lean_inc(v___y_831_);
lean_inc_ref(v___y_830_);
v___x_854_ = lean_apply_8(v___f_828_, v___x_853_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, lean_box(0));
return v___x_854_;
}
}
}
case 2:
{
lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; 
lean_dec_ref_known(v_loc_827_, 2);
lean_dec_ref(v___f_828_);
v___x_857_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___closed__1));
v___x_858_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_token_829_, v___x_857_);
v___x_859_ = lean_box(0);
v___x_860_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_860_, 0, v___x_859_);
return v___x_860_;
}
default: 
{
lean_object* v_a_861_; lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
lean_dec_ref(v_token_829_);
v_a_861_ = lean_ctor_get(v_loc_827_, 0);
lean_inc(v_a_861_);
lean_dec_ref_known(v_loc_827_, 1);
v___x_862_ = lean_box(0);
v___x_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_863_, 0, v___x_862_);
lean_ctor_set(v___x_863_, 1, v_a_861_);
lean_inc(v___y_835_);
lean_inc_ref(v___y_834_);
lean_inc(v___y_833_);
lean_inc_ref(v___y_832_);
lean_inc(v___y_831_);
lean_inc_ref(v___y_830_);
v___x_864_ = lean_apply_8(v___f_828_, v___x_863_, v___y_830_, v___y_831_, v___y_832_, v___y_833_, v___y_834_, v___y_835_, lean_box(0));
return v___x_864_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___boxed(lean_object* v_loc_865_, lean_object* v___f_866_, lean_object* v_token_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_, lean_object* v___y_873_, lean_object* v___y_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2(v_loc_865_, v___f_866_, v_token_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_, v___y_872_, v___y_873_);
lean_dec(v___y_873_);
lean_dec_ref(v___y_872_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3(lean_object* v_token_877_, lean_object* v_parentDecl_x3f_878_, lean_object* v_mvarId_879_, lean_object* v_loc_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_){
_start:
{
lean_object* v_lctx_888_; lean_object* v_options_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v___x_892_; lean_object* v_fst_893_; lean_object* v___f_894_; lean_object* v___y_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
v_lctx_888_ = lean_ctor_get(v___y_883_, 2);
v_options_889_ = lean_ctor_get(v___y_885_, 2);
v___x_890_ = lean_box(1);
lean_inc_ref(v_options_889_);
v___x_891_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_891_, 0, v_options_889_);
lean_ctor_set(v___x_891_, 1, v___x_890_);
lean_ctor_set(v___x_891_, 2, v___x_890_);
lean_inc_ref(v_lctx_888_);
v___x_892_ = l_Lean_LocalContext_sanitizeNames(v_lctx_888_, v___x_891_);
v_fst_893_ = lean_ctor_get(v___x_892_, 0);
lean_inc_n(v_fst_893_, 2);
lean_dec_ref(v___x_892_);
lean_inc_ref(v_token_877_);
v___f_894_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__1___boxed), 12, 4);
lean_closure_set(v___f_894_, 0, v_token_877_);
lean_closure_set(v___f_894_, 1, v_fst_893_);
lean_closure_set(v___f_894_, 2, v_parentDecl_x3f_878_);
lean_closure_set(v___f_894_, 3, v_mvarId_879_);
v___y_895_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__2___boxed), 10, 3);
lean_closure_set(v___y_895_, 0, v_loc_880_);
lean_closure_set(v___y_895_, 1, v___f_894_);
lean_closure_set(v___y_895_, 2, v_token_877_);
v___x_896_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___closed__0));
v___x_897_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_trackingComputation___boxed), 10, 3);
lean_closure_set(v___x_897_, 0, lean_box(0));
lean_closure_set(v___x_897_, 1, v___x_896_);
lean_closure_set(v___x_897_, 2, v___y_895_);
v___x_898_ = lp_mathlib_Lean_Meta_withLCtx_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__2___redArg(v_fst_893_, v___x_897_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___boxed(lean_object* v_token_899_, lean_object* v_parentDecl_x3f_900_, lean_object* v_mvarId_901_, lean_object* v_loc_902_, lean_object* v___y_903_, lean_object* v___y_904_, lean_object* v___y_905_, lean_object* v___y_906_, lean_object* v___y_907_, lean_object* v___y_908_, lean_object* v___y_909_){
_start:
{
lean_object* v_res_910_; 
v_res_910_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3(v_token_899_, v_parentDecl_x3f_900_, v_mvarId_901_, v_loc_902_, v___y_903_, v___y_904_, v___y_905_, v___y_906_, v___y_907_, v___y_908_);
lean_dec(v___y_908_);
lean_dec_ref(v___y_907_);
lean_dec(v___y_906_);
lean_dec_ref(v___y_905_);
lean_dec(v___y_904_);
lean_dec_ref(v___y_903_);
return v_res_910_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(lean_object* v_e_911_, lean_object* v___y_912_){
_start:
{
uint8_t v___x_914_; 
v___x_914_ = l_Lean_Expr_hasMVar(v_e_911_);
if (v___x_914_ == 0)
{
lean_object* v___x_915_; 
v___x_915_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_915_, 0, v_e_911_);
return v___x_915_;
}
else
{
lean_object* v___x_916_; lean_object* v_mctx_917_; lean_object* v___x_918_; lean_object* v_fst_919_; lean_object* v_snd_920_; lean_object* v___x_921_; lean_object* v_cache_922_; lean_object* v_zetaDeltaFVarIds_923_; lean_object* v_postponed_924_; lean_object* v_diag_925_; lean_object* v___x_927_; uint8_t v_isShared_928_; uint8_t v_isSharedCheck_934_; 
v___x_916_ = lean_st_ref_get(v___y_912_);
v_mctx_917_ = lean_ctor_get(v___x_916_, 0);
lean_inc_ref(v_mctx_917_);
lean_dec(v___x_916_);
v___x_918_ = l_Lean_instantiateMVarsCore(v_mctx_917_, v_e_911_);
v_fst_919_ = lean_ctor_get(v___x_918_, 0);
lean_inc(v_fst_919_);
v_snd_920_ = lean_ctor_get(v___x_918_, 1);
lean_inc(v_snd_920_);
lean_dec_ref(v___x_918_);
v___x_921_ = lean_st_ref_take(v___y_912_);
v_cache_922_ = lean_ctor_get(v___x_921_, 1);
v_zetaDeltaFVarIds_923_ = lean_ctor_get(v___x_921_, 2);
v_postponed_924_ = lean_ctor_get(v___x_921_, 3);
v_diag_925_ = lean_ctor_get(v___x_921_, 4);
v_isSharedCheck_934_ = !lean_is_exclusive(v___x_921_);
if (v_isSharedCheck_934_ == 0)
{
lean_object* v_unused_935_; 
v_unused_935_ = lean_ctor_get(v___x_921_, 0);
lean_dec(v_unused_935_);
v___x_927_ = v___x_921_;
v_isShared_928_ = v_isSharedCheck_934_;
goto v_resetjp_926_;
}
else
{
lean_inc(v_diag_925_);
lean_inc(v_postponed_924_);
lean_inc(v_zetaDeltaFVarIds_923_);
lean_inc(v_cache_922_);
lean_dec(v___x_921_);
v___x_927_ = lean_box(0);
v_isShared_928_ = v_isSharedCheck_934_;
goto v_resetjp_926_;
}
v_resetjp_926_:
{
lean_object* v___x_930_; 
if (v_isShared_928_ == 0)
{
lean_ctor_set(v___x_927_, 0, v_snd_920_);
v___x_930_ = v___x_927_;
goto v_reusejp_929_;
}
else
{
lean_object* v_reuseFailAlloc_933_; 
v_reuseFailAlloc_933_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_933_, 0, v_snd_920_);
lean_ctor_set(v_reuseFailAlloc_933_, 1, v_cache_922_);
lean_ctor_set(v_reuseFailAlloc_933_, 2, v_zetaDeltaFVarIds_923_);
lean_ctor_set(v_reuseFailAlloc_933_, 3, v_postponed_924_);
lean_ctor_set(v_reuseFailAlloc_933_, 4, v_diag_925_);
v___x_930_ = v_reuseFailAlloc_933_;
goto v_reusejp_929_;
}
v_reusejp_929_:
{
lean_object* v___x_931_; lean_object* v___x_932_; 
v___x_931_ = lean_st_ref_set(v___y_912_, v___x_930_);
v___x_932_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_932_, 0, v_fst_919_);
return v___x_932_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg___boxed(lean_object* v_e_936_, lean_object* v___y_937_, lean_object* v___y_938_){
_start:
{
lean_object* v_res_939_; 
v_res_939_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_e_936_, v___y_937_);
lean_dec(v___y_937_);
return v_res_939_;
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0(void){
_start:
{
lean_object* v___x_940_; 
v___x_940_ = l_instMonadEIO(lean_box(0));
return v___x_940_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4(lean_object* v_msg_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_, lean_object* v___y_950_, lean_object* v___y_951_){
_start:
{
lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v_toApplicative_955_; lean_object* v___x_957_; uint8_t v_isShared_958_; uint8_t v_isSharedCheck_1018_; 
v___x_953_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0, &lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0_once, _init_lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__0);
v___x_954_ = l_StateRefT_x27_instMonad___redArg(v___x_953_);
v_toApplicative_955_ = lean_ctor_get(v___x_954_, 0);
v_isSharedCheck_1018_ = !lean_is_exclusive(v___x_954_);
if (v_isSharedCheck_1018_ == 0)
{
lean_object* v_unused_1019_; 
v_unused_1019_ = lean_ctor_get(v___x_954_, 1);
lean_dec(v_unused_1019_);
v___x_957_ = v___x_954_;
v_isShared_958_ = v_isSharedCheck_1018_;
goto v_resetjp_956_;
}
else
{
lean_inc(v_toApplicative_955_);
lean_dec(v___x_954_);
v___x_957_ = lean_box(0);
v_isShared_958_ = v_isSharedCheck_1018_;
goto v_resetjp_956_;
}
v_resetjp_956_:
{
lean_object* v_toFunctor_959_; lean_object* v_toSeq_960_; lean_object* v_toSeqLeft_961_; lean_object* v_toSeqRight_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_1016_; 
v_toFunctor_959_ = lean_ctor_get(v_toApplicative_955_, 0);
v_toSeq_960_ = lean_ctor_get(v_toApplicative_955_, 2);
v_toSeqLeft_961_ = lean_ctor_get(v_toApplicative_955_, 3);
v_toSeqRight_962_ = lean_ctor_get(v_toApplicative_955_, 4);
v_isSharedCheck_1016_ = !lean_is_exclusive(v_toApplicative_955_);
if (v_isSharedCheck_1016_ == 0)
{
lean_object* v_unused_1017_; 
v_unused_1017_ = lean_ctor_get(v_toApplicative_955_, 1);
lean_dec(v_unused_1017_);
v___x_964_ = v_toApplicative_955_;
v_isShared_965_ = v_isSharedCheck_1016_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_toSeqRight_962_);
lean_inc(v_toSeqLeft_961_);
lean_inc(v_toSeq_960_);
lean_inc(v_toFunctor_959_);
lean_dec(v_toApplicative_955_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_1016_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v___f_966_; lean_object* v___f_967_; lean_object* v___f_968_; lean_object* v___f_969_; lean_object* v___x_970_; lean_object* v___f_971_; lean_object* v___f_972_; lean_object* v___f_973_; lean_object* v___x_975_; 
v___f_966_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__1));
v___f_967_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__2));
lean_inc_ref(v_toFunctor_959_);
v___f_968_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_968_, 0, v_toFunctor_959_);
v___f_969_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_969_, 0, v_toFunctor_959_);
v___x_970_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_970_, 0, v___f_968_);
lean_ctor_set(v___x_970_, 1, v___f_969_);
v___f_971_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_971_, 0, v_toSeqRight_962_);
v___f_972_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_972_, 0, v_toSeqLeft_961_);
v___f_973_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_973_, 0, v_toSeq_960_);
if (v_isShared_965_ == 0)
{
lean_ctor_set(v___x_964_, 4, v___f_971_);
lean_ctor_set(v___x_964_, 3, v___f_972_);
lean_ctor_set(v___x_964_, 2, v___f_973_);
lean_ctor_set(v___x_964_, 1, v___f_966_);
lean_ctor_set(v___x_964_, 0, v___x_970_);
v___x_975_ = v___x_964_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_1015_; 
v_reuseFailAlloc_1015_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1015_, 0, v___x_970_);
lean_ctor_set(v_reuseFailAlloc_1015_, 1, v___f_966_);
lean_ctor_set(v_reuseFailAlloc_1015_, 2, v___f_973_);
lean_ctor_set(v_reuseFailAlloc_1015_, 3, v___f_972_);
lean_ctor_set(v_reuseFailAlloc_1015_, 4, v___f_971_);
v___x_975_ = v_reuseFailAlloc_1015_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
lean_object* v___x_977_; 
if (v_isShared_958_ == 0)
{
lean_ctor_set(v___x_957_, 1, v___f_967_);
lean_ctor_set(v___x_957_, 0, v___x_975_);
v___x_977_ = v___x_957_;
goto v_reusejp_976_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_1014_, 1, v___f_967_);
v___x_977_ = v_reuseFailAlloc_1014_;
goto v_reusejp_976_;
}
v_reusejp_976_:
{
lean_object* v___x_978_; lean_object* v_toApplicative_979_; lean_object* v___x_981_; uint8_t v_isShared_982_; uint8_t v_isSharedCheck_1012_; 
v___x_978_ = l_StateRefT_x27_instMonad___redArg(v___x_977_);
v_toApplicative_979_ = lean_ctor_get(v___x_978_, 0);
v_isSharedCheck_1012_ = !lean_is_exclusive(v___x_978_);
if (v_isSharedCheck_1012_ == 0)
{
lean_object* v_unused_1013_; 
v_unused_1013_ = lean_ctor_get(v___x_978_, 1);
lean_dec(v_unused_1013_);
v___x_981_ = v___x_978_;
v_isShared_982_ = v_isSharedCheck_1012_;
goto v_resetjp_980_;
}
else
{
lean_inc(v_toApplicative_979_);
lean_dec(v___x_978_);
v___x_981_ = lean_box(0);
v_isShared_982_ = v_isSharedCheck_1012_;
goto v_resetjp_980_;
}
v_resetjp_980_:
{
lean_object* v_toFunctor_983_; lean_object* v_toSeq_984_; lean_object* v_toSeqLeft_985_; lean_object* v_toSeqRight_986_; lean_object* v___x_988_; uint8_t v_isShared_989_; uint8_t v_isSharedCheck_1010_; 
v_toFunctor_983_ = lean_ctor_get(v_toApplicative_979_, 0);
v_toSeq_984_ = lean_ctor_get(v_toApplicative_979_, 2);
v_toSeqLeft_985_ = lean_ctor_get(v_toApplicative_979_, 3);
v_toSeqRight_986_ = lean_ctor_get(v_toApplicative_979_, 4);
v_isSharedCheck_1010_ = !lean_is_exclusive(v_toApplicative_979_);
if (v_isSharedCheck_1010_ == 0)
{
lean_object* v_unused_1011_; 
v_unused_1011_ = lean_ctor_get(v_toApplicative_979_, 1);
lean_dec(v_unused_1011_);
v___x_988_ = v_toApplicative_979_;
v_isShared_989_ = v_isSharedCheck_1010_;
goto v_resetjp_987_;
}
else
{
lean_inc(v_toSeqRight_986_);
lean_inc(v_toSeqLeft_985_);
lean_inc(v_toSeq_984_);
lean_inc(v_toFunctor_983_);
lean_dec(v_toApplicative_979_);
v___x_988_ = lean_box(0);
v_isShared_989_ = v_isSharedCheck_1010_;
goto v_resetjp_987_;
}
v_resetjp_987_:
{
lean_object* v___f_990_; lean_object* v___f_991_; lean_object* v___f_992_; lean_object* v___f_993_; lean_object* v___x_994_; lean_object* v___f_995_; lean_object* v___f_996_; lean_object* v___f_997_; lean_object* v___x_999_; 
v___f_990_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__3));
v___f_991_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___closed__4));
lean_inc_ref(v_toFunctor_983_);
v___f_992_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_992_, 0, v_toFunctor_983_);
v___f_993_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_993_, 0, v_toFunctor_983_);
v___x_994_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_994_, 0, v___f_992_);
lean_ctor_set(v___x_994_, 1, v___f_993_);
v___f_995_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_995_, 0, v_toSeqRight_986_);
v___f_996_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_996_, 0, v_toSeqLeft_985_);
v___f_997_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_997_, 0, v_toSeq_984_);
if (v_isShared_989_ == 0)
{
lean_ctor_set(v___x_988_, 4, v___f_995_);
lean_ctor_set(v___x_988_, 3, v___f_996_);
lean_ctor_set(v___x_988_, 2, v___f_997_);
lean_ctor_set(v___x_988_, 1, v___f_990_);
lean_ctor_set(v___x_988_, 0, v___x_994_);
v___x_999_ = v___x_988_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v___x_994_);
lean_ctor_set(v_reuseFailAlloc_1009_, 1, v___f_990_);
lean_ctor_set(v_reuseFailAlloc_1009_, 2, v___f_997_);
lean_ctor_set(v_reuseFailAlloc_1009_, 3, v___f_996_);
lean_ctor_set(v_reuseFailAlloc_1009_, 4, v___f_995_);
v___x_999_ = v_reuseFailAlloc_1009_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
lean_object* v___x_1001_; 
if (v_isShared_982_ == 0)
{
lean_ctor_set(v___x_981_, 1, v___f_991_);
lean_ctor_set(v___x_981_, 0, v___x_999_);
v___x_1001_ = v___x_981_;
goto v_reusejp_1000_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v___x_999_);
lean_ctor_set(v_reuseFailAlloc_1008_, 1, v___f_991_);
v___x_1001_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1000_;
}
v_reusejp_1000_:
{
lean_object* v___x_1002_; lean_object* v___x_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_15820__overap_1006_; lean_object* v___x_1007_; 
v___x_1002_ = l_ReaderT_instMonad___redArg(v___x_1001_);
v___x_1003_ = l_ReaderT_instMonad___redArg(v___x_1002_);
v___x_1004_ = l_Lean_instInhabitedLocalContext_default;
v___x_1005_ = l_instInhabitedOfMonad___redArg(v___x_1003_, v___x_1004_);
v___x_15820__overap_1006_ = lean_panic_fn_borrowed(v___x_1005_, v_msg_945_);
lean_dec(v___x_1005_);
lean_inc(v___y_951_);
lean_inc_ref(v___y_950_);
lean_inc(v___y_949_);
lean_inc_ref(v___y_948_);
lean_inc(v___y_947_);
lean_inc_ref(v___y_946_);
v___x_1007_ = lean_apply_7(v___x_15820__overap_1006_, v___y_946_, v___y_947_, v___y_948_, v___y_949_, v___y_950_, v___y_951_, lean_box(0));
return v___x_1007_;
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
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4___boxed(lean_object* v_msg_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4(v_msg_1020_, v___y_1021_, v___y_1022_, v___y_1023_, v___y_1024_, v___y_1025_, v___y_1026_);
lean_dec(v___y_1026_);
lean_dec_ref(v___y_1025_);
lean_dec(v___y_1024_);
lean_dec_ref(v___y_1023_);
lean_dec(v___y_1022_);
lean_dec_ref(v___y_1021_);
return v_res_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg(lean_object* v_t_1029_, lean_object* v_k_1030_){
_start:
{
if (lean_obj_tag(v_t_1029_) == 0)
{
lean_object* v_k_1031_; lean_object* v_v_1032_; lean_object* v_l_1033_; lean_object* v_r_1034_; uint8_t v___x_1035_; 
v_k_1031_ = lean_ctor_get(v_t_1029_, 1);
v_v_1032_ = lean_ctor_get(v_t_1029_, 2);
v_l_1033_ = lean_ctor_get(v_t_1029_, 3);
v_r_1034_ = lean_ctor_get(v_t_1029_, 4);
v___x_1035_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_1030_, v_k_1031_);
switch(v___x_1035_)
{
case 0:
{
v_t_1029_ = v_l_1033_;
goto _start;
}
case 1:
{
lean_object* v___x_1037_; 
lean_inc(v_v_1032_);
v___x_1037_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1037_, 0, v_v_1032_);
return v___x_1037_;
}
default: 
{
v_t_1029_ = v_r_1034_;
goto _start;
}
}
}
else
{
lean_object* v___x_1039_; 
v___x_1039_ = lean_box(0);
return v___x_1039_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_t_1040_, lean_object* v_k_1041_){
_start:
{
lean_object* v_res_1042_; 
v_res_1042_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg(v_t_1040_, v_k_1041_);
lean_dec(v_k_1041_);
lean_dec(v_t_1040_);
return v_res_1042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(lean_object* v_auxDeclToFullName_1047_, lean_object* v_as_1048_, size_t v_i_1049_, size_t v_stop_1050_, lean_object* v_b_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_){
_start:
{
lean_object* v_a_1060_; uint8_t v___x_1064_; 
v___x_1064_ = lean_usize_dec_eq(v_i_1049_, v_stop_1050_);
if (v___x_1064_ == 0)
{
lean_object* v___x_1065_; 
v___x_1065_ = lean_array_uget_borrowed(v_as_1048_, v_i_1049_);
if (lean_obj_tag(v___x_1065_) == 0)
{
v_a_1060_ = v_b_1051_;
goto v___jp_1059_;
}
else
{
lean_object* v_val_1066_; 
v_val_1066_ = lean_ctor_get(v___x_1065_, 0);
if (lean_obj_tag(v_val_1066_) == 0)
{
uint8_t v_kind_1067_; 
v_kind_1067_ = lean_ctor_get_uint8(v_val_1066_, sizeof(void*)*4 + 1);
if (v_kind_1067_ == 2)
{
lean_object* v_fvarId_1068_; lean_object* v_userName_1069_; lean_object* v_type_1070_; lean_object* v___x_1071_; 
v_fvarId_1068_ = lean_ctor_get(v_val_1066_, 1);
v_userName_1069_ = lean_ctor_get(v_val_1066_, 2);
v_type_1070_ = lean_ctor_get(v_val_1066_, 3);
lean_inc_ref(v_type_1070_);
v___x_1071_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_type_1070_, v___y_1055_);
if (lean_obj_tag(v___x_1071_) == 0)
{
lean_object* v_a_1072_; lean_object* v___x_1073_; 
v_a_1072_ = lean_ctor_get(v___x_1071_, 0);
lean_inc(v_a_1072_);
lean_dec_ref_known(v___x_1071_, 1);
v___x_1073_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg(v_auxDeclToFullName_1047_, v_fvarId_1068_);
if (lean_obj_tag(v___x_1073_) == 1)
{
lean_object* v_val_1074_; lean_object* v___x_1075_; 
v_val_1074_ = lean_ctor_get(v___x_1073_, 0);
lean_inc(v_val_1074_);
lean_dec_ref_known(v___x_1073_, 1);
lean_inc(v_userName_1069_);
lean_inc(v_fvarId_1068_);
v___x_1075_ = l_Lean_LocalContext_mkAuxDecl(v_b_1051_, v_fvarId_1068_, v_userName_1069_, v_a_1072_, v_val_1074_);
v_a_1060_ = v___x_1075_;
goto v___jp_1059_;
}
else
{
lean_object* v___x_1076_; lean_object* v___x_1077_; lean_object* v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; uint8_t v___x_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
lean_dec(v___x_1073_);
lean_dec(v_a_1072_);
lean_dec_ref(v_b_1051_);
v___x_1076_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__0));
v___x_1077_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__1));
v___x_1078_ = lean_unsigned_to_nat(635u);
v___x_1079_ = lean_unsigned_to_nat(12u);
v___x_1080_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__2));
v___x_1081_ = 1;
lean_inc(v_userName_1069_);
v___x_1082_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_userName_1069_, v___x_1081_);
v___x_1083_ = lean_string_append(v___x_1080_, v___x_1082_);
lean_dec_ref(v___x_1082_);
v___x_1084_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___closed__3));
v___x_1085_ = lean_string_append(v___x_1083_, v___x_1084_);
v___x_1086_ = l_mkPanicMessageWithDecl(v___x_1076_, v___x_1077_, v___x_1078_, v___x_1079_, v___x_1085_);
lean_dec_ref(v___x_1085_);
v___x_1087_ = lp_mathlib_panic___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__4(v___x_1086_, v___y_1052_, v___y_1053_, v___y_1054_, v___y_1055_, v___y_1056_, v___y_1057_);
if (lean_obj_tag(v___x_1087_) == 0)
{
lean_object* v_a_1088_; 
v_a_1088_ = lean_ctor_get(v___x_1087_, 0);
lean_inc(v_a_1088_);
lean_dec_ref_known(v___x_1087_, 1);
v_a_1060_ = v_a_1088_;
goto v___jp_1059_;
}
else
{
return v___x_1087_;
}
}
}
else
{
lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1096_; 
lean_dec_ref(v_b_1051_);
v_a_1089_ = lean_ctor_get(v___x_1071_, 0);
v_isSharedCheck_1096_ = !lean_is_exclusive(v___x_1071_);
if (v_isSharedCheck_1096_ == 0)
{
v___x_1091_ = v___x_1071_;
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_dec(v___x_1071_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1096_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
lean_object* v___x_1094_; 
if (v_isShared_1092_ == 0)
{
v___x_1094_ = v___x_1091_;
goto v_reusejp_1093_;
}
else
{
lean_object* v_reuseFailAlloc_1095_; 
v_reuseFailAlloc_1095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1095_, 0, v_a_1089_);
v___x_1094_ = v_reuseFailAlloc_1095_;
goto v_reusejp_1093_;
}
v_reusejp_1093_:
{
return v___x_1094_;
}
}
}
}
else
{
lean_object* v_fvarId_1097_; lean_object* v_userName_1098_; lean_object* v_type_1099_; uint8_t v_bi_1100_; lean_object* v___x_1101_; 
v_fvarId_1097_ = lean_ctor_get(v_val_1066_, 1);
v_userName_1098_ = lean_ctor_get(v_val_1066_, 2);
v_type_1099_ = lean_ctor_get(v_val_1066_, 3);
v_bi_1100_ = lean_ctor_get_uint8(v_val_1066_, sizeof(void*)*4);
lean_inc_ref(v_type_1099_);
v___x_1101_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_type_1099_, v___y_1055_);
if (lean_obj_tag(v___x_1101_) == 0)
{
lean_object* v_a_1102_; lean_object* v___x_1103_; 
v_a_1102_ = lean_ctor_get(v___x_1101_, 0);
lean_inc(v_a_1102_);
lean_dec_ref_known(v___x_1101_, 1);
lean_inc(v_userName_1098_);
lean_inc(v_fvarId_1097_);
v___x_1103_ = l_Lean_LocalContext_mkLocalDecl(v_b_1051_, v_fvarId_1097_, v_userName_1098_, v_a_1102_, v_bi_1100_, v_kind_1067_);
v_a_1060_ = v___x_1103_;
goto v___jp_1059_;
}
else
{
lean_object* v_a_1104_; lean_object* v___x_1106_; uint8_t v_isShared_1107_; uint8_t v_isSharedCheck_1111_; 
lean_dec_ref(v_b_1051_);
v_a_1104_ = lean_ctor_get(v___x_1101_, 0);
v_isSharedCheck_1111_ = !lean_is_exclusive(v___x_1101_);
if (v_isSharedCheck_1111_ == 0)
{
v___x_1106_ = v___x_1101_;
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
else
{
lean_inc(v_a_1104_);
lean_dec(v___x_1101_);
v___x_1106_ = lean_box(0);
v_isShared_1107_ = v_isSharedCheck_1111_;
goto v_resetjp_1105_;
}
v_resetjp_1105_:
{
lean_object* v___x_1109_; 
if (v_isShared_1107_ == 0)
{
v___x_1109_ = v___x_1106_;
goto v_reusejp_1108_;
}
else
{
lean_object* v_reuseFailAlloc_1110_; 
v_reuseFailAlloc_1110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1110_, 0, v_a_1104_);
v___x_1109_ = v_reuseFailAlloc_1110_;
goto v_reusejp_1108_;
}
v_reusejp_1108_:
{
return v___x_1109_;
}
}
}
}
}
else
{
lean_object* v_fvarId_1112_; lean_object* v_userName_1113_; lean_object* v_type_1114_; lean_object* v_value_1115_; uint8_t v_nondep_1116_; uint8_t v_kind_1117_; lean_object* v___x_1118_; 
v_fvarId_1112_ = lean_ctor_get(v_val_1066_, 1);
v_userName_1113_ = lean_ctor_get(v_val_1066_, 2);
v_type_1114_ = lean_ctor_get(v_val_1066_, 3);
v_value_1115_ = lean_ctor_get(v_val_1066_, 4);
v_nondep_1116_ = lean_ctor_get_uint8(v_val_1066_, sizeof(void*)*5);
v_kind_1117_ = lean_ctor_get_uint8(v_val_1066_, sizeof(void*)*5 + 1);
lean_inc_ref(v_type_1114_);
v___x_1118_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_type_1114_, v___y_1055_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_a_1119_; lean_object* v___x_1120_; 
v_a_1119_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_a_1119_);
lean_dec_ref_known(v___x_1118_, 1);
lean_inc_ref(v_value_1115_);
v___x_1120_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_value_1115_, v___y_1055_);
if (lean_obj_tag(v___x_1120_) == 0)
{
lean_object* v_a_1121_; lean_object* v___x_1122_; 
v_a_1121_ = lean_ctor_get(v___x_1120_, 0);
lean_inc(v_a_1121_);
lean_dec_ref_known(v___x_1120_, 1);
lean_inc(v_userName_1113_);
lean_inc(v_fvarId_1112_);
v___x_1122_ = l_Lean_LocalContext_mkLetDecl(v_b_1051_, v_fvarId_1112_, v_userName_1113_, v_a_1119_, v_a_1121_, v_nondep_1116_, v_kind_1117_);
v_a_1060_ = v___x_1122_;
goto v___jp_1059_;
}
else
{
lean_object* v_a_1123_; lean_object* v___x_1125_; uint8_t v_isShared_1126_; uint8_t v_isSharedCheck_1130_; 
lean_dec(v_a_1119_);
lean_dec_ref(v_b_1051_);
v_a_1123_ = lean_ctor_get(v___x_1120_, 0);
v_isSharedCheck_1130_ = !lean_is_exclusive(v___x_1120_);
if (v_isSharedCheck_1130_ == 0)
{
v___x_1125_ = v___x_1120_;
v_isShared_1126_ = v_isSharedCheck_1130_;
goto v_resetjp_1124_;
}
else
{
lean_inc(v_a_1123_);
lean_dec(v___x_1120_);
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
else
{
lean_object* v_a_1131_; lean_object* v___x_1133_; uint8_t v_isShared_1134_; uint8_t v_isSharedCheck_1138_; 
lean_dec_ref(v_b_1051_);
v_a_1131_ = lean_ctor_get(v___x_1118_, 0);
v_isSharedCheck_1138_ = !lean_is_exclusive(v___x_1118_);
if (v_isSharedCheck_1138_ == 0)
{
v___x_1133_ = v___x_1118_;
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
else
{
lean_inc(v_a_1131_);
lean_dec(v___x_1118_);
v___x_1133_ = lean_box(0);
v_isShared_1134_ = v_isSharedCheck_1138_;
goto v_resetjp_1132_;
}
v_resetjp_1132_:
{
lean_object* v___x_1136_; 
if (v_isShared_1134_ == 0)
{
v___x_1136_ = v___x_1133_;
goto v_reusejp_1135_;
}
else
{
lean_object* v_reuseFailAlloc_1137_; 
v_reuseFailAlloc_1137_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1137_, 0, v_a_1131_);
v___x_1136_ = v_reuseFailAlloc_1137_;
goto v_reusejp_1135_;
}
v_reusejp_1135_:
{
return v___x_1136_;
}
}
}
}
}
}
else
{
lean_object* v___x_1139_; 
v___x_1139_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1139_, 0, v_b_1051_);
return v___x_1139_;
}
v___jp_1059_:
{
size_t v___x_1061_; size_t v___x_1062_; 
v___x_1061_ = ((size_t)1ULL);
v___x_1062_ = lean_usize_add(v_i_1049_, v___x_1061_);
v_i_1049_ = v___x_1062_;
v_b_1051_ = v_a_1060_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12___boxed(lean_object* v_auxDeclToFullName_1140_, lean_object* v_as_1141_, lean_object* v_i_1142_, lean_object* v_stop_1143_, lean_object* v_b_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_, lean_object* v___y_1148_, lean_object* v___y_1149_, lean_object* v___y_1150_, lean_object* v___y_1151_){
_start:
{
size_t v_i_boxed_1152_; size_t v_stop_boxed_1153_; lean_object* v_res_1154_; 
v_i_boxed_1152_ = lean_unbox_usize(v_i_1142_);
lean_dec(v_i_1142_);
v_stop_boxed_1153_ = lean_unbox_usize(v_stop_1143_);
lean_dec(v_stop_1143_);
v_res_1154_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1140_, v_as_1141_, v_i_boxed_1152_, v_stop_boxed_1153_, v_b_1144_, v___y_1145_, v___y_1146_, v___y_1147_, v___y_1148_, v___y_1149_, v___y_1150_);
lean_dec(v___y_1150_);
lean_dec_ref(v___y_1149_);
lean_dec(v___y_1148_);
lean_dec_ref(v___y_1147_);
lean_dec(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec_ref(v_as_1141_);
lean_dec(v_auxDeclToFullName_1140_);
return v_res_1154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13(lean_object* v_auxDeclToFullName_1155_, lean_object* v_x_1156_, lean_object* v_x_1157_, lean_object* v___y_1158_, lean_object* v___y_1159_, lean_object* v___y_1160_, lean_object* v___y_1161_, lean_object* v___y_1162_, lean_object* v___y_1163_){
_start:
{
if (lean_obj_tag(v_x_1156_) == 0)
{
lean_object* v_cs_1165_; lean_object* v___x_1167_; uint8_t v_isShared_1168_; uint8_t v_isSharedCheck_1185_; 
v_cs_1165_ = lean_ctor_get(v_x_1156_, 0);
v_isSharedCheck_1185_ = !lean_is_exclusive(v_x_1156_);
if (v_isSharedCheck_1185_ == 0)
{
v___x_1167_ = v_x_1156_;
v_isShared_1168_ = v_isSharedCheck_1185_;
goto v_resetjp_1166_;
}
else
{
lean_inc(v_cs_1165_);
lean_dec(v_x_1156_);
v___x_1167_ = lean_box(0);
v_isShared_1168_ = v_isSharedCheck_1185_;
goto v_resetjp_1166_;
}
v_resetjp_1166_:
{
lean_object* v___x_1169_; lean_object* v___x_1170_; uint8_t v___x_1171_; 
v___x_1169_ = lean_unsigned_to_nat(0u);
v___x_1170_ = lean_array_get_size(v_cs_1165_);
v___x_1171_ = lean_nat_dec_lt(v___x_1169_, v___x_1170_);
if (v___x_1171_ == 0)
{
lean_object* v___x_1173_; 
lean_dec_ref(v_cs_1165_);
if (v_isShared_1168_ == 0)
{
lean_ctor_set(v___x_1167_, 0, v_x_1157_);
v___x_1173_ = v___x_1167_;
goto v_reusejp_1172_;
}
else
{
lean_object* v_reuseFailAlloc_1174_; 
v_reuseFailAlloc_1174_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1174_, 0, v_x_1157_);
v___x_1173_ = v_reuseFailAlloc_1174_;
goto v_reusejp_1172_;
}
v_reusejp_1172_:
{
return v___x_1173_;
}
}
else
{
uint8_t v___x_1175_; 
v___x_1175_ = lean_nat_dec_le(v___x_1170_, v___x_1170_);
if (v___x_1175_ == 0)
{
if (v___x_1171_ == 0)
{
lean_object* v___x_1177_; 
lean_dec_ref(v_cs_1165_);
if (v_isShared_1168_ == 0)
{
lean_ctor_set(v___x_1167_, 0, v_x_1157_);
v___x_1177_ = v___x_1167_;
goto v_reusejp_1176_;
}
else
{
lean_object* v_reuseFailAlloc_1178_; 
v_reuseFailAlloc_1178_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1178_, 0, v_x_1157_);
v___x_1177_ = v_reuseFailAlloc_1178_;
goto v_reusejp_1176_;
}
v_reusejp_1176_:
{
return v___x_1177_;
}
}
else
{
size_t v___x_1179_; size_t v___x_1180_; lean_object* v___x_1181_; 
lean_del_object(v___x_1167_);
v___x_1179_ = ((size_t)0ULL);
v___x_1180_ = lean_usize_of_nat(v___x_1170_);
v___x_1181_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(v_auxDeclToFullName_1155_, v_cs_1165_, v___x_1179_, v___x_1180_, v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec_ref(v_cs_1165_);
return v___x_1181_;
}
}
else
{
size_t v___x_1182_; size_t v___x_1183_; lean_object* v___x_1184_; 
lean_del_object(v___x_1167_);
v___x_1182_ = ((size_t)0ULL);
v___x_1183_ = lean_usize_of_nat(v___x_1170_);
v___x_1184_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(v_auxDeclToFullName_1155_, v_cs_1165_, v___x_1182_, v___x_1183_, v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec_ref(v_cs_1165_);
return v___x_1184_;
}
}
}
}
else
{
lean_object* v_vs_1186_; lean_object* v___x_1188_; uint8_t v_isShared_1189_; uint8_t v_isSharedCheck_1206_; 
v_vs_1186_ = lean_ctor_get(v_x_1156_, 0);
v_isSharedCheck_1206_ = !lean_is_exclusive(v_x_1156_);
if (v_isSharedCheck_1206_ == 0)
{
v___x_1188_ = v_x_1156_;
v_isShared_1189_ = v_isSharedCheck_1206_;
goto v_resetjp_1187_;
}
else
{
lean_inc(v_vs_1186_);
lean_dec(v_x_1156_);
v___x_1188_ = lean_box(0);
v_isShared_1189_ = v_isSharedCheck_1206_;
goto v_resetjp_1187_;
}
v_resetjp_1187_:
{
lean_object* v___x_1190_; lean_object* v___x_1191_; uint8_t v___x_1192_; 
v___x_1190_ = lean_unsigned_to_nat(0u);
v___x_1191_ = lean_array_get_size(v_vs_1186_);
v___x_1192_ = lean_nat_dec_lt(v___x_1190_, v___x_1191_);
if (v___x_1192_ == 0)
{
lean_object* v___x_1194_; 
lean_dec_ref(v_vs_1186_);
if (v_isShared_1189_ == 0)
{
lean_ctor_set_tag(v___x_1188_, 0);
lean_ctor_set(v___x_1188_, 0, v_x_1157_);
v___x_1194_ = v___x_1188_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_x_1157_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
else
{
uint8_t v___x_1196_; 
v___x_1196_ = lean_nat_dec_le(v___x_1191_, v___x_1191_);
if (v___x_1196_ == 0)
{
if (v___x_1192_ == 0)
{
lean_object* v___x_1198_; 
lean_dec_ref(v_vs_1186_);
if (v_isShared_1189_ == 0)
{
lean_ctor_set_tag(v___x_1188_, 0);
lean_ctor_set(v___x_1188_, 0, v_x_1157_);
v___x_1198_ = v___x_1188_;
goto v_reusejp_1197_;
}
else
{
lean_object* v_reuseFailAlloc_1199_; 
v_reuseFailAlloc_1199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1199_, 0, v_x_1157_);
v___x_1198_ = v_reuseFailAlloc_1199_;
goto v_reusejp_1197_;
}
v_reusejp_1197_:
{
return v___x_1198_;
}
}
else
{
size_t v___x_1200_; size_t v___x_1201_; lean_object* v___x_1202_; 
lean_del_object(v___x_1188_);
v___x_1200_ = ((size_t)0ULL);
v___x_1201_ = lean_usize_of_nat(v___x_1191_);
v___x_1202_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1155_, v_vs_1186_, v___x_1200_, v___x_1201_, v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec_ref(v_vs_1186_);
return v___x_1202_;
}
}
else
{
size_t v___x_1203_; size_t v___x_1204_; lean_object* v___x_1205_; 
lean_del_object(v___x_1188_);
v___x_1203_ = ((size_t)0ULL);
v___x_1204_ = lean_usize_of_nat(v___x_1191_);
v___x_1205_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1155_, v_vs_1186_, v___x_1203_, v___x_1204_, v_x_1157_, v___y_1158_, v___y_1159_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
lean_dec_ref(v_vs_1186_);
return v___x_1205_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(lean_object* v_auxDeclToFullName_1207_, lean_object* v_as_1208_, size_t v_i_1209_, size_t v_stop_1210_, lean_object* v_b_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_, lean_object* v___y_1215_, lean_object* v___y_1216_, lean_object* v___y_1217_){
_start:
{
uint8_t v___x_1219_; 
v___x_1219_ = lean_usize_dec_eq(v_i_1209_, v_stop_1210_);
if (v___x_1219_ == 0)
{
lean_object* v___x_1220_; lean_object* v___x_1221_; 
v___x_1220_ = lean_array_uget_borrowed(v_as_1208_, v_i_1209_);
lean_inc(v___x_1220_);
v___x_1221_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13(v_auxDeclToFullName_1207_, v___x_1220_, v_b_1211_, v___y_1212_, v___y_1213_, v___y_1214_, v___y_1215_, v___y_1216_, v___y_1217_);
if (lean_obj_tag(v___x_1221_) == 0)
{
lean_object* v_a_1222_; size_t v___x_1223_; size_t v___x_1224_; 
v_a_1222_ = lean_ctor_get(v___x_1221_, 0);
lean_inc(v_a_1222_);
lean_dec_ref_known(v___x_1221_, 1);
v___x_1223_ = ((size_t)1ULL);
v___x_1224_ = lean_usize_add(v_i_1209_, v___x_1223_);
v_i_1209_ = v___x_1224_;
v_b_1211_ = v_a_1222_;
goto _start;
}
else
{
return v___x_1221_;
}
}
else
{
lean_object* v___x_1226_; 
v___x_1226_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1226_, 0, v_b_1211_);
return v___x_1226_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14___boxed(lean_object* v_auxDeclToFullName_1227_, lean_object* v_as_1228_, lean_object* v_i_1229_, lean_object* v_stop_1230_, lean_object* v_b_1231_, lean_object* v___y_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_, lean_object* v___y_1237_, lean_object* v___y_1238_){
_start:
{
size_t v_i_boxed_1239_; size_t v_stop_boxed_1240_; lean_object* v_res_1241_; 
v_i_boxed_1239_ = lean_unbox_usize(v_i_1229_);
lean_dec(v_i_1229_);
v_stop_boxed_1240_ = lean_unbox_usize(v_stop_1230_);
lean_dec(v_stop_1230_);
v_res_1241_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(v_auxDeclToFullName_1227_, v_as_1228_, v_i_boxed_1239_, v_stop_boxed_1240_, v_b_1231_, v___y_1232_, v___y_1233_, v___y_1234_, v___y_1235_, v___y_1236_, v___y_1237_);
lean_dec(v___y_1237_);
lean_dec_ref(v___y_1236_);
lean_dec(v___y_1235_);
lean_dec_ref(v___y_1234_);
lean_dec(v___y_1233_);
lean_dec_ref(v___y_1232_);
lean_dec_ref(v_as_1228_);
lean_dec(v_auxDeclToFullName_1227_);
return v_res_1241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13___boxed(lean_object* v_auxDeclToFullName_1242_, lean_object* v_x_1243_, lean_object* v_x_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_, lean_object* v___y_1249_, lean_object* v___y_1250_, lean_object* v___y_1251_){
_start:
{
lean_object* v_res_1252_; 
v_res_1252_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13(v_auxDeclToFullName_1242_, v_x_1243_, v_x_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_);
lean_dec(v___y_1250_);
lean_dec_ref(v___y_1249_);
lean_dec(v___y_1248_);
lean_dec_ref(v___y_1247_);
lean_dec(v___y_1246_);
lean_dec_ref(v___y_1245_);
lean_dec(v_auxDeclToFullName_1242_);
return v_res_1252_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0(void){
_start:
{
lean_object* v___x_1253_; 
v___x_1253_ = l_Lean_instInhabitedPersistentArrayNode_default(lean_box(0));
return v___x_1253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11(lean_object* v_auxDeclToFullName_1254_, lean_object* v_x_1255_, size_t v_x_1256_, size_t v_x_1257_, lean_object* v_x_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_, lean_object* v___y_1261_, lean_object* v___y_1262_, lean_object* v___y_1263_, lean_object* v___y_1264_){
_start:
{
if (lean_obj_tag(v_x_1255_) == 0)
{
lean_object* v_cs_1266_; lean_object* v___x_1267_; size_t v___x_1268_; lean_object* v_j_1269_; lean_object* v___x_1270_; size_t v___x_1271_; size_t v___x_1272_; size_t v___x_1273_; size_t v___x_1274_; size_t v___x_1275_; size_t v___x_1276_; lean_object* v___x_1277_; 
v_cs_1266_ = lean_ctor_get(v_x_1255_, 0);
lean_inc_ref(v_cs_1266_);
lean_dec_ref_known(v_x_1255_, 1);
v___x_1267_ = lean_obj_once(&lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0, &lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0_once, _init_lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___closed__0);
v___x_1268_ = lean_usize_shift_right(v_x_1256_, v_x_1257_);
v_j_1269_ = lean_usize_to_nat(v___x_1268_);
v___x_1270_ = lean_array_get_borrowed(v___x_1267_, v_cs_1266_, v_j_1269_);
v___x_1271_ = ((size_t)1ULL);
v___x_1272_ = lean_usize_shift_left(v___x_1271_, v_x_1257_);
v___x_1273_ = lean_usize_sub(v___x_1272_, v___x_1271_);
v___x_1274_ = lean_usize_land(v_x_1256_, v___x_1273_);
v___x_1275_ = ((size_t)5ULL);
v___x_1276_ = lean_usize_sub(v_x_1257_, v___x_1275_);
lean_inc(v___x_1270_);
v___x_1277_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11(v_auxDeclToFullName_1254_, v___x_1270_, v___x_1274_, v___x_1276_, v_x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_);
if (lean_obj_tag(v___x_1277_) == 0)
{
lean_object* v_a_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; uint8_t v___x_1282_; 
v_a_1278_ = lean_ctor_get(v___x_1277_, 0);
lean_inc(v_a_1278_);
v___x_1279_ = lean_unsigned_to_nat(1u);
v___x_1280_ = lean_nat_add(v_j_1269_, v___x_1279_);
lean_dec(v_j_1269_);
v___x_1281_ = lean_array_get_size(v_cs_1266_);
v___x_1282_ = lean_nat_dec_lt(v___x_1280_, v___x_1281_);
if (v___x_1282_ == 0)
{
lean_dec(v___x_1280_);
lean_dec(v_a_1278_);
lean_dec_ref(v_cs_1266_);
return v___x_1277_;
}
else
{
uint8_t v___x_1283_; 
v___x_1283_ = lean_nat_dec_le(v___x_1281_, v___x_1281_);
if (v___x_1283_ == 0)
{
if (v___x_1282_ == 0)
{
lean_dec(v___x_1280_);
lean_dec(v_a_1278_);
lean_dec_ref(v_cs_1266_);
return v___x_1277_;
}
else
{
size_t v___x_1284_; size_t v___x_1285_; lean_object* v___x_1286_; 
lean_dec_ref_known(v___x_1277_, 1);
v___x_1284_ = lean_usize_of_nat(v___x_1280_);
lean_dec(v___x_1280_);
v___x_1285_ = lean_usize_of_nat(v___x_1281_);
v___x_1286_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(v_auxDeclToFullName_1254_, v_cs_1266_, v___x_1284_, v___x_1285_, v_a_1278_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_);
lean_dec_ref(v_cs_1266_);
return v___x_1286_;
}
}
else
{
size_t v___x_1287_; size_t v___x_1288_; lean_object* v___x_1289_; 
lean_dec_ref_known(v___x_1277_, 1);
v___x_1287_ = lean_usize_of_nat(v___x_1280_);
lean_dec(v___x_1280_);
v___x_1288_ = lean_usize_of_nat(v___x_1281_);
v___x_1289_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11_spec__14(v_auxDeclToFullName_1254_, v_cs_1266_, v___x_1287_, v___x_1288_, v_a_1278_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_);
lean_dec_ref(v_cs_1266_);
return v___x_1289_;
}
}
}
else
{
lean_dec(v_j_1269_);
lean_dec_ref(v_cs_1266_);
return v___x_1277_;
}
}
else
{
lean_object* v_vs_1290_; lean_object* v___x_1292_; uint8_t v_isShared_1293_; uint8_t v_isSharedCheck_1310_; 
v_vs_1290_ = lean_ctor_get(v_x_1255_, 0);
v_isSharedCheck_1310_ = !lean_is_exclusive(v_x_1255_);
if (v_isSharedCheck_1310_ == 0)
{
v___x_1292_ = v_x_1255_;
v_isShared_1293_ = v_isSharedCheck_1310_;
goto v_resetjp_1291_;
}
else
{
lean_inc(v_vs_1290_);
lean_dec(v_x_1255_);
v___x_1292_ = lean_box(0);
v_isShared_1293_ = v_isSharedCheck_1310_;
goto v_resetjp_1291_;
}
v_resetjp_1291_:
{
lean_object* v___x_1294_; lean_object* v___x_1295_; uint8_t v___x_1296_; 
v___x_1294_ = lean_usize_to_nat(v_x_1256_);
v___x_1295_ = lean_array_get_size(v_vs_1290_);
v___x_1296_ = lean_nat_dec_lt(v___x_1294_, v___x_1295_);
if (v___x_1296_ == 0)
{
lean_object* v___x_1298_; 
lean_dec(v___x_1294_);
lean_dec_ref(v_vs_1290_);
if (v_isShared_1293_ == 0)
{
lean_ctor_set_tag(v___x_1292_, 0);
lean_ctor_set(v___x_1292_, 0, v_x_1258_);
v___x_1298_ = v___x_1292_;
goto v_reusejp_1297_;
}
else
{
lean_object* v_reuseFailAlloc_1299_; 
v_reuseFailAlloc_1299_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1299_, 0, v_x_1258_);
v___x_1298_ = v_reuseFailAlloc_1299_;
goto v_reusejp_1297_;
}
v_reusejp_1297_:
{
return v___x_1298_;
}
}
else
{
uint8_t v___x_1300_; 
v___x_1300_ = lean_nat_dec_le(v___x_1295_, v___x_1295_);
if (v___x_1300_ == 0)
{
if (v___x_1296_ == 0)
{
lean_object* v___x_1302_; 
lean_dec(v___x_1294_);
lean_dec_ref(v_vs_1290_);
if (v_isShared_1293_ == 0)
{
lean_ctor_set_tag(v___x_1292_, 0);
lean_ctor_set(v___x_1292_, 0, v_x_1258_);
v___x_1302_ = v___x_1292_;
goto v_reusejp_1301_;
}
else
{
lean_object* v_reuseFailAlloc_1303_; 
v_reuseFailAlloc_1303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1303_, 0, v_x_1258_);
v___x_1302_ = v_reuseFailAlloc_1303_;
goto v_reusejp_1301_;
}
v_reusejp_1301_:
{
return v___x_1302_;
}
}
else
{
size_t v___x_1304_; size_t v___x_1305_; lean_object* v___x_1306_; 
lean_del_object(v___x_1292_);
v___x_1304_ = lean_usize_of_nat(v___x_1294_);
lean_dec(v___x_1294_);
v___x_1305_ = lean_usize_of_nat(v___x_1295_);
v___x_1306_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1254_, v_vs_1290_, v___x_1304_, v___x_1305_, v_x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_);
lean_dec_ref(v_vs_1290_);
return v___x_1306_;
}
}
else
{
size_t v___x_1307_; size_t v___x_1308_; lean_object* v___x_1309_; 
lean_del_object(v___x_1292_);
v___x_1307_ = lean_usize_of_nat(v___x_1294_);
lean_dec(v___x_1294_);
v___x_1308_ = lean_usize_of_nat(v___x_1295_);
v___x_1309_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1254_, v_vs_1290_, v___x_1307_, v___x_1308_, v_x_1258_, v___y_1259_, v___y_1260_, v___y_1261_, v___y_1262_, v___y_1263_, v___y_1264_);
lean_dec_ref(v_vs_1290_);
return v___x_1309_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11___boxed(lean_object* v_auxDeclToFullName_1311_, lean_object* v_x_1312_, lean_object* v_x_1313_, lean_object* v_x_1314_, lean_object* v_x_1315_, lean_object* v___y_1316_, lean_object* v___y_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_, lean_object* v___y_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_){
_start:
{
size_t v_x_27323__boxed_1323_; size_t v_x_27324__boxed_1324_; lean_object* v_res_1325_; 
v_x_27323__boxed_1323_ = lean_unbox_usize(v_x_1313_);
lean_dec(v_x_1313_);
v_x_27324__boxed_1324_ = lean_unbox_usize(v_x_1314_);
lean_dec(v_x_1314_);
v_res_1325_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11(v_auxDeclToFullName_1311_, v_x_1312_, v_x_27323__boxed_1323_, v_x_27324__boxed_1324_, v_x_1315_, v___y_1316_, v___y_1317_, v___y_1318_, v___y_1319_, v___y_1320_, v___y_1321_);
lean_dec(v___y_1321_);
lean_dec_ref(v___y_1320_);
lean_dec(v___y_1319_);
lean_dec_ref(v___y_1318_);
lean_dec(v___y_1317_);
lean_dec_ref(v___y_1316_);
lean_dec(v_auxDeclToFullName_1311_);
return v_res_1325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8(lean_object* v_auxDeclToFullName_1326_, lean_object* v_t_1327_, lean_object* v_init_1328_, lean_object* v_start_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_){
_start:
{
lean_object* v___x_1337_; uint8_t v___x_1338_; 
v___x_1337_ = lean_unsigned_to_nat(0u);
v___x_1338_ = lean_nat_dec_eq(v_start_1329_, v___x_1337_);
if (v___x_1338_ == 0)
{
lean_object* v_root_1339_; lean_object* v_tail_1340_; size_t v_shift_1341_; lean_object* v_tailOff_1342_; uint8_t v___x_1343_; 
v_root_1339_ = lean_ctor_get(v_t_1327_, 0);
lean_inc_ref(v_root_1339_);
v_tail_1340_ = lean_ctor_get(v_t_1327_, 1);
lean_inc_ref(v_tail_1340_);
v_shift_1341_ = lean_ctor_get_usize(v_t_1327_, 4);
v_tailOff_1342_ = lean_ctor_get(v_t_1327_, 3);
lean_inc(v_tailOff_1342_);
lean_dec_ref(v_t_1327_);
v___x_1343_ = lean_nat_dec_le(v_tailOff_1342_, v_start_1329_);
if (v___x_1343_ == 0)
{
size_t v___x_1344_; lean_object* v___x_1345_; 
lean_dec(v_tailOff_1342_);
v___x_1344_ = lean_usize_of_nat(v_start_1329_);
v___x_1345_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlFromMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__11(v_auxDeclToFullName_1326_, v_root_1339_, v___x_1344_, v_shift_1341_, v_init_1328_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
if (lean_obj_tag(v___x_1345_) == 0)
{
lean_object* v_a_1346_; lean_object* v___x_1347_; uint8_t v___x_1348_; 
v_a_1346_ = lean_ctor_get(v___x_1345_, 0);
lean_inc(v_a_1346_);
v___x_1347_ = lean_array_get_size(v_tail_1340_);
v___x_1348_ = lean_nat_dec_lt(v___x_1337_, v___x_1347_);
if (v___x_1348_ == 0)
{
lean_dec(v_a_1346_);
lean_dec_ref(v_tail_1340_);
return v___x_1345_;
}
else
{
uint8_t v___x_1349_; 
v___x_1349_ = lean_nat_dec_le(v___x_1347_, v___x_1347_);
if (v___x_1349_ == 0)
{
if (v___x_1348_ == 0)
{
lean_dec(v_a_1346_);
lean_dec_ref(v_tail_1340_);
return v___x_1345_;
}
else
{
size_t v___x_1350_; size_t v___x_1351_; lean_object* v___x_1352_; 
lean_dec_ref_known(v___x_1345_, 1);
v___x_1350_ = ((size_t)0ULL);
v___x_1351_ = lean_usize_of_nat(v___x_1347_);
v___x_1352_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1340_, v___x_1350_, v___x_1351_, v_a_1346_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1340_);
return v___x_1352_;
}
}
else
{
size_t v___x_1353_; size_t v___x_1354_; lean_object* v___x_1355_; 
lean_dec_ref_known(v___x_1345_, 1);
v___x_1353_ = ((size_t)0ULL);
v___x_1354_ = lean_usize_of_nat(v___x_1347_);
v___x_1355_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1340_, v___x_1353_, v___x_1354_, v_a_1346_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1340_);
return v___x_1355_;
}
}
}
else
{
lean_dec_ref(v_tail_1340_);
return v___x_1345_;
}
}
else
{
lean_object* v___x_1356_; lean_object* v___x_1357_; uint8_t v___x_1358_; 
lean_dec_ref(v_root_1339_);
v___x_1356_ = lean_nat_sub(v_start_1329_, v_tailOff_1342_);
lean_dec(v_tailOff_1342_);
v___x_1357_ = lean_array_get_size(v_tail_1340_);
v___x_1358_ = lean_nat_dec_lt(v___x_1356_, v___x_1357_);
if (v___x_1358_ == 0)
{
lean_object* v___x_1359_; 
lean_dec(v___x_1356_);
lean_dec_ref(v_tail_1340_);
v___x_1359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1359_, 0, v_init_1328_);
return v___x_1359_;
}
else
{
uint8_t v___x_1360_; 
v___x_1360_ = lean_nat_dec_le(v___x_1357_, v___x_1357_);
if (v___x_1360_ == 0)
{
if (v___x_1358_ == 0)
{
lean_object* v___x_1361_; 
lean_dec(v___x_1356_);
lean_dec_ref(v_tail_1340_);
v___x_1361_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1361_, 0, v_init_1328_);
return v___x_1361_;
}
else
{
size_t v___x_1362_; size_t v___x_1363_; lean_object* v___x_1364_; 
v___x_1362_ = lean_usize_of_nat(v___x_1356_);
lean_dec(v___x_1356_);
v___x_1363_ = lean_usize_of_nat(v___x_1357_);
v___x_1364_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1340_, v___x_1362_, v___x_1363_, v_init_1328_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1340_);
return v___x_1364_;
}
}
else
{
size_t v___x_1365_; size_t v___x_1366_; lean_object* v___x_1367_; 
v___x_1365_ = lean_usize_of_nat(v___x_1356_);
lean_dec(v___x_1356_);
v___x_1366_ = lean_usize_of_nat(v___x_1357_);
v___x_1367_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1340_, v___x_1365_, v___x_1366_, v_init_1328_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1340_);
return v___x_1367_;
}
}
}
}
else
{
lean_object* v_root_1368_; lean_object* v_tail_1369_; lean_object* v___x_1370_; 
v_root_1368_ = lean_ctor_get(v_t_1327_, 0);
lean_inc_ref(v_root_1368_);
v_tail_1369_ = lean_ctor_get(v_t_1327_, 1);
lean_inc_ref(v_tail_1369_);
lean_dec_ref(v_t_1327_);
v___x_1370_ = lp_mathlib___private_Lean_Data_PersistentArray_0__Lean_PersistentArray_foldlMAux___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__13(v_auxDeclToFullName_1326_, v_root_1368_, v_init_1328_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
if (lean_obj_tag(v___x_1370_) == 0)
{
lean_object* v_a_1371_; lean_object* v___x_1372_; uint8_t v___x_1373_; 
v_a_1371_ = lean_ctor_get(v___x_1370_, 0);
lean_inc(v_a_1371_);
v___x_1372_ = lean_array_get_size(v_tail_1369_);
v___x_1373_ = lean_nat_dec_lt(v___x_1337_, v___x_1372_);
if (v___x_1373_ == 0)
{
lean_dec(v_a_1371_);
lean_dec_ref(v_tail_1369_);
return v___x_1370_;
}
else
{
uint8_t v___x_1374_; 
v___x_1374_ = lean_nat_dec_le(v___x_1372_, v___x_1372_);
if (v___x_1374_ == 0)
{
if (v___x_1373_ == 0)
{
lean_dec(v_a_1371_);
lean_dec_ref(v_tail_1369_);
return v___x_1370_;
}
else
{
size_t v___x_1375_; size_t v___x_1376_; lean_object* v___x_1377_; 
lean_dec_ref_known(v___x_1370_, 1);
v___x_1375_ = ((size_t)0ULL);
v___x_1376_ = lean_usize_of_nat(v___x_1372_);
v___x_1377_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1369_, v___x_1375_, v___x_1376_, v_a_1371_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1369_);
return v___x_1377_;
}
}
else
{
size_t v___x_1378_; size_t v___x_1379_; lean_object* v___x_1380_; 
lean_dec_ref_known(v___x_1370_, 1);
v___x_1378_ = ((size_t)0ULL);
v___x_1379_ = lean_usize_of_nat(v___x_1372_);
v___x_1380_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8_spec__12(v_auxDeclToFullName_1326_, v_tail_1369_, v___x_1378_, v___x_1379_, v_a_1371_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_, v___y_1335_);
lean_dec_ref(v_tail_1369_);
return v___x_1380_;
}
}
}
else
{
lean_dec_ref(v_tail_1369_);
return v___x_1370_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8___boxed(lean_object* v_auxDeclToFullName_1381_, lean_object* v_t_1382_, lean_object* v_init_1383_, lean_object* v_start_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8(v_auxDeclToFullName_1381_, v_t_1382_, v_init_1383_, v_start_1384_, v___y_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_);
lean_dec(v___y_1390_);
lean_dec_ref(v___y_1389_);
lean_dec(v___y_1388_);
lean_dec_ref(v___y_1387_);
lean_dec(v___y_1386_);
lean_dec_ref(v___y_1385_);
lean_dec(v_start_1384_);
lean_dec(v_auxDeclToFullName_1381_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5(lean_object* v_auxDeclToFullName_1393_, lean_object* v_lctx_1394_, lean_object* v_init_1395_, lean_object* v_start_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
lean_object* v_decls_1404_; lean_object* v___x_1405_; 
v_decls_1404_ = lean_ctor_get(v_lctx_1394_, 1);
lean_inc_ref(v_decls_1404_);
lean_dec_ref(v_lctx_1394_);
v___x_1405_ = lp_mathlib_Lean_PersistentArray_foldlM___at___00Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5_spec__8(v_auxDeclToFullName_1393_, v_decls_1404_, v_init_1395_, v_start_1396_, v___y_1397_, v___y_1398_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_);
return v___x_1405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5___boxed(lean_object* v_auxDeclToFullName_1406_, lean_object* v_lctx_1407_, lean_object* v_init_1408_, lean_object* v_start_1409_, lean_object* v___y_1410_, lean_object* v___y_1411_, lean_object* v___y_1412_, lean_object* v___y_1413_, lean_object* v___y_1414_, lean_object* v___y_1415_, lean_object* v___y_1416_){
_start:
{
lean_object* v_res_1417_; 
v_res_1417_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5(v_auxDeclToFullName_1406_, v_lctx_1407_, v_init_1408_, v_start_1409_, v___y_1410_, v___y_1411_, v___y_1412_, v___y_1413_, v___y_1414_, v___y_1415_);
lean_dec(v___y_1415_);
lean_dec_ref(v___y_1414_);
lean_dec(v___y_1413_);
lean_dec_ref(v___y_1412_);
lean_dec(v___y_1411_);
lean_dec_ref(v___y_1410_);
lean_dec(v_start_1409_);
lean_dec(v_auxDeclToFullName_1406_);
return v_res_1417_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0(void){
_start:
{
lean_object* v___x_1418_; 
v___x_1418_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1418_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_1419_; lean_object* v___x_1420_; 
v___x_1419_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__0);
v___x_1420_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1420_, 0, v___x_1419_);
return v___x_1420_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; 
v___x_1421_ = lean_unsigned_to_nat(32u);
v___x_1422_ = lean_mk_empty_array_with_capacity(v___x_1421_);
v___x_1423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1423_, 0, v___x_1422_);
return v___x_1423_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3(void){
_start:
{
size_t v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; 
v___x_1424_ = ((size_t)5ULL);
v___x_1425_ = lean_unsigned_to_nat(0u);
v___x_1426_ = lean_unsigned_to_nat(32u);
v___x_1427_ = lean_mk_empty_array_with_capacity(v___x_1426_);
v___x_1428_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__2);
v___x_1429_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1429_, 0, v___x_1428_);
lean_ctor_set(v___x_1429_, 1, v___x_1427_);
lean_ctor_set(v___x_1429_, 2, v___x_1425_);
lean_ctor_set(v___x_1429_, 3, v___x_1425_);
lean_ctor_set_usize(v___x_1429_, 4, v___x_1424_);
return v___x_1429_;
}
}
static lean_object* _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4(void){
_start:
{
lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; lean_object* v___x_1433_; 
v___x_1430_ = lean_box(1);
v___x_1431_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__3);
v___x_1432_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__1);
v___x_1433_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1433_, 0, v___x_1432_);
lean_ctor_set(v___x_1433_, 1, v___x_1431_);
lean_ctor_set(v___x_1433_, 2, v___x_1430_);
return v___x_1433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0(lean_object* v_lctx_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_, lean_object* v___y_1437_, lean_object* v___y_1438_, lean_object* v___y_1439_, lean_object* v___y_1440_){
_start:
{
lean_object* v_auxDeclToFullName_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; 
v_auxDeclToFullName_1442_ = lean_ctor_get(v_lctx_1434_, 2);
lean_inc(v_auxDeclToFullName_1442_);
v___x_1443_ = lean_unsigned_to_nat(0u);
v___x_1444_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4);
v___x_1445_ = lp_mathlib_Lean_LocalContext_foldlM___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__5(v_auxDeclToFullName_1442_, v_lctx_1434_, v___x_1444_, v___x_1443_, v___y_1435_, v___y_1436_, v___y_1437_, v___y_1438_, v___y_1439_, v___y_1440_);
lean_dec(v_auxDeclToFullName_1442_);
return v___x_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___boxed(lean_object* v_lctx_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_, lean_object* v___y_1451_, lean_object* v___y_1452_, lean_object* v___y_1453_){
_start:
{
lean_object* v_res_1454_; 
v_res_1454_ = lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0(v_lctx_1446_, v___y_1447_, v___y_1448_, v___y_1449_, v___y_1450_, v___y_1451_, v___y_1452_);
lean_dec(v___y_1452_);
lean_dec_ref(v___y_1451_);
lean_dec(v___y_1450_);
lean_dec_ref(v___y_1449_);
lean_dec(v___y_1448_);
lean_dec_ref(v___y_1447_);
return v_res_1454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16___redArg(lean_object* v_x_1455_, lean_object* v_x_1456_, lean_object* v_x_1457_, lean_object* v_x_1458_){
_start:
{
lean_object* v_ks_1459_; lean_object* v_vs_1460_; lean_object* v___x_1462_; uint8_t v_isShared_1463_; uint8_t v_isSharedCheck_1484_; 
v_ks_1459_ = lean_ctor_get(v_x_1455_, 0);
v_vs_1460_ = lean_ctor_get(v_x_1455_, 1);
v_isSharedCheck_1484_ = !lean_is_exclusive(v_x_1455_);
if (v_isSharedCheck_1484_ == 0)
{
v___x_1462_ = v_x_1455_;
v_isShared_1463_ = v_isSharedCheck_1484_;
goto v_resetjp_1461_;
}
else
{
lean_inc(v_vs_1460_);
lean_inc(v_ks_1459_);
lean_dec(v_x_1455_);
v___x_1462_ = lean_box(0);
v_isShared_1463_ = v_isSharedCheck_1484_;
goto v_resetjp_1461_;
}
v_resetjp_1461_:
{
lean_object* v___x_1464_; uint8_t v___x_1465_; 
v___x_1464_ = lean_array_get_size(v_ks_1459_);
v___x_1465_ = lean_nat_dec_lt(v_x_1456_, v___x_1464_);
if (v___x_1465_ == 0)
{
lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1469_; 
lean_dec(v_x_1456_);
v___x_1466_ = lean_array_push(v_ks_1459_, v_x_1457_);
v___x_1467_ = lean_array_push(v_vs_1460_, v_x_1458_);
if (v_isShared_1463_ == 0)
{
lean_ctor_set(v___x_1462_, 1, v___x_1467_);
lean_ctor_set(v___x_1462_, 0, v___x_1466_);
v___x_1469_ = v___x_1462_;
goto v_reusejp_1468_;
}
else
{
lean_object* v_reuseFailAlloc_1470_; 
v_reuseFailAlloc_1470_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1470_, 0, v___x_1466_);
lean_ctor_set(v_reuseFailAlloc_1470_, 1, v___x_1467_);
v___x_1469_ = v_reuseFailAlloc_1470_;
goto v_reusejp_1468_;
}
v_reusejp_1468_:
{
return v___x_1469_;
}
}
else
{
lean_object* v_k_x27_1471_; uint8_t v___x_1472_; 
v_k_x27_1471_ = lean_array_fget_borrowed(v_ks_1459_, v_x_1456_);
v___x_1472_ = l_Lean_instBEqMVarId_beq(v_x_1457_, v_k_x27_1471_);
if (v___x_1472_ == 0)
{
lean_object* v___x_1474_; 
if (v_isShared_1463_ == 0)
{
v___x_1474_ = v___x_1462_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1478_; 
v_reuseFailAlloc_1478_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1478_, 0, v_ks_1459_);
lean_ctor_set(v_reuseFailAlloc_1478_, 1, v_vs_1460_);
v___x_1474_ = v_reuseFailAlloc_1478_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
lean_object* v___x_1475_; lean_object* v___x_1476_; 
v___x_1475_ = lean_unsigned_to_nat(1u);
v___x_1476_ = lean_nat_add(v_x_1456_, v___x_1475_);
lean_dec(v_x_1456_);
v_x_1455_ = v___x_1474_;
v_x_1456_ = v___x_1476_;
goto _start;
}
}
else
{
lean_object* v___x_1479_; lean_object* v___x_1480_; lean_object* v___x_1482_; 
v___x_1479_ = lean_array_fset(v_ks_1459_, v_x_1456_, v_x_1457_);
v___x_1480_ = lean_array_fset(v_vs_1460_, v_x_1456_, v_x_1458_);
lean_dec(v_x_1456_);
if (v_isShared_1463_ == 0)
{
lean_ctor_set(v___x_1462_, 1, v___x_1480_);
lean_ctor_set(v___x_1462_, 0, v___x_1479_);
v___x_1482_ = v___x_1462_;
goto v_reusejp_1481_;
}
else
{
lean_object* v_reuseFailAlloc_1483_; 
v_reuseFailAlloc_1483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1483_, 0, v___x_1479_);
lean_ctor_set(v_reuseFailAlloc_1483_, 1, v___x_1480_);
v___x_1482_ = v_reuseFailAlloc_1483_;
goto v_reusejp_1481_;
}
v_reusejp_1481_:
{
return v___x_1482_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11___redArg(lean_object* v_n_1485_, lean_object* v_k_1486_, lean_object* v_v_1487_){
_start:
{
lean_object* v___x_1488_; lean_object* v___x_1489_; 
v___x_1488_ = lean_unsigned_to_nat(0u);
v___x_1489_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16___redArg(v_n_1485_, v___x_1488_, v_k_1486_, v_v_1487_);
return v___x_1489_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0(void){
_start:
{
lean_object* v___x_1490_; 
v___x_1490_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(lean_object* v_x_1491_, size_t v_x_1492_, size_t v_x_1493_, lean_object* v_x_1494_, lean_object* v_x_1495_){
_start:
{
if (lean_obj_tag(v_x_1491_) == 0)
{
lean_object* v_es_1496_; size_t v___x_1497_; size_t v___x_1498_; lean_object* v_j_1499_; lean_object* v___x_1500_; uint8_t v___x_1501_; 
v_es_1496_ = lean_ctor_get(v_x_1491_, 0);
v___x_1497_ = ((size_t)31ULL);
v___x_1498_ = lean_usize_land(v_x_1492_, v___x_1497_);
v_j_1499_ = lean_usize_to_nat(v___x_1498_);
v___x_1500_ = lean_array_get_size(v_es_1496_);
v___x_1501_ = lean_nat_dec_lt(v_j_1499_, v___x_1500_);
if (v___x_1501_ == 0)
{
lean_dec(v_j_1499_);
lean_dec(v_x_1495_);
lean_dec(v_x_1494_);
return v_x_1491_;
}
else
{
lean_object* v___x_1503_; uint8_t v_isShared_1504_; uint8_t v_isSharedCheck_1540_; 
lean_inc_ref(v_es_1496_);
v_isSharedCheck_1540_ = !lean_is_exclusive(v_x_1491_);
if (v_isSharedCheck_1540_ == 0)
{
lean_object* v_unused_1541_; 
v_unused_1541_ = lean_ctor_get(v_x_1491_, 0);
lean_dec(v_unused_1541_);
v___x_1503_ = v_x_1491_;
v_isShared_1504_ = v_isSharedCheck_1540_;
goto v_resetjp_1502_;
}
else
{
lean_dec(v_x_1491_);
v___x_1503_ = lean_box(0);
v_isShared_1504_ = v_isSharedCheck_1540_;
goto v_resetjp_1502_;
}
v_resetjp_1502_:
{
lean_object* v_v_1505_; lean_object* v___x_1506_; lean_object* v_xs_x27_1507_; lean_object* v___y_1509_; 
v_v_1505_ = lean_array_fget(v_es_1496_, v_j_1499_);
v___x_1506_ = lean_box(0);
v_xs_x27_1507_ = lean_array_fset(v_es_1496_, v_j_1499_, v___x_1506_);
switch(lean_obj_tag(v_v_1505_))
{
case 0:
{
lean_object* v_key_1514_; lean_object* v_val_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1525_; 
v_key_1514_ = lean_ctor_get(v_v_1505_, 0);
v_val_1515_ = lean_ctor_get(v_v_1505_, 1);
v_isSharedCheck_1525_ = !lean_is_exclusive(v_v_1505_);
if (v_isSharedCheck_1525_ == 0)
{
v___x_1517_ = v_v_1505_;
v_isShared_1518_ = v_isSharedCheck_1525_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_val_1515_);
lean_inc(v_key_1514_);
lean_dec(v_v_1505_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1525_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
uint8_t v___x_1519_; 
v___x_1519_ = l_Lean_instBEqMVarId_beq(v_x_1494_, v_key_1514_);
if (v___x_1519_ == 0)
{
lean_object* v___x_1520_; lean_object* v___x_1521_; 
lean_del_object(v___x_1517_);
v___x_1520_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1514_, v_val_1515_, v_x_1494_, v_x_1495_);
v___x_1521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1521_, 0, v___x_1520_);
v___y_1509_ = v___x_1521_;
goto v___jp_1508_;
}
else
{
lean_object* v___x_1523_; 
lean_dec(v_val_1515_);
lean_dec(v_key_1514_);
if (v_isShared_1518_ == 0)
{
lean_ctor_set(v___x_1517_, 1, v_x_1495_);
lean_ctor_set(v___x_1517_, 0, v_x_1494_);
v___x_1523_ = v___x_1517_;
goto v_reusejp_1522_;
}
else
{
lean_object* v_reuseFailAlloc_1524_; 
v_reuseFailAlloc_1524_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1524_, 0, v_x_1494_);
lean_ctor_set(v_reuseFailAlloc_1524_, 1, v_x_1495_);
v___x_1523_ = v_reuseFailAlloc_1524_;
goto v_reusejp_1522_;
}
v_reusejp_1522_:
{
v___y_1509_ = v___x_1523_;
goto v___jp_1508_;
}
}
}
}
case 1:
{
lean_object* v_node_1526_; lean_object* v___x_1528_; uint8_t v_isShared_1529_; uint8_t v_isSharedCheck_1538_; 
v_node_1526_ = lean_ctor_get(v_v_1505_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v_v_1505_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1528_ = v_v_1505_;
v_isShared_1529_ = v_isSharedCheck_1538_;
goto v_resetjp_1527_;
}
else
{
lean_inc(v_node_1526_);
lean_dec(v_v_1505_);
v___x_1528_ = lean_box(0);
v_isShared_1529_ = v_isSharedCheck_1538_;
goto v_resetjp_1527_;
}
v_resetjp_1527_:
{
size_t v___x_1530_; size_t v___x_1531_; size_t v___x_1532_; size_t v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1536_; 
v___x_1530_ = ((size_t)5ULL);
v___x_1531_ = lean_usize_shift_right(v_x_1492_, v___x_1530_);
v___x_1532_ = ((size_t)1ULL);
v___x_1533_ = lean_usize_add(v_x_1493_, v___x_1532_);
v___x_1534_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(v_node_1526_, v___x_1531_, v___x_1533_, v_x_1494_, v_x_1495_);
if (v_isShared_1529_ == 0)
{
lean_ctor_set(v___x_1528_, 0, v___x_1534_);
v___x_1536_ = v___x_1528_;
goto v_reusejp_1535_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v___x_1534_);
v___x_1536_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1535_;
}
v_reusejp_1535_:
{
v___y_1509_ = v___x_1536_;
goto v___jp_1508_;
}
}
}
default: 
{
lean_object* v___x_1539_; 
v___x_1539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1539_, 0, v_x_1494_);
lean_ctor_set(v___x_1539_, 1, v_x_1495_);
v___y_1509_ = v___x_1539_;
goto v___jp_1508_;
}
}
v___jp_1508_:
{
lean_object* v___x_1510_; lean_object* v___x_1512_; 
v___x_1510_ = lean_array_fset(v_xs_x27_1507_, v_j_1499_, v___y_1509_);
lean_dec(v_j_1499_);
if (v_isShared_1504_ == 0)
{
lean_ctor_set(v___x_1503_, 0, v___x_1510_);
v___x_1512_ = v___x_1503_;
goto v_reusejp_1511_;
}
else
{
lean_object* v_reuseFailAlloc_1513_; 
v_reuseFailAlloc_1513_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1513_, 0, v___x_1510_);
v___x_1512_ = v_reuseFailAlloc_1513_;
goto v_reusejp_1511_;
}
v_reusejp_1511_:
{
return v___x_1512_;
}
}
}
}
}
else
{
lean_object* v_ks_1542_; lean_object* v_vs_1543_; lean_object* v___x_1545_; uint8_t v_isShared_1546_; uint8_t v_isSharedCheck_1563_; 
v_ks_1542_ = lean_ctor_get(v_x_1491_, 0);
v_vs_1543_ = lean_ctor_get(v_x_1491_, 1);
v_isSharedCheck_1563_ = !lean_is_exclusive(v_x_1491_);
if (v_isSharedCheck_1563_ == 0)
{
v___x_1545_ = v_x_1491_;
v_isShared_1546_ = v_isSharedCheck_1563_;
goto v_resetjp_1544_;
}
else
{
lean_inc(v_vs_1543_);
lean_inc(v_ks_1542_);
lean_dec(v_x_1491_);
v___x_1545_ = lean_box(0);
v_isShared_1546_ = v_isSharedCheck_1563_;
goto v_resetjp_1544_;
}
v_resetjp_1544_:
{
lean_object* v___x_1548_; 
if (v_isShared_1546_ == 0)
{
v___x_1548_ = v___x_1545_;
goto v_reusejp_1547_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_ks_1542_);
lean_ctor_set(v_reuseFailAlloc_1562_, 1, v_vs_1543_);
v___x_1548_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1547_;
}
v_reusejp_1547_:
{
lean_object* v_newNode_1549_; uint8_t v___y_1551_; size_t v___x_1557_; uint8_t v___x_1558_; 
v_newNode_1549_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11___redArg(v___x_1548_, v_x_1494_, v_x_1495_);
v___x_1557_ = ((size_t)7ULL);
v___x_1558_ = lean_usize_dec_le(v___x_1557_, v_x_1493_);
if (v___x_1558_ == 0)
{
lean_object* v___x_1559_; lean_object* v___x_1560_; uint8_t v___x_1561_; 
v___x_1559_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1549_);
v___x_1560_ = lean_unsigned_to_nat(4u);
v___x_1561_ = lean_nat_dec_lt(v___x_1559_, v___x_1560_);
lean_dec(v___x_1559_);
v___y_1551_ = v___x_1561_;
goto v___jp_1550_;
}
else
{
v___y_1551_ = v___x_1558_;
goto v___jp_1550_;
}
v___jp_1550_:
{
if (v___y_1551_ == 0)
{
lean_object* v_ks_1552_; lean_object* v_vs_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; 
v_ks_1552_ = lean_ctor_get(v_newNode_1549_, 0);
lean_inc_ref(v_ks_1552_);
v_vs_1553_ = lean_ctor_get(v_newNode_1549_, 1);
lean_inc_ref(v_vs_1553_);
lean_dec_ref(v_newNode_1549_);
v___x_1554_ = lean_unsigned_to_nat(0u);
v___x_1555_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___closed__0);
v___x_1556_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg(v_x_1493_, v_ks_1552_, v_vs_1553_, v___x_1554_, v___x_1555_);
lean_dec_ref(v_vs_1553_);
lean_dec_ref(v_ks_1552_);
return v___x_1556_;
}
else
{
return v_newNode_1549_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg(size_t v_depth_1564_, lean_object* v_keys_1565_, lean_object* v_vals_1566_, lean_object* v_i_1567_, lean_object* v_entries_1568_){
_start:
{
lean_object* v___x_1569_; uint8_t v___x_1570_; 
v___x_1569_ = lean_array_get_size(v_keys_1565_);
v___x_1570_ = lean_nat_dec_lt(v_i_1567_, v___x_1569_);
if (v___x_1570_ == 0)
{
lean_dec(v_i_1567_);
return v_entries_1568_;
}
else
{
lean_object* v_k_1571_; lean_object* v_v_1572_; uint64_t v___x_1573_; size_t v_h_1574_; size_t v___x_1575_; lean_object* v___x_1576_; size_t v___x_1577_; size_t v___x_1578_; size_t v___x_1579_; size_t v_h_1580_; lean_object* v___x_1581_; lean_object* v___x_1582_; 
v_k_1571_ = lean_array_fget_borrowed(v_keys_1565_, v_i_1567_);
v_v_1572_ = lean_array_fget_borrowed(v_vals_1566_, v_i_1567_);
v___x_1573_ = l_Lean_instHashableMVarId_hash(v_k_1571_);
v_h_1574_ = lean_uint64_to_usize(v___x_1573_);
v___x_1575_ = ((size_t)5ULL);
v___x_1576_ = lean_unsigned_to_nat(1u);
v___x_1577_ = ((size_t)1ULL);
v___x_1578_ = lean_usize_sub(v_depth_1564_, v___x_1577_);
v___x_1579_ = lean_usize_mul(v___x_1575_, v___x_1578_);
v_h_1580_ = lean_usize_shift_right(v_h_1574_, v___x_1579_);
v___x_1581_ = lean_nat_add(v_i_1567_, v___x_1576_);
lean_dec(v_i_1567_);
lean_inc(v_v_1572_);
lean_inc(v_k_1571_);
v___x_1582_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(v_entries_1568_, v_h_1580_, v_depth_1564_, v_k_1571_, v_v_1572_);
v_i_1567_ = v___x_1581_;
v_entries_1568_ = v___x_1582_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg___boxed(lean_object* v_depth_1584_, lean_object* v_keys_1585_, lean_object* v_vals_1586_, lean_object* v_i_1587_, lean_object* v_entries_1588_){
_start:
{
size_t v_depth_boxed_1589_; lean_object* v_res_1590_; 
v_depth_boxed_1589_ = lean_unbox_usize(v_depth_1584_);
lean_dec(v_depth_1584_);
v_res_1590_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg(v_depth_boxed_1589_, v_keys_1585_, v_vals_1586_, v_i_1587_, v_entries_1588_);
lean_dec_ref(v_vals_1586_);
lean_dec_ref(v_keys_1585_);
return v_res_1590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg___boxed(lean_object* v_x_1591_, lean_object* v_x_1592_, lean_object* v_x_1593_, lean_object* v_x_1594_, lean_object* v_x_1595_){
_start:
{
size_t v_x_27721__boxed_1596_; size_t v_x_27722__boxed_1597_; lean_object* v_res_1598_; 
v_x_27721__boxed_1596_ = lean_unbox_usize(v_x_1592_);
lean_dec(v_x_1592_);
v_x_27722__boxed_1597_ = lean_unbox_usize(v_x_1593_);
lean_dec(v_x_1593_);
v_res_1598_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(v_x_1591_, v_x_27721__boxed_1596_, v_x_27722__boxed_1597_, v_x_1594_, v_x_1595_);
return v_res_1598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2___redArg(lean_object* v_x_1599_, lean_object* v_x_1600_, lean_object* v_x_1601_){
_start:
{
uint64_t v___x_1602_; size_t v___x_1603_; size_t v___x_1604_; lean_object* v___x_1605_; 
v___x_1602_ = l_Lean_instHashableMVarId_hash(v_x_1600_);
v___x_1603_ = lean_uint64_to_usize(v___x_1602_);
v___x_1604_ = ((size_t)1ULL);
v___x_1605_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(v_x_1599_, v___x_1603_, v___x_1604_, v_x_1600_, v_x_1601_);
return v___x_1605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0(lean_object* v_mvarId_1606_, lean_object* v___y_1607_, lean_object* v___y_1608_, lean_object* v___y_1609_, lean_object* v___y_1610_, lean_object* v___y_1611_, lean_object* v___y_1612_){
_start:
{
lean_object* v___x_1614_; lean_object* v_mctx_1615_; lean_object* v_mvarDecl_1616_; lean_object* v_userName_1617_; lean_object* v_lctx_1618_; lean_object* v_type_1619_; lean_object* v_depth_1620_; lean_object* v_localInstances_1621_; uint8_t v_kind_1622_; lean_object* v_numScopeArgs_1623_; lean_object* v_index_1624_; lean_object* v___x_1626_; uint8_t v_isShared_1627_; uint8_t v_isSharedCheck_1687_; 
v___x_1614_ = lean_st_ref_get(v___y_1610_);
v_mctx_1615_ = lean_ctor_get(v___x_1614_, 0);
lean_inc_ref(v_mctx_1615_);
lean_dec(v___x_1614_);
lean_inc(v_mvarId_1606_);
v_mvarDecl_1616_ = l_Lean_MetavarContext_getDecl(v_mctx_1615_, v_mvarId_1606_);
lean_dec_ref(v_mctx_1615_);
v_userName_1617_ = lean_ctor_get(v_mvarDecl_1616_, 0);
v_lctx_1618_ = lean_ctor_get(v_mvarDecl_1616_, 1);
v_type_1619_ = lean_ctor_get(v_mvarDecl_1616_, 2);
v_depth_1620_ = lean_ctor_get(v_mvarDecl_1616_, 3);
v_localInstances_1621_ = lean_ctor_get(v_mvarDecl_1616_, 4);
v_kind_1622_ = lean_ctor_get_uint8(v_mvarDecl_1616_, sizeof(void*)*7);
v_numScopeArgs_1623_ = lean_ctor_get(v_mvarDecl_1616_, 5);
v_index_1624_ = lean_ctor_get(v_mvarDecl_1616_, 6);
v_isSharedCheck_1687_ = !lean_is_exclusive(v_mvarDecl_1616_);
if (v_isSharedCheck_1687_ == 0)
{
v___x_1626_ = v_mvarDecl_1616_;
v_isShared_1627_ = v_isSharedCheck_1687_;
goto v_resetjp_1625_;
}
else
{
lean_inc(v_index_1624_);
lean_inc(v_numScopeArgs_1623_);
lean_inc(v_localInstances_1621_);
lean_inc(v_depth_1620_);
lean_inc(v_type_1619_);
lean_inc(v_lctx_1618_);
lean_inc(v_userName_1617_);
lean_dec(v_mvarDecl_1616_);
v___x_1626_ = lean_box(0);
v_isShared_1627_ = v_isSharedCheck_1687_;
goto v_resetjp_1625_;
}
v_resetjp_1625_:
{
lean_object* v___x_1628_; 
v___x_1628_ = lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0(v_lctx_1618_, v___y_1607_, v___y_1608_, v___y_1609_, v___y_1610_, v___y_1611_, v___y_1612_);
if (lean_obj_tag(v___x_1628_) == 0)
{
lean_object* v_a_1629_; lean_object* v___x_1630_; lean_object* v_a_1631_; lean_object* v___x_1633_; uint8_t v_isShared_1634_; uint8_t v_isSharedCheck_1678_; 
v_a_1629_ = lean_ctor_get(v___x_1628_, 0);
lean_inc(v_a_1629_);
lean_dec_ref_known(v___x_1628_, 1);
v___x_1630_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_type_1619_, v___y_1610_);
v_a_1631_ = lean_ctor_get(v___x_1630_, 0);
v_isSharedCheck_1678_ = !lean_is_exclusive(v___x_1630_);
if (v_isSharedCheck_1678_ == 0)
{
v___x_1633_ = v___x_1630_;
v_isShared_1634_ = v_isSharedCheck_1678_;
goto v_resetjp_1632_;
}
else
{
lean_inc(v_a_1631_);
lean_dec(v___x_1630_);
v___x_1633_ = lean_box(0);
v_isShared_1634_ = v_isSharedCheck_1678_;
goto v_resetjp_1632_;
}
v_resetjp_1632_:
{
lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v_fst_1637_; lean_object* v_snd_1638_; lean_object* v___x_1639_; lean_object* v_mctx_1640_; lean_object* v_cache_1641_; lean_object* v_zetaDeltaFVarIds_1642_; lean_object* v_postponed_1643_; lean_object* v_diag_1644_; lean_object* v___x_1646_; uint8_t v_isShared_1647_; uint8_t v_isSharedCheck_1677_; 
v___x_1635_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1635_, 0, v_a_1629_);
lean_ctor_set(v___x_1635_, 1, v_a_1631_);
v___x_1636_ = lean_sharecommon_quick(v___x_1635_);
lean_dec_ref_known(v___x_1635_, 2);
v_fst_1637_ = lean_ctor_get(v___x_1636_, 0);
lean_inc(v_fst_1637_);
v_snd_1638_ = lean_ctor_get(v___x_1636_, 1);
lean_inc(v_snd_1638_);
lean_dec(v___x_1636_);
v___x_1639_ = lean_st_ref_take(v___y_1610_);
v_mctx_1640_ = lean_ctor_get(v___x_1639_, 0);
v_cache_1641_ = lean_ctor_get(v___x_1639_, 1);
v_zetaDeltaFVarIds_1642_ = lean_ctor_get(v___x_1639_, 2);
v_postponed_1643_ = lean_ctor_get(v___x_1639_, 3);
v_diag_1644_ = lean_ctor_get(v___x_1639_, 4);
v_isSharedCheck_1677_ = !lean_is_exclusive(v___x_1639_);
if (v_isSharedCheck_1677_ == 0)
{
v___x_1646_ = v___x_1639_;
v_isShared_1647_ = v_isSharedCheck_1677_;
goto v_resetjp_1645_;
}
else
{
lean_inc(v_diag_1644_);
lean_inc(v_postponed_1643_);
lean_inc(v_zetaDeltaFVarIds_1642_);
lean_inc(v_cache_1641_);
lean_inc(v_mctx_1640_);
lean_dec(v___x_1639_);
v___x_1646_ = lean_box(0);
v_isShared_1647_ = v_isSharedCheck_1677_;
goto v_resetjp_1645_;
}
v_resetjp_1645_:
{
lean_object* v_depth_1648_; lean_object* v_levelAssignDepth_1649_; lean_object* v_lmvarCounter_1650_; lean_object* v_mvarCounter_1651_; lean_object* v_lDecls_1652_; lean_object* v_decls_1653_; lean_object* v_userNames_1654_; lean_object* v_lAssignment_1655_; lean_object* v_eAssignment_1656_; lean_object* v_dAssignment_1657_; lean_object* v___x_1659_; uint8_t v_isShared_1660_; uint8_t v_isSharedCheck_1676_; 
v_depth_1648_ = lean_ctor_get(v_mctx_1640_, 0);
v_levelAssignDepth_1649_ = lean_ctor_get(v_mctx_1640_, 1);
v_lmvarCounter_1650_ = lean_ctor_get(v_mctx_1640_, 2);
v_mvarCounter_1651_ = lean_ctor_get(v_mctx_1640_, 3);
v_lDecls_1652_ = lean_ctor_get(v_mctx_1640_, 4);
v_decls_1653_ = lean_ctor_get(v_mctx_1640_, 5);
v_userNames_1654_ = lean_ctor_get(v_mctx_1640_, 6);
v_lAssignment_1655_ = lean_ctor_get(v_mctx_1640_, 7);
v_eAssignment_1656_ = lean_ctor_get(v_mctx_1640_, 8);
v_dAssignment_1657_ = lean_ctor_get(v_mctx_1640_, 9);
v_isSharedCheck_1676_ = !lean_is_exclusive(v_mctx_1640_);
if (v_isSharedCheck_1676_ == 0)
{
v___x_1659_ = v_mctx_1640_;
v_isShared_1660_ = v_isSharedCheck_1676_;
goto v_resetjp_1658_;
}
else
{
lean_inc(v_dAssignment_1657_);
lean_inc(v_eAssignment_1656_);
lean_inc(v_lAssignment_1655_);
lean_inc(v_userNames_1654_);
lean_inc(v_decls_1653_);
lean_inc(v_lDecls_1652_);
lean_inc(v_mvarCounter_1651_);
lean_inc(v_lmvarCounter_1650_);
lean_inc(v_levelAssignDepth_1649_);
lean_inc(v_depth_1648_);
lean_dec(v_mctx_1640_);
v___x_1659_ = lean_box(0);
v_isShared_1660_ = v_isSharedCheck_1676_;
goto v_resetjp_1658_;
}
v_resetjp_1658_:
{
lean_object* v___x_1662_; 
if (v_isShared_1627_ == 0)
{
lean_ctor_set(v___x_1626_, 2, v_snd_1638_);
lean_ctor_set(v___x_1626_, 1, v_fst_1637_);
v___x_1662_ = v___x_1626_;
goto v_reusejp_1661_;
}
else
{
lean_object* v_reuseFailAlloc_1675_; 
v_reuseFailAlloc_1675_ = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(v_reuseFailAlloc_1675_, 0, v_userName_1617_);
lean_ctor_set(v_reuseFailAlloc_1675_, 1, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1675_, 2, v_snd_1638_);
lean_ctor_set(v_reuseFailAlloc_1675_, 3, v_depth_1620_);
lean_ctor_set(v_reuseFailAlloc_1675_, 4, v_localInstances_1621_);
lean_ctor_set(v_reuseFailAlloc_1675_, 5, v_numScopeArgs_1623_);
lean_ctor_set(v_reuseFailAlloc_1675_, 6, v_index_1624_);
lean_ctor_set_uint8(v_reuseFailAlloc_1675_, sizeof(void*)*7, v_kind_1622_);
v___x_1662_ = v_reuseFailAlloc_1675_;
goto v_reusejp_1661_;
}
v_reusejp_1661_:
{
lean_object* v___x_1663_; lean_object* v___x_1665_; 
v___x_1663_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2___redArg(v_decls_1653_, v_mvarId_1606_, v___x_1662_);
if (v_isShared_1660_ == 0)
{
lean_ctor_set(v___x_1659_, 5, v___x_1663_);
v___x_1665_ = v___x_1659_;
goto v_reusejp_1664_;
}
else
{
lean_object* v_reuseFailAlloc_1674_; 
v_reuseFailAlloc_1674_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1674_, 0, v_depth_1648_);
lean_ctor_set(v_reuseFailAlloc_1674_, 1, v_levelAssignDepth_1649_);
lean_ctor_set(v_reuseFailAlloc_1674_, 2, v_lmvarCounter_1650_);
lean_ctor_set(v_reuseFailAlloc_1674_, 3, v_mvarCounter_1651_);
lean_ctor_set(v_reuseFailAlloc_1674_, 4, v_lDecls_1652_);
lean_ctor_set(v_reuseFailAlloc_1674_, 5, v___x_1663_);
lean_ctor_set(v_reuseFailAlloc_1674_, 6, v_userNames_1654_);
lean_ctor_set(v_reuseFailAlloc_1674_, 7, v_lAssignment_1655_);
lean_ctor_set(v_reuseFailAlloc_1674_, 8, v_eAssignment_1656_);
lean_ctor_set(v_reuseFailAlloc_1674_, 9, v_dAssignment_1657_);
v___x_1665_ = v_reuseFailAlloc_1674_;
goto v_reusejp_1664_;
}
v_reusejp_1664_:
{
lean_object* v___x_1667_; 
if (v_isShared_1647_ == 0)
{
lean_ctor_set(v___x_1646_, 0, v___x_1665_);
v___x_1667_ = v___x_1646_;
goto v_reusejp_1666_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v___x_1665_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v_cache_1641_);
lean_ctor_set(v_reuseFailAlloc_1673_, 2, v_zetaDeltaFVarIds_1642_);
lean_ctor_set(v_reuseFailAlloc_1673_, 3, v_postponed_1643_);
lean_ctor_set(v_reuseFailAlloc_1673_, 4, v_diag_1644_);
v___x_1667_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1666_;
}
v_reusejp_1666_:
{
lean_object* v___x_1668_; lean_object* v___x_1669_; lean_object* v___x_1671_; 
v___x_1668_ = lean_st_ref_set(v___y_1610_, v___x_1667_);
v___x_1669_ = lean_box(0);
if (v_isShared_1634_ == 0)
{
lean_ctor_set(v___x_1633_, 0, v___x_1669_);
v___x_1671_ = v___x_1633_;
goto v_reusejp_1670_;
}
else
{
lean_object* v_reuseFailAlloc_1672_; 
v_reuseFailAlloc_1672_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1672_, 0, v___x_1669_);
v___x_1671_ = v_reuseFailAlloc_1672_;
goto v_reusejp_1670_;
}
v_reusejp_1670_:
{
return v___x_1671_;
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
lean_object* v_a_1679_; lean_object* v___x_1681_; uint8_t v_isShared_1682_; uint8_t v_isSharedCheck_1686_; 
lean_del_object(v___x_1626_);
lean_dec(v_index_1624_);
lean_dec(v_numScopeArgs_1623_);
lean_dec_ref(v_localInstances_1621_);
lean_dec(v_depth_1620_);
lean_dec_ref(v_type_1619_);
lean_dec(v_userName_1617_);
lean_dec(v_mvarId_1606_);
v_a_1679_ = lean_ctor_get(v___x_1628_, 0);
v_isSharedCheck_1686_ = !lean_is_exclusive(v___x_1628_);
if (v_isSharedCheck_1686_ == 0)
{
v___x_1681_ = v___x_1628_;
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
else
{
lean_inc(v_a_1679_);
lean_dec(v___x_1628_);
v___x_1681_ = lean_box(0);
v_isShared_1682_ = v_isSharedCheck_1686_;
goto v_resetjp_1680_;
}
v_resetjp_1680_:
{
lean_object* v___x_1684_; 
if (v_isShared_1682_ == 0)
{
v___x_1684_ = v___x_1681_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v_a_1679_);
v___x_1684_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
return v___x_1684_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0___boxed(lean_object* v_mvarId_1688_, lean_object* v___y_1689_, lean_object* v___y_1690_, lean_object* v___y_1691_, lean_object* v___y_1692_, lean_object* v___y_1693_, lean_object* v___y_1694_, lean_object* v___y_1695_){
_start:
{
lean_object* v_res_1696_; 
v_res_1696_ = lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0(v_mvarId_1688_, v___y_1689_, v___y_1690_, v___y_1691_, v___y_1692_, v___y_1693_, v___y_1694_);
lean_dec(v___y_1694_);
lean_dec_ref(v___y_1693_);
lean_dec(v___y_1692_);
lean_dec_ref(v___y_1691_);
lean_dec(v___y_1690_);
lean_dec_ref(v___y_1689_);
return v_res_1696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions(lean_object* v_loc_1697_, lean_object* v_parentDecl_x3f_1698_, lean_object* v_token_1699_, lean_object* v_a_1700_, lean_object* v_a_1701_, lean_object* v_a_1702_, lean_object* v_a_1703_, lean_object* v_a_1704_, lean_object* v_a_1705_){
_start:
{
lean_object* v___y_1708_; lean_object* v_mvarId_1717_; lean_object* v_loc_1718_; lean_object* v_keyedConfig_1719_; uint8_t v_trackZetaDelta_1720_; lean_object* v_zetaDeltaSet_1721_; lean_object* v_lctx_1722_; lean_object* v_localInstances_1723_; lean_object* v_defEqCtx_x3f_1724_; lean_object* v_synthPendingDepth_1725_; lean_object* v_customCanUnfoldPredicate_x3f_1726_; uint8_t v_univApprox_1727_; uint8_t v_inTypeClassResolution_1728_; uint8_t v_cacheInferType_1729_; uint8_t v___x_1730_; lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; 
v_mvarId_1717_ = lean_ctor_get(v_loc_1697_, 0);
lean_inc_n(v_mvarId_1717_, 2);
v_loc_1718_ = lean_ctor_get(v_loc_1697_, 1);
lean_inc_ref(v_loc_1718_);
lean_dec_ref(v_loc_1697_);
v_keyedConfig_1719_ = lean_ctor_get(v_a_1702_, 0);
v_trackZetaDelta_1720_ = lean_ctor_get_uint8(v_a_1702_, sizeof(void*)*7);
v_zetaDeltaSet_1721_ = lean_ctor_get(v_a_1702_, 1);
v_lctx_1722_ = lean_ctor_get(v_a_1702_, 2);
v_localInstances_1723_ = lean_ctor_get(v_a_1702_, 3);
v_defEqCtx_x3f_1724_ = lean_ctor_get(v_a_1702_, 4);
v_synthPendingDepth_1725_ = lean_ctor_get(v_a_1702_, 5);
v_customCanUnfoldPredicate_x3f_1726_ = lean_ctor_get(v_a_1702_, 6);
v_univApprox_1727_ = lean_ctor_get_uint8(v_a_1702_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1728_ = lean_ctor_get_uint8(v_a_1702_, sizeof(void*)*7 + 2);
v_cacheInferType_1729_ = lean_ctor_get_uint8(v_a_1702_, sizeof(void*)*7 + 3);
v___x_1730_ = 2;
lean_inc_ref(v_keyedConfig_1719_);
v___x_1731_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1730_, v_keyedConfig_1719_);
lean_inc(v_customCanUnfoldPredicate_x3f_1726_);
lean_inc(v_synthPendingDepth_1725_);
lean_inc(v_defEqCtx_x3f_1724_);
lean_inc_ref(v_localInstances_1723_);
lean_inc_ref(v_lctx_1722_);
lean_inc(v_zetaDeltaSet_1721_);
v___x_1732_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_1732_, 0, v___x_1731_);
lean_ctor_set(v___x_1732_, 1, v_zetaDeltaSet_1721_);
lean_ctor_set(v___x_1732_, 2, v_lctx_1722_);
lean_ctor_set(v___x_1732_, 3, v_localInstances_1723_);
lean_ctor_set(v___x_1732_, 4, v_defEqCtx_x3f_1724_);
lean_ctor_set(v___x_1732_, 5, v_synthPendingDepth_1725_);
lean_ctor_set(v___x_1732_, 6, v_customCanUnfoldPredicate_x3f_1726_);
lean_ctor_set_uint8(v___x_1732_, sizeof(void*)*7, v_trackZetaDelta_1720_);
lean_ctor_set_uint8(v___x_1732_, sizeof(void*)*7 + 1, v_univApprox_1727_);
lean_ctor_set_uint8(v___x_1732_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1728_);
lean_ctor_set_uint8(v___x_1732_, sizeof(void*)*7 + 3, v_cacheInferType_1729_);
v___x_1733_ = lp_mathlib_Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0(v_mvarId_1717_, v_a_1700_, v_a_1701_, v___x_1732_, v_a_1703_, v_a_1704_, v_a_1705_);
if (lean_obj_tag(v___x_1733_) == 0)
{
lean_object* v___f_1734_; lean_object* v___x_1735_; 
lean_dec_ref_known(v___x_1733_, 1);
lean_inc(v_mvarId_1717_);
v___f_1734_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__3___boxed), 11, 4);
lean_closure_set(v___f_1734_, 0, v_token_1699_);
lean_closure_set(v___f_1734_, 1, v_parentDecl_x3f_1698_);
lean_closure_set(v___f_1734_, 2, v_mvarId_1717_);
lean_closure_set(v___f_1734_, 3, v_loc_1718_);
v___x_1735_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__3___redArg(v_mvarId_1717_, v___f_1734_, v_a_1700_, v_a_1701_, v___x_1732_, v_a_1703_, v_a_1704_, v_a_1705_);
lean_dec_ref_known(v___x_1732_, 7);
v___y_1708_ = v___x_1735_;
goto v___jp_1707_;
}
else
{
lean_dec_ref_known(v___x_1732_, 7);
lean_dec_ref(v_loc_1718_);
lean_dec(v_mvarId_1717_);
lean_dec_ref(v_token_1699_);
lean_dec(v_parentDecl_x3f_1698_);
v___y_1708_ = v___x_1733_;
goto v___jp_1707_;
}
v___jp_1707_:
{
if (lean_obj_tag(v___y_1708_) == 0)
{
lean_object* v_a_1709_; lean_object* v___x_1711_; uint8_t v_isShared_1712_; uint8_t v_isSharedCheck_1716_; 
v_a_1709_ = lean_ctor_get(v___y_1708_, 0);
v_isSharedCheck_1716_ = !lean_is_exclusive(v___y_1708_);
if (v_isSharedCheck_1716_ == 0)
{
v___x_1711_ = v___y_1708_;
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
else
{
lean_inc(v_a_1709_);
lean_dec(v___y_1708_);
v___x_1711_ = lean_box(0);
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
v_resetjp_1710_:
{
lean_object* v___x_1714_; 
if (v_isShared_1712_ == 0)
{
v___x_1714_ = v___x_1711_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v_a_1709_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
}
else
{
return v___y_1708_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___boxed(lean_object* v_loc_1736_, lean_object* v_parentDecl_x3f_1737_, lean_object* v_token_1738_, lean_object* v_a_1739_, lean_object* v_a_1740_, lean_object* v_a_1741_, lean_object* v_a_1742_, lean_object* v_a_1743_, lean_object* v_a_1744_, lean_object* v_a_1745_){
_start:
{
lean_object* v_res_1746_; 
v_res_1746_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions(v_loc_1736_, v_parentDecl_x3f_1737_, v_token_1738_, v_a_1739_, v_a_1740_, v_a_1741_, v_a_1742_, v_a_1743_, v_a_1744_);
lean_dec(v_a_1744_);
lean_dec_ref(v_a_1743_);
lean_dec(v_a_1742_);
lean_dec_ref(v_a_1741_);
lean_dec(v_a_1740_);
lean_dec_ref(v_a_1739_);
return v_res_1746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1(lean_object* v_e_1747_, lean_object* v___y_1748_, lean_object* v___y_1749_, lean_object* v___y_1750_, lean_object* v___y_1751_, lean_object* v___y_1752_, lean_object* v___y_1753_){
_start:
{
lean_object* v___x_1755_; 
v___x_1755_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___redArg(v_e_1747_, v___y_1751_);
return v___x_1755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1___boxed(lean_object* v_e_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_, lean_object* v___y_1761_, lean_object* v___y_1762_, lean_object* v___y_1763_){
_start:
{
lean_object* v_res_1764_; 
v_res_1764_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__1(v_e_1756_, v___y_1757_, v___y_1758_, v___y_1759_, v___y_1760_, v___y_1761_, v___y_1762_);
lean_dec(v___y_1762_);
lean_dec_ref(v___y_1761_);
lean_dec(v___y_1760_);
lean_dec_ref(v___y_1759_);
lean_dec(v___y_1758_);
lean_dec_ref(v___y_1757_);
return v_res_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1(lean_object* v_00_u03b1_1765_, lean_object* v_e_1766_, lean_object* v_pos_1767_, lean_object* v_k_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_, lean_object* v___y_1771_, lean_object* v___y_1772_, lean_object* v___y_1773_, lean_object* v___y_1774_){
_start:
{
lean_object* v___x_1776_; 
v___x_1776_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___redArg(v_e_1766_, v_pos_1767_, v_k_1768_, v___y_1769_, v___y_1770_, v___y_1771_, v___y_1772_, v___y_1773_, v___y_1774_);
return v___x_1776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1___boxed(lean_object* v_00_u03b1_1777_, lean_object* v_e_1778_, lean_object* v_pos_1779_, lean_object* v_k_1780_, lean_object* v___y_1781_, lean_object* v___y_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v_res_1788_; 
v_res_1788_ = lp_mathlib___private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1(v_00_u03b1_1777_, v_e_1778_, v_pos_1779_, v_k_1780_, v___y_1781_, v___y_1782_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec(v___y_1784_);
lean_dec_ref(v___y_1783_);
lean_dec(v___y_1782_);
lean_dec_ref(v___y_1781_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2(lean_object* v_00_u03b2_1789_, lean_object* v_x_1790_, lean_object* v_x_1791_, lean_object* v_x_1792_){
_start:
{
lean_object* v___x_1793_; 
v___x_1793_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2___redArg(v_x_1790_, v_x_1791_, v_x_1792_);
return v___x_1793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4(lean_object* v_00_u03b1_1794_, lean_object* v_visit_1795_, lean_object* v_p_1796_, lean_object* v_root_1797_, lean_object* v___y_1798_, lean_object* v___y_1799_, lean_object* v___y_1800_, lean_object* v___y_1801_, lean_object* v___y_1802_, lean_object* v___y_1803_){
_start:
{
lean_object* v___x_1805_; 
v___x_1805_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___redArg(v_visit_1795_, v_p_1796_, v_root_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_);
return v___x_1805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4___boxed(lean_object* v_00_u03b1_1806_, lean_object* v_visit_1807_, lean_object* v_p_1808_, lean_object* v_root_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_, lean_object* v___y_1812_, lean_object* v___y_1813_, lean_object* v___y_1814_, lean_object* v___y_1815_, lean_object* v___y_1816_){
_start:
{
lean_object* v_res_1817_; 
v_res_1817_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4(v_00_u03b1_1806_, v_visit_1807_, v_p_1808_, v_root_1809_, v___y_1810_, v___y_1811_, v___y_1812_, v___y_1813_, v___y_1814_, v___y_1815_);
lean_dec(v___y_1815_);
lean_dec_ref(v___y_1814_);
lean_dec(v___y_1813_);
lean_dec_ref(v___y_1812_);
lean_dec(v___y_1811_);
lean_dec_ref(v___y_1810_);
lean_dec(v_p_1808_);
return v_res_1817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3(lean_object* v_00_u03b4_1818_, lean_object* v_t_1819_, lean_object* v_k_1820_){
_start:
{
lean_object* v___x_1821_; 
v___x_1821_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___redArg(v_t_1819_, v_k_1820_);
return v___x_1821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b4_1822_, lean_object* v_t_1823_, lean_object* v_k_1824_){
_start:
{
lean_object* v_res_1825_; 
v_res_1825_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0_spec__3(v_00_u03b4_1822_, v_t_1823_, v_k_1824_);
lean_dec(v_k_1824_);
lean_dec(v_t_1823_);
return v_res_1825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8(lean_object* v_00_u03b2_1826_, lean_object* v_x_1827_, size_t v_x_1828_, size_t v_x_1829_, lean_object* v_x_1830_, lean_object* v_x_1831_){
_start:
{
lean_object* v___x_1832_; 
v___x_1832_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___redArg(v_x_1827_, v_x_1828_, v_x_1829_, v_x_1830_, v_x_1831_);
return v___x_1832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8___boxed(lean_object* v_00_u03b2_1833_, lean_object* v_x_1834_, lean_object* v_x_1835_, lean_object* v_x_1836_, lean_object* v_x_1837_, lean_object* v_x_1838_){
_start:
{
size_t v_x_28128__boxed_1839_; size_t v_x_28129__boxed_1840_; lean_object* v_res_1841_; 
v_x_28128__boxed_1839_ = lean_unbox_usize(v_x_1835_);
lean_dec(v_x_1835_);
v_x_28129__boxed_1840_ = lean_unbox_usize(v_x_1836_);
lean_dec(v_x_1836_);
v_res_1841_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8(v_00_u03b2_1833_, v_x_1834_, v_x_28128__boxed_1839_, v_x_28129__boxed_1840_, v_x_1837_, v_x_1838_);
return v_res_1841_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11(lean_object* v_00_u03b1_1842_, lean_object* v_k_1843_, lean_object* v_fvars_1844_, lean_object* v_x_1845_, lean_object* v_x_1846_, lean_object* v___y_1847_, lean_object* v___y_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_){
_start:
{
lean_object* v___x_1854_; 
v___x_1854_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg(v_k_1843_, v_fvars_1844_, v_x_1845_, v_x_1846_, v___y_1847_, v___y_1848_, v___y_1849_, v___y_1850_, v___y_1851_, v___y_1852_);
return v___x_1854_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___boxed(lean_object* v_00_u03b1_1855_, lean_object* v_k_1856_, lean_object* v_fvars_1857_, lean_object* v_x_1858_, lean_object* v_x_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_){
_start:
{
lean_object* v_res_1867_; 
v_res_1867_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11(v_00_u03b1_1855_, v_k_1856_, v_fvars_1857_, v_x_1858_, v_x_1859_, v___y_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_, v___y_1865_);
lean_dec(v___y_1865_);
lean_dec_ref(v___y_1864_);
lean_dec(v___y_1863_);
lean_dec_ref(v___y_1862_);
lean_dec(v___y_1861_);
lean_dec_ref(v___y_1860_);
return v_res_1867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11(lean_object* v_00_u03b2_1868_, lean_object* v_n_1869_, lean_object* v_k_1870_, lean_object* v_v_1871_){
_start:
{
lean_object* v___x_1872_; 
v___x_1872_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11___redArg(v_n_1869_, v_k_1870_, v_v_1871_);
return v___x_1872_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12(lean_object* v_00_u03b2_1873_, size_t v_depth_1874_, lean_object* v_keys_1875_, lean_object* v_vals_1876_, lean_object* v_heq_1877_, lean_object* v_i_1878_, lean_object* v_entries_1879_){
_start:
{
lean_object* v___x_1880_; 
v___x_1880_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___redArg(v_depth_1874_, v_keys_1875_, v_vals_1876_, v_i_1878_, v_entries_1879_);
return v___x_1880_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12___boxed(lean_object* v_00_u03b2_1881_, lean_object* v_depth_1882_, lean_object* v_keys_1883_, lean_object* v_vals_1884_, lean_object* v_heq_1885_, lean_object* v_i_1886_, lean_object* v_entries_1887_){
_start:
{
size_t v_depth_boxed_1888_; lean_object* v_res_1889_; 
v_depth_boxed_1888_ = lean_unbox_usize(v_depth_1882_);
lean_dec(v_depth_1882_);
v_res_1889_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__12(v_00_u03b2_1881_, v_depth_boxed_1888_, v_keys_1883_, v_vals_1884_, v_heq_1885_, v_i_1886_, v_entries_1887_);
lean_dec_ref(v_vals_1884_);
lean_dec_ref(v_keys_1883_);
return v_res_1889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21(lean_object* v_00_u03b1_1890_, lean_object* v_name_1891_, uint8_t v_bi_1892_, lean_object* v_type_1893_, lean_object* v_k_1894_, uint8_t v_kind_1895_, lean_object* v___y_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_){
_start:
{
lean_object* v___x_1903_; 
v___x_1903_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___redArg(v_name_1891_, v_bi_1892_, v_type_1893_, v_k_1894_, v_kind_1895_, v___y_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_);
return v___x_1903_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21___boxed(lean_object* v_00_u03b1_1904_, lean_object* v_name_1905_, lean_object* v_bi_1906_, lean_object* v_type_1907_, lean_object* v_k_1908_, lean_object* v_kind_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_, lean_object* v___y_1914_, lean_object* v___y_1915_, lean_object* v___y_1916_){
_start:
{
uint8_t v_bi_boxed_1917_; uint8_t v_kind_boxed_1918_; lean_object* v_res_1919_; 
v_bi_boxed_1917_ = lean_unbox(v_bi_1906_);
v_kind_boxed_1918_ = lean_unbox(v_kind_1909_);
v_res_1919_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__21(v_00_u03b1_1904_, v_name_1905_, v_bi_boxed_1917_, v_type_1907_, v_k_1908_, v_kind_boxed_1918_, v___y_1910_, v___y_1911_, v___y_1912_, v___y_1913_, v___y_1914_, v___y_1915_);
lean_dec(v___y_1915_);
lean_dec_ref(v___y_1914_);
lean_dec(v___y_1913_);
lean_dec_ref(v___y_1912_);
lean_dec(v___y_1911_);
lean_dec_ref(v___y_1910_);
return v_res_1919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22(lean_object* v_00_u03b1_1920_, lean_object* v_name_1921_, lean_object* v_type_1922_, lean_object* v_val_1923_, lean_object* v_k_1924_, uint8_t v_nondep_1925_, uint8_t v_kind_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_){
_start:
{
lean_object* v___x_1934_; 
v___x_1934_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___redArg(v_name_1921_, v_type_1922_, v_val_1923_, v_k_1924_, v_nondep_1925_, v_kind_1926_, v___y_1927_, v___y_1928_, v___y_1929_, v___y_1930_, v___y_1931_, v___y_1932_);
return v___x_1934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22___boxed(lean_object* v_00_u03b1_1935_, lean_object* v_name_1936_, lean_object* v_type_1937_, lean_object* v_val_1938_, lean_object* v_k_1939_, lean_object* v_nondep_1940_, lean_object* v_kind_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
uint8_t v_nondep_boxed_1949_; uint8_t v_kind_boxed_1950_; lean_object* v_res_1951_; 
v_nondep_boxed_1949_ = lean_unbox(v_nondep_1940_);
v_kind_boxed_1950_ = lean_unbox(v_kind_1941_);
v_res_1951_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__22(v_00_u03b1_1935_, v_name_1936_, v_type_1937_, v_val_1938_, v_k_1939_, v_nondep_boxed_1949_, v_kind_boxed_1950_, v___y_1942_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1946_);
lean_dec(v___y_1945_);
lean_dec_ref(v___y_1944_);
lean_dec(v___y_1943_);
lean_dec_ref(v___y_1942_);
return v_res_1951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15(lean_object* v_00_u03b1_1952_, lean_object* v_k_1953_, lean_object* v_fvars_1954_, lean_object* v_n_1955_, lean_object* v_e_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_, lean_object* v___y_1962_){
_start:
{
lean_object* v___x_1964_; 
v___x_1964_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg(v_k_1953_, v_fvars_1954_, v_n_1955_, v_e_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_, v___y_1961_, v___y_1962_);
return v___x_1964_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___boxed(lean_object* v_00_u03b1_1965_, lean_object* v_k_1966_, lean_object* v_fvars_1967_, lean_object* v_n_1968_, lean_object* v_e_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
lean_object* v_res_1977_; 
v_res_1977_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15(v_00_u03b1_1965_, v_k_1966_, v_fvars_1967_, v_n_1968_, v_e_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec(v___y_1971_);
lean_dec_ref(v___y_1970_);
return v_res_1977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16(lean_object* v_00_u03b2_1978_, lean_object* v_x_1979_, lean_object* v_x_1980_, lean_object* v_x_1981_, lean_object* v_x_1982_){
_start:
{
lean_object* v___x_1983_; 
v___x_1983_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__2_spec__8_spec__11_spec__16___redArg(v_x_1979_, v_x_1980_, v_x_1981_, v_x_1982_);
return v___x_1983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20(lean_object* v_00_u03b1_1984_, lean_object* v_msg_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_){
_start:
{
lean_object* v___x_1993_; 
v___x_1993_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___redArg(v_msg_1985_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_);
return v___x_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20___boxed(lean_object* v_00_u03b1_1994_, lean_object* v_msg_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_){
_start:
{
lean_object* v_res_2003_; 
v_res_2003_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20(v_00_u03b1_1994_, v_msg_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_, v___y_2001_);
lean_dec(v___y_2001_);
lean_dec_ref(v___y_2000_);
lean_dec(v___y_1999_);
lean_dec_ref(v___y_1998_);
lean_dec(v___y_1997_);
lean_dec_ref(v___y_1996_);
return v_res_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1(lean_object* v___y_2004_){
_start:
{
lean_object* v_doc_2006_; lean_object* v___x_2007_; 
v_doc_2006_ = lean_ctor_get(v___y_2004_, 1);
lean_inc_ref(v_doc_2006_);
v___x_2007_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2007_, 0, v_doc_2006_);
return v___x_2007_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1___boxed(lean_object* v___y_2008_, lean_object* v___y_2009_){
_start:
{
lean_object* v_res_2010_; 
v_res_2010_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1(v___y_2008_);
lean_dec_ref(v___y_2008_);
return v_res_2010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg(lean_object* v_mvarId_2011_, lean_object* v_x_2012_, lean_object* v___y_2013_, lean_object* v___y_2014_, lean_object* v___y_2015_, lean_object* v___y_2016_){
_start:
{
lean_object* v___x_2018_; 
v___x_2018_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2011_, v_x_2012_, v___y_2013_, v___y_2014_, v___y_2015_, v___y_2016_);
if (lean_obj_tag(v___x_2018_) == 0)
{
lean_object* v_a_2019_; lean_object* v___x_2021_; uint8_t v_isShared_2022_; uint8_t v_isSharedCheck_2026_; 
v_a_2019_ = lean_ctor_get(v___x_2018_, 0);
v_isSharedCheck_2026_ = !lean_is_exclusive(v___x_2018_);
if (v_isSharedCheck_2026_ == 0)
{
v___x_2021_ = v___x_2018_;
v_isShared_2022_ = v_isSharedCheck_2026_;
goto v_resetjp_2020_;
}
else
{
lean_inc(v_a_2019_);
lean_dec(v___x_2018_);
v___x_2021_ = lean_box(0);
v_isShared_2022_ = v_isSharedCheck_2026_;
goto v_resetjp_2020_;
}
v_resetjp_2020_:
{
lean_object* v___x_2024_; 
if (v_isShared_2022_ == 0)
{
v___x_2024_ = v___x_2021_;
goto v_reusejp_2023_;
}
else
{
lean_object* v_reuseFailAlloc_2025_; 
v_reuseFailAlloc_2025_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2025_, 0, v_a_2019_);
v___x_2024_ = v_reuseFailAlloc_2025_;
goto v_reusejp_2023_;
}
v_reusejp_2023_:
{
return v___x_2024_;
}
}
}
else
{
lean_object* v_a_2027_; lean_object* v___x_2029_; uint8_t v_isShared_2030_; uint8_t v_isSharedCheck_2034_; 
v_a_2027_ = lean_ctor_get(v___x_2018_, 0);
v_isSharedCheck_2034_ = !lean_is_exclusive(v___x_2018_);
if (v_isSharedCheck_2034_ == 0)
{
v___x_2029_ = v___x_2018_;
v_isShared_2030_ = v_isSharedCheck_2034_;
goto v_resetjp_2028_;
}
else
{
lean_inc(v_a_2027_);
lean_dec(v___x_2018_);
v___x_2029_ = lean_box(0);
v_isShared_2030_ = v_isSharedCheck_2034_;
goto v_resetjp_2028_;
}
v_resetjp_2028_:
{
lean_object* v___x_2032_; 
if (v_isShared_2030_ == 0)
{
v___x_2032_ = v___x_2029_;
goto v_reusejp_2031_;
}
else
{
lean_object* v_reuseFailAlloc_2033_; 
v_reuseFailAlloc_2033_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2033_, 0, v_a_2027_);
v___x_2032_ = v_reuseFailAlloc_2033_;
goto v_reusejp_2031_;
}
v_reusejp_2031_:
{
return v___x_2032_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg___boxed(lean_object* v_mvarId_2035_, lean_object* v_x_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_, lean_object* v___y_2041_){
_start:
{
lean_object* v_res_2042_; 
v_res_2042_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg(v_mvarId_2035_, v_x_2036_, v___y_2037_, v___y_2038_, v___y_2039_, v___y_2040_);
lean_dec(v___y_2040_);
lean_dec_ref(v___y_2039_);
lean_dec(v___y_2038_);
lean_dec_ref(v___y_2037_);
return v_res_2042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5(lean_object* v_00_u03b1_2043_, lean_object* v_mvarId_2044_, lean_object* v_x_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg(v_mvarId_2044_, v_x_2045_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___boxed(lean_object* v_00_u03b1_2052_, lean_object* v_mvarId_2053_, lean_object* v_x_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v___y_2057_, lean_object* v___y_2058_, lean_object* v___y_2059_){
_start:
{
lean_object* v_res_2060_; 
v_res_2060_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5(v_00_u03b1_2052_, v_mvarId_2053_, v_x_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_);
lean_dec(v___y_2058_);
lean_dec_ref(v___y_2057_);
lean_dec(v___y_2056_);
lean_dec_ref(v___y_2055_);
return v_res_2060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0(lean_object* v_x_2061_, lean_object* v_e_2062_, lean_object* v___y_2063_, lean_object* v___y_2064_, lean_object* v___y_2065_, lean_object* v___y_2066_){
_start:
{
lean_object* v___x_2068_; 
v___x_2068_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v_e_2062_, v___y_2063_, v___y_2064_, v___y_2065_, v___y_2066_);
return v___x_2068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0___boxed(lean_object* v_x_2069_, lean_object* v_e_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_, lean_object* v___y_2075_){
_start:
{
lean_object* v_res_2076_; 
v_res_2076_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__0(v_x_2069_, v_e_2070_, v___y_2071_, v___y_2072_, v___y_2073_, v___y_2074_);
lean_dec(v___y_2074_);
lean_dec_ref(v___y_2073_);
lean_dec(v___y_2072_);
lean_dec_ref(v___y_2071_);
lean_dec_ref(v_x_2069_);
return v_res_2076_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1(lean_object* v_mvarId_2077_, lean_object* v_x_2078_){
_start:
{
lean_object* v_mvarId_2079_; uint8_t v___x_2080_; 
v_mvarId_2079_ = lean_ctor_get(v_x_2078_, 3);
v___x_2080_ = l_Lean_instBEqMVarId_beq(v_mvarId_2079_, v_mvarId_2077_);
return v___x_2080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1___boxed(lean_object* v_mvarId_2081_, lean_object* v_x_2082_){
_start:
{
uint8_t v_res_2083_; lean_object* v_r_2084_; 
v_res_2083_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1(v_mvarId_2081_, v_x_2082_);
lean_dec_ref(v_x_2082_);
lean_dec(v_mvarId_2081_);
v_r_2084_ = lean_box(v_res_2083_);
return v_r_2084_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0(void){
_start:
{
lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; 
v___x_2085_ = lean_box(0);
v___x_2086_ = lean_unsigned_to_nat(16u);
v___x_2087_ = lean_mk_array(v___x_2086_, v___x_2085_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2(lean_object* v___x_2088_, uint8_t v___x_2089_, lean_object* v_val_2090_, lean_object* v_meta_2091_, lean_object* v_pos_2092_, lean_object* v___y_2093_, lean_object* v_snd_2094_, lean_object* v_snd_2095_, lean_object* v_mvarId_2096_, lean_object* v_parentDecl_x3f_2097_, uint8_t v_useAfter_2098_, lean_object* v_stx_2099_, lean_object* v_masterToken_2100_, lean_object* v___y_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_){
_start:
{
lean_object* v___x_2106_; lean_object* v___x_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; lean_object* v___y_2112_; 
v___x_2106_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___closed__0);
lean_inc(v___x_2088_);
v___x_2107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2107_, 0, v___x_2088_);
lean_ctor_set(v___x_2107_, 1, v___x_2106_);
v___x_2108_ = lean_mk_empty_array_with_capacity(v___x_2088_);
lean_dec(v___x_2088_);
v___x_2109_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_2109_, 0, v___x_2107_);
lean_ctor_set(v___x_2109_, 1, v___x_2108_);
lean_ctor_set_uint8(v___x_2109_, sizeof(void*)*2, v___x_2089_);
v___x_2110_ = lean_st_mk_ref(v___x_2109_);
if (v_useAfter_2098_ == 0)
{
lean_object* v___x_2117_; 
lean_dec(v_stx_2099_);
v___x_2117_ = lean_box(0);
v___y_2112_ = v___x_2117_;
goto v___jp_2111_;
}
else
{
lean_object* v___x_2118_; 
v___x_2118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2118_, 0, v_stx_2099_);
v___y_2112_ = v___x_2118_;
goto v___jp_2111_;
}
v___jp_2111_:
{
lean_object* v___x_2113_; lean_object* v___x_2114_; lean_object* v___x_2115_; lean_object* v___x_2116_; 
lean_inc_ref(v_val_2090_);
v___x_2113_ = lp_mathlib_Lean_SubExpr_GoalsLocation_fvarId_x3f(v_val_2090_);
v___x_2114_ = lp_mathlib_Lean_SubExpr_GoalsLocation_pos(v_val_2090_);
lean_inc_ref(v_masterToken_2100_);
v___x_2115_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2115_, 0, v_meta_2091_);
lean_ctor_set(v___x_2115_, 1, v_pos_2092_);
lean_ctor_set(v___x_2115_, 2, v___y_2093_);
lean_ctor_set(v___x_2115_, 3, v___y_2112_);
lean_ctor_set(v___x_2115_, 4, v_masterToken_2100_);
lean_ctor_set(v___x_2115_, 5, v_snd_2094_);
lean_ctor_set(v___x_2115_, 6, v_snd_2095_);
lean_ctor_set(v___x_2115_, 7, v_mvarId_2096_);
lean_ctor_set(v___x_2115_, 8, v___x_2113_);
lean_ctor_set(v___x_2115_, 9, v___x_2114_);
v___x_2116_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions(v_val_2090_, v_parentDecl_x3f_2097_, v_masterToken_2100_, v___x_2115_, v___x_2110_, v___y_2101_, v___y_2102_, v___y_2103_, v___y_2104_);
lean_dec(v___x_2110_);
lean_dec_ref_known(v___x_2115_, 10);
return v___x_2116_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___boxed(lean_object** _args){
lean_object* v___x_2119_ = _args[0];
lean_object* v___x_2120_ = _args[1];
lean_object* v_val_2121_ = _args[2];
lean_object* v_meta_2122_ = _args[3];
lean_object* v_pos_2123_ = _args[4];
lean_object* v___y_2124_ = _args[5];
lean_object* v_snd_2125_ = _args[6];
lean_object* v_snd_2126_ = _args[7];
lean_object* v_mvarId_2127_ = _args[8];
lean_object* v_parentDecl_x3f_2128_ = _args[9];
lean_object* v_useAfter_2129_ = _args[10];
lean_object* v_stx_2130_ = _args[11];
lean_object* v_masterToken_2131_ = _args[12];
lean_object* v___y_2132_ = _args[13];
lean_object* v___y_2133_ = _args[14];
lean_object* v___y_2134_ = _args[15];
lean_object* v___y_2135_ = _args[16];
lean_object* v___y_2136_ = _args[17];
_start:
{
uint8_t v___x_14635__boxed_2137_; uint8_t v_useAfter_14641__boxed_2138_; lean_object* v_res_2139_; 
v___x_14635__boxed_2137_ = lean_unbox(v___x_2120_);
v_useAfter_14641__boxed_2138_ = lean_unbox(v_useAfter_2129_);
v_res_2139_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2(v___x_2119_, v___x_14635__boxed_2137_, v_val_2121_, v_meta_2122_, v_pos_2123_, v___y_2124_, v_snd_2125_, v_snd_2126_, v_mvarId_2127_, v_parentDecl_x3f_2128_, v_useAfter_14641__boxed_2138_, v_stx_2130_, v_masterToken_2131_, v___y_2132_, v___y_2133_, v___y_2134_, v___y_2135_);
lean_dec(v___y_2135_);
lean_dec_ref(v___y_2134_);
lean_dec(v___y_2133_);
lean_dec_ref(v___y_2132_);
return v_res_2139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1(lean_object* v_fvars_2140_, lean_object* v_k_2141_, lean_object* v_otherFvars_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_){
_start:
{
lean_object* v___x_2149_; lean_object* v___x_2150_; 
v___x_2149_ = l_Array_append___redArg(v_fvars_2140_, v_otherFvars_2142_);
lean_inc(v___y_2147_);
lean_inc_ref(v___y_2146_);
lean_inc(v___y_2145_);
lean_inc_ref(v___y_2144_);
v___x_2150_ = lean_apply_7(v_k_2141_, v___x_2149_, v___y_2143_, v___y_2144_, v___y_2145_, v___y_2146_, v___y_2147_, lean_box(0));
return v___x_2150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1___boxed(lean_object* v_fvars_2151_, lean_object* v_k_2152_, lean_object* v_otherFvars_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_){
_start:
{
lean_object* v_res_2160_; 
v_res_2160_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1(v_fvars_2151_, v_k_2152_, v_otherFvars_2153_, v___y_2154_, v___y_2155_, v___y_2156_, v___y_2157_, v___y_2158_);
lean_dec(v___y_2158_);
lean_dec_ref(v___y_2157_);
lean_dec(v___y_2156_);
lean_dec_ref(v___y_2155_);
lean_dec_ref(v_otherFvars_2153_);
return v_res_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0(lean_object* v_k_2161_, lean_object* v_b_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_){
_start:
{
lean_object* v___x_2168_; 
lean_inc(v___y_2166_);
lean_inc_ref(v___y_2165_);
lean_inc(v___y_2164_);
lean_inc_ref(v___y_2163_);
v___x_2168_ = lean_apply_6(v_k_2161_, v_b_2162_, v___y_2163_, v___y_2164_, v___y_2165_, v___y_2166_, lean_box(0));
return v___x_2168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0___boxed(lean_object* v_k_2169_, lean_object* v_b_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_){
_start:
{
lean_object* v_res_2176_; 
v_res_2176_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0(v_k_2169_, v_b_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
lean_dec(v___y_2172_);
lean_dec_ref(v___y_2171_);
return v_res_2176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg(lean_object* v_name_2177_, uint8_t v_bi_2178_, lean_object* v_type_2179_, lean_object* v_k_2180_, uint8_t v_kind_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_){
_start:
{
lean_object* v___f_2187_; lean_object* v___x_2188_; 
v___f_2187_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_2187_, 0, v_k_2180_);
v___x_2188_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_2177_, v_bi_2178_, v_type_2179_, v___f_2187_, v_kind_2181_, v___y_2182_, v___y_2183_, v___y_2184_, v___y_2185_);
if (lean_obj_tag(v___x_2188_) == 0)
{
lean_object* v_a_2189_; lean_object* v___x_2191_; uint8_t v_isShared_2192_; uint8_t v_isSharedCheck_2196_; 
v_a_2189_ = lean_ctor_get(v___x_2188_, 0);
v_isSharedCheck_2196_ = !lean_is_exclusive(v___x_2188_);
if (v_isSharedCheck_2196_ == 0)
{
v___x_2191_ = v___x_2188_;
v_isShared_2192_ = v_isSharedCheck_2196_;
goto v_resetjp_2190_;
}
else
{
lean_inc(v_a_2189_);
lean_dec(v___x_2188_);
v___x_2191_ = lean_box(0);
v_isShared_2192_ = v_isSharedCheck_2196_;
goto v_resetjp_2190_;
}
v_resetjp_2190_:
{
lean_object* v___x_2194_; 
if (v_isShared_2192_ == 0)
{
v___x_2194_ = v___x_2191_;
goto v_reusejp_2193_;
}
else
{
lean_object* v_reuseFailAlloc_2195_; 
v_reuseFailAlloc_2195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2195_, 0, v_a_2189_);
v___x_2194_ = v_reuseFailAlloc_2195_;
goto v_reusejp_2193_;
}
v_reusejp_2193_:
{
return v___x_2194_;
}
}
}
else
{
lean_object* v_a_2197_; lean_object* v___x_2199_; uint8_t v_isShared_2200_; uint8_t v_isSharedCheck_2204_; 
v_a_2197_ = lean_ctor_get(v___x_2188_, 0);
v_isSharedCheck_2204_ = !lean_is_exclusive(v___x_2188_);
if (v_isSharedCheck_2204_ == 0)
{
v___x_2199_ = v___x_2188_;
v_isShared_2200_ = v_isSharedCheck_2204_;
goto v_resetjp_2198_;
}
else
{
lean_inc(v_a_2197_);
lean_dec(v___x_2188_);
v___x_2199_ = lean_box(0);
v_isShared_2200_ = v_isSharedCheck_2204_;
goto v_resetjp_2198_;
}
v_resetjp_2198_:
{
lean_object* v___x_2202_; 
if (v_isShared_2200_ == 0)
{
v___x_2202_ = v___x_2199_;
goto v_reusejp_2201_;
}
else
{
lean_object* v_reuseFailAlloc_2203_; 
v_reuseFailAlloc_2203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2203_, 0, v_a_2197_);
v___x_2202_ = v_reuseFailAlloc_2203_;
goto v_reusejp_2201_;
}
v_reusejp_2201_:
{
return v___x_2202_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___boxed(lean_object* v_name_2205_, lean_object* v_bi_2206_, lean_object* v_type_2207_, lean_object* v_k_2208_, lean_object* v_kind_2209_, lean_object* v___y_2210_, lean_object* v___y_2211_, lean_object* v___y_2212_, lean_object* v___y_2213_, lean_object* v___y_2214_){
_start:
{
uint8_t v_bi_boxed_2215_; uint8_t v_kind_boxed_2216_; lean_object* v_res_2217_; 
v_bi_boxed_2215_ = lean_unbox(v_bi_2206_);
v_kind_boxed_2216_ = lean_unbox(v_kind_2209_);
v_res_2217_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg(v_name_2205_, v_bi_boxed_2215_, v_type_2207_, v_k_2208_, v_kind_boxed_2216_, v___y_2210_, v___y_2211_, v___y_2212_, v___y_2213_);
lean_dec(v___y_2213_);
lean_dec_ref(v___y_2212_);
lean_dec(v___y_2211_);
lean_dec_ref(v___y_2210_);
return v_res_2217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg(lean_object* v_name_2218_, lean_object* v_type_2219_, lean_object* v_val_2220_, lean_object* v_k_2221_, uint8_t v_nondep_2222_, uint8_t v_kind_2223_, lean_object* v___y_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_){
_start:
{
lean_object* v___f_2229_; lean_object* v___x_2230_; 
v___f_2229_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_2229_, 0, v_k_2221_);
v___x_2230_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_2218_, v_type_2219_, v_val_2220_, v___f_2229_, v_nondep_2222_, v_kind_2223_, v___y_2224_, v___y_2225_, v___y_2226_, v___y_2227_);
if (lean_obj_tag(v___x_2230_) == 0)
{
lean_object* v_a_2231_; lean_object* v___x_2233_; uint8_t v_isShared_2234_; uint8_t v_isSharedCheck_2238_; 
v_a_2231_ = lean_ctor_get(v___x_2230_, 0);
v_isSharedCheck_2238_ = !lean_is_exclusive(v___x_2230_);
if (v_isSharedCheck_2238_ == 0)
{
v___x_2233_ = v___x_2230_;
v_isShared_2234_ = v_isSharedCheck_2238_;
goto v_resetjp_2232_;
}
else
{
lean_inc(v_a_2231_);
lean_dec(v___x_2230_);
v___x_2233_ = lean_box(0);
v_isShared_2234_ = v_isSharedCheck_2238_;
goto v_resetjp_2232_;
}
v_resetjp_2232_:
{
lean_object* v___x_2236_; 
if (v_isShared_2234_ == 0)
{
v___x_2236_ = v___x_2233_;
goto v_reusejp_2235_;
}
else
{
lean_object* v_reuseFailAlloc_2237_; 
v_reuseFailAlloc_2237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2237_, 0, v_a_2231_);
v___x_2236_ = v_reuseFailAlloc_2237_;
goto v_reusejp_2235_;
}
v_reusejp_2235_:
{
return v___x_2236_;
}
}
}
else
{
lean_object* v_a_2239_; lean_object* v___x_2241_; uint8_t v_isShared_2242_; uint8_t v_isSharedCheck_2246_; 
v_a_2239_ = lean_ctor_get(v___x_2230_, 0);
v_isSharedCheck_2246_ = !lean_is_exclusive(v___x_2230_);
if (v_isSharedCheck_2246_ == 0)
{
v___x_2241_ = v___x_2230_;
v_isShared_2242_ = v_isSharedCheck_2246_;
goto v_resetjp_2240_;
}
else
{
lean_inc(v_a_2239_);
lean_dec(v___x_2230_);
v___x_2241_ = lean_box(0);
v_isShared_2242_ = v_isSharedCheck_2246_;
goto v_resetjp_2240_;
}
v_resetjp_2240_:
{
lean_object* v___x_2244_; 
if (v_isShared_2242_ == 0)
{
v___x_2244_ = v___x_2241_;
goto v_reusejp_2243_;
}
else
{
lean_object* v_reuseFailAlloc_2245_; 
v_reuseFailAlloc_2245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2245_, 0, v_a_2239_);
v___x_2244_ = v_reuseFailAlloc_2245_;
goto v_reusejp_2243_;
}
v_reusejp_2243_:
{
return v___x_2244_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg___boxed(lean_object* v_name_2247_, lean_object* v_type_2248_, lean_object* v_val_2249_, lean_object* v_k_2250_, lean_object* v_nondep_2251_, lean_object* v_kind_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_){
_start:
{
uint8_t v_nondep_boxed_2258_; uint8_t v_kind_boxed_2259_; lean_object* v_res_2260_; 
v_nondep_boxed_2258_ = lean_unbox(v_nondep_2251_);
v_kind_boxed_2259_ = lean_unbox(v_kind_2252_);
v_res_2260_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg(v_name_2247_, v_type_2248_, v_val_2249_, v_k_2250_, v_nondep_boxed_2258_, v_kind_boxed_2259_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_);
lean_dec(v___y_2256_);
lean_dec_ref(v___y_2255_);
lean_dec(v___y_2254_);
lean_dec_ref(v___y_2253_);
return v_res_2260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0(lean_object* v_fvars_2261_, lean_object* v_k_2262_, lean_object* v_body_2263_, lean_object* v_x_2264_, lean_object* v___y_2265_, lean_object* v___y_2266_, lean_object* v___y_2267_, lean_object* v___y_2268_){
_start:
{
lean_object* v___x_2270_; lean_object* v___x_2271_; 
v___x_2270_ = lean_array_push(v_fvars_2261_, v_x_2264_);
lean_inc(v___y_2268_);
lean_inc_ref(v___y_2267_);
lean_inc(v___y_2266_);
lean_inc_ref(v___y_2265_);
v___x_2271_ = lean_apply_7(v_k_2262_, v___x_2270_, v_body_2263_, v___y_2265_, v___y_2266_, v___y_2267_, v___y_2268_, lean_box(0));
return v___x_2271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0___boxed(lean_object* v_fvars_2272_, lean_object* v_k_2273_, lean_object* v_body_2274_, lean_object* v_x_2275_, lean_object* v___y_2276_, lean_object* v___y_2277_, lean_object* v___y_2278_, lean_object* v___y_2279_, lean_object* v___y_2280_){
_start:
{
lean_object* v_res_2281_; 
v_res_2281_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0(v_fvars_2272_, v_k_2273_, v_body_2274_, v_x_2275_, v___y_2276_, v___y_2277_, v___y_2278_, v___y_2279_);
lean_dec(v___y_2279_);
lean_dec_ref(v___y_2278_);
lean_dec(v___y_2277_);
lean_dec_ref(v___y_2276_);
return v_res_2281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1(lean_object* v_fvars_2282_, lean_object* v_k_2283_, lean_object* v_b_2284_, lean_object* v_x_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_, lean_object* v___y_2289_){
_start:
{
lean_object* v___x_2291_; lean_object* v___x_2292_; 
v___x_2291_ = lean_array_push(v_fvars_2282_, v_x_2285_);
lean_inc(v___y_2289_);
lean_inc_ref(v___y_2288_);
lean_inc(v___y_2287_);
lean_inc_ref(v___y_2286_);
v___x_2292_ = lean_apply_7(v_k_2283_, v___x_2291_, v_b_2284_, v___y_2286_, v___y_2287_, v___y_2288_, v___y_2289_, lean_box(0));
return v___x_2292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1___boxed(lean_object* v_fvars_2293_, lean_object* v_k_2294_, lean_object* v_b_2295_, lean_object* v_x_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_){
_start:
{
lean_object* v_res_2302_; 
v_res_2302_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1(v_fvars_2293_, v_k_2294_, v_b_2295_, v_x_2296_, v___y_2297_, v___y_2298_, v___y_2299_, v___y_2300_);
lean_dec(v___y_2300_);
lean_dec_ref(v___y_2299_);
lean_dec(v___y_2298_);
lean_dec_ref(v___y_2297_);
return v_res_2302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(lean_object* v_msg_2303_, lean_object* v___y_2304_, lean_object* v___y_2305_, lean_object* v___y_2306_, lean_object* v___y_2307_){
_start:
{
lean_object* v_ref_2309_; lean_object* v___x_2310_; lean_object* v_a_2311_; lean_object* v___x_2313_; uint8_t v_isShared_2314_; uint8_t v_isSharedCheck_2319_; 
v_ref_2309_ = lean_ctor_get(v___y_2306_, 5);
v___x_2310_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15_spec__20_spec__22(v_msg_2303_, v___y_2304_, v___y_2305_, v___y_2306_, v___y_2307_);
v_a_2311_ = lean_ctor_get(v___x_2310_, 0);
v_isSharedCheck_2319_ = !lean_is_exclusive(v___x_2310_);
if (v_isSharedCheck_2319_ == 0)
{
v___x_2313_ = v___x_2310_;
v_isShared_2314_ = v_isSharedCheck_2319_;
goto v_resetjp_2312_;
}
else
{
lean_inc(v_a_2311_);
lean_dec(v___x_2310_);
v___x_2313_ = lean_box(0);
v_isShared_2314_ = v_isSharedCheck_2319_;
goto v_resetjp_2312_;
}
v_resetjp_2312_:
{
lean_object* v___x_2315_; lean_object* v___x_2317_; 
lean_inc(v_ref_2309_);
v___x_2315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2315_, 0, v_ref_2309_);
lean_ctor_set(v___x_2315_, 1, v_a_2311_);
if (v_isShared_2314_ == 0)
{
lean_ctor_set_tag(v___x_2313_, 1);
lean_ctor_set(v___x_2313_, 0, v___x_2315_);
v___x_2317_ = v___x_2313_;
goto v_reusejp_2316_;
}
else
{
lean_object* v_reuseFailAlloc_2318_; 
v_reuseFailAlloc_2318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2318_, 0, v___x_2315_);
v___x_2317_ = v_reuseFailAlloc_2318_;
goto v_reusejp_2316_;
}
v_reusejp_2316_:
{
return v___x_2317_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg___boxed(lean_object* v_msg_2320_, lean_object* v___y_2321_, lean_object* v___y_2322_, lean_object* v___y_2323_, lean_object* v___y_2324_, lean_object* v___y_2325_){
_start:
{
lean_object* v_res_2326_; 
v_res_2326_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(v_msg_2320_, v___y_2321_, v___y_2322_, v___y_2323_, v___y_2324_);
lean_dec(v___y_2324_);
lean_dec_ref(v___y_2323_);
lean_dec(v___y_2322_);
lean_dec_ref(v___y_2321_);
return v_res_2326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg(lean_object* v_k_2327_, lean_object* v_fvars_2328_, lean_object* v_n_2329_, lean_object* v_e_2330_, lean_object* v___y_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_){
_start:
{
lean_object* v_c_2337_; lean_object* v_e_2338_; lean_object* v_n_2350_; lean_object* v_y_2351_; lean_object* v_b_2352_; uint8_t v_c_2353_; lean_object* v___x_2358_; uint8_t v___x_2359_; 
v___x_2358_ = lean_unsigned_to_nat(3u);
v___x_2359_ = lean_nat_dec_eq(v_n_2329_, v___x_2358_);
if (v___x_2359_ == 0)
{
lean_object* v___x_2360_; uint8_t v___x_2361_; 
v___x_2360_ = lean_unsigned_to_nat(0u);
v___x_2361_ = lean_nat_dec_eq(v_n_2329_, v___x_2360_);
if (v___x_2361_ == 0)
{
lean_object* v___x_2362_; uint8_t v___x_2363_; 
v___x_2362_ = lean_unsigned_to_nat(1u);
v___x_2363_ = lean_nat_dec_eq(v_n_2329_, v___x_2362_);
if (v___x_2363_ == 0)
{
lean_object* v___x_2364_; uint8_t v___x_2365_; 
v___x_2364_ = lean_unsigned_to_nat(2u);
v___x_2365_ = lean_nat_dec_eq(v_n_2329_, v___x_2364_);
if (v___x_2365_ == 0)
{
if (lean_obj_tag(v_e_2330_) == 10)
{
lean_object* v_expr_2366_; 
v_expr_2366_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_expr_2366_);
lean_dec_ref_known(v_e_2330_, 2);
v_e_2330_ = v_expr_2366_;
goto _start;
}
else
{
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_k_2327_);
v_c_2337_ = v_n_2329_;
v_e_2338_ = v_e_2330_;
goto v___jp_2336_;
}
}
else
{
lean_dec(v_n_2329_);
switch(lean_obj_tag(v_e_2330_))
{
case 8:
{
lean_object* v_declName_2368_; lean_object* v_type_2369_; lean_object* v_value_2370_; lean_object* v_body_2371_; lean_object* v___f_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; uint8_t v___x_2375_; lean_object* v___x_2376_; 
v_declName_2368_ = lean_ctor_get(v_e_2330_, 0);
lean_inc(v_declName_2368_);
v_type_2369_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_type_2369_);
v_value_2370_ = lean_ctor_get(v_e_2330_, 2);
lean_inc_ref(v_value_2370_);
v_body_2371_ = lean_ctor_get(v_e_2330_, 3);
lean_inc_ref(v_body_2371_);
lean_dec_ref_known(v_e_2330_, 4);
lean_inc_ref(v_fvars_2328_);
v___f_2372_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_2372_, 0, v_fvars_2328_);
lean_closure_set(v___f_2372_, 1, v_k_2327_);
lean_closure_set(v___f_2372_, 2, v_body_2371_);
v___x_2373_ = lean_expr_instantiate_rev(v_type_2369_, v_fvars_2328_);
lean_dec_ref(v_type_2369_);
v___x_2374_ = lean_expr_instantiate_rev(v_value_2370_, v_fvars_2328_);
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_value_2370_);
v___x_2375_ = 0;
v___x_2376_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg(v_declName_2368_, v___x_2373_, v___x_2374_, v___f_2372_, v___x_2363_, v___x_2375_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2376_;
}
case 10:
{
lean_object* v_expr_2377_; 
v_expr_2377_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_expr_2377_);
lean_dec_ref_known(v_e_2330_, 2);
v_n_2329_ = v___x_2364_;
v_e_2330_ = v_expr_2377_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_k_2327_);
v_c_2337_ = v___x_2364_;
v_e_2338_ = v_e_2330_;
goto v___jp_2336_;
}
}
}
}
else
{
lean_dec(v_n_2329_);
switch(lean_obj_tag(v_e_2330_))
{
case 5:
{
lean_object* v_arg_2379_; lean_object* v___x_2380_; 
v_arg_2379_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_arg_2379_);
lean_dec_ref_known(v_e_2330_, 2);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2380_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_arg_2379_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2380_;
}
case 6:
{
lean_object* v_binderName_2381_; lean_object* v_binderType_2382_; lean_object* v_body_2383_; uint8_t v_binderInfo_2384_; 
v_binderName_2381_ = lean_ctor_get(v_e_2330_, 0);
lean_inc(v_binderName_2381_);
v_binderType_2382_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_binderType_2382_);
v_body_2383_ = lean_ctor_get(v_e_2330_, 2);
lean_inc_ref(v_body_2383_);
v_binderInfo_2384_ = lean_ctor_get_uint8(v_e_2330_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2330_, 3);
v_n_2350_ = v_binderName_2381_;
v_y_2351_ = v_binderType_2382_;
v_b_2352_ = v_body_2383_;
v_c_2353_ = v_binderInfo_2384_;
goto v___jp_2349_;
}
case 7:
{
lean_object* v_binderName_2385_; lean_object* v_binderType_2386_; lean_object* v_body_2387_; uint8_t v_binderInfo_2388_; 
v_binderName_2385_ = lean_ctor_get(v_e_2330_, 0);
lean_inc(v_binderName_2385_);
v_binderType_2386_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_binderType_2386_);
v_body_2387_ = lean_ctor_get(v_e_2330_, 2);
lean_inc_ref(v_body_2387_);
v_binderInfo_2388_ = lean_ctor_get_uint8(v_e_2330_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_2330_, 3);
v_n_2350_ = v_binderName_2385_;
v_y_2351_ = v_binderType_2386_;
v_b_2352_ = v_body_2387_;
v_c_2353_ = v_binderInfo_2388_;
goto v___jp_2349_;
}
case 8:
{
lean_object* v_value_2389_; lean_object* v___x_2390_; 
v_value_2389_ = lean_ctor_get(v_e_2330_, 2);
lean_inc_ref(v_value_2389_);
lean_dec_ref_known(v_e_2330_, 4);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2390_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_value_2389_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2390_;
}
case 10:
{
lean_object* v_expr_2391_; 
v_expr_2391_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_expr_2391_);
lean_dec_ref_known(v_e_2330_, 2);
v_n_2329_ = v___x_2362_;
v_e_2330_ = v_expr_2391_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_k_2327_);
v_c_2337_ = v___x_2362_;
v_e_2338_ = v_e_2330_;
goto v___jp_2336_;
}
}
}
}
else
{
lean_dec(v_n_2329_);
switch(lean_obj_tag(v_e_2330_))
{
case 5:
{
lean_object* v_fn_2393_; lean_object* v___x_2394_; 
v_fn_2393_ = lean_ctor_get(v_e_2330_, 0);
lean_inc_ref(v_fn_2393_);
lean_dec_ref_known(v_e_2330_, 2);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2394_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_fn_2393_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2394_;
}
case 6:
{
lean_object* v_binderType_2395_; lean_object* v___x_2396_; 
v_binderType_2395_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_binderType_2395_);
lean_dec_ref_known(v_e_2330_, 3);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2396_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_binderType_2395_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2396_;
}
case 7:
{
lean_object* v_binderType_2397_; lean_object* v___x_2398_; 
v_binderType_2397_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_binderType_2397_);
lean_dec_ref_known(v_e_2330_, 3);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2398_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_binderType_2397_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2398_;
}
case 8:
{
lean_object* v_type_2399_; lean_object* v___x_2400_; 
v_type_2399_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_type_2399_);
lean_dec_ref_known(v_e_2330_, 4);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2400_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_type_2399_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2400_;
}
case 11:
{
lean_object* v_struct_2401_; lean_object* v___x_2402_; 
v_struct_2401_ = lean_ctor_get(v_e_2330_, 2);
lean_inc_ref(v_struct_2401_);
lean_dec_ref_known(v_e_2330_, 3);
lean_inc(v___y_2334_);
lean_inc_ref(v___y_2333_);
lean_inc(v___y_2332_);
lean_inc_ref(v___y_2331_);
v___x_2402_ = lean_apply_7(v_k_2327_, v_fvars_2328_, v_struct_2401_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_, lean_box(0));
return v___x_2402_;
}
case 10:
{
lean_object* v_expr_2403_; 
v_expr_2403_ = lean_ctor_get(v_e_2330_, 1);
lean_inc_ref(v_expr_2403_);
lean_dec_ref_known(v_e_2330_, 2);
v_n_2329_ = v___x_2360_;
v_e_2330_ = v_expr_2403_;
goto _start;
}
default: 
{
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_k_2327_);
v_c_2337_ = v___x_2360_;
v_e_2338_ = v_e_2330_;
goto v___jp_2336_;
}
}
}
}
else
{
lean_object* v___x_2405_; lean_object* v___x_2406_; 
lean_dec_ref(v_e_2330_);
lean_dec(v_n_2329_);
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_k_2327_);
v___x_2405_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__5);
v___x_2406_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(v___x_2405_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2406_;
}
v___jp_2336_:
{
lean_object* v___x_2339_; lean_object* v___x_2340_; lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; 
v___x_2339_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__1);
v___x_2340_ = l_Nat_reprFast(v_c_2337_);
v___x_2341_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2341_, 0, v___x_2340_);
v___x_2342_ = l_Lean_MessageData_ofFormat(v___x_2341_);
v___x_2343_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2343_, 0, v___x_2339_);
lean_ctor_set(v___x_2343_, 1, v___x_2342_);
v___x_2344_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3, &lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3_once, _init_lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11_spec__15___redArg___closed__3);
v___x_2345_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2345_, 0, v___x_2343_);
lean_ctor_set(v___x_2345_, 1, v___x_2344_);
v___x_2346_ = l_Lean_MessageData_ofExpr(v_e_2338_);
v___x_2347_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2347_, 0, v___x_2345_);
lean_ctor_set(v___x_2347_, 1, v___x_2346_);
v___x_2348_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(v___x_2347_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2348_;
}
v___jp_2349_:
{
lean_object* v___f_2354_; lean_object* v___x_2355_; uint8_t v___x_2356_; lean_object* v___x_2357_; 
lean_inc_ref(v_fvars_2328_);
v___f_2354_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___lam__1___boxed), 9, 3);
lean_closure_set(v___f_2354_, 0, v_fvars_2328_);
lean_closure_set(v___f_2354_, 1, v_k_2327_);
lean_closure_set(v___f_2354_, 2, v_b_2352_);
v___x_2355_ = lean_expr_instantiate_rev(v_y_2351_, v_fvars_2328_);
lean_dec_ref(v_fvars_2328_);
lean_dec_ref(v_y_2351_);
v___x_2356_ = 0;
v___x_2357_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg(v_n_2350_, v_c_2353_, v___x_2355_, v___f_2354_, v___x_2356_, v___y_2331_, v___y_2332_, v___y_2333_, v___y_2334_);
return v___x_2357_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg___boxed(lean_object* v_k_2407_, lean_object* v_fvars_2408_, lean_object* v_n_2409_, lean_object* v_e_2410_, lean_object* v___y_2411_, lean_object* v___y_2412_, lean_object* v___y_2413_, lean_object* v___y_2414_, lean_object* v___y_2415_){
_start:
{
lean_object* v_res_2416_; 
v_res_2416_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg(v_k_2407_, v_fvars_2408_, v_n_2409_, v_e_2410_, v___y_2411_, v___y_2412_, v___y_2413_, v___y_2414_);
lean_dec(v___y_2414_);
lean_dec_ref(v___y_2413_);
lean_dec(v___y_2412_);
lean_dec_ref(v___y_2411_);
return v_res_2416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0___boxed(lean_object* v_k_2417_, lean_object* v_tail_2418_, lean_object* v_fvars_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_){
_start:
{
lean_object* v_res_2426_; 
v_res_2426_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0(v_k_2417_, v_tail_2418_, v_fvars_2419_, v___y_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_);
lean_dec(v___y_2424_);
lean_dec_ref(v___y_2423_);
lean_dec(v___y_2422_);
lean_dec_ref(v___y_2421_);
return v_res_2426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(lean_object* v_k_2427_, lean_object* v_fvars_2428_, lean_object* v_x_2429_, lean_object* v_x_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_){
_start:
{
if (lean_obj_tag(v_x_2429_) == 0)
{
lean_object* v___x_2436_; lean_object* v___x_2437_; 
v___x_2436_ = lean_expr_instantiate_rev(v_x_2430_, v_fvars_2428_);
lean_dec_ref(v_x_2430_);
lean_inc(v___y_2434_);
lean_inc_ref(v___y_2433_);
lean_inc(v___y_2432_);
lean_inc_ref(v___y_2431_);
v___x_2437_ = lean_apply_7(v_k_2427_, v_fvars_2428_, v___x_2436_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_, lean_box(0));
return v___x_2437_;
}
else
{
lean_object* v_head_2438_; lean_object* v_tail_2439_; lean_object* v___x_2440_; uint8_t v___x_2441_; 
v_head_2438_ = lean_ctor_get(v_x_2429_, 0);
lean_inc(v_head_2438_);
v_tail_2439_ = lean_ctor_get(v_x_2429_, 1);
lean_inc(v_tail_2439_);
lean_dec_ref_known(v_x_2429_, 2);
v___x_2440_ = lean_unsigned_to_nat(3u);
v___x_2441_ = lean_nat_dec_eq(v_head_2438_, v___x_2440_);
if (v___x_2441_ == 0)
{
lean_object* v___f_2442_; lean_object* v___x_2443_; 
v___f_2442_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0___boxed), 9, 2);
lean_closure_set(v___f_2442_, 0, v_k_2427_);
lean_closure_set(v___f_2442_, 1, v_tail_2439_);
v___x_2443_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg(v___f_2442_, v_fvars_2428_, v_head_2438_, v_x_2430_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_);
return v___x_2443_;
}
else
{
lean_object* v___x_2444_; lean_object* v___x_2445_; 
lean_dec(v_head_2438_);
v___x_2444_ = lean_expr_instantiate_rev(v_x_2430_, v_fvars_2428_);
lean_dec_ref(v_x_2430_);
lean_inc(v___y_2434_);
lean_inc_ref(v___y_2433_);
lean_inc(v___y_2432_);
lean_inc_ref(v___y_2431_);
v___x_2445_ = lean_infer_type(v___x_2444_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_);
if (lean_obj_tag(v___x_2445_) == 0)
{
lean_object* v_a_2446_; lean_object* v___f_2447_; lean_object* v___x_2448_; 
v_a_2446_ = lean_ctor_get(v___x_2445_, 0);
lean_inc(v_a_2446_);
lean_dec_ref_known(v___x_2445_, 1);
v___f_2447_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__1___boxed), 9, 2);
lean_closure_set(v___f_2447_, 0, v_fvars_2428_);
lean_closure_set(v___f_2447_, 1, v_k_2427_);
v___x_2448_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0));
v_k_2427_ = v___f_2447_;
v_fvars_2428_ = v___x_2448_;
v_x_2429_ = v_tail_2439_;
v_x_2430_ = v_a_2446_;
goto _start;
}
else
{
lean_object* v_a_2450_; lean_object* v___x_2452_; uint8_t v_isShared_2453_; uint8_t v_isSharedCheck_2457_; 
lean_dec(v_tail_2439_);
lean_dec_ref(v_fvars_2428_);
lean_dec_ref(v_k_2427_);
v_a_2450_ = lean_ctor_get(v___x_2445_, 0);
v_isSharedCheck_2457_ = !lean_is_exclusive(v___x_2445_);
if (v_isSharedCheck_2457_ == 0)
{
v___x_2452_ = v___x_2445_;
v_isShared_2453_ = v_isSharedCheck_2457_;
goto v_resetjp_2451_;
}
else
{
lean_inc(v_a_2450_);
lean_dec(v___x_2445_);
v___x_2452_ = lean_box(0);
v_isShared_2453_ = v_isSharedCheck_2457_;
goto v_resetjp_2451_;
}
v_resetjp_2451_:
{
lean_object* v___x_2455_; 
if (v_isShared_2453_ == 0)
{
v___x_2455_ = v___x_2452_;
goto v_reusejp_2454_;
}
else
{
lean_object* v_reuseFailAlloc_2456_; 
v_reuseFailAlloc_2456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2456_, 0, v_a_2450_);
v___x_2455_ = v_reuseFailAlloc_2456_;
goto v_reusejp_2454_;
}
v_reusejp_2454_:
{
return v___x_2455_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___lam__0(lean_object* v_k_2458_, lean_object* v_tail_2459_, lean_object* v_fvars_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_, lean_object* v___y_2464_, lean_object* v___y_2465_){
_start:
{
lean_object* v___x_2467_; 
v___x_2467_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(v_k_2458_, v_fvars_2460_, v_tail_2459_, v___y_2461_, v___y_2462_, v___y_2463_, v___y_2464_, v___y_2465_);
return v___x_2467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg___boxed(lean_object* v_k_2468_, lean_object* v_fvars_2469_, lean_object* v_x_2470_, lean_object* v_x_2471_, lean_object* v___y_2472_, lean_object* v___y_2473_, lean_object* v___y_2474_, lean_object* v___y_2475_, lean_object* v___y_2476_){
_start:
{
lean_object* v_res_2477_; 
v_res_2477_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(v_k_2468_, v_fvars_2469_, v_x_2470_, v_x_2471_, v___y_2472_, v___y_2473_, v___y_2474_, v___y_2475_);
lean_dec(v___y_2475_);
lean_dec_ref(v___y_2474_);
lean_dec(v___y_2473_);
lean_dec_ref(v___y_2472_);
return v_res_2477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg(lean_object* v_visit_2478_, lean_object* v_p_2479_, lean_object* v_root_2480_, lean_object* v___y_2481_, lean_object* v___y_2482_, lean_object* v___y_2483_, lean_object* v___y_2484_){
_start:
{
lean_object* v___x_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; lean_object* v___x_2489_; 
v___x_2486_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00__private_Mathlib_Tactic_ClickSuggestions_0__Mathlib_Tactic_ClickSuggestions_viewKAbstractSubExpr_x27___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__1_spec__4_spec__11___redArg___closed__0));
v___x_2487_ = l_Lean_SubExpr_Pos_toArray(v_p_2479_);
v___x_2488_ = lean_array_to_list(v___x_2487_);
v___x_2489_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(v_visit_2478_, v___x_2486_, v___x_2488_, v_root_2480_, v___y_2481_, v___y_2482_, v___y_2483_, v___y_2484_);
return v___x_2489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg___boxed(lean_object* v_visit_2490_, lean_object* v_p_2491_, lean_object* v_root_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_, lean_object* v___y_2497_){
_start:
{
lean_object* v_res_2498_; 
v_res_2498_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg(v_visit_2490_, v_p_2491_, v_root_2492_, v___y_2493_, v___y_2494_, v___y_2495_, v___y_2496_);
lean_dec(v___y_2496_);
lean_dec_ref(v___y_2495_);
lean_dec(v___y_2494_);
lean_dec_ref(v___y_2493_);
lean_dec(v_p_2491_);
return v_res_2498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3(lean_object* v_c_2499_, lean_object* v_props_2500_, lean_object* v_children_2501_){
_start:
{
lean_object* v_toModule_2502_; lean_object* v_export_2503_; lean_object* v_javascript_2504_; uint64_t v___x_2505_; lean_object* v___x_2506_; lean_object* v___x_2507_; 
v_toModule_2502_ = lean_ctor_get(v_c_2499_, 0);
v_export_2503_ = lean_ctor_get(v_c_2499_, 1);
v_javascript_2504_ = lean_ctor_get(v_toModule_2502_, 0);
v___x_2505_ = lean_string_hash(v_javascript_2504_);
v___x_2506_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveMessageProps_enc_00___x40_ProofWidgets_Component_Basic_2277670097____hygCtx___hyg_1____boxed), 2, 1);
lean_closure_set(v___x_2506_, 0, v_props_2500_);
lean_inc_ref(v_export_2503_);
v___x_2507_ = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(v___x_2507_, 0, v_export_2503_);
lean_ctor_set(v___x_2507_, 1, v___x_2506_);
lean_ctor_set(v___x_2507_, 2, v_children_2501_);
lean_ctor_set_uint64(v___x_2507_, sizeof(void*)*3, v___x_2505_);
return v___x_2507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3___boxed(lean_object* v_c_2508_, lean_object* v_props_2509_, lean_object* v_children_2510_){
_start:
{
lean_object* v_res_2511_; 
v_res_2511_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3(v_c_2508_, v_props_2509_, v_children_2510_);
lean_dec_ref(v_c_2508_);
return v_res_2511_;
}
}
static lean_object* _init_lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6(void){
_start:
{
lean_object* v___x_2521_; lean_object* v___x_2522_; lean_object* v___x_2523_; lean_object* v___x_2524_; 
v___x_2521_ = ((lean_object*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__5));
v___x_2522_ = lean_unsigned_to_nat(2u);
v___x_2523_ = lean_mk_empty_array_with_capacity(v___x_2522_);
v___x_2524_ = lean_array_push(v___x_2523_, v___x_2521_);
return v___x_2524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0(lean_object* v_val_2525_, lean_object* v_val_2526_, lean_object* v___y_2527_, lean_object* v_snd_2528_, lean_object* v_k_2529_, lean_object* v___y_2530_){
_start:
{
lean_object* v___x_2532_; lean_object* v___x_2533_; lean_object* v_fileName_2534_; lean_object* v_fileMap_2535_; lean_object* v_options_2536_; lean_object* v_currRecDepth_2537_; lean_object* v_maxRecDepth_2538_; lean_object* v_ref_2539_; lean_object* v_currNamespace_2540_; lean_object* v_openDecls_2541_; lean_object* v_initHeartbeats_2542_; lean_object* v_maxHeartbeats_2543_; lean_object* v_quotContext_2544_; lean_object* v_currMacroScope_2545_; uint8_t v_diag_2546_; uint8_t v_suppressElabErrors_2547_; lean_object* v_inheritedTraceOptions_2548_; lean_object* v_cancelTk_2549_; lean_object* v___x_2550_; lean_object* v___x_2551_; lean_object* v___x_2552_; 
v___x_2532_ = lean_st_mk_ref(v_val_2525_);
v___x_2533_ = lean_st_mk_ref(v_val_2526_);
v_fileName_2534_ = lean_ctor_get(v___y_2527_, 0);
v_fileMap_2535_ = lean_ctor_get(v___y_2527_, 1);
v_options_2536_ = lean_ctor_get(v___y_2527_, 2);
v_currRecDepth_2537_ = lean_ctor_get(v___y_2527_, 3);
v_maxRecDepth_2538_ = lean_ctor_get(v___y_2527_, 4);
v_ref_2539_ = lean_ctor_get(v___y_2527_, 5);
v_currNamespace_2540_ = lean_ctor_get(v___y_2527_, 6);
v_openDecls_2541_ = lean_ctor_get(v___y_2527_, 7);
v_initHeartbeats_2542_ = lean_ctor_get(v___y_2527_, 8);
v_maxHeartbeats_2543_ = lean_ctor_get(v___y_2527_, 9);
v_quotContext_2544_ = lean_ctor_get(v___y_2527_, 10);
v_currMacroScope_2545_ = lean_ctor_get(v___y_2527_, 11);
v_diag_2546_ = lean_ctor_get_uint8(v___y_2527_, sizeof(void*)*14);
v_suppressElabErrors_2547_ = lean_ctor_get_uint8(v___y_2527_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2548_ = lean_ctor_get(v___y_2527_, 13);
v_cancelTk_2549_ = lean_ctor_get(v_snd_2528_, 1);
lean_inc_ref(v_cancelTk_2549_);
v___x_2550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2550_, 0, v_cancelTk_2549_);
lean_inc_ref(v_inheritedTraceOptions_2548_);
lean_inc(v_currMacroScope_2545_);
lean_inc(v_quotContext_2544_);
lean_inc(v_maxHeartbeats_2543_);
lean_inc(v_initHeartbeats_2542_);
lean_inc(v_openDecls_2541_);
lean_inc(v_currNamespace_2540_);
lean_inc(v_ref_2539_);
lean_inc(v_maxRecDepth_2538_);
lean_inc(v_currRecDepth_2537_);
lean_inc_ref(v_options_2536_);
lean_inc_ref(v_fileMap_2535_);
lean_inc_ref(v_fileName_2534_);
v___x_2551_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2551_, 0, v_fileName_2534_);
lean_ctor_set(v___x_2551_, 1, v_fileMap_2535_);
lean_ctor_set(v___x_2551_, 2, v_options_2536_);
lean_ctor_set(v___x_2551_, 3, v_currRecDepth_2537_);
lean_ctor_set(v___x_2551_, 4, v_maxRecDepth_2538_);
lean_ctor_set(v___x_2551_, 5, v_ref_2539_);
lean_ctor_set(v___x_2551_, 6, v_currNamespace_2540_);
lean_ctor_set(v___x_2551_, 7, v_openDecls_2541_);
lean_ctor_set(v___x_2551_, 8, v_initHeartbeats_2542_);
lean_ctor_set(v___x_2551_, 9, v_maxHeartbeats_2543_);
lean_ctor_set(v___x_2551_, 10, v_quotContext_2544_);
lean_ctor_set(v___x_2551_, 11, v_currMacroScope_2545_);
lean_ctor_set(v___x_2551_, 12, v___x_2550_);
lean_ctor_set(v___x_2551_, 13, v_inheritedTraceOptions_2548_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*14, v_diag_2546_);
lean_ctor_set_uint8(v___x_2551_, sizeof(void*)*14 + 1, v_suppressElabErrors_2547_);
lean_inc(v___x_2532_);
lean_inc(v___x_2533_);
lean_inc_ref(v___y_2530_);
lean_inc_ref(v_snd_2528_);
v___x_2552_ = lean_apply_6(v_k_2529_, v_snd_2528_, v___y_2530_, v___x_2533_, v___x_2551_, v___x_2532_, lean_box(0));
if (lean_obj_tag(v___x_2552_) == 0)
{
lean_object* v_a_2553_; lean_object* v___x_2554_; lean_object* v___x_2555_; 
lean_dec_ref(v_snd_2528_);
v_a_2553_ = lean_ctor_get(v___x_2552_, 0);
lean_inc(v_a_2553_);
lean_dec_ref_known(v___x_2552_, 1);
v___x_2554_ = lean_st_ref_get(v___x_2533_);
lean_dec(v___x_2533_);
lean_dec(v___x_2554_);
v___x_2555_ = lean_st_ref_get(v___x_2532_);
lean_dec(v___x_2532_);
lean_dec(v___x_2555_);
return v_a_2553_;
}
else
{
lean_object* v_a_2556_; 
lean_dec(v___x_2533_);
lean_dec(v___x_2532_);
v_a_2556_ = lean_ctor_get(v___x_2552_, 0);
lean_inc(v_a_2556_);
lean_dec_ref_known(v___x_2552_, 1);
if (lean_obj_tag(v_a_2556_) == 1)
{
lean_object* v_id_2557_; lean_object* v___x_2558_; uint8_t v___x_2559_; 
v_id_2557_ = lean_ctor_get(v_a_2556_, 0);
lean_inc(v_id_2557_);
lean_dec_ref_known(v_a_2556_, 2);
v___x_2558_ = l_Lean_interruptExceptionId;
v___x_2559_ = l_Lean_instBEqInternalExceptionId_beq(v_id_2557_, v___x_2558_);
lean_dec(v_id_2557_);
if (v___x_2559_ == 0)
{
lean_object* v___x_2560_; 
lean_dec_ref(v_snd_2528_);
v___x_2560_ = lean_box(0);
return v___x_2560_;
}
else
{
lean_object* v___x_2561_; lean_object* v___x_2562_; 
v___x_2561_ = ((lean_object*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__1));
v___x_2562_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_snd_2528_, v___x_2561_);
return v___x_2562_;
}
}
else
{
lean_object* v___x_2563_; lean_object* v___x_2564_; lean_object* v___x_2565_; lean_object* v___x_2566_; lean_object* v___x_2567_; lean_object* v___x_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___x_2571_; lean_object* v___x_2572_; 
v___x_2563_ = l_Lean_Exception_toMessageData(v_a_2556_);
v___x_2564_ = l_Lean_Server_WithRpcRef_mk___redArg(v___x_2563_);
v___x_2565_ = ((lean_object*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__2));
v___x_2566_ = ((lean_object*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__3));
v___x_2567_ = lp_proofwidgets_ProofWidgets_InteractiveMessage;
v___x_2568_ = lp_mathlib_ProofWidgets_Html_ofComponent___at___00ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3_spec__3(v___x_2567_, v___x_2564_, v___x_2566_);
v___x_2569_ = lean_obj_once(&lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6, &lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6_once, _init_lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__6);
v___x_2570_ = lean_array_push(v___x_2569_, v___x_2568_);
v___x_2571_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2571_, 0, v___x_2565_);
lean_ctor_set(v___x_2571_, 1, v___x_2566_);
lean_ctor_set(v___x_2571_, 2, v___x_2570_);
v___x_2572_ = lp_proofwidgets_ProofWidgets_RefreshToken_update(v_snd_2528_, v___x_2571_);
return v___x_2572_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___boxed(lean_object* v_val_2573_, lean_object* v_val_2574_, lean_object* v___y_2575_, lean_object* v_snd_2576_, lean_object* v_k_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_){
_start:
{
lean_object* v_res_2580_; 
v_res_2580_ = lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0(v_val_2573_, v_val_2574_, v___y_2575_, v_snd_2576_, v_k_2577_, v___y_2578_);
lean_dec_ref(v___y_2578_);
lean_dec_ref(v___y_2575_);
return v_res_2580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3(lean_object* v_initial_2581_, lean_object* v_k_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v___x_2588_; lean_object* v_fst_2589_; lean_object* v_snd_2590_; lean_object* v___x_2591_; lean_object* v___x_2592_; lean_object* v___f_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; 
v___x_2588_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent(v_initial_2581_);
v_fst_2589_ = lean_ctor_get(v___x_2588_, 0);
lean_inc(v_fst_2589_);
v_snd_2590_ = lean_ctor_get(v___x_2588_, 1);
lean_inc(v_snd_2590_);
lean_dec_ref(v___x_2588_);
v___x_2591_ = lean_st_ref_get(v___y_2584_);
v___x_2592_ = lean_st_ref_get(v___y_2586_);
lean_inc_ref(v___y_2583_);
lean_inc_ref(v___y_2585_);
v___f_2593_ = lean_alloc_closure((void*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___boxed), 7, 6);
lean_closure_set(v___f_2593_, 0, v___x_2592_);
lean_closure_set(v___f_2593_, 1, v___x_2591_);
lean_closure_set(v___f_2593_, 2, v___y_2585_);
lean_closure_set(v___f_2593_, 3, v_snd_2590_);
lean_closure_set(v___f_2593_, 4, v_k_2582_);
lean_closure_set(v___f_2593_, 5, v___y_2583_);
v___x_2594_ = lean_unsigned_to_nat(9u);
v___x_2595_ = lean_io_as_task(v___f_2593_, v___x_2594_);
lean_dec_ref(v___x_2595_);
v___x_2596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2596_, 0, v_fst_2589_);
return v___x_2596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___boxed(lean_object* v_initial_2597_, lean_object* v_k_2598_, lean_object* v___y_2599_, lean_object* v___y_2600_, lean_object* v___y_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_){
_start:
{
lean_object* v_res_2604_; 
v_res_2604_ = lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3(v_initial_2597_, v_k_2598_, v___y_2599_, v___y_2600_, v___y_2601_, v___y_2602_);
lean_dec(v___y_2602_);
lean_dec_ref(v___y_2601_);
lean_dec(v___y_2600_);
lean_dec_ref(v___y_2599_);
return v_res_2604_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14(void){
_start:
{
lean_object* v___x_2628_; lean_object* v___x_2629_; lean_object* v___x_2630_; lean_object* v___x_2631_; 
v___x_2628_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__11));
v___x_2629_ = lean_unsigned_to_nat(4u);
v___x_2630_ = lean_mk_empty_array_with_capacity(v___x_2629_);
v___x_2631_ = lean_array_push(v___x_2630_, v___x_2628_);
return v___x_2631_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17(void){
_start:
{
lean_object* v___x_2635_; lean_object* v___x_2636_; lean_object* v___x_2637_; lean_object* v___x_2638_; 
v___x_2635_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__16));
v___x_2636_ = lean_unsigned_to_nat(2u);
v___x_2637_ = lean_mk_empty_array_with_capacity(v___x_2636_);
v___x_2638_ = lean_array_push(v___x_2637_, v___x_2635_);
return v___x_2638_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3(lean_object* v___x_2639_, lean_object* v___x_2640_, uint8_t v___x_2641_, lean_object* v_val_2642_, lean_object* v_meta_2643_, lean_object* v_pos_2644_, lean_object* v___y_2645_, lean_object* v_mvarId_2646_, lean_object* v_parentDecl_x3f_2647_, uint8_t v_useAfter_2648_, lean_object* v_stx_2649_, uint8_t v___x_2650_, lean_object* v_loc_2651_, lean_object* v___f_2652_, lean_object* v___y_2653_, lean_object* v___y_2654_, lean_object* v___y_2655_, lean_object* v___y_2656_){
_start:
{
lean_object* v___x_2658_; lean_object* v_fst_2659_; lean_object* v_snd_2660_; lean_object* v___x_2661_; lean_object* v_fst_2662_; lean_object* v_snd_2663_; lean_object* v___x_2665_; uint8_t v_isShared_2666_; uint8_t v_isSharedCheck_2731_; 
lean_inc_ref(v___x_2639_);
v___x_2658_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent(v___x_2639_);
v_fst_2659_ = lean_ctor_get(v___x_2658_, 0);
lean_inc(v_fst_2659_);
v_snd_2660_ = lean_ctor_get(v___x_2658_, 1);
lean_inc(v_snd_2660_);
lean_dec_ref(v___x_2658_);
v___x_2661_ = lp_proofwidgets_ProofWidgets_mkRefreshComponent(v___x_2639_);
v_fst_2662_ = lean_ctor_get(v___x_2661_, 0);
v_snd_2663_ = lean_ctor_get(v___x_2661_, 1);
v_isSharedCheck_2731_ = !lean_is_exclusive(v___x_2661_);
if (v_isSharedCheck_2731_ == 0)
{
v___x_2665_ = v___x_2661_;
v_isShared_2666_ = v_isSharedCheck_2731_;
goto v_resetjp_2664_;
}
else
{
lean_inc(v_snd_2663_);
lean_inc(v_fst_2662_);
lean_dec(v___x_2661_);
v___x_2665_ = lean_box(0);
v_isShared_2666_ = v_isSharedCheck_2731_;
goto v_resetjp_2664_;
}
v_resetjp_2664_:
{
lean_object* v___x_2667_; lean_object* v___x_2668_; lean_object* v___f_2669_; lean_object* v_targetHtml_2671_; lean_object* v___y_2672_; lean_object* v___y_2673_; lean_object* v___y_2674_; lean_object* v___y_2675_; 
v___x_2667_ = lean_box(v___x_2641_);
v___x_2668_ = lean_box(v_useAfter_2648_);
lean_inc_ref(v_val_2642_);
lean_inc(v___x_2640_);
v___f_2669_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__2___boxed), 18, 12);
lean_closure_set(v___f_2669_, 0, v___x_2640_);
lean_closure_set(v___f_2669_, 1, v___x_2667_);
lean_closure_set(v___f_2669_, 2, v_val_2642_);
lean_closure_set(v___f_2669_, 3, v_meta_2643_);
lean_closure_set(v___f_2669_, 4, v_pos_2644_);
lean_closure_set(v___f_2669_, 5, v___y_2645_);
lean_closure_set(v___f_2669_, 6, v_snd_2660_);
lean_closure_set(v___f_2669_, 7, v_snd_2663_);
lean_closure_set(v___f_2669_, 8, v_mvarId_2646_);
lean_closure_set(v___f_2669_, 9, v_parentDecl_x3f_2647_);
lean_closure_set(v___f_2669_, 10, v___x_2668_);
lean_closure_set(v___f_2669_, 11, v_stx_2649_);
if (lean_obj_tag(v_loc_2651_) == 0)
{
lean_object* v_a_2709_; lean_object* v___x_2710_; lean_object* v___x_2711_; 
lean_dec_ref(v___f_2652_);
lean_dec_ref(v_val_2642_);
v_a_2709_ = lean_ctor_get(v_loc_2651_, 0);
lean_inc(v_a_2709_);
lean_dec_ref_known(v_loc_2651_, 1);
v___x_2710_ = l_Lean_Expr_fvar___override(v_a_2709_);
v___x_2711_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_exprToHtml(v___x_2710_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2711_) == 0)
{
lean_object* v_a_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2716_; lean_object* v___x_2717_; 
v_a_2712_ = lean_ctor_get(v___x_2711_, 0);
lean_inc(v_a_2712_);
lean_dec_ref_known(v___x_2711_, 1);
v___x_2713_ = ((lean_object*)(lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3___lam__0___closed__2));
v___x_2714_ = lean_mk_empty_array_with_capacity(v___x_2640_);
lean_dec(v___x_2640_);
v___x_2715_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__17);
v___x_2716_ = lean_array_push(v___x_2715_, v_a_2712_);
v___x_2717_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2717_, 0, v___x_2713_);
lean_ctor_set(v___x_2717_, 1, v___x_2714_);
lean_ctor_set(v___x_2717_, 2, v___x_2716_);
v_targetHtml_2671_ = v___x_2717_;
v___y_2672_ = v___y_2653_;
v___y_2673_ = v___y_2654_;
v___y_2674_ = v___y_2655_;
v___y_2675_ = v___y_2656_;
goto v___jp_2670_;
}
else
{
lean_dec_ref(v___f_2669_);
lean_del_object(v___x_2665_);
lean_dec(v_fst_2662_);
lean_dec(v_fst_2659_);
lean_dec(v___x_2640_);
return v___x_2711_;
}
}
else
{
lean_object* v___x_2718_; 
lean_dec_ref(v_loc_2651_);
lean_dec(v___x_2640_);
lean_inc_ref(v_val_2642_);
v___x_2718_ = lp_mathlib_Lean_SubExpr_GoalsLocation_rootExpr(v_val_2642_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
if (lean_obj_tag(v___x_2718_) == 0)
{
lean_object* v_a_2719_; lean_object* v___x_2720_; lean_object* v___x_2721_; 
v_a_2719_ = lean_ctor_get(v___x_2718_, 0);
lean_inc(v_a_2719_);
lean_dec_ref_known(v___x_2718_, 1);
v___x_2720_ = lp_mathlib_Lean_SubExpr_GoalsLocation_pos(v_val_2642_);
lean_dec_ref(v_val_2642_);
v___x_2721_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg(v___f_2652_, v___x_2720_, v_a_2719_, v___y_2653_, v___y_2654_, v___y_2655_, v___y_2656_);
lean_dec(v___x_2720_);
if (lean_obj_tag(v___x_2721_) == 0)
{
lean_object* v_a_2722_; 
v_a_2722_ = lean_ctor_get(v___x_2721_, 0);
lean_inc(v_a_2722_);
lean_dec_ref_known(v___x_2721_, 1);
v_targetHtml_2671_ = v_a_2722_;
v___y_2672_ = v___y_2653_;
v___y_2673_ = v___y_2654_;
v___y_2674_ = v___y_2655_;
v___y_2675_ = v___y_2656_;
goto v___jp_2670_;
}
else
{
lean_dec_ref(v___f_2669_);
lean_del_object(v___x_2665_);
lean_dec(v_fst_2662_);
lean_dec(v_fst_2659_);
return v___x_2721_;
}
}
else
{
lean_object* v_a_2723_; lean_object* v___x_2725_; uint8_t v_isShared_2726_; uint8_t v_isSharedCheck_2730_; 
lean_dec_ref(v___f_2669_);
lean_del_object(v___x_2665_);
lean_dec(v_fst_2662_);
lean_dec(v_fst_2659_);
lean_dec_ref(v___f_2652_);
lean_dec_ref(v_val_2642_);
v_a_2723_ = lean_ctor_get(v___x_2718_, 0);
v_isSharedCheck_2730_ = !lean_is_exclusive(v___x_2718_);
if (v_isSharedCheck_2730_ == 0)
{
v___x_2725_ = v___x_2718_;
v_isShared_2726_ = v_isSharedCheck_2730_;
goto v_resetjp_2724_;
}
else
{
lean_inc(v_a_2723_);
lean_dec(v___x_2718_);
v___x_2725_ = lean_box(0);
v_isShared_2726_ = v_isSharedCheck_2730_;
goto v_resetjp_2724_;
}
v_resetjp_2724_:
{
lean_object* v___x_2728_; 
if (v_isShared_2726_ == 0)
{
v___x_2728_ = v___x_2725_;
goto v_reusejp_2727_;
}
else
{
lean_object* v_reuseFailAlloc_2729_; 
v_reuseFailAlloc_2729_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2729_, 0, v_a_2723_);
v___x_2728_ = v_reuseFailAlloc_2729_;
goto v_reusejp_2727_;
}
v_reusejp_2727_:
{
return v___x_2728_;
}
}
}
}
v___jp_2670_:
{
lean_object* v___x_2676_; lean_object* v___x_2677_; lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2708_; 
v___x_2676_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__1));
v___x_2677_ = lp_mathlib_ProofWidgets_mkRefreshComponentM___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__3(v___x_2676_, v___f_2669_, v___y_2672_, v___y_2673_, v___y_2674_, v___y_2675_);
v_a_2678_ = lean_ctor_get(v___x_2677_, 0);
v_isSharedCheck_2708_ = !lean_is_exclusive(v___x_2677_);
if (v_isSharedCheck_2708_ == 0)
{
v___x_2680_ = v___x_2677_;
v_isShared_2681_ = v_isSharedCheck_2708_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_dec(v___x_2677_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2708_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2686_; 
v___x_2682_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__2));
v___x_2683_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__3));
v___x_2684_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_2684_, 0, v___x_2650_);
if (v_isShared_2666_ == 0)
{
lean_ctor_set(v___x_2665_, 1, v___x_2684_);
lean_ctor_set(v___x_2665_, 0, v___x_2683_);
v___x_2686_ = v___x_2665_;
goto v_reusejp_2685_;
}
else
{
lean_object* v_reuseFailAlloc_2707_; 
v_reuseFailAlloc_2707_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2707_, 0, v___x_2683_);
lean_ctor_set(v_reuseFailAlloc_2707_, 1, v___x_2684_);
v___x_2686_ = v_reuseFailAlloc_2707_;
goto v_reusejp_2685_;
}
v_reusejp_2685_:
{
lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___x_2692_; lean_object* v___x_2693_; lean_object* v___x_2694_; lean_object* v___x_2695_; lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; lean_object* v___x_2701_; lean_object* v___x_2702_; lean_object* v___x_2703_; lean_object* v___x_2705_; 
v___x_2687_ = lean_unsigned_to_nat(1u);
v___x_2688_ = lean_mk_empty_array_with_capacity(v___x_2687_);
v___x_2689_ = lean_array_push(v___x_2688_, v___x_2686_);
v___x_2690_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__4));
v___x_2691_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__9));
v___x_2692_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__13));
v___x_2693_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___closed__14);
v___x_2694_ = lean_array_push(v___x_2693_, v_targetHtml_2671_);
v___x_2695_ = lean_array_push(v___x_2694_, v___x_2692_);
v___x_2696_ = lean_array_push(v___x_2695_, v_fst_2659_);
v___x_2697_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2697_, 0, v___x_2690_);
lean_ctor_set(v___x_2697_, 1, v___x_2691_);
lean_ctor_set(v___x_2697_, 2, v___x_2696_);
v___x_2698_ = lean_unsigned_to_nat(3u);
v___x_2699_ = lean_mk_empty_array_with_capacity(v___x_2698_);
v___x_2700_ = lean_array_push(v___x_2699_, v___x_2697_);
v___x_2701_ = lean_array_push(v___x_2700_, v_fst_2662_);
v___x_2702_ = lean_array_push(v___x_2701_, v_a_2678_);
v___x_2703_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2703_, 0, v___x_2682_);
lean_ctor_set(v___x_2703_, 1, v___x_2689_);
lean_ctor_set(v___x_2703_, 2, v___x_2702_);
if (v_isShared_2681_ == 0)
{
lean_ctor_set(v___x_2680_, 0, v___x_2703_);
v___x_2705_ = v___x_2680_;
goto v_reusejp_2704_;
}
else
{
lean_object* v_reuseFailAlloc_2706_; 
v_reuseFailAlloc_2706_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2706_, 0, v___x_2703_);
v___x_2705_ = v_reuseFailAlloc_2706_;
goto v_reusejp_2704_;
}
v_reusejp_2704_:
{
return v___x_2705_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___boxed(lean_object** _args){
lean_object* v___x_2732_ = _args[0];
lean_object* v___x_2733_ = _args[1];
lean_object* v___x_2734_ = _args[2];
lean_object* v_val_2735_ = _args[3];
lean_object* v_meta_2736_ = _args[4];
lean_object* v_pos_2737_ = _args[5];
lean_object* v___y_2738_ = _args[6];
lean_object* v_mvarId_2739_ = _args[7];
lean_object* v_parentDecl_x3f_2740_ = _args[8];
lean_object* v_useAfter_2741_ = _args[9];
lean_object* v_stx_2742_ = _args[10];
lean_object* v___x_2743_ = _args[11];
lean_object* v_loc_2744_ = _args[12];
lean_object* v___f_2745_ = _args[13];
lean_object* v___y_2746_ = _args[14];
lean_object* v___y_2747_ = _args[15];
lean_object* v___y_2748_ = _args[16];
lean_object* v___y_2749_ = _args[17];
lean_object* v___y_2750_ = _args[18];
_start:
{
uint8_t v___x_15438__boxed_2751_; uint8_t v_useAfter_15442__boxed_2752_; uint8_t v___x_15444__boxed_2753_; lean_object* v_res_2754_; 
v___x_15438__boxed_2751_ = lean_unbox(v___x_2734_);
v_useAfter_15442__boxed_2752_ = lean_unbox(v_useAfter_2741_);
v___x_15444__boxed_2753_ = lean_unbox(v___x_2743_);
v_res_2754_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3(v___x_2732_, v___x_2733_, v___x_15438__boxed_2751_, v_val_2735_, v_meta_2736_, v_pos_2737_, v___y_2738_, v_mvarId_2739_, v_parentDecl_x3f_2740_, v_useAfter_15442__boxed_2752_, v_stx_2742_, v___x_15444__boxed_2753_, v_loc_2744_, v___f_2745_, v___y_2746_, v___y_2747_, v___y_2748_, v___y_2749_);
lean_dec(v___y_2749_);
lean_dec_ref(v___y_2748_);
lean_dec(v___y_2747_);
lean_dec_ref(v___y_2746_);
return v_res_2754_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4(lean_object* v_mvarId_2755_, lean_object* v___f_2756_, lean_object* v___y_2757_, lean_object* v___y_2758_, lean_object* v___y_2759_, lean_object* v___y_2760_){
_start:
{
uint8_t v_trackZetaDelta_2762_; lean_object* v_zetaDeltaSet_2763_; lean_object* v_lctx_2764_; lean_object* v_localInstances_2765_; lean_object* v_defEqCtx_x3f_2766_; lean_object* v_synthPendingDepth_2767_; lean_object* v_customCanUnfoldPredicate_x3f_2768_; uint8_t v_univApprox_2769_; uint8_t v_inTypeClassResolution_2770_; uint8_t v_cacheInferType_2771_; lean_object* v___x_2772_; lean_object* v___x_2774_; uint8_t v_isShared_2775_; uint8_t v_isSharedCheck_2783_; 
v_trackZetaDelta_2762_ = lean_ctor_get_uint8(v___y_2757_, sizeof(void*)*7);
v_zetaDeltaSet_2763_ = lean_ctor_get(v___y_2757_, 1);
lean_inc(v_zetaDeltaSet_2763_);
v_lctx_2764_ = lean_ctor_get(v___y_2757_, 2);
lean_inc_ref(v_lctx_2764_);
v_localInstances_2765_ = lean_ctor_get(v___y_2757_, 3);
lean_inc_ref(v_localInstances_2765_);
v_defEqCtx_x3f_2766_ = lean_ctor_get(v___y_2757_, 4);
lean_inc(v_defEqCtx_x3f_2766_);
v_synthPendingDepth_2767_ = lean_ctor_get(v___y_2757_, 5);
lean_inc(v_synthPendingDepth_2767_);
v_customCanUnfoldPredicate_x3f_2768_ = lean_ctor_get(v___y_2757_, 6);
lean_inc(v_customCanUnfoldPredicate_x3f_2768_);
v_univApprox_2769_ = lean_ctor_get_uint8(v___y_2757_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2770_ = lean_ctor_get_uint8(v___y_2757_, sizeof(void*)*7 + 2);
v_cacheInferType_2771_ = lean_ctor_get_uint8(v___y_2757_, sizeof(void*)*7 + 3);
v___x_2772_ = l_Lean_Meta_Context_config(v___y_2757_);
v_isSharedCheck_2783_ = !lean_is_exclusive(v___y_2757_);
if (v_isSharedCheck_2783_ == 0)
{
lean_object* v_unused_2784_; lean_object* v_unused_2785_; lean_object* v_unused_2786_; lean_object* v_unused_2787_; lean_object* v_unused_2788_; lean_object* v_unused_2789_; lean_object* v_unused_2790_; 
v_unused_2784_ = lean_ctor_get(v___y_2757_, 6);
lean_dec(v_unused_2784_);
v_unused_2785_ = lean_ctor_get(v___y_2757_, 5);
lean_dec(v_unused_2785_);
v_unused_2786_ = lean_ctor_get(v___y_2757_, 4);
lean_dec(v_unused_2786_);
v_unused_2787_ = lean_ctor_get(v___y_2757_, 3);
lean_dec(v_unused_2787_);
v_unused_2788_ = lean_ctor_get(v___y_2757_, 2);
lean_dec(v_unused_2788_);
v_unused_2789_ = lean_ctor_get(v___y_2757_, 1);
lean_dec(v_unused_2789_);
v_unused_2790_ = lean_ctor_get(v___y_2757_, 0);
lean_dec(v_unused_2790_);
v___x_2774_ = v___y_2757_;
v_isShared_2775_ = v_isSharedCheck_2783_;
goto v_resetjp_2773_;
}
else
{
lean_dec(v___y_2757_);
v___x_2774_ = lean_box(0);
v_isShared_2775_ = v_isSharedCheck_2783_;
goto v_resetjp_2773_;
}
v_resetjp_2773_:
{
lean_object* v___x_2776_; uint64_t v___x_2777_; lean_object* v___x_2778_; lean_object* v___x_2780_; 
v___x_2776_ = l_Lean_Elab_Term_setElabConfig(v___x_2772_);
v___x_2777_ = l___private_Lean_Meta_Basic_0__Lean_Meta_Config_toKey(v___x_2776_);
v___x_2778_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_2778_, 0, v___x_2776_);
lean_ctor_set_uint64(v___x_2778_, sizeof(void*)*1, v___x_2777_);
if (v_isShared_2775_ == 0)
{
lean_ctor_set(v___x_2774_, 0, v___x_2778_);
v___x_2780_ = v___x_2774_;
goto v_reusejp_2779_;
}
else
{
lean_object* v_reuseFailAlloc_2782_; 
v_reuseFailAlloc_2782_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2782_, 0, v___x_2778_);
lean_ctor_set(v_reuseFailAlloc_2782_, 1, v_zetaDeltaSet_2763_);
lean_ctor_set(v_reuseFailAlloc_2782_, 2, v_lctx_2764_);
lean_ctor_set(v_reuseFailAlloc_2782_, 3, v_localInstances_2765_);
lean_ctor_set(v_reuseFailAlloc_2782_, 4, v_defEqCtx_x3f_2766_);
lean_ctor_set(v_reuseFailAlloc_2782_, 5, v_synthPendingDepth_2767_);
lean_ctor_set(v_reuseFailAlloc_2782_, 6, v_customCanUnfoldPredicate_x3f_2768_);
lean_ctor_set_uint8(v_reuseFailAlloc_2782_, sizeof(void*)*7, v_trackZetaDelta_2762_);
lean_ctor_set_uint8(v_reuseFailAlloc_2782_, sizeof(void*)*7 + 1, v_univApprox_2769_);
lean_ctor_set_uint8(v_reuseFailAlloc_2782_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2770_);
lean_ctor_set_uint8(v_reuseFailAlloc_2782_, sizeof(void*)*7 + 3, v_cacheInferType_2771_);
v___x_2780_ = v_reuseFailAlloc_2782_;
goto v_reusejp_2779_;
}
v_reusejp_2779_:
{
lean_object* v___x_2781_; 
v___x_2781_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__5___redArg(v_mvarId_2755_, v___f_2756_, v___x_2780_, v___y_2758_, v___y_2759_, v___y_2760_);
lean_dec_ref(v___x_2780_);
return v___x_2781_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4___boxed(lean_object* v_mvarId_2791_, lean_object* v___f_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_, lean_object* v___y_2797_){
_start:
{
lean_object* v_res_2798_; 
v_res_2798_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4(v_mvarId_2791_, v___f_2792_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_);
lean_dec(v___y_2796_);
lean_dec_ref(v___y_2795_);
lean_dec(v___y_2794_);
return v_res_2798_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0(lean_object* v_a_2799_, lean_object* v_x_2800_){
_start:
{
if (lean_obj_tag(v_x_2800_) == 0)
{
uint8_t v___x_2801_; 
v___x_2801_ = 0;
return v___x_2801_;
}
else
{
lean_object* v_head_2802_; lean_object* v_tail_2803_; uint8_t v___x_2804_; 
v_head_2802_ = lean_ctor_get(v_x_2800_, 0);
v_tail_2803_ = lean_ctor_get(v_x_2800_, 1);
v___x_2804_ = l_Lean_instBEqMVarId_beq(v_a_2799_, v_head_2802_);
if (v___x_2804_ == 0)
{
v_x_2800_ = v_tail_2803_;
goto _start;
}
else
{
return v___x_2804_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0___boxed(lean_object* v_a_2806_, lean_object* v_x_2807_){
_start:
{
uint8_t v_res_2808_; lean_object* v_r_2809_; 
v_res_2808_ = lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0(v_a_2806_, v_x_2807_);
lean_dec(v_x_2807_);
lean_dec(v_a_2806_);
v_r_2809_ = lean_box(v_res_2808_);
return v_r_2809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2(lean_object* v_val_2810_, lean_object* v_x_2811_){
_start:
{
if (lean_obj_tag(v_x_2811_) == 0)
{
lean_object* v___x_2812_; 
v___x_2812_ = lean_box(0);
return v___x_2812_;
}
else
{
lean_object* v_head_2813_; lean_object* v_tail_2814_; lean_object* v___y_2816_; uint8_t v_useAfter_2821_; 
v_head_2813_ = lean_ctor_get(v_x_2811_, 0);
v_tail_2814_ = lean_ctor_get(v_x_2811_, 1);
v_useAfter_2821_ = lean_ctor_get_uint8(v_head_2813_, sizeof(void*)*3);
if (v_useAfter_2821_ == 0)
{
lean_object* v_tacticInfo_2822_; lean_object* v_goalsBefore_2823_; 
v_tacticInfo_2822_ = lean_ctor_get(v_head_2813_, 1);
v_goalsBefore_2823_ = lean_ctor_get(v_tacticInfo_2822_, 2);
v___y_2816_ = v_goalsBefore_2823_;
goto v___jp_2815_;
}
else
{
lean_object* v_tacticInfo_2824_; lean_object* v_goalsAfter_2825_; 
v_tacticInfo_2824_ = lean_ctor_get(v_head_2813_, 1);
v_goalsAfter_2825_ = lean_ctor_get(v_tacticInfo_2824_, 4);
v___y_2816_ = v_goalsAfter_2825_;
goto v___jp_2815_;
}
v___jp_2815_:
{
lean_object* v_mvarId_2817_; uint8_t v___x_2818_; 
v_mvarId_2817_ = lean_ctor_get(v_val_2810_, 0);
v___x_2818_ = lp_mathlib_List_elem___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__0(v_mvarId_2817_, v___y_2816_);
if (v___x_2818_ == 0)
{
v_x_2811_ = v_tail_2814_;
goto _start;
}
else
{
lean_object* v___x_2820_; 
lean_inc(v_head_2813_);
v___x_2820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2820_, 0, v_head_2813_);
return v___x_2820_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2___boxed(lean_object* v_val_2826_, lean_object* v_x_2827_){
_start:
{
lean_object* v_res_2828_; 
v_res_2828_ = lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2(v_val_2826_, v_x_2827_);
lean_dec(v_x_2827_);
lean_dec_ref(v_val_2826_);
return v_res_2828_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5(lean_object* v___y_2844_, lean_object* v_goals_2845_, lean_object* v_pos_2846_, lean_object* v___f_2847_, lean_object* v___y_2848_){
_start:
{
if (lean_obj_tag(v___y_2844_) == 1)
{
lean_object* v_val_2850_; lean_object* v___x_2851_; lean_object* v_loc_2852_; 
v_val_2850_ = lean_ctor_get(v___y_2844_, 0);
lean_inc(v_val_2850_);
lean_dec_ref_known(v___y_2844_, 1);
v___x_2851_ = lp_mathlib_Lean_Server_RequestM_readDoc___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__1(v___y_2848_);
v_loc_2852_ = lean_ctor_get(v_val_2850_, 1);
lean_inc_ref(v_loc_2852_);
if (lean_obj_tag(v_loc_2852_) == 2)
{
lean_object* v___x_2854_; uint8_t v_isShared_2855_; uint8_t v_isSharedCheck_2860_; 
lean_dec_ref_known(v_loc_2852_, 2);
lean_dec(v_val_2850_);
lean_dec_ref(v___f_2847_);
lean_dec_ref(v_pos_2846_);
v_isSharedCheck_2860_ = !lean_is_exclusive(v___x_2851_);
if (v_isSharedCheck_2860_ == 0)
{
lean_object* v_unused_2861_; 
v_unused_2861_ = lean_ctor_get(v___x_2851_, 0);
lean_dec(v_unused_2861_);
v___x_2854_ = v___x_2851_;
v_isShared_2855_ = v_isSharedCheck_2860_;
goto v_resetjp_2853_;
}
else
{
lean_dec(v___x_2851_);
v___x_2854_ = lean_box(0);
v_isShared_2855_ = v_isSharedCheck_2860_;
goto v_resetjp_2853_;
}
v_resetjp_2853_:
{
lean_object* v___x_2856_; lean_object* v___x_2858_; 
v___x_2856_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__1));
if (v_isShared_2855_ == 0)
{
lean_ctor_set(v___x_2854_, 0, v___x_2856_);
v___x_2858_ = v___x_2854_;
goto v_reusejp_2857_;
}
else
{
lean_object* v_reuseFailAlloc_2859_; 
v_reuseFailAlloc_2859_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2859_, 0, v___x_2856_);
v___x_2858_ = v_reuseFailAlloc_2859_;
goto v_reusejp_2857_;
}
v_reusejp_2857_:
{
return v___x_2858_;
}
}
}
else
{
lean_object* v_a_2862_; lean_object* v___x_2864_; uint8_t v_isShared_2865_; uint8_t v_isSharedCheck_2940_; 
v_a_2862_ = lean_ctor_get(v___x_2851_, 0);
v_isSharedCheck_2940_ = !lean_is_exclusive(v___x_2851_);
if (v_isSharedCheck_2940_ == 0)
{
v___x_2864_ = v___x_2851_;
v_isShared_2865_ = v_isSharedCheck_2940_;
goto v_resetjp_2863_;
}
else
{
lean_inc(v_a_2862_);
lean_dec(v___x_2851_);
v___x_2864_ = lean_box(0);
v_isShared_2865_ = v_isSharedCheck_2940_;
goto v_resetjp_2863_;
}
v_resetjp_2863_:
{
lean_object* v_mvarId_2866_; lean_object* v___f_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; 
v_mvarId_2866_ = lean_ctor_get(v_val_2850_, 0);
lean_inc_n(v_mvarId_2866_, 2);
v___f_2867_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__1___boxed), 2, 1);
lean_closure_set(v___f_2867_, 0, v_mvarId_2866_);
v___x_2868_ = lean_unsigned_to_nat(0u);
v___x_2869_ = l___private_Init_Data_Array_Basic_0__Array_findFinIdx_x3f_loop(lean_box(0), v___f_2867_, v_goals_2845_, v___x_2868_);
if (lean_obj_tag(v___x_2869_) == 1)
{
lean_object* v_val_2870_; lean_object* v___x_2872_; uint8_t v_isShared_2873_; uint8_t v_isSharedCheck_2935_; 
v_val_2870_ = lean_ctor_get(v___x_2869_, 0);
v_isSharedCheck_2935_ = !lean_is_exclusive(v___x_2869_);
if (v_isSharedCheck_2935_ == 0)
{
v___x_2872_ = v___x_2869_;
v_isShared_2873_ = v_isSharedCheck_2935_;
goto v_resetjp_2871_;
}
else
{
lean_inc(v_val_2870_);
lean_dec(v___x_2869_);
v___x_2872_ = lean_box(0);
v_isShared_2873_ = v_isSharedCheck_2935_;
goto v_resetjp_2871_;
}
v_resetjp_2871_:
{
uint8_t v___x_2874_; uint8_t v___x_2875_; lean_object* v___x_2876_; lean_object* v___y_2878_; uint8_t v___x_2930_; 
v___x_2874_ = 0;
v___x_2875_ = 1;
v___x_2876_ = lean_array_fget_borrowed(v_goals_2845_, v_val_2870_);
v___x_2930_ = lean_nat_dec_eq(v_val_2870_, v___x_2868_);
if (v___x_2930_ == 0)
{
lean_object* v___x_2932_; 
if (v_isShared_2873_ == 0)
{
v___x_2932_ = v___x_2872_;
goto v_reusejp_2931_;
}
else
{
lean_object* v_reuseFailAlloc_2933_; 
v_reuseFailAlloc_2933_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2933_, 0, v_val_2870_);
v___x_2932_ = v_reuseFailAlloc_2933_;
goto v_reusejp_2931_;
}
v_reusejp_2931_:
{
v___y_2878_ = v___x_2932_;
goto v___jp_2877_;
}
}
else
{
lean_object* v___x_2934_; 
lean_del_object(v___x_2872_);
lean_dec(v_val_2870_);
v___x_2934_ = lean_box(0);
v___y_2878_ = v___x_2934_;
goto v___jp_2877_;
}
v___jp_2877_:
{
lean_object* v_toEditableDocumentCore_2879_; lean_object* v_meta_2880_; lean_object* v_text_2881_; lean_object* v___x_2882_; lean_object* v___x_2883_; lean_object* v___x_2884_; 
v_toEditableDocumentCore_2879_ = lean_ctor_get(v_a_2862_, 0);
v_meta_2880_ = lean_ctor_get(v_toEditableDocumentCore_2879_, 0);
lean_inc_ref(v_meta_2880_);
v_text_2881_ = lean_ctor_get(v_meta_2880_, 3);
lean_inc_ref(v_pos_2846_);
v___x_2882_ = l_Lean_FileMap_lspPosToUtf8Pos(v_text_2881_, v_pos_2846_);
v___x_2883_ = l_Lean_Server_FileWorker_findGoalsAt_x3f(v_a_2862_, v___x_2882_);
v___x_2884_ = lean_task_get_own(v___x_2883_);
if (lean_obj_tag(v___x_2884_) == 1)
{
lean_object* v_val_2885_; lean_object* v___x_2886_; 
v_val_2885_ = lean_ctor_get(v___x_2884_, 0);
lean_inc(v_val_2885_);
lean_dec_ref_known(v___x_2884_, 1);
v___x_2886_ = lp_mathlib_List_find_x3f___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__2(v_val_2850_, v_val_2885_);
lean_dec(v_val_2885_);
if (lean_obj_tag(v___x_2886_) == 1)
{
lean_object* v_val_2887_; lean_object* v_ctxInfo_2888_; lean_object* v_tacticInfo_2889_; lean_object* v_toElabInfo_2890_; lean_object* v_toInteractiveGoalCore_2891_; lean_object* v_ctx_2892_; uint8_t v_useAfter_2893_; lean_object* v_parentDecl_x3f_2894_; lean_object* v_stx_2895_; lean_object* v_val_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; lean_object* v___x_2900_; lean_object* v___x_2901_; lean_object* v___f_2902_; lean_object* v___f_2903_; lean_object* v___x_2904_; 
lean_del_object(v___x_2864_);
v_val_2887_ = lean_ctor_get(v___x_2886_, 0);
lean_inc(v_val_2887_);
lean_dec_ref_known(v___x_2886_, 1);
v_ctxInfo_2888_ = lean_ctor_get(v_val_2887_, 0);
lean_inc_ref(v_ctxInfo_2888_);
v_tacticInfo_2889_ = lean_ctor_get(v_val_2887_, 1);
v_toElabInfo_2890_ = lean_ctor_get(v_tacticInfo_2889_, 0);
lean_inc_ref(v_toElabInfo_2890_);
v_toInteractiveGoalCore_2891_ = lean_ctor_get(v___x_2876_, 0);
v_ctx_2892_ = lean_ctor_get(v_toInteractiveGoalCore_2891_, 2);
v_useAfter_2893_ = lean_ctor_get_uint8(v_val_2887_, sizeof(void*)*3);
lean_dec(v_val_2887_);
v_parentDecl_x3f_2894_ = lean_ctor_get(v_ctxInfo_2888_, 1);
lean_inc(v_parentDecl_x3f_2894_);
lean_dec_ref(v_ctxInfo_2888_);
v_stx_2895_ = lean_ctor_get(v_toElabInfo_2890_, 1);
lean_inc(v_stx_2895_);
lean_dec_ref(v_toElabInfo_2890_);
v_val_2896_ = lean_ctor_get(v_ctx_2892_, 0);
v___x_2897_ = lean_obj_once(&lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4, &lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4_once, _init_lp_mathlib_Lean_instantiateLCtxMVars___at___00Lean_instantiateMVarDeclMVars___at___00Mathlib_Tactic_ClickSuggestions_generateSuggestions_spec__0_spec__0___closed__4);
v___x_2898_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_generateSuggestions___lam__0___closed__1));
v___x_2899_ = lean_box(v___x_2874_);
v___x_2900_ = lean_box(v_useAfter_2893_);
v___x_2901_ = lean_box(v___x_2875_);
lean_inc(v_mvarId_2866_);
v___f_2902_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__3___boxed), 19, 14);
lean_closure_set(v___f_2902_, 0, v___x_2898_);
lean_closure_set(v___f_2902_, 1, v___x_2868_);
lean_closure_set(v___f_2902_, 2, v___x_2899_);
lean_closure_set(v___f_2902_, 3, v_val_2850_);
lean_closure_set(v___f_2902_, 4, v_meta_2880_);
lean_closure_set(v___f_2902_, 5, v_pos_2846_);
lean_closure_set(v___f_2902_, 6, v___y_2878_);
lean_closure_set(v___f_2902_, 7, v_mvarId_2866_);
lean_closure_set(v___f_2902_, 8, v_parentDecl_x3f_2894_);
lean_closure_set(v___f_2902_, 9, v___x_2900_);
lean_closure_set(v___f_2902_, 10, v_stx_2895_);
lean_closure_set(v___f_2902_, 11, v___x_2901_);
lean_closure_set(v___f_2902_, 12, v_loc_2852_);
lean_closure_set(v___f_2902_, 13, v___f_2847_);
v___f_2903_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__4___boxed), 7, 2);
lean_closure_set(v___f_2903_, 0, v_mvarId_2866_);
lean_closure_set(v___f_2903_, 1, v___f_2902_);
lean_inc(v_val_2896_);
v___x_2904_ = l_Lean_Elab_ContextInfo_runMetaM___redArg(v_val_2896_, v___x_2897_, v___f_2903_);
if (lean_obj_tag(v___x_2904_) == 0)
{
lean_object* v_a_2905_; lean_object* v___x_2907_; uint8_t v_isShared_2908_; uint8_t v_isSharedCheck_2912_; 
v_a_2905_ = lean_ctor_get(v___x_2904_, 0);
v_isSharedCheck_2912_ = !lean_is_exclusive(v___x_2904_);
if (v_isSharedCheck_2912_ == 0)
{
v___x_2907_ = v___x_2904_;
v_isShared_2908_ = v_isSharedCheck_2912_;
goto v_resetjp_2906_;
}
else
{
lean_inc(v_a_2905_);
lean_dec(v___x_2904_);
v___x_2907_ = lean_box(0);
v_isShared_2908_ = v_isSharedCheck_2912_;
goto v_resetjp_2906_;
}
v_resetjp_2906_:
{
lean_object* v___x_2910_; 
if (v_isShared_2908_ == 0)
{
v___x_2910_ = v___x_2907_;
goto v_reusejp_2909_;
}
else
{
lean_object* v_reuseFailAlloc_2911_; 
v_reuseFailAlloc_2911_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2911_, 0, v_a_2905_);
v___x_2910_ = v_reuseFailAlloc_2911_;
goto v_reusejp_2909_;
}
v_reusejp_2909_:
{
return v___x_2910_;
}
}
}
else
{
lean_object* v_a_2913_; lean_object* v___x_2915_; uint8_t v_isShared_2916_; uint8_t v_isSharedCheck_2921_; 
v_a_2913_ = lean_ctor_get(v___x_2904_, 0);
v_isSharedCheck_2921_ = !lean_is_exclusive(v___x_2904_);
if (v_isSharedCheck_2921_ == 0)
{
v___x_2915_ = v___x_2904_;
v_isShared_2916_ = v_isSharedCheck_2921_;
goto v_resetjp_2914_;
}
else
{
lean_inc(v_a_2913_);
lean_dec(v___x_2904_);
v___x_2915_ = lean_box(0);
v_isShared_2916_ = v_isSharedCheck_2921_;
goto v_resetjp_2914_;
}
v_resetjp_2914_:
{
lean_object* v___x_2917_; lean_object* v___x_2919_; 
v___x_2917_ = l_Lean_Server_RequestError_ofIoError(v_a_2913_);
if (v_isShared_2916_ == 0)
{
lean_ctor_set(v___x_2915_, 0, v___x_2917_);
v___x_2919_ = v___x_2915_;
goto v_reusejp_2918_;
}
else
{
lean_object* v_reuseFailAlloc_2920_; 
v_reuseFailAlloc_2920_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2920_, 0, v___x_2917_);
v___x_2919_ = v_reuseFailAlloc_2920_;
goto v_reusejp_2918_;
}
v_reusejp_2918_:
{
return v___x_2919_;
}
}
}
}
else
{
lean_object* v___x_2922_; lean_object* v___x_2924_; 
lean_dec(v___x_2886_);
lean_dec_ref(v_meta_2880_);
lean_dec(v___y_2878_);
lean_dec(v_mvarId_2866_);
lean_dec_ref(v_loc_2852_);
lean_dec(v_val_2850_);
lean_dec_ref(v___f_2847_);
lean_dec_ref(v_pos_2846_);
v___x_2922_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__3));
if (v_isShared_2865_ == 0)
{
lean_ctor_set(v___x_2864_, 0, v___x_2922_);
v___x_2924_ = v___x_2864_;
goto v_reusejp_2923_;
}
else
{
lean_object* v_reuseFailAlloc_2925_; 
v_reuseFailAlloc_2925_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2925_, 0, v___x_2922_);
v___x_2924_ = v_reuseFailAlloc_2925_;
goto v_reusejp_2923_;
}
v_reusejp_2923_:
{
return v___x_2924_;
}
}
}
else
{
lean_object* v___x_2926_; lean_object* v___x_2928_; 
lean_dec(v___x_2884_);
lean_dec_ref(v_meta_2880_);
lean_dec(v___y_2878_);
lean_dec(v_mvarId_2866_);
lean_dec_ref(v_loc_2852_);
lean_dec(v_val_2850_);
lean_dec_ref(v___f_2847_);
lean_dec_ref(v_pos_2846_);
v___x_2926_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__5));
if (v_isShared_2865_ == 0)
{
lean_ctor_set(v___x_2864_, 0, v___x_2926_);
v___x_2928_ = v___x_2864_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v___x_2926_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
return v___x_2928_;
}
}
}
}
}
else
{
lean_object* v___x_2936_; lean_object* v___x_2938_; 
lean_dec(v___x_2869_);
lean_dec(v_mvarId_2866_);
lean_dec(v_a_2862_);
lean_dec_ref(v_loc_2852_);
lean_dec(v_val_2850_);
lean_dec_ref(v___f_2847_);
lean_dec_ref(v_pos_2846_);
v___x_2936_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__7));
if (v_isShared_2865_ == 0)
{
lean_ctor_set(v___x_2864_, 0, v___x_2936_);
v___x_2938_ = v___x_2864_;
goto v_reusejp_2937_;
}
else
{
lean_object* v_reuseFailAlloc_2939_; 
v_reuseFailAlloc_2939_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2939_, 0, v___x_2936_);
v___x_2938_ = v_reuseFailAlloc_2939_;
goto v_reusejp_2937_;
}
v_reusejp_2937_:
{
return v___x_2938_;
}
}
}
}
}
else
{
lean_object* v___x_2941_; lean_object* v___x_2942_; 
lean_dec_ref(v___f_2847_);
lean_dec_ref(v_pos_2846_);
lean_dec(v___y_2844_);
v___x_2941_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___closed__9));
v___x_2942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2942_, 0, v___x_2941_);
return v___x_2942_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___boxed(lean_object* v___y_2943_, lean_object* v_goals_2944_, lean_object* v_pos_2945_, lean_object* v___f_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_){
_start:
{
lean_object* v_res_2949_; 
v_res_2949_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5(v___y_2943_, v_goals_2944_, v_pos_2945_, v___f_2946_, v___y_2947_);
lean_dec_ref(v___y_2947_);
lean_dec_ref(v_goals_2944_);
return v_res_2949_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc(lean_object* v_props_2951_, lean_object* v_a_2952_){
_start:
{
lean_object* v_pos_2954_; lean_object* v_goals_2955_; lean_object* v_selectedLocations_2956_; lean_object* v___f_2957_; lean_object* v___y_2959_; lean_object* v___x_2962_; lean_object* v___x_2963_; lean_object* v___x_2964_; uint8_t v___x_2965_; 
v_pos_2954_ = lean_ctor_get(v_props_2951_, 0);
lean_inc_ref(v_pos_2954_);
v_goals_2955_ = lean_ctor_get(v_props_2951_, 1);
lean_inc_ref(v_goals_2955_);
v_selectedLocations_2956_ = lean_ctor_get(v_props_2951_, 3);
lean_inc_ref(v_selectedLocations_2956_);
lean_dec_ref(v_props_2951_);
v___f_2957_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___closed__0));
v___x_2962_ = lean_array_get_size(v_selectedLocations_2956_);
v___x_2963_ = lean_unsigned_to_nat(1u);
v___x_2964_ = lean_nat_sub(v___x_2962_, v___x_2963_);
v___x_2965_ = lean_nat_dec_lt(v___x_2964_, v___x_2962_);
if (v___x_2965_ == 0)
{
lean_object* v___x_2966_; 
lean_dec(v___x_2964_);
lean_dec_ref(v_selectedLocations_2956_);
v___x_2966_ = lean_box(0);
v___y_2959_ = v___x_2966_;
goto v___jp_2958_;
}
else
{
lean_object* v___x_2967_; lean_object* v___x_2968_; 
v___x_2967_ = lean_array_fget(v_selectedLocations_2956_, v___x_2964_);
lean_dec(v___x_2964_);
lean_dec_ref(v_selectedLocations_2956_);
v___x_2968_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2968_, 0, v___x_2967_);
v___y_2959_ = v___x_2968_;
goto v___jp_2958_;
}
v___jp_2958_:
{
lean_object* v___y_2960_; lean_object* v___x_2961_; 
v___y_2960_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___lam__5___boxed), 6, 4);
lean_closure_set(v___y_2960_, 0, v___y_2959_);
lean_closure_set(v___y_2960_, 1, v_goals_2955_);
lean_closure_set(v___y_2960_, 2, v_pos_2954_);
lean_closure_set(v___y_2960_, 3, v___f_2957_);
v___x_2961_ = l_Lean_Server_RequestM_asTask___redArg(v___y_2960_, v_a_2952_);
return v___x_2961_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___boxed(lean_object* v_props_2969_, lean_object* v_a_2970_, lean_object* v_a_2971_){
_start:
{
lean_object* v_res_2972_; 
v_res_2972_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc(v_props_2969_, v_a_2970_);
lean_dec_ref(v_a_2970_);
return v_res_2972_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4(lean_object* v_00_u03b1_2973_, lean_object* v_visit_2974_, lean_object* v_p_2975_, lean_object* v_root_2976_, lean_object* v___y_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_){
_start:
{
lean_object* v___x_2982_; 
v___x_2982_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___redArg(v_visit_2974_, v_p_2975_, v_root_2976_, v___y_2977_, v___y_2978_, v___y_2979_, v___y_2980_);
return v___x_2982_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4___boxed(lean_object* v_00_u03b1_2983_, lean_object* v_visit_2984_, lean_object* v_p_2985_, lean_object* v_root_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_, lean_object* v___y_2991_){
_start:
{
lean_object* v_res_2992_; 
v_res_2992_ = lp_mathlib_Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4(v_00_u03b1_2983_, v_visit_2984_, v_p_2985_, v_root_2986_, v___y_2987_, v___y_2988_, v___y_2989_, v___y_2990_);
lean_dec(v___y_2990_);
lean_dec_ref(v___y_2989_);
lean_dec(v___y_2988_);
lean_dec_ref(v___y_2987_);
lean_dec(v_p_2985_);
return v_res_2992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5(lean_object* v_00_u03b1_2993_, lean_object* v_k_2994_, lean_object* v_fvars_2995_, lean_object* v_x_2996_, lean_object* v_x_2997_, lean_object* v___y_2998_, lean_object* v___y_2999_, lean_object* v___y_3000_, lean_object* v___y_3001_){
_start:
{
lean_object* v___x_3003_; 
v___x_3003_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___redArg(v_k_2994_, v_fvars_2995_, v_x_2996_, v_x_2997_, v___y_2998_, v___y_2999_, v___y_3000_, v___y_3001_);
return v___x_3003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5___boxed(lean_object* v_00_u03b1_3004_, lean_object* v_k_3005_, lean_object* v_fvars_3006_, lean_object* v_x_3007_, lean_object* v_x_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_){
_start:
{
lean_object* v_res_3014_; 
v_res_3014_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5(v_00_u03b1_3004_, v_k_3005_, v_fvars_3006_, v_x_3007_, v_x_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_);
lean_dec(v___y_3012_);
lean_dec_ref(v___y_3011_);
lean_dec(v___y_3010_);
lean_dec_ref(v___y_3009_);
return v_res_3014_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9(lean_object* v_00_u03b1_3015_, lean_object* v_name_3016_, uint8_t v_bi_3017_, lean_object* v_type_3018_, lean_object* v_k_3019_, uint8_t v_kind_3020_, lean_object* v___y_3021_, lean_object* v___y_3022_, lean_object* v___y_3023_, lean_object* v___y_3024_){
_start:
{
lean_object* v___x_3026_; 
v___x_3026_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___redArg(v_name_3016_, v_bi_3017_, v_type_3018_, v_k_3019_, v_kind_3020_, v___y_3021_, v___y_3022_, v___y_3023_, v___y_3024_);
return v___x_3026_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9___boxed(lean_object* v_00_u03b1_3027_, lean_object* v_name_3028_, lean_object* v_bi_3029_, lean_object* v_type_3030_, lean_object* v_k_3031_, lean_object* v_kind_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_, lean_object* v___y_3035_, lean_object* v___y_3036_, lean_object* v___y_3037_){
_start:
{
uint8_t v_bi_boxed_3038_; uint8_t v_kind_boxed_3039_; lean_object* v_res_3040_; 
v_bi_boxed_3038_ = lean_unbox(v_bi_3029_);
v_kind_boxed_3039_ = lean_unbox(v_kind_3032_);
v_res_3040_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__9(v_00_u03b1_3027_, v_name_3028_, v_bi_boxed_3038_, v_type_3030_, v_k_3031_, v_kind_boxed_3039_, v___y_3033_, v___y_3034_, v___y_3035_, v___y_3036_);
lean_dec(v___y_3036_);
lean_dec_ref(v___y_3035_);
lean_dec(v___y_3034_);
lean_dec_ref(v___y_3033_);
return v_res_3040_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10(lean_object* v_00_u03b1_3041_, lean_object* v_name_3042_, lean_object* v_type_3043_, lean_object* v_val_3044_, lean_object* v_k_3045_, uint8_t v_nondep_3046_, uint8_t v_kind_3047_, lean_object* v___y_3048_, lean_object* v___y_3049_, lean_object* v___y_3050_, lean_object* v___y_3051_){
_start:
{
lean_object* v___x_3053_; 
v___x_3053_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___redArg(v_name_3042_, v_type_3043_, v_val_3044_, v_k_3045_, v_nondep_3046_, v_kind_3047_, v___y_3048_, v___y_3049_, v___y_3050_, v___y_3051_);
return v___x_3053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10___boxed(lean_object* v_00_u03b1_3054_, lean_object* v_name_3055_, lean_object* v_type_3056_, lean_object* v_val_3057_, lean_object* v_k_3058_, lean_object* v_nondep_3059_, lean_object* v_kind_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_, lean_object* v___y_3064_, lean_object* v___y_3065_){
_start:
{
uint8_t v_nondep_boxed_3066_; uint8_t v_kind_boxed_3067_; lean_object* v_res_3068_; 
v_nondep_boxed_3066_ = lean_unbox(v_nondep_3059_);
v_kind_boxed_3067_ = lean_unbox(v_kind_3060_);
v_res_3068_ = lp_mathlib_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__10(v_00_u03b1_3054_, v_name_3055_, v_type_3056_, v_val_3057_, v_k_3058_, v_nondep_boxed_3066_, v_kind_boxed_3067_, v___y_3061_, v___y_3062_, v___y_3063_, v___y_3064_);
lean_dec(v___y_3064_);
lean_dec_ref(v___y_3063_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
return v_res_3068_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7(lean_object* v_00_u03b1_3069_, lean_object* v_k_3070_, lean_object* v_fvars_3071_, lean_object* v_n_3072_, lean_object* v_e_3073_, lean_object* v___y_3074_, lean_object* v___y_3075_, lean_object* v___y_3076_, lean_object* v___y_3077_){
_start:
{
lean_object* v___x_3079_; 
v___x_3079_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___redArg(v_k_3070_, v_fvars_3071_, v_n_3072_, v_e_3073_, v___y_3074_, v___y_3075_, v___y_3076_, v___y_3077_);
return v___x_3079_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7___boxed(lean_object* v_00_u03b1_3080_, lean_object* v_k_3081_, lean_object* v_fvars_3082_, lean_object* v_n_3083_, lean_object* v_e_3084_, lean_object* v___y_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_, lean_object* v___y_3088_, lean_object* v___y_3089_){
_start:
{
lean_object* v_res_3090_; 
v_res_3090_ = lp_mathlib___private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7(v_00_u03b1_3080_, v_k_3081_, v_fvars_3082_, v_n_3083_, v_e_3084_, v___y_3085_, v___y_3086_, v___y_3087_, v___y_3088_);
lean_dec(v___y_3088_);
lean_dec_ref(v___y_3087_);
lean_dec(v___y_3086_);
lean_dec_ref(v___y_3085_);
return v_res_3090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8(lean_object* v_00_u03b1_3091_, lean_object* v_msg_3092_, lean_object* v___y_3093_, lean_object* v___y_3094_, lean_object* v___y_3095_, lean_object* v___y_3096_){
_start:
{
lean_object* v___x_3098_; 
v___x_3098_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___redArg(v_msg_3092_, v___y_3093_, v___y_3094_, v___y_3095_, v___y_3096_);
return v___x_3098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8___boxed(lean_object* v_00_u03b1_3099_, lean_object* v_msg_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_){
_start:
{
lean_object* v_res_3106_; 
v_res_3106_ = lp_mathlib_Lean_throwError___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewCoordAux___at___00__private_Lean_Meta_ExprLens_0__Lean_Meta_viewAux___at___00Lean_Meta_viewSubexpr___at___00Mathlib_Tactic_ClickSuggestions_rpc_spec__4_spec__5_spec__7_spec__8(v_00_u03b1_3099_, v_msg_3100_, v___y_3101_, v___y_3102_, v___y_3103_, v___y_3104_);
lean_dec(v___y_3104_);
lean_dec_ref(v___y_3103_);
lean_dec(v___y_3102_);
lean_dec_ref(v___y_3101_);
return v_res_3106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg(lean_object* v_t_3107_, uint64_t v_k_3108_){
_start:
{
if (lean_obj_tag(v_t_3107_) == 0)
{
lean_object* v_k_3109_; lean_object* v_v_3110_; lean_object* v_l_3111_; lean_object* v_r_3112_; uint64_t v___x_3113_; uint8_t v___x_3114_; 
v_k_3109_ = lean_ctor_get(v_t_3107_, 1);
v_v_3110_ = lean_ctor_get(v_t_3107_, 2);
v_l_3111_ = lean_ctor_get(v_t_3107_, 3);
v_r_3112_ = lean_ctor_get(v_t_3107_, 4);
v___x_3113_ = lean_unbox_uint64(v_k_3109_);
v___x_3114_ = lean_uint64_dec_lt(v_k_3108_, v___x_3113_);
if (v___x_3114_ == 0)
{
uint64_t v___x_3115_; uint8_t v___x_3116_; 
v___x_3115_ = lean_unbox_uint64(v_k_3109_);
v___x_3116_ = lean_uint64_dec_eq(v_k_3108_, v___x_3115_);
if (v___x_3116_ == 0)
{
v_t_3107_ = v_r_3112_;
goto _start;
}
else
{
lean_object* v___x_3118_; 
lean_inc(v_v_3110_);
v___x_3118_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3118_, 0, v_v_3110_);
return v___x_3118_;
}
}
else
{
v_t_3107_ = v_l_3111_;
goto _start;
}
}
else
{
lean_object* v___x_3120_; 
v___x_3120_ = lean_box(0);
return v___x_3120_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg___boxed(lean_object* v_t_3121_, lean_object* v_k_3122_){
_start:
{
uint64_t v_k_boxed_3123_; lean_object* v_res_3124_; 
v_k_boxed_3123_ = lean_unbox_uint64(v_k_3122_);
lean_dec_ref(v_k_3122_);
v_res_3124_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_3121_, v_k_boxed_3123_);
lean_dec(v_t_3121_);
return v_res_3124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__1(lean_object* v_expireTime_3125_, lean_object* v_x_3126_){
_start:
{
lean_object* v___x_3127_; 
v___x_3127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3127_, 0, v_x_3126_);
lean_ctor_set(v___x_3127_, 1, v_expireTime_3125_);
return v___x_3127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2(lean_object* v_val_3128_, lean_object* v___f_3129_, lean_object* v_x_3130_, lean_object* v___y_3131_){
_start:
{
if (lean_obj_tag(v_x_3130_) == 0)
{
lean_object* v_a_3133_; lean_object* v___x_3135_; uint8_t v_isShared_3136_; uint8_t v_isSharedCheck_3140_; 
lean_dec_ref(v___f_3129_);
v_a_3133_ = lean_ctor_get(v_x_3130_, 0);
v_isSharedCheck_3140_ = !lean_is_exclusive(v_x_3130_);
if (v_isSharedCheck_3140_ == 0)
{
v___x_3135_ = v_x_3130_;
v_isShared_3136_ = v_isSharedCheck_3140_;
goto v_resetjp_3134_;
}
else
{
lean_inc(v_a_3133_);
lean_dec(v_x_3130_);
v___x_3135_ = lean_box(0);
v_isShared_3136_ = v_isSharedCheck_3140_;
goto v_resetjp_3134_;
}
v_resetjp_3134_:
{
lean_object* v___x_3138_; 
if (v_isShared_3136_ == 0)
{
lean_ctor_set_tag(v___x_3135_, 1);
v___x_3138_ = v___x_3135_;
goto v_reusejp_3137_;
}
else
{
lean_object* v_reuseFailAlloc_3139_; 
v_reuseFailAlloc_3139_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3139_, 0, v_a_3133_);
v___x_3138_ = v_reuseFailAlloc_3139_;
goto v_reusejp_3137_;
}
v_reusejp_3137_:
{
return v___x_3138_;
}
}
}
else
{
lean_object* v_a_3141_; lean_object* v___x_3143_; uint8_t v_isShared_3144_; uint8_t v_isSharedCheck_3157_; 
v_a_3141_ = lean_ctor_get(v_x_3130_, 0);
v_isSharedCheck_3157_ = !lean_is_exclusive(v_x_3130_);
if (v_isSharedCheck_3157_ == 0)
{
v___x_3143_ = v_x_3130_;
v_isShared_3144_ = v_isSharedCheck_3157_;
goto v_resetjp_3142_;
}
else
{
lean_inc(v_a_3141_);
lean_dec(v_x_3130_);
v___x_3143_ = lean_box(0);
v_isShared_3144_ = v_isSharedCheck_3157_;
goto v_resetjp_3142_;
}
v_resetjp_3142_:
{
lean_object* v___x_3145_; lean_object* v_objects_3146_; lean_object* v_expireTime_3147_; lean_object* v___f_3148_; lean_object* v___x_3149_; lean_object* v___x_3150_; lean_object* v_fst_3151_; lean_object* v_snd_3152_; lean_object* v___x_3153_; lean_object* v___x_3155_; 
v___x_3145_ = lean_st_ref_take(v_val_3128_);
v_objects_3146_ = lean_ctor_get(v___x_3145_, 0);
lean_inc_ref(v_objects_3146_);
v_expireTime_3147_ = lean_ctor_get(v___x_3145_, 1);
lean_inc(v_expireTime_3147_);
lean_dec(v___x_3145_);
v___f_3148_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__1), 2, 1);
lean_closure_set(v___f_3148_, 0, v_expireTime_3147_);
v___x_3149_ = lp_proofwidgets_ProofWidgets_instRpcEncodableHtml_enc_00___x40_ProofWidgets_Data_Html_2686543190____hygCtx___hyg_1_(v_a_3141_, v_objects_3146_);
v___x_3150_ = l_Prod_map___redArg(v___f_3129_, v___f_3148_, v___x_3149_);
v_fst_3151_ = lean_ctor_get(v___x_3150_, 0);
lean_inc(v_fst_3151_);
v_snd_3152_ = lean_ctor_get(v___x_3150_, 1);
lean_inc(v_snd_3152_);
lean_dec_ref(v___x_3150_);
v___x_3153_ = lean_st_ref_set(v_val_3128_, v_snd_3152_);
if (v_isShared_3144_ == 0)
{
lean_ctor_set_tag(v___x_3143_, 0);
lean_ctor_set(v___x_3143_, 0, v_fst_3151_);
v___x_3155_ = v___x_3143_;
goto v_reusejp_3154_;
}
else
{
lean_object* v_reuseFailAlloc_3156_; 
v_reuseFailAlloc_3156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3156_, 0, v_fst_3151_);
v___x_3155_ = v_reuseFailAlloc_3156_;
goto v_reusejp_3154_;
}
v_reusejp_3154_:
{
return v___x_3155_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2___boxed(lean_object* v_val_3158_, lean_object* v___f_3159_, lean_object* v_x_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_){
_start:
{
lean_object* v_res_3163_; 
v_res_3163_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2(v_val_3158_, v___f_3159_, v_x_3160_, v___y_3161_);
lean_dec_ref(v___y_3161_);
lean_dec(v_val_3158_);
return v_res_3163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3(lean_object* v_method_3171_, lean_object* v_handler_3172_, lean_object* v___f_3173_, uint64_t v_seshId_3174_, lean_object* v_j_3175_, lean_object* v___y_3176_){
_start:
{
lean_object* v_rpcSessions_3178_; lean_object* v___x_3179_; 
v_rpcSessions_3178_ = lean_ctor_get(v___y_3176_, 0);
v___x_3179_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_rpcSessions_3178_, v_seshId_3174_);
if (lean_obj_tag(v___x_3179_) == 1)
{
lean_object* v_val_3180_; lean_object* v___x_3181_; lean_object* v_objects_3182_; lean_object* v___x_3183_; 
v_val_3180_ = lean_ctor_get(v___x_3179_, 0);
lean_inc(v_val_3180_);
lean_dec_ref_known(v___x_3179_, 1);
v___x_3181_ = lean_st_ref_get(v_val_3180_);
v_objects_3182_ = lean_ctor_get(v___x_3181_, 0);
lean_inc_ref(v_objects_3182_);
lean_dec(v___x_3181_);
lean_inc(v_j_3175_);
v___x_3183_ = lp_proofwidgets_ProofWidgets_instRpcEncodablePanelWidgetProps_dec_00___x40_ProofWidgets_Component_Panel_Basic_2840189264____hygCtx___hyg_1_(v_j_3175_, v_objects_3182_);
lean_dec_ref(v_objects_3182_);
if (lean_obj_tag(v___x_3183_) == 0)
{
lean_object* v_a_3184_; lean_object* v___x_3186_; uint8_t v_isShared_3187_; uint8_t v_isSharedCheck_3204_; 
lean_dec(v_val_3180_);
lean_dec_ref(v___f_3173_);
lean_dec_ref(v_handler_3172_);
v_a_3184_ = lean_ctor_get(v___x_3183_, 0);
v_isSharedCheck_3204_ = !lean_is_exclusive(v___x_3183_);
if (v_isSharedCheck_3204_ == 0)
{
v___x_3186_ = v___x_3183_;
v_isShared_3187_ = v_isSharedCheck_3204_;
goto v_resetjp_3185_;
}
else
{
lean_inc(v_a_3184_);
lean_dec(v___x_3183_);
v___x_3186_ = lean_box(0);
v_isShared_3187_ = v_isSharedCheck_3204_;
goto v_resetjp_3185_;
}
v_resetjp_3185_:
{
uint8_t v___x_3188_; lean_object* v___x_3189_; uint8_t v___x_3190_; lean_object* v___x_3191_; lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3202_; 
v___x_3188_ = 3;
v___x_3189_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__0));
v___x_3190_ = 1;
v___x_3191_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_method_3171_, v___x_3190_);
v___x_3192_ = lean_string_append(v___x_3189_, v___x_3191_);
lean_dec_ref(v___x_3191_);
v___x_3193_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__1));
v___x_3194_ = lean_string_append(v___x_3192_, v___x_3193_);
v___x_3195_ = l_Lean_Json_compress(v_j_3175_);
v___x_3196_ = lean_string_append(v___x_3194_, v___x_3195_);
lean_dec_ref(v___x_3195_);
v___x_3197_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__2));
v___x_3198_ = lean_string_append(v___x_3196_, v___x_3197_);
v___x_3199_ = lean_string_append(v___x_3198_, v_a_3184_);
lean_dec(v_a_3184_);
v___x_3200_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_3200_, 0, v___x_3199_);
lean_ctor_set_uint8(v___x_3200_, sizeof(void*)*1, v___x_3188_);
if (v_isShared_3187_ == 0)
{
lean_ctor_set_tag(v___x_3186_, 1);
lean_ctor_set(v___x_3186_, 0, v___x_3200_);
v___x_3202_ = v___x_3186_;
goto v_reusejp_3201_;
}
else
{
lean_object* v_reuseFailAlloc_3203_; 
v_reuseFailAlloc_3203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3203_, 0, v___x_3200_);
v___x_3202_ = v_reuseFailAlloc_3203_;
goto v_reusejp_3201_;
}
v_reusejp_3201_:
{
return v___x_3202_;
}
}
}
else
{
lean_object* v_a_3205_; lean_object* v___x_3206_; 
lean_dec(v_j_3175_);
lean_dec(v_method_3171_);
v_a_3205_ = lean_ctor_get(v___x_3183_, 0);
lean_inc(v_a_3205_);
lean_dec_ref_known(v___x_3183_, 1);
lean_inc_ref(v___y_3176_);
v___x_3206_ = lean_apply_3(v_handler_3172_, v_a_3205_, v___y_3176_, lean_box(0));
if (lean_obj_tag(v___x_3206_) == 0)
{
lean_object* v_a_3207_; lean_object* v___f_3208_; lean_object* v___x_3209_; 
v_a_3207_ = lean_ctor_get(v___x_3206_, 0);
lean_inc(v_a_3207_);
lean_dec_ref_known(v___x_3206_, 1);
v___f_3208_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__2___boxed), 5, 2);
lean_closure_set(v___f_3208_, 0, v_val_3180_);
lean_closure_set(v___f_3208_, 1, v___f_3173_);
v___x_3209_ = l_Lean_Server_RequestM_mapTaskCheap___redArg(v_a_3207_, v___f_3208_, v___y_3176_);
return v___x_3209_;
}
else
{
lean_object* v_a_3210_; lean_object* v___x_3212_; uint8_t v_isShared_3213_; uint8_t v_isSharedCheck_3217_; 
lean_dec(v_val_3180_);
lean_dec_ref(v___f_3173_);
v_a_3210_ = lean_ctor_get(v___x_3206_, 0);
v_isSharedCheck_3217_ = !lean_is_exclusive(v___x_3206_);
if (v_isSharedCheck_3217_ == 0)
{
v___x_3212_ = v___x_3206_;
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
else
{
lean_inc(v_a_3210_);
lean_dec(v___x_3206_);
v___x_3212_ = lean_box(0);
v_isShared_3213_ = v_isSharedCheck_3217_;
goto v_resetjp_3211_;
}
v_resetjp_3211_:
{
lean_object* v___x_3215_; 
if (v_isShared_3213_ == 0)
{
v___x_3215_ = v___x_3212_;
goto v_reusejp_3214_;
}
else
{
lean_object* v_reuseFailAlloc_3216_; 
v_reuseFailAlloc_3216_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3216_, 0, v_a_3210_);
v___x_3215_ = v_reuseFailAlloc_3216_;
goto v_reusejp_3214_;
}
v_reusejp_3214_:
{
return v___x_3215_;
}
}
}
}
}
else
{
lean_object* v___x_3218_; lean_object* v___x_3219_; 
lean_dec(v___x_3179_);
lean_dec(v_j_3175_);
lean_dec_ref(v___f_3173_);
lean_dec_ref(v_handler_3172_);
lean_dec(v_method_3171_);
v___x_3218_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___closed__4));
v___x_3219_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3219_, 0, v___x_3218_);
return v___x_3219_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___boxed(lean_object* v_method_3220_, lean_object* v_handler_3221_, lean_object* v___f_3222_, lean_object* v_seshId_3223_, lean_object* v_j_3224_, lean_object* v___y_3225_, lean_object* v___y_3226_){
_start:
{
uint64_t v_seshId_boxed_3227_; lean_object* v_res_3228_; 
v_seshId_boxed_3227_ = lean_unbox_uint64(v_seshId_3223_);
lean_dec_ref(v_seshId_3223_);
v_res_3228_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3(v_method_3220_, v_handler_3221_, v___f_3222_, v_seshId_boxed_3227_, v_j_3224_, v___y_3225_);
lean_dec_ref(v___y_3225_);
return v_res_3228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0(lean_object* v___y_3229_){
_start:
{
lean_inc(v___y_3229_);
return v___y_3229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0___boxed(lean_object* v___y_3230_){
_start:
{
lean_object* v_res_3231_; 
v_res_3231_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__0(v___y_3230_);
lean_dec(v___y_3230_);
return v_res_3231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0(lean_object* v_method_3233_, lean_object* v_handler_3234_){
_start:
{
lean_object* v___f_3235_; lean_object* v___f_3236_; 
v___f_3235_ = ((lean_object*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___closed__0));
v___f_3236_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0___lam__3___boxed), 7, 3);
lean_closure_set(v___f_3236_, 0, v_method_3233_);
lean_closure_set(v___f_3236_, 1, v_handler_3234_);
lean_closure_set(v___f_3236_, 2, v___f_3235_);
return v___f_3236_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6(void){
_start:
{
lean_object* v___x_3247_; lean_object* v___x_3248_; lean_object* v___x_3249_; 
v___x_3247_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__5));
v___x_3248_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__4));
v___x_3249_ = lp_mathlib_Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0(v___x_3248_, v___x_3247_);
return v___x_3249_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped(void){
_start:
{
lean_object* v___x_3250_; 
v___x_3250_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped___closed__6);
return v___x_3250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0(lean_object* v_00_u03b4_3251_, lean_object* v_t_3252_, uint64_t v_k_3253_){
_start:
{
lean_object* v___x_3254_; 
v___x_3254_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___redArg(v_t_3252_, v_k_3253_);
return v___x_3254_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0___boxed(lean_object* v_00_u03b4_3255_, lean_object* v_t_3256_, lean_object* v_k_3257_){
_start:
{
uint64_t v_k_boxed_3258_; lean_object* v_res_3259_; 
v_k_boxed_3258_ = lean_unbox_uint64(v_k_3257_);
lean_dec_ref(v_k_3257_);
v_res_3259_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_Server_wrapRpcProcedure___at___00Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped_spec__0_spec__0(v_00_u03b4_3255_, v_t_3256_, v_k_boxed_3258_);
lean_dec(v_t_3256_);
return v_res_3259_;
}
}
static uint64_t _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1(void){
_start:
{
lean_object* v___x_3261_; uint64_t v___x_3262_; 
v___x_3261_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__0));
v___x_3262_ = lean_string_hash(v___x_3261_);
return v___x_3262_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2(void){
_start:
{
uint64_t v___x_3263_; lean_object* v___x_3264_; lean_object* v___x_3265_; 
v___x_3263_ = lean_uint64_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__1);
v___x_3264_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__0));
v___x_3265_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v___x_3265_, 0, v___x_3264_);
lean_ctor_set_uint64(v___x_3265_, sizeof(void*)*1, v___x_3263_);
return v___x_3265_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4(void){
_start:
{
lean_object* v___x_3267_; lean_object* v___x_3268_; lean_object* v___x_3269_; 
v___x_3267_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__3));
v___x_3268_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__2);
v___x_3269_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3269_, 0, v___x_3268_);
lean_ctor_set(v___x_3269_, 1, v___x_3267_);
return v___x_3269_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent(void){
_start:
{
lean_object* v___x_3270_; 
v___x_3270_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4, &lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent___closed__4);
return v___x_3270_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_3285_; lean_object* v___x_3286_; lean_object* v___x_3287_; 
v___x_3285_ = lean_box(0);
v___x_3286_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3287_, 0, v___x_3286_);
lean_ctor_set(v___x_3287_, 1, v___x_3285_);
return v___x_3287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg(){
_start:
{
lean_object* v___x_3289_; lean_object* v___x_3290_; 
v___x_3289_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___closed__0);
v___x_3290_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3290_, 0, v___x_3289_);
return v___x_3290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg___boxed(lean_object* v___y_3291_){
_start:
{
lean_object* v_res_3292_; 
v_res_3292_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg();
return v_res_3292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0(lean_object* v_00_u03b1_3293_, lean_object* v___y_3294_, lean_object* v___y_3295_){
_start:
{
lean_object* v___x_3297_; 
v___x_3297_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg();
return v___x_3297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___boxed(lean_object* v_00_u03b1_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_, lean_object* v___y_3301_){
_start:
{
lean_object* v_res_3302_; 
v_res_3302_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0(v_00_u03b1_3298_, v___y_3299_, v___y_3300_);
lean_dec(v___y_3300_);
lean_dec_ref(v___y_3299_);
return v_res_3302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___lam__0(lean_object* v___x_3303_, lean_object* v___y_3304_){
_start:
{
lean_object* v___x_3305_; 
v___x_3305_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3305_, 0, v___x_3303_);
lean_ctor_set(v___x_3305_, 1, v___y_3304_);
return v___x_3305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg(lean_object* v_t_3306_, uint64_t v_k_3307_, lean_object* v_fallback_3308_){
_start:
{
if (lean_obj_tag(v_t_3306_) == 0)
{
lean_object* v_k_3309_; lean_object* v_v_3310_; lean_object* v_l_3311_; lean_object* v_r_3312_; uint64_t v___x_3313_; uint8_t v___x_3314_; 
v_k_3309_ = lean_ctor_get(v_t_3306_, 1);
v_v_3310_ = lean_ctor_get(v_t_3306_, 2);
v_l_3311_ = lean_ctor_get(v_t_3306_, 3);
v_r_3312_ = lean_ctor_get(v_t_3306_, 4);
v___x_3313_ = lean_unbox_uint64(v_k_3309_);
v___x_3314_ = lean_uint64_dec_lt(v_k_3307_, v___x_3313_);
if (v___x_3314_ == 0)
{
uint64_t v___x_3315_; uint8_t v___x_3316_; 
v___x_3315_ = lean_unbox_uint64(v_k_3309_);
v___x_3316_ = lean_uint64_dec_eq(v_k_3307_, v___x_3315_);
if (v___x_3316_ == 0)
{
v_t_3306_ = v_r_3312_;
goto _start;
}
else
{
lean_inc(v_v_3310_);
return v_v_3310_;
}
}
else
{
v_t_3306_ = v_l_3311_;
goto _start;
}
}
else
{
lean_inc(v_fallback_3308_);
return v_fallback_3308_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg___boxed(lean_object* v_t_3319_, lean_object* v_k_3320_, lean_object* v_fallback_3321_){
_start:
{
uint64_t v_k_boxed_3322_; lean_object* v_res_3323_; 
v_k_boxed_3322_ = lean_unbox_uint64(v_k_3320_);
lean_dec_ref(v_k_3320_);
v_res_3323_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg(v_t_3319_, v_k_boxed_3322_, v_fallback_3321_);
lean_dec(v_fallback_3321_);
lean_dec(v_t_3319_);
return v_res_3323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(uint64_t v_k_3324_, lean_object* v_v_3325_, lean_object* v_t_3326_){
_start:
{
if (lean_obj_tag(v_t_3326_) == 0)
{
lean_object* v_size_3327_; lean_object* v_k_3328_; lean_object* v_v_3329_; lean_object* v_l_3330_; lean_object* v_r_3331_; lean_object* v___x_3333_; uint8_t v_isShared_3334_; uint8_t v_isSharedCheck_3615_; 
v_size_3327_ = lean_ctor_get(v_t_3326_, 0);
v_k_3328_ = lean_ctor_get(v_t_3326_, 1);
v_v_3329_ = lean_ctor_get(v_t_3326_, 2);
v_l_3330_ = lean_ctor_get(v_t_3326_, 3);
v_r_3331_ = lean_ctor_get(v_t_3326_, 4);
v_isSharedCheck_3615_ = !lean_is_exclusive(v_t_3326_);
if (v_isSharedCheck_3615_ == 0)
{
v___x_3333_ = v_t_3326_;
v_isShared_3334_ = v_isSharedCheck_3615_;
goto v_resetjp_3332_;
}
else
{
lean_inc(v_r_3331_);
lean_inc(v_l_3330_);
lean_inc(v_v_3329_);
lean_inc(v_k_3328_);
lean_inc(v_size_3327_);
lean_dec(v_t_3326_);
v___x_3333_ = lean_box(0);
v_isShared_3334_ = v_isSharedCheck_3615_;
goto v_resetjp_3332_;
}
v_resetjp_3332_:
{
uint64_t v___x_3335_; uint8_t v___x_3336_; 
v___x_3335_ = lean_unbox_uint64(v_k_3328_);
v___x_3336_ = lean_uint64_dec_lt(v_k_3324_, v___x_3335_);
if (v___x_3336_ == 0)
{
uint64_t v___x_3337_; uint8_t v___x_3338_; 
v___x_3337_ = lean_unbox_uint64(v_k_3328_);
v___x_3338_ = lean_uint64_dec_eq(v_k_3324_, v___x_3337_);
if (v___x_3338_ == 0)
{
lean_object* v_impl_3339_; lean_object* v___x_3340_; 
lean_dec(v_size_3327_);
v_impl_3339_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(v_k_3324_, v_v_3325_, v_r_3331_);
v___x_3340_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_l_3330_) == 0)
{
lean_object* v_size_3341_; lean_object* v_size_3342_; lean_object* v_k_3343_; lean_object* v_v_3344_; lean_object* v_l_3345_; lean_object* v_r_3346_; lean_object* v___x_3347_; lean_object* v___x_3348_; uint8_t v___x_3349_; 
v_size_3341_ = lean_ctor_get(v_l_3330_, 0);
v_size_3342_ = lean_ctor_get(v_impl_3339_, 0);
lean_inc(v_size_3342_);
v_k_3343_ = lean_ctor_get(v_impl_3339_, 1);
lean_inc(v_k_3343_);
v_v_3344_ = lean_ctor_get(v_impl_3339_, 2);
lean_inc(v_v_3344_);
v_l_3345_ = lean_ctor_get(v_impl_3339_, 3);
lean_inc(v_l_3345_);
v_r_3346_ = lean_ctor_get(v_impl_3339_, 4);
lean_inc(v_r_3346_);
v___x_3347_ = lean_unsigned_to_nat(3u);
v___x_3348_ = lean_nat_mul(v___x_3347_, v_size_3341_);
v___x_3349_ = lean_nat_dec_lt(v___x_3348_, v_size_3342_);
lean_dec(v___x_3348_);
if (v___x_3349_ == 0)
{
lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3353_; 
lean_dec(v_r_3346_);
lean_dec(v_l_3345_);
lean_dec(v_v_3344_);
lean_dec(v_k_3343_);
v___x_3350_ = lean_nat_add(v___x_3340_, v_size_3341_);
v___x_3351_ = lean_nat_add(v___x_3350_, v_size_3342_);
lean_dec(v_size_3342_);
lean_dec(v___x_3350_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_impl_3339_);
lean_ctor_set(v___x_3333_, 0, v___x_3351_);
v___x_3353_ = v___x_3333_;
goto v_reusejp_3352_;
}
else
{
lean_object* v_reuseFailAlloc_3354_; 
v_reuseFailAlloc_3354_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3354_, 0, v___x_3351_);
lean_ctor_set(v_reuseFailAlloc_3354_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3354_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3354_, 3, v_l_3330_);
lean_ctor_set(v_reuseFailAlloc_3354_, 4, v_impl_3339_);
v___x_3353_ = v_reuseFailAlloc_3354_;
goto v_reusejp_3352_;
}
v_reusejp_3352_:
{
return v___x_3353_;
}
}
else
{
lean_object* v___x_3356_; uint8_t v_isShared_3357_; uint8_t v_isSharedCheck_3418_; 
v_isSharedCheck_3418_ = !lean_is_exclusive(v_impl_3339_);
if (v_isSharedCheck_3418_ == 0)
{
lean_object* v_unused_3419_; lean_object* v_unused_3420_; lean_object* v_unused_3421_; lean_object* v_unused_3422_; lean_object* v_unused_3423_; 
v_unused_3419_ = lean_ctor_get(v_impl_3339_, 4);
lean_dec(v_unused_3419_);
v_unused_3420_ = lean_ctor_get(v_impl_3339_, 3);
lean_dec(v_unused_3420_);
v_unused_3421_ = lean_ctor_get(v_impl_3339_, 2);
lean_dec(v_unused_3421_);
v_unused_3422_ = lean_ctor_get(v_impl_3339_, 1);
lean_dec(v_unused_3422_);
v_unused_3423_ = lean_ctor_get(v_impl_3339_, 0);
lean_dec(v_unused_3423_);
v___x_3356_ = v_impl_3339_;
v_isShared_3357_ = v_isSharedCheck_3418_;
goto v_resetjp_3355_;
}
else
{
lean_dec(v_impl_3339_);
v___x_3356_ = lean_box(0);
v_isShared_3357_ = v_isSharedCheck_3418_;
goto v_resetjp_3355_;
}
v_resetjp_3355_:
{
lean_object* v_size_3358_; lean_object* v_k_3359_; lean_object* v_v_3360_; lean_object* v_l_3361_; lean_object* v_r_3362_; lean_object* v_size_3363_; lean_object* v___x_3364_; lean_object* v___x_3365_; uint8_t v___x_3366_; 
v_size_3358_ = lean_ctor_get(v_l_3345_, 0);
v_k_3359_ = lean_ctor_get(v_l_3345_, 1);
v_v_3360_ = lean_ctor_get(v_l_3345_, 2);
v_l_3361_ = lean_ctor_get(v_l_3345_, 3);
v_r_3362_ = lean_ctor_get(v_l_3345_, 4);
v_size_3363_ = lean_ctor_get(v_r_3346_, 0);
v___x_3364_ = lean_unsigned_to_nat(2u);
v___x_3365_ = lean_nat_mul(v___x_3364_, v_size_3363_);
v___x_3366_ = lean_nat_dec_lt(v_size_3358_, v___x_3365_);
lean_dec(v___x_3365_);
if (v___x_3366_ == 0)
{
lean_object* v___x_3368_; uint8_t v_isShared_3369_; uint8_t v_isSharedCheck_3394_; 
lean_inc(v_r_3362_);
lean_inc(v_l_3361_);
lean_inc(v_v_3360_);
lean_inc(v_k_3359_);
v_isSharedCheck_3394_ = !lean_is_exclusive(v_l_3345_);
if (v_isSharedCheck_3394_ == 0)
{
lean_object* v_unused_3395_; lean_object* v_unused_3396_; lean_object* v_unused_3397_; lean_object* v_unused_3398_; lean_object* v_unused_3399_; 
v_unused_3395_ = lean_ctor_get(v_l_3345_, 4);
lean_dec(v_unused_3395_);
v_unused_3396_ = lean_ctor_get(v_l_3345_, 3);
lean_dec(v_unused_3396_);
v_unused_3397_ = lean_ctor_get(v_l_3345_, 2);
lean_dec(v_unused_3397_);
v_unused_3398_ = lean_ctor_get(v_l_3345_, 1);
lean_dec(v_unused_3398_);
v_unused_3399_ = lean_ctor_get(v_l_3345_, 0);
lean_dec(v_unused_3399_);
v___x_3368_ = v_l_3345_;
v_isShared_3369_ = v_isSharedCheck_3394_;
goto v_resetjp_3367_;
}
else
{
lean_dec(v_l_3345_);
v___x_3368_ = lean_box(0);
v_isShared_3369_ = v_isSharedCheck_3394_;
goto v_resetjp_3367_;
}
v_resetjp_3367_:
{
lean_object* v___x_3370_; lean_object* v___x_3371_; lean_object* v___y_3373_; lean_object* v___y_3374_; lean_object* v___y_3375_; lean_object* v___y_3384_; 
v___x_3370_ = lean_nat_add(v___x_3340_, v_size_3341_);
v___x_3371_ = lean_nat_add(v___x_3370_, v_size_3342_);
lean_dec(v_size_3342_);
if (lean_obj_tag(v_l_3361_) == 0)
{
lean_object* v_size_3392_; 
v_size_3392_ = lean_ctor_get(v_l_3361_, 0);
lean_inc(v_size_3392_);
v___y_3384_ = v_size_3392_;
goto v___jp_3383_;
}
else
{
lean_object* v___x_3393_; 
v___x_3393_ = lean_unsigned_to_nat(0u);
v___y_3384_ = v___x_3393_;
goto v___jp_3383_;
}
v___jp_3372_:
{
lean_object* v___x_3376_; lean_object* v___x_3378_; 
v___x_3376_ = lean_nat_add(v___y_3374_, v___y_3375_);
lean_dec(v___y_3375_);
lean_dec(v___y_3374_);
if (v_isShared_3369_ == 0)
{
lean_ctor_set(v___x_3368_, 4, v_r_3346_);
lean_ctor_set(v___x_3368_, 3, v_r_3362_);
lean_ctor_set(v___x_3368_, 2, v_v_3344_);
lean_ctor_set(v___x_3368_, 1, v_k_3343_);
lean_ctor_set(v___x_3368_, 0, v___x_3376_);
v___x_3378_ = v___x_3368_;
goto v_reusejp_3377_;
}
else
{
lean_object* v_reuseFailAlloc_3382_; 
v_reuseFailAlloc_3382_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3382_, 0, v___x_3376_);
lean_ctor_set(v_reuseFailAlloc_3382_, 1, v_k_3343_);
lean_ctor_set(v_reuseFailAlloc_3382_, 2, v_v_3344_);
lean_ctor_set(v_reuseFailAlloc_3382_, 3, v_r_3362_);
lean_ctor_set(v_reuseFailAlloc_3382_, 4, v_r_3346_);
v___x_3378_ = v_reuseFailAlloc_3382_;
goto v_reusejp_3377_;
}
v_reusejp_3377_:
{
lean_object* v___x_3380_; 
if (v_isShared_3357_ == 0)
{
lean_ctor_set(v___x_3356_, 4, v___x_3378_);
lean_ctor_set(v___x_3356_, 3, v___y_3373_);
lean_ctor_set(v___x_3356_, 2, v_v_3360_);
lean_ctor_set(v___x_3356_, 1, v_k_3359_);
lean_ctor_set(v___x_3356_, 0, v___x_3371_);
v___x_3380_ = v___x_3356_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3381_; 
v_reuseFailAlloc_3381_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3381_, 0, v___x_3371_);
lean_ctor_set(v_reuseFailAlloc_3381_, 1, v_k_3359_);
lean_ctor_set(v_reuseFailAlloc_3381_, 2, v_v_3360_);
lean_ctor_set(v_reuseFailAlloc_3381_, 3, v___y_3373_);
lean_ctor_set(v_reuseFailAlloc_3381_, 4, v___x_3378_);
v___x_3380_ = v_reuseFailAlloc_3381_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
return v___x_3380_;
}
}
}
v___jp_3383_:
{
lean_object* v___x_3385_; lean_object* v___x_3387_; 
v___x_3385_ = lean_nat_add(v___x_3370_, v___y_3384_);
lean_dec(v___y_3384_);
lean_dec(v___x_3370_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_l_3361_);
lean_ctor_set(v___x_3333_, 0, v___x_3385_);
v___x_3387_ = v___x_3333_;
goto v_reusejp_3386_;
}
else
{
lean_object* v_reuseFailAlloc_3391_; 
v_reuseFailAlloc_3391_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3391_, 0, v___x_3385_);
lean_ctor_set(v_reuseFailAlloc_3391_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3391_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3391_, 3, v_l_3330_);
lean_ctor_set(v_reuseFailAlloc_3391_, 4, v_l_3361_);
v___x_3387_ = v_reuseFailAlloc_3391_;
goto v_reusejp_3386_;
}
v_reusejp_3386_:
{
lean_object* v___x_3388_; 
v___x_3388_ = lean_nat_add(v___x_3340_, v_size_3363_);
if (lean_obj_tag(v_r_3362_) == 0)
{
lean_object* v_size_3389_; 
v_size_3389_ = lean_ctor_get(v_r_3362_, 0);
lean_inc(v_size_3389_);
v___y_3373_ = v___x_3387_;
v___y_3374_ = v___x_3388_;
v___y_3375_ = v_size_3389_;
goto v___jp_3372_;
}
else
{
lean_object* v___x_3390_; 
v___x_3390_ = lean_unsigned_to_nat(0u);
v___y_3373_ = v___x_3387_;
v___y_3374_ = v___x_3388_;
v___y_3375_ = v___x_3390_;
goto v___jp_3372_;
}
}
}
}
}
else
{
lean_object* v___x_3400_; lean_object* v___x_3401_; lean_object* v___x_3402_; lean_object* v___x_3404_; 
lean_del_object(v___x_3333_);
v___x_3400_ = lean_nat_add(v___x_3340_, v_size_3341_);
v___x_3401_ = lean_nat_add(v___x_3400_, v_size_3342_);
lean_dec(v_size_3342_);
v___x_3402_ = lean_nat_add(v___x_3400_, v_size_3358_);
lean_dec(v___x_3400_);
lean_inc_ref(v_l_3330_);
if (v_isShared_3357_ == 0)
{
lean_ctor_set(v___x_3356_, 4, v_l_3345_);
lean_ctor_set(v___x_3356_, 3, v_l_3330_);
lean_ctor_set(v___x_3356_, 2, v_v_3329_);
lean_ctor_set(v___x_3356_, 1, v_k_3328_);
lean_ctor_set(v___x_3356_, 0, v___x_3402_);
v___x_3404_ = v___x_3356_;
goto v_reusejp_3403_;
}
else
{
lean_object* v_reuseFailAlloc_3417_; 
v_reuseFailAlloc_3417_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3417_, 0, v___x_3402_);
lean_ctor_set(v_reuseFailAlloc_3417_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3417_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3417_, 3, v_l_3330_);
lean_ctor_set(v_reuseFailAlloc_3417_, 4, v_l_3345_);
v___x_3404_ = v_reuseFailAlloc_3417_;
goto v_reusejp_3403_;
}
v_reusejp_3403_:
{
lean_object* v___x_3406_; uint8_t v_isShared_3407_; uint8_t v_isSharedCheck_3411_; 
v_isSharedCheck_3411_ = !lean_is_exclusive(v_l_3330_);
if (v_isSharedCheck_3411_ == 0)
{
lean_object* v_unused_3412_; lean_object* v_unused_3413_; lean_object* v_unused_3414_; lean_object* v_unused_3415_; lean_object* v_unused_3416_; 
v_unused_3412_ = lean_ctor_get(v_l_3330_, 4);
lean_dec(v_unused_3412_);
v_unused_3413_ = lean_ctor_get(v_l_3330_, 3);
lean_dec(v_unused_3413_);
v_unused_3414_ = lean_ctor_get(v_l_3330_, 2);
lean_dec(v_unused_3414_);
v_unused_3415_ = lean_ctor_get(v_l_3330_, 1);
lean_dec(v_unused_3415_);
v_unused_3416_ = lean_ctor_get(v_l_3330_, 0);
lean_dec(v_unused_3416_);
v___x_3406_ = v_l_3330_;
v_isShared_3407_ = v_isSharedCheck_3411_;
goto v_resetjp_3405_;
}
else
{
lean_dec(v_l_3330_);
v___x_3406_ = lean_box(0);
v_isShared_3407_ = v_isSharedCheck_3411_;
goto v_resetjp_3405_;
}
v_resetjp_3405_:
{
lean_object* v___x_3409_; 
if (v_isShared_3407_ == 0)
{
lean_ctor_set(v___x_3406_, 4, v_r_3346_);
lean_ctor_set(v___x_3406_, 3, v___x_3404_);
lean_ctor_set(v___x_3406_, 2, v_v_3344_);
lean_ctor_set(v___x_3406_, 1, v_k_3343_);
lean_ctor_set(v___x_3406_, 0, v___x_3401_);
v___x_3409_ = v___x_3406_;
goto v_reusejp_3408_;
}
else
{
lean_object* v_reuseFailAlloc_3410_; 
v_reuseFailAlloc_3410_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3410_, 0, v___x_3401_);
lean_ctor_set(v_reuseFailAlloc_3410_, 1, v_k_3343_);
lean_ctor_set(v_reuseFailAlloc_3410_, 2, v_v_3344_);
lean_ctor_set(v_reuseFailAlloc_3410_, 3, v___x_3404_);
lean_ctor_set(v_reuseFailAlloc_3410_, 4, v_r_3346_);
v___x_3409_ = v_reuseFailAlloc_3410_;
goto v_reusejp_3408_;
}
v_reusejp_3408_:
{
return v___x_3409_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_3424_; 
v_l_3424_ = lean_ctor_get(v_impl_3339_, 3);
lean_inc(v_l_3424_);
if (lean_obj_tag(v_l_3424_) == 0)
{
lean_object* v_r_3425_; lean_object* v_k_3426_; lean_object* v_v_3427_; lean_object* v___x_3429_; uint8_t v_isShared_3430_; uint8_t v_isSharedCheck_3450_; 
v_r_3425_ = lean_ctor_get(v_impl_3339_, 4);
v_k_3426_ = lean_ctor_get(v_impl_3339_, 1);
v_v_3427_ = lean_ctor_get(v_impl_3339_, 2);
v_isSharedCheck_3450_ = !lean_is_exclusive(v_impl_3339_);
if (v_isSharedCheck_3450_ == 0)
{
lean_object* v_unused_3451_; lean_object* v_unused_3452_; 
v_unused_3451_ = lean_ctor_get(v_impl_3339_, 3);
lean_dec(v_unused_3451_);
v_unused_3452_ = lean_ctor_get(v_impl_3339_, 0);
lean_dec(v_unused_3452_);
v___x_3429_ = v_impl_3339_;
v_isShared_3430_ = v_isSharedCheck_3450_;
goto v_resetjp_3428_;
}
else
{
lean_inc(v_r_3425_);
lean_inc(v_v_3427_);
lean_inc(v_k_3426_);
lean_dec(v_impl_3339_);
v___x_3429_ = lean_box(0);
v_isShared_3430_ = v_isSharedCheck_3450_;
goto v_resetjp_3428_;
}
v_resetjp_3428_:
{
lean_object* v_k_3431_; lean_object* v_v_3432_; lean_object* v___x_3434_; uint8_t v_isShared_3435_; uint8_t v_isSharedCheck_3446_; 
v_k_3431_ = lean_ctor_get(v_l_3424_, 1);
v_v_3432_ = lean_ctor_get(v_l_3424_, 2);
v_isSharedCheck_3446_ = !lean_is_exclusive(v_l_3424_);
if (v_isSharedCheck_3446_ == 0)
{
lean_object* v_unused_3447_; lean_object* v_unused_3448_; lean_object* v_unused_3449_; 
v_unused_3447_ = lean_ctor_get(v_l_3424_, 4);
lean_dec(v_unused_3447_);
v_unused_3448_ = lean_ctor_get(v_l_3424_, 3);
lean_dec(v_unused_3448_);
v_unused_3449_ = lean_ctor_get(v_l_3424_, 0);
lean_dec(v_unused_3449_);
v___x_3434_ = v_l_3424_;
v_isShared_3435_ = v_isSharedCheck_3446_;
goto v_resetjp_3433_;
}
else
{
lean_inc(v_v_3432_);
lean_inc(v_k_3431_);
lean_dec(v_l_3424_);
v___x_3434_ = lean_box(0);
v_isShared_3435_ = v_isSharedCheck_3446_;
goto v_resetjp_3433_;
}
v_resetjp_3433_:
{
lean_object* v___x_3436_; lean_object* v___x_3438_; 
v___x_3436_ = lean_unsigned_to_nat(3u);
lean_inc_n(v_r_3425_, 2);
if (v_isShared_3435_ == 0)
{
lean_ctor_set(v___x_3434_, 4, v_r_3425_);
lean_ctor_set(v___x_3434_, 3, v_r_3425_);
lean_ctor_set(v___x_3434_, 2, v_v_3329_);
lean_ctor_set(v___x_3434_, 1, v_k_3328_);
lean_ctor_set(v___x_3434_, 0, v___x_3340_);
v___x_3438_ = v___x_3434_;
goto v_reusejp_3437_;
}
else
{
lean_object* v_reuseFailAlloc_3445_; 
v_reuseFailAlloc_3445_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3445_, 0, v___x_3340_);
lean_ctor_set(v_reuseFailAlloc_3445_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3445_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3445_, 3, v_r_3425_);
lean_ctor_set(v_reuseFailAlloc_3445_, 4, v_r_3425_);
v___x_3438_ = v_reuseFailAlloc_3445_;
goto v_reusejp_3437_;
}
v_reusejp_3437_:
{
lean_object* v___x_3440_; 
lean_inc(v_r_3425_);
if (v_isShared_3430_ == 0)
{
lean_ctor_set(v___x_3429_, 3, v_r_3425_);
lean_ctor_set(v___x_3429_, 0, v___x_3340_);
v___x_3440_ = v___x_3429_;
goto v_reusejp_3439_;
}
else
{
lean_object* v_reuseFailAlloc_3444_; 
v_reuseFailAlloc_3444_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3444_, 0, v___x_3340_);
lean_ctor_set(v_reuseFailAlloc_3444_, 1, v_k_3426_);
lean_ctor_set(v_reuseFailAlloc_3444_, 2, v_v_3427_);
lean_ctor_set(v_reuseFailAlloc_3444_, 3, v_r_3425_);
lean_ctor_set(v_reuseFailAlloc_3444_, 4, v_r_3425_);
v___x_3440_ = v_reuseFailAlloc_3444_;
goto v_reusejp_3439_;
}
v_reusejp_3439_:
{
lean_object* v___x_3442_; 
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v___x_3440_);
lean_ctor_set(v___x_3333_, 3, v___x_3438_);
lean_ctor_set(v___x_3333_, 2, v_v_3432_);
lean_ctor_set(v___x_3333_, 1, v_k_3431_);
lean_ctor_set(v___x_3333_, 0, v___x_3436_);
v___x_3442_ = v___x_3333_;
goto v_reusejp_3441_;
}
else
{
lean_object* v_reuseFailAlloc_3443_; 
v_reuseFailAlloc_3443_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3443_, 0, v___x_3436_);
lean_ctor_set(v_reuseFailAlloc_3443_, 1, v_k_3431_);
lean_ctor_set(v_reuseFailAlloc_3443_, 2, v_v_3432_);
lean_ctor_set(v_reuseFailAlloc_3443_, 3, v___x_3438_);
lean_ctor_set(v_reuseFailAlloc_3443_, 4, v___x_3440_);
v___x_3442_ = v_reuseFailAlloc_3443_;
goto v_reusejp_3441_;
}
v_reusejp_3441_:
{
return v___x_3442_;
}
}
}
}
}
}
else
{
lean_object* v_r_3453_; 
v_r_3453_ = lean_ctor_get(v_impl_3339_, 4);
lean_inc(v_r_3453_);
if (lean_obj_tag(v_r_3453_) == 0)
{
lean_object* v_k_3454_; lean_object* v_v_3455_; lean_object* v___x_3457_; uint8_t v_isShared_3458_; uint8_t v_isSharedCheck_3466_; 
v_k_3454_ = lean_ctor_get(v_impl_3339_, 1);
v_v_3455_ = lean_ctor_get(v_impl_3339_, 2);
v_isSharedCheck_3466_ = !lean_is_exclusive(v_impl_3339_);
if (v_isSharedCheck_3466_ == 0)
{
lean_object* v_unused_3467_; lean_object* v_unused_3468_; lean_object* v_unused_3469_; 
v_unused_3467_ = lean_ctor_get(v_impl_3339_, 4);
lean_dec(v_unused_3467_);
v_unused_3468_ = lean_ctor_get(v_impl_3339_, 3);
lean_dec(v_unused_3468_);
v_unused_3469_ = lean_ctor_get(v_impl_3339_, 0);
lean_dec(v_unused_3469_);
v___x_3457_ = v_impl_3339_;
v_isShared_3458_ = v_isSharedCheck_3466_;
goto v_resetjp_3456_;
}
else
{
lean_inc(v_v_3455_);
lean_inc(v_k_3454_);
lean_dec(v_impl_3339_);
v___x_3457_ = lean_box(0);
v_isShared_3458_ = v_isSharedCheck_3466_;
goto v_resetjp_3456_;
}
v_resetjp_3456_:
{
lean_object* v___x_3459_; lean_object* v___x_3461_; 
v___x_3459_ = lean_unsigned_to_nat(3u);
if (v_isShared_3458_ == 0)
{
lean_ctor_set(v___x_3457_, 4, v_l_3424_);
lean_ctor_set(v___x_3457_, 2, v_v_3329_);
lean_ctor_set(v___x_3457_, 1, v_k_3328_);
lean_ctor_set(v___x_3457_, 0, v___x_3340_);
v___x_3461_ = v___x_3457_;
goto v_reusejp_3460_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v___x_3340_);
lean_ctor_set(v_reuseFailAlloc_3465_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3465_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3465_, 3, v_l_3424_);
lean_ctor_set(v_reuseFailAlloc_3465_, 4, v_l_3424_);
v___x_3461_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3460_;
}
v_reusejp_3460_:
{
lean_object* v___x_3463_; 
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_r_3453_);
lean_ctor_set(v___x_3333_, 3, v___x_3461_);
lean_ctor_set(v___x_3333_, 2, v_v_3455_);
lean_ctor_set(v___x_3333_, 1, v_k_3454_);
lean_ctor_set(v___x_3333_, 0, v___x_3459_);
v___x_3463_ = v___x_3333_;
goto v_reusejp_3462_;
}
else
{
lean_object* v_reuseFailAlloc_3464_; 
v_reuseFailAlloc_3464_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3464_, 0, v___x_3459_);
lean_ctor_set(v_reuseFailAlloc_3464_, 1, v_k_3454_);
lean_ctor_set(v_reuseFailAlloc_3464_, 2, v_v_3455_);
lean_ctor_set(v_reuseFailAlloc_3464_, 3, v___x_3461_);
lean_ctor_set(v_reuseFailAlloc_3464_, 4, v_r_3453_);
v___x_3463_ = v_reuseFailAlloc_3464_;
goto v_reusejp_3462_;
}
v_reusejp_3462_:
{
return v___x_3463_;
}
}
}
}
else
{
lean_object* v___x_3470_; lean_object* v___x_3472_; 
v___x_3470_ = lean_unsigned_to_nat(2u);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_impl_3339_);
lean_ctor_set(v___x_3333_, 3, v_r_3453_);
lean_ctor_set(v___x_3333_, 0, v___x_3470_);
v___x_3472_ = v___x_3333_;
goto v_reusejp_3471_;
}
else
{
lean_object* v_reuseFailAlloc_3473_; 
v_reuseFailAlloc_3473_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3473_, 0, v___x_3470_);
lean_ctor_set(v_reuseFailAlloc_3473_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3473_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3473_, 3, v_r_3453_);
lean_ctor_set(v_reuseFailAlloc_3473_, 4, v_impl_3339_);
v___x_3472_ = v_reuseFailAlloc_3473_;
goto v_reusejp_3471_;
}
v_reusejp_3471_:
{
return v___x_3472_;
}
}
}
}
}
else
{
lean_object* v___x_3474_; lean_object* v___x_3476_; 
lean_dec(v_v_3329_);
lean_dec(v_k_3328_);
v___x_3474_ = lean_box_uint64(v_k_3324_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 2, v_v_3325_);
lean_ctor_set(v___x_3333_, 1, v___x_3474_);
v___x_3476_ = v___x_3333_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3477_; 
v_reuseFailAlloc_3477_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3477_, 0, v_size_3327_);
lean_ctor_set(v_reuseFailAlloc_3477_, 1, v___x_3474_);
lean_ctor_set(v_reuseFailAlloc_3477_, 2, v_v_3325_);
lean_ctor_set(v_reuseFailAlloc_3477_, 3, v_l_3330_);
lean_ctor_set(v_reuseFailAlloc_3477_, 4, v_r_3331_);
v___x_3476_ = v_reuseFailAlloc_3477_;
goto v_reusejp_3475_;
}
v_reusejp_3475_:
{
return v___x_3476_;
}
}
}
else
{
lean_object* v_impl_3478_; lean_object* v___x_3479_; 
lean_dec(v_size_3327_);
v_impl_3478_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(v_k_3324_, v_v_3325_, v_l_3330_);
v___x_3479_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_r_3331_) == 0)
{
lean_object* v_size_3480_; lean_object* v_size_3481_; lean_object* v_k_3482_; lean_object* v_v_3483_; lean_object* v_l_3484_; lean_object* v_r_3485_; lean_object* v___x_3486_; lean_object* v___x_3487_; uint8_t v___x_3488_; 
v_size_3480_ = lean_ctor_get(v_r_3331_, 0);
v_size_3481_ = lean_ctor_get(v_impl_3478_, 0);
lean_inc(v_size_3481_);
v_k_3482_ = lean_ctor_get(v_impl_3478_, 1);
lean_inc(v_k_3482_);
v_v_3483_ = lean_ctor_get(v_impl_3478_, 2);
lean_inc(v_v_3483_);
v_l_3484_ = lean_ctor_get(v_impl_3478_, 3);
lean_inc(v_l_3484_);
v_r_3485_ = lean_ctor_get(v_impl_3478_, 4);
lean_inc(v_r_3485_);
v___x_3486_ = lean_unsigned_to_nat(3u);
v___x_3487_ = lean_nat_mul(v___x_3486_, v_size_3480_);
v___x_3488_ = lean_nat_dec_lt(v___x_3487_, v_size_3481_);
lean_dec(v___x_3487_);
if (v___x_3488_ == 0)
{
lean_object* v___x_3489_; lean_object* v___x_3490_; lean_object* v___x_3492_; 
lean_dec(v_r_3485_);
lean_dec(v_l_3484_);
lean_dec(v_v_3483_);
lean_dec(v_k_3482_);
v___x_3489_ = lean_nat_add(v___x_3479_, v_size_3481_);
lean_dec(v_size_3481_);
v___x_3490_ = lean_nat_add(v___x_3489_, v_size_3480_);
lean_dec(v___x_3489_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 3, v_impl_3478_);
lean_ctor_set(v___x_3333_, 0, v___x_3490_);
v___x_3492_ = v___x_3333_;
goto v_reusejp_3491_;
}
else
{
lean_object* v_reuseFailAlloc_3493_; 
v_reuseFailAlloc_3493_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3493_, 0, v___x_3490_);
lean_ctor_set(v_reuseFailAlloc_3493_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3493_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3493_, 3, v_impl_3478_);
lean_ctor_set(v_reuseFailAlloc_3493_, 4, v_r_3331_);
v___x_3492_ = v_reuseFailAlloc_3493_;
goto v_reusejp_3491_;
}
v_reusejp_3491_:
{
return v___x_3492_;
}
}
else
{
lean_object* v___x_3495_; uint8_t v_isShared_3496_; uint8_t v_isSharedCheck_3559_; 
v_isSharedCheck_3559_ = !lean_is_exclusive(v_impl_3478_);
if (v_isSharedCheck_3559_ == 0)
{
lean_object* v_unused_3560_; lean_object* v_unused_3561_; lean_object* v_unused_3562_; lean_object* v_unused_3563_; lean_object* v_unused_3564_; 
v_unused_3560_ = lean_ctor_get(v_impl_3478_, 4);
lean_dec(v_unused_3560_);
v_unused_3561_ = lean_ctor_get(v_impl_3478_, 3);
lean_dec(v_unused_3561_);
v_unused_3562_ = lean_ctor_get(v_impl_3478_, 2);
lean_dec(v_unused_3562_);
v_unused_3563_ = lean_ctor_get(v_impl_3478_, 1);
lean_dec(v_unused_3563_);
v_unused_3564_ = lean_ctor_get(v_impl_3478_, 0);
lean_dec(v_unused_3564_);
v___x_3495_ = v_impl_3478_;
v_isShared_3496_ = v_isSharedCheck_3559_;
goto v_resetjp_3494_;
}
else
{
lean_dec(v_impl_3478_);
v___x_3495_ = lean_box(0);
v_isShared_3496_ = v_isSharedCheck_3559_;
goto v_resetjp_3494_;
}
v_resetjp_3494_:
{
lean_object* v_size_3497_; lean_object* v_size_3498_; lean_object* v_k_3499_; lean_object* v_v_3500_; lean_object* v_l_3501_; lean_object* v_r_3502_; lean_object* v___x_3503_; lean_object* v___x_3504_; uint8_t v___x_3505_; 
v_size_3497_ = lean_ctor_get(v_l_3484_, 0);
v_size_3498_ = lean_ctor_get(v_r_3485_, 0);
v_k_3499_ = lean_ctor_get(v_r_3485_, 1);
v_v_3500_ = lean_ctor_get(v_r_3485_, 2);
v_l_3501_ = lean_ctor_get(v_r_3485_, 3);
v_r_3502_ = lean_ctor_get(v_r_3485_, 4);
v___x_3503_ = lean_unsigned_to_nat(2u);
v___x_3504_ = lean_nat_mul(v___x_3503_, v_size_3497_);
v___x_3505_ = lean_nat_dec_lt(v_size_3498_, v___x_3504_);
lean_dec(v___x_3504_);
if (v___x_3505_ == 0)
{
lean_object* v___x_3507_; uint8_t v_isShared_3508_; uint8_t v_isSharedCheck_3534_; 
lean_inc(v_r_3502_);
lean_inc(v_l_3501_);
lean_inc(v_v_3500_);
lean_inc(v_k_3499_);
v_isSharedCheck_3534_ = !lean_is_exclusive(v_r_3485_);
if (v_isSharedCheck_3534_ == 0)
{
lean_object* v_unused_3535_; lean_object* v_unused_3536_; lean_object* v_unused_3537_; lean_object* v_unused_3538_; lean_object* v_unused_3539_; 
v_unused_3535_ = lean_ctor_get(v_r_3485_, 4);
lean_dec(v_unused_3535_);
v_unused_3536_ = lean_ctor_get(v_r_3485_, 3);
lean_dec(v_unused_3536_);
v_unused_3537_ = lean_ctor_get(v_r_3485_, 2);
lean_dec(v_unused_3537_);
v_unused_3538_ = lean_ctor_get(v_r_3485_, 1);
lean_dec(v_unused_3538_);
v_unused_3539_ = lean_ctor_get(v_r_3485_, 0);
lean_dec(v_unused_3539_);
v___x_3507_ = v_r_3485_;
v_isShared_3508_ = v_isSharedCheck_3534_;
goto v_resetjp_3506_;
}
else
{
lean_dec(v_r_3485_);
v___x_3507_ = lean_box(0);
v_isShared_3508_ = v_isSharedCheck_3534_;
goto v_resetjp_3506_;
}
v_resetjp_3506_:
{
lean_object* v___x_3509_; lean_object* v___x_3510_; lean_object* v___y_3512_; lean_object* v___y_3513_; lean_object* v___y_3514_; lean_object* v___x_3522_; lean_object* v___y_3524_; 
v___x_3509_ = lean_nat_add(v___x_3479_, v_size_3481_);
lean_dec(v_size_3481_);
v___x_3510_ = lean_nat_add(v___x_3509_, v_size_3480_);
lean_dec(v___x_3509_);
v___x_3522_ = lean_nat_add(v___x_3479_, v_size_3497_);
if (lean_obj_tag(v_l_3501_) == 0)
{
lean_object* v_size_3532_; 
v_size_3532_ = lean_ctor_get(v_l_3501_, 0);
lean_inc(v_size_3532_);
v___y_3524_ = v_size_3532_;
goto v___jp_3523_;
}
else
{
lean_object* v___x_3533_; 
v___x_3533_ = lean_unsigned_to_nat(0u);
v___y_3524_ = v___x_3533_;
goto v___jp_3523_;
}
v___jp_3511_:
{
lean_object* v___x_3515_; lean_object* v___x_3517_; 
v___x_3515_ = lean_nat_add(v___y_3513_, v___y_3514_);
lean_dec(v___y_3514_);
lean_dec(v___y_3513_);
if (v_isShared_3508_ == 0)
{
lean_ctor_set(v___x_3507_, 4, v_r_3331_);
lean_ctor_set(v___x_3507_, 3, v_r_3502_);
lean_ctor_set(v___x_3507_, 2, v_v_3329_);
lean_ctor_set(v___x_3507_, 1, v_k_3328_);
lean_ctor_set(v___x_3507_, 0, v___x_3515_);
v___x_3517_ = v___x_3507_;
goto v_reusejp_3516_;
}
else
{
lean_object* v_reuseFailAlloc_3521_; 
v_reuseFailAlloc_3521_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3521_, 0, v___x_3515_);
lean_ctor_set(v_reuseFailAlloc_3521_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3521_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3521_, 3, v_r_3502_);
lean_ctor_set(v_reuseFailAlloc_3521_, 4, v_r_3331_);
v___x_3517_ = v_reuseFailAlloc_3521_;
goto v_reusejp_3516_;
}
v_reusejp_3516_:
{
lean_object* v___x_3519_; 
if (v_isShared_3496_ == 0)
{
lean_ctor_set(v___x_3495_, 4, v___x_3517_);
lean_ctor_set(v___x_3495_, 3, v___y_3512_);
lean_ctor_set(v___x_3495_, 2, v_v_3500_);
lean_ctor_set(v___x_3495_, 1, v_k_3499_);
lean_ctor_set(v___x_3495_, 0, v___x_3510_);
v___x_3519_ = v___x_3495_;
goto v_reusejp_3518_;
}
else
{
lean_object* v_reuseFailAlloc_3520_; 
v_reuseFailAlloc_3520_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3520_, 0, v___x_3510_);
lean_ctor_set(v_reuseFailAlloc_3520_, 1, v_k_3499_);
lean_ctor_set(v_reuseFailAlloc_3520_, 2, v_v_3500_);
lean_ctor_set(v_reuseFailAlloc_3520_, 3, v___y_3512_);
lean_ctor_set(v_reuseFailAlloc_3520_, 4, v___x_3517_);
v___x_3519_ = v_reuseFailAlloc_3520_;
goto v_reusejp_3518_;
}
v_reusejp_3518_:
{
return v___x_3519_;
}
}
}
v___jp_3523_:
{
lean_object* v___x_3525_; lean_object* v___x_3527_; 
v___x_3525_ = lean_nat_add(v___x_3522_, v___y_3524_);
lean_dec(v___y_3524_);
lean_dec(v___x_3522_);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_l_3501_);
lean_ctor_set(v___x_3333_, 3, v_l_3484_);
lean_ctor_set(v___x_3333_, 2, v_v_3483_);
lean_ctor_set(v___x_3333_, 1, v_k_3482_);
lean_ctor_set(v___x_3333_, 0, v___x_3525_);
v___x_3527_ = v___x_3333_;
goto v_reusejp_3526_;
}
else
{
lean_object* v_reuseFailAlloc_3531_; 
v_reuseFailAlloc_3531_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3531_, 0, v___x_3525_);
lean_ctor_set(v_reuseFailAlloc_3531_, 1, v_k_3482_);
lean_ctor_set(v_reuseFailAlloc_3531_, 2, v_v_3483_);
lean_ctor_set(v_reuseFailAlloc_3531_, 3, v_l_3484_);
lean_ctor_set(v_reuseFailAlloc_3531_, 4, v_l_3501_);
v___x_3527_ = v_reuseFailAlloc_3531_;
goto v_reusejp_3526_;
}
v_reusejp_3526_:
{
lean_object* v___x_3528_; 
v___x_3528_ = lean_nat_add(v___x_3479_, v_size_3480_);
if (lean_obj_tag(v_r_3502_) == 0)
{
lean_object* v_size_3529_; 
v_size_3529_ = lean_ctor_get(v_r_3502_, 0);
lean_inc(v_size_3529_);
v___y_3512_ = v___x_3527_;
v___y_3513_ = v___x_3528_;
v___y_3514_ = v_size_3529_;
goto v___jp_3511_;
}
else
{
lean_object* v___x_3530_; 
v___x_3530_ = lean_unsigned_to_nat(0u);
v___y_3512_ = v___x_3527_;
v___y_3513_ = v___x_3528_;
v___y_3514_ = v___x_3530_;
goto v___jp_3511_;
}
}
}
}
}
else
{
lean_object* v___x_3540_; lean_object* v___x_3541_; lean_object* v___x_3542_; lean_object* v___x_3543_; lean_object* v___x_3545_; 
lean_del_object(v___x_3333_);
v___x_3540_ = lean_nat_add(v___x_3479_, v_size_3481_);
lean_dec(v_size_3481_);
v___x_3541_ = lean_nat_add(v___x_3540_, v_size_3480_);
lean_dec(v___x_3540_);
v___x_3542_ = lean_nat_add(v___x_3479_, v_size_3480_);
v___x_3543_ = lean_nat_add(v___x_3542_, v_size_3498_);
lean_dec(v___x_3542_);
lean_inc_ref(v_r_3331_);
if (v_isShared_3496_ == 0)
{
lean_ctor_set(v___x_3495_, 4, v_r_3331_);
lean_ctor_set(v___x_3495_, 3, v_r_3485_);
lean_ctor_set(v___x_3495_, 2, v_v_3329_);
lean_ctor_set(v___x_3495_, 1, v_k_3328_);
lean_ctor_set(v___x_3495_, 0, v___x_3543_);
v___x_3545_ = v___x_3495_;
goto v_reusejp_3544_;
}
else
{
lean_object* v_reuseFailAlloc_3558_; 
v_reuseFailAlloc_3558_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3558_, 0, v___x_3543_);
lean_ctor_set(v_reuseFailAlloc_3558_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3558_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3558_, 3, v_r_3485_);
lean_ctor_set(v_reuseFailAlloc_3558_, 4, v_r_3331_);
v___x_3545_ = v_reuseFailAlloc_3558_;
goto v_reusejp_3544_;
}
v_reusejp_3544_:
{
lean_object* v___x_3547_; uint8_t v_isShared_3548_; uint8_t v_isSharedCheck_3552_; 
v_isSharedCheck_3552_ = !lean_is_exclusive(v_r_3331_);
if (v_isSharedCheck_3552_ == 0)
{
lean_object* v_unused_3553_; lean_object* v_unused_3554_; lean_object* v_unused_3555_; lean_object* v_unused_3556_; lean_object* v_unused_3557_; 
v_unused_3553_ = lean_ctor_get(v_r_3331_, 4);
lean_dec(v_unused_3553_);
v_unused_3554_ = lean_ctor_get(v_r_3331_, 3);
lean_dec(v_unused_3554_);
v_unused_3555_ = lean_ctor_get(v_r_3331_, 2);
lean_dec(v_unused_3555_);
v_unused_3556_ = lean_ctor_get(v_r_3331_, 1);
lean_dec(v_unused_3556_);
v_unused_3557_ = lean_ctor_get(v_r_3331_, 0);
lean_dec(v_unused_3557_);
v___x_3547_ = v_r_3331_;
v_isShared_3548_ = v_isSharedCheck_3552_;
goto v_resetjp_3546_;
}
else
{
lean_dec(v_r_3331_);
v___x_3547_ = lean_box(0);
v_isShared_3548_ = v_isSharedCheck_3552_;
goto v_resetjp_3546_;
}
v_resetjp_3546_:
{
lean_object* v___x_3550_; 
if (v_isShared_3548_ == 0)
{
lean_ctor_set(v___x_3547_, 4, v___x_3545_);
lean_ctor_set(v___x_3547_, 3, v_l_3484_);
lean_ctor_set(v___x_3547_, 2, v_v_3483_);
lean_ctor_set(v___x_3547_, 1, v_k_3482_);
lean_ctor_set(v___x_3547_, 0, v___x_3541_);
v___x_3550_ = v___x_3547_;
goto v_reusejp_3549_;
}
else
{
lean_object* v_reuseFailAlloc_3551_; 
v_reuseFailAlloc_3551_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3551_, 0, v___x_3541_);
lean_ctor_set(v_reuseFailAlloc_3551_, 1, v_k_3482_);
lean_ctor_set(v_reuseFailAlloc_3551_, 2, v_v_3483_);
lean_ctor_set(v_reuseFailAlloc_3551_, 3, v_l_3484_);
lean_ctor_set(v_reuseFailAlloc_3551_, 4, v___x_3545_);
v___x_3550_ = v_reuseFailAlloc_3551_;
goto v_reusejp_3549_;
}
v_reusejp_3549_:
{
return v___x_3550_;
}
}
}
}
}
}
}
else
{
lean_object* v_l_3565_; 
v_l_3565_ = lean_ctor_get(v_impl_3478_, 3);
lean_inc(v_l_3565_);
if (lean_obj_tag(v_l_3565_) == 0)
{
lean_object* v_r_3566_; lean_object* v_k_3567_; lean_object* v_v_3568_; lean_object* v___x_3570_; uint8_t v_isShared_3571_; uint8_t v_isSharedCheck_3579_; 
v_r_3566_ = lean_ctor_get(v_impl_3478_, 4);
v_k_3567_ = lean_ctor_get(v_impl_3478_, 1);
v_v_3568_ = lean_ctor_get(v_impl_3478_, 2);
v_isSharedCheck_3579_ = !lean_is_exclusive(v_impl_3478_);
if (v_isSharedCheck_3579_ == 0)
{
lean_object* v_unused_3580_; lean_object* v_unused_3581_; 
v_unused_3580_ = lean_ctor_get(v_impl_3478_, 3);
lean_dec(v_unused_3580_);
v_unused_3581_ = lean_ctor_get(v_impl_3478_, 0);
lean_dec(v_unused_3581_);
v___x_3570_ = v_impl_3478_;
v_isShared_3571_ = v_isSharedCheck_3579_;
goto v_resetjp_3569_;
}
else
{
lean_inc(v_r_3566_);
lean_inc(v_v_3568_);
lean_inc(v_k_3567_);
lean_dec(v_impl_3478_);
v___x_3570_ = lean_box(0);
v_isShared_3571_ = v_isSharedCheck_3579_;
goto v_resetjp_3569_;
}
v_resetjp_3569_:
{
lean_object* v___x_3572_; lean_object* v___x_3574_; 
v___x_3572_ = lean_unsigned_to_nat(3u);
lean_inc(v_r_3566_);
if (v_isShared_3571_ == 0)
{
lean_ctor_set(v___x_3570_, 3, v_r_3566_);
lean_ctor_set(v___x_3570_, 2, v_v_3329_);
lean_ctor_set(v___x_3570_, 1, v_k_3328_);
lean_ctor_set(v___x_3570_, 0, v___x_3479_);
v___x_3574_ = v___x_3570_;
goto v_reusejp_3573_;
}
else
{
lean_object* v_reuseFailAlloc_3578_; 
v_reuseFailAlloc_3578_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3578_, 0, v___x_3479_);
lean_ctor_set(v_reuseFailAlloc_3578_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3578_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3578_, 3, v_r_3566_);
lean_ctor_set(v_reuseFailAlloc_3578_, 4, v_r_3566_);
v___x_3574_ = v_reuseFailAlloc_3578_;
goto v_reusejp_3573_;
}
v_reusejp_3573_:
{
lean_object* v___x_3576_; 
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v___x_3574_);
lean_ctor_set(v___x_3333_, 3, v_l_3565_);
lean_ctor_set(v___x_3333_, 2, v_v_3568_);
lean_ctor_set(v___x_3333_, 1, v_k_3567_);
lean_ctor_set(v___x_3333_, 0, v___x_3572_);
v___x_3576_ = v___x_3333_;
goto v_reusejp_3575_;
}
else
{
lean_object* v_reuseFailAlloc_3577_; 
v_reuseFailAlloc_3577_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3577_, 0, v___x_3572_);
lean_ctor_set(v_reuseFailAlloc_3577_, 1, v_k_3567_);
lean_ctor_set(v_reuseFailAlloc_3577_, 2, v_v_3568_);
lean_ctor_set(v_reuseFailAlloc_3577_, 3, v_l_3565_);
lean_ctor_set(v_reuseFailAlloc_3577_, 4, v___x_3574_);
v___x_3576_ = v_reuseFailAlloc_3577_;
goto v_reusejp_3575_;
}
v_reusejp_3575_:
{
return v___x_3576_;
}
}
}
}
else
{
lean_object* v_r_3582_; 
v_r_3582_ = lean_ctor_get(v_impl_3478_, 4);
lean_inc(v_r_3582_);
if (lean_obj_tag(v_r_3582_) == 0)
{
lean_object* v_k_3583_; lean_object* v_v_3584_; lean_object* v___x_3586_; uint8_t v_isShared_3587_; uint8_t v_isSharedCheck_3607_; 
v_k_3583_ = lean_ctor_get(v_impl_3478_, 1);
v_v_3584_ = lean_ctor_get(v_impl_3478_, 2);
v_isSharedCheck_3607_ = !lean_is_exclusive(v_impl_3478_);
if (v_isSharedCheck_3607_ == 0)
{
lean_object* v_unused_3608_; lean_object* v_unused_3609_; lean_object* v_unused_3610_; 
v_unused_3608_ = lean_ctor_get(v_impl_3478_, 4);
lean_dec(v_unused_3608_);
v_unused_3609_ = lean_ctor_get(v_impl_3478_, 3);
lean_dec(v_unused_3609_);
v_unused_3610_ = lean_ctor_get(v_impl_3478_, 0);
lean_dec(v_unused_3610_);
v___x_3586_ = v_impl_3478_;
v_isShared_3587_ = v_isSharedCheck_3607_;
goto v_resetjp_3585_;
}
else
{
lean_inc(v_v_3584_);
lean_inc(v_k_3583_);
lean_dec(v_impl_3478_);
v___x_3586_ = lean_box(0);
v_isShared_3587_ = v_isSharedCheck_3607_;
goto v_resetjp_3585_;
}
v_resetjp_3585_:
{
lean_object* v_k_3588_; lean_object* v_v_3589_; lean_object* v___x_3591_; uint8_t v_isShared_3592_; uint8_t v_isSharedCheck_3603_; 
v_k_3588_ = lean_ctor_get(v_r_3582_, 1);
v_v_3589_ = lean_ctor_get(v_r_3582_, 2);
v_isSharedCheck_3603_ = !lean_is_exclusive(v_r_3582_);
if (v_isSharedCheck_3603_ == 0)
{
lean_object* v_unused_3604_; lean_object* v_unused_3605_; lean_object* v_unused_3606_; 
v_unused_3604_ = lean_ctor_get(v_r_3582_, 4);
lean_dec(v_unused_3604_);
v_unused_3605_ = lean_ctor_get(v_r_3582_, 3);
lean_dec(v_unused_3605_);
v_unused_3606_ = lean_ctor_get(v_r_3582_, 0);
lean_dec(v_unused_3606_);
v___x_3591_ = v_r_3582_;
v_isShared_3592_ = v_isSharedCheck_3603_;
goto v_resetjp_3590_;
}
else
{
lean_inc(v_v_3589_);
lean_inc(v_k_3588_);
lean_dec(v_r_3582_);
v___x_3591_ = lean_box(0);
v_isShared_3592_ = v_isSharedCheck_3603_;
goto v_resetjp_3590_;
}
v_resetjp_3590_:
{
lean_object* v___x_3593_; lean_object* v___x_3595_; 
v___x_3593_ = lean_unsigned_to_nat(3u);
if (v_isShared_3592_ == 0)
{
lean_ctor_set(v___x_3591_, 4, v_l_3565_);
lean_ctor_set(v___x_3591_, 3, v_l_3565_);
lean_ctor_set(v___x_3591_, 2, v_v_3584_);
lean_ctor_set(v___x_3591_, 1, v_k_3583_);
lean_ctor_set(v___x_3591_, 0, v___x_3479_);
v___x_3595_ = v___x_3591_;
goto v_reusejp_3594_;
}
else
{
lean_object* v_reuseFailAlloc_3602_; 
v_reuseFailAlloc_3602_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3602_, 0, v___x_3479_);
lean_ctor_set(v_reuseFailAlloc_3602_, 1, v_k_3583_);
lean_ctor_set(v_reuseFailAlloc_3602_, 2, v_v_3584_);
lean_ctor_set(v_reuseFailAlloc_3602_, 3, v_l_3565_);
lean_ctor_set(v_reuseFailAlloc_3602_, 4, v_l_3565_);
v___x_3595_ = v_reuseFailAlloc_3602_;
goto v_reusejp_3594_;
}
v_reusejp_3594_:
{
lean_object* v___x_3597_; 
if (v_isShared_3587_ == 0)
{
lean_ctor_set(v___x_3586_, 4, v_l_3565_);
lean_ctor_set(v___x_3586_, 2, v_v_3329_);
lean_ctor_set(v___x_3586_, 1, v_k_3328_);
lean_ctor_set(v___x_3586_, 0, v___x_3479_);
v___x_3597_ = v___x_3586_;
goto v_reusejp_3596_;
}
else
{
lean_object* v_reuseFailAlloc_3601_; 
v_reuseFailAlloc_3601_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3601_, 0, v___x_3479_);
lean_ctor_set(v_reuseFailAlloc_3601_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3601_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3601_, 3, v_l_3565_);
lean_ctor_set(v_reuseFailAlloc_3601_, 4, v_l_3565_);
v___x_3597_ = v_reuseFailAlloc_3601_;
goto v_reusejp_3596_;
}
v_reusejp_3596_:
{
lean_object* v___x_3599_; 
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v___x_3597_);
lean_ctor_set(v___x_3333_, 3, v___x_3595_);
lean_ctor_set(v___x_3333_, 2, v_v_3589_);
lean_ctor_set(v___x_3333_, 1, v_k_3588_);
lean_ctor_set(v___x_3333_, 0, v___x_3593_);
v___x_3599_ = v___x_3333_;
goto v_reusejp_3598_;
}
else
{
lean_object* v_reuseFailAlloc_3600_; 
v_reuseFailAlloc_3600_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3600_, 0, v___x_3593_);
lean_ctor_set(v_reuseFailAlloc_3600_, 1, v_k_3588_);
lean_ctor_set(v_reuseFailAlloc_3600_, 2, v_v_3589_);
lean_ctor_set(v_reuseFailAlloc_3600_, 3, v___x_3595_);
lean_ctor_set(v_reuseFailAlloc_3600_, 4, v___x_3597_);
v___x_3599_ = v_reuseFailAlloc_3600_;
goto v_reusejp_3598_;
}
v_reusejp_3598_:
{
return v___x_3599_;
}
}
}
}
}
}
else
{
lean_object* v___x_3611_; lean_object* v___x_3613_; 
v___x_3611_ = lean_unsigned_to_nat(2u);
if (v_isShared_3334_ == 0)
{
lean_ctor_set(v___x_3333_, 4, v_r_3582_);
lean_ctor_set(v___x_3333_, 3, v_impl_3478_);
lean_ctor_set(v___x_3333_, 0, v___x_3611_);
v___x_3613_ = v___x_3333_;
goto v_reusejp_3612_;
}
else
{
lean_object* v_reuseFailAlloc_3614_; 
v_reuseFailAlloc_3614_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3614_, 0, v___x_3611_);
lean_ctor_set(v_reuseFailAlloc_3614_, 1, v_k_3328_);
lean_ctor_set(v_reuseFailAlloc_3614_, 2, v_v_3329_);
lean_ctor_set(v_reuseFailAlloc_3614_, 3, v_impl_3478_);
lean_ctor_set(v_reuseFailAlloc_3614_, 4, v_r_3582_);
v___x_3613_ = v_reuseFailAlloc_3614_;
goto v_reusejp_3612_;
}
v_reusejp_3612_:
{
return v___x_3613_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3616_; lean_object* v___x_3617_; lean_object* v___x_3618_; 
v___x_3616_ = lean_unsigned_to_nat(1u);
v___x_3617_ = lean_box_uint64(v_k_3324_);
v___x_3618_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3618_, 0, v___x_3616_);
lean_ctor_set(v___x_3618_, 1, v___x_3617_);
lean_ctor_set(v___x_3618_, 2, v_v_3325_);
lean_ctor_set(v___x_3618_, 3, v_t_3326_);
lean_ctor_set(v___x_3618_, 4, v_t_3326_);
return v___x_3618_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg___boxed(lean_object* v_k_3619_, lean_object* v_v_3620_, lean_object* v_t_3621_){
_start:
{
uint64_t v_k_boxed_3622_; lean_object* v_res_3623_; 
v_k_boxed_3622_ = lean_unbox_uint64(v_k_3619_);
lean_dec_ref(v_k_3619_);
v_res_3623_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(v_k_boxed_3622_, v_v_3620_, v_t_3621_);
return v_res_3623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg___lam__0(lean_object* v_wi_3624_, lean_object* v_s_3625_){
_start:
{
uint64_t v_javascriptHash_3626_; lean_object* v___x_3627_; lean_object* v___x_3628_; lean_object* v___x_3629_; lean_object* v___x_3630_; lean_object* v___x_3631_; 
v_javascriptHash_3626_ = lean_ctor_get_uint64(v_wi_3624_, sizeof(void*)*2);
v___x_3627_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3627_, 0, v_wi_3624_);
v___x_3628_ = lean_box(0);
v___x_3629_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg(v_s_3625_, v_javascriptHash_3626_, v___x_3628_);
v___x_3630_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3630_, 0, v___x_3627_);
lean_ctor_set(v___x_3630_, 1, v___x_3629_);
v___x_3631_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(v_javascriptHash_3626_, v___x_3630_, v_s_3625_);
return v___x_3631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg(lean_object* v_wi_3632_, lean_object* v___y_3633_){
_start:
{
lean_object* v___x_3635_; lean_object* v_env_3636_; lean_object* v_messages_3637_; lean_object* v_scopes_3638_; lean_object* v_usedQuotCtxts_3639_; lean_object* v_nextMacroScope_3640_; lean_object* v_maxRecDepth_3641_; lean_object* v_ngen_3642_; lean_object* v_auxDeclNGen_3643_; lean_object* v_infoState_3644_; lean_object* v_traceState_3645_; lean_object* v_snapshotTasks_3646_; lean_object* v_prevLinterStates_3647_; lean_object* v___x_3649_; uint8_t v_isShared_3650_; uint8_t v_isSharedCheck_3660_; 
v___x_3635_ = lean_st_ref_take(v___y_3633_);
v_env_3636_ = lean_ctor_get(v___x_3635_, 0);
v_messages_3637_ = lean_ctor_get(v___x_3635_, 1);
v_scopes_3638_ = lean_ctor_get(v___x_3635_, 2);
v_usedQuotCtxts_3639_ = lean_ctor_get(v___x_3635_, 3);
v_nextMacroScope_3640_ = lean_ctor_get(v___x_3635_, 4);
v_maxRecDepth_3641_ = lean_ctor_get(v___x_3635_, 5);
v_ngen_3642_ = lean_ctor_get(v___x_3635_, 6);
v_auxDeclNGen_3643_ = lean_ctor_get(v___x_3635_, 7);
v_infoState_3644_ = lean_ctor_get(v___x_3635_, 8);
v_traceState_3645_ = lean_ctor_get(v___x_3635_, 9);
v_snapshotTasks_3646_ = lean_ctor_get(v___x_3635_, 10);
v_prevLinterStates_3647_ = lean_ctor_get(v___x_3635_, 11);
v_isSharedCheck_3660_ = !lean_is_exclusive(v___x_3635_);
if (v_isSharedCheck_3660_ == 0)
{
v___x_3649_ = v___x_3635_;
v_isShared_3650_ = v_isSharedCheck_3660_;
goto v_resetjp_3648_;
}
else
{
lean_inc(v_prevLinterStates_3647_);
lean_inc(v_snapshotTasks_3646_);
lean_inc(v_traceState_3645_);
lean_inc(v_infoState_3644_);
lean_inc(v_auxDeclNGen_3643_);
lean_inc(v_ngen_3642_);
lean_inc(v_maxRecDepth_3641_);
lean_inc(v_nextMacroScope_3640_);
lean_inc(v_usedQuotCtxts_3639_);
lean_inc(v_scopes_3638_);
lean_inc(v_messages_3637_);
lean_inc(v_env_3636_);
lean_dec(v___x_3635_);
v___x_3649_ = lean_box(0);
v_isShared_3650_ = v_isSharedCheck_3660_;
goto v_resetjp_3648_;
}
v_resetjp_3648_:
{
lean_object* v___f_3651_; lean_object* v___x_3652_; lean_object* v___x_3653_; lean_object* v___x_3655_; 
v___f_3651_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg___lam__0), 2, 1);
lean_closure_set(v___f_3651_, 0, v_wi_3632_);
v___x_3652_ = l___private_Lean_Widget_UserWidget_0__Lean_Widget_panelWidgetsExt;
v___x_3653_ = l_Lean_ScopedEnvExtension_modifyState___redArg(v___x_3652_, v_env_3636_, v___f_3651_);
if (v_isShared_3650_ == 0)
{
lean_ctor_set(v___x_3649_, 0, v___x_3653_);
v___x_3655_ = v___x_3649_;
goto v_reusejp_3654_;
}
else
{
lean_object* v_reuseFailAlloc_3659_; 
v_reuseFailAlloc_3659_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_3659_, 0, v___x_3653_);
lean_ctor_set(v_reuseFailAlloc_3659_, 1, v_messages_3637_);
lean_ctor_set(v_reuseFailAlloc_3659_, 2, v_scopes_3638_);
lean_ctor_set(v_reuseFailAlloc_3659_, 3, v_usedQuotCtxts_3639_);
lean_ctor_set(v_reuseFailAlloc_3659_, 4, v_nextMacroScope_3640_);
lean_ctor_set(v_reuseFailAlloc_3659_, 5, v_maxRecDepth_3641_);
lean_ctor_set(v_reuseFailAlloc_3659_, 6, v_ngen_3642_);
lean_ctor_set(v_reuseFailAlloc_3659_, 7, v_auxDeclNGen_3643_);
lean_ctor_set(v_reuseFailAlloc_3659_, 8, v_infoState_3644_);
lean_ctor_set(v_reuseFailAlloc_3659_, 9, v_traceState_3645_);
lean_ctor_set(v_reuseFailAlloc_3659_, 10, v_snapshotTasks_3646_);
lean_ctor_set(v_reuseFailAlloc_3659_, 11, v_prevLinterStates_3647_);
v___x_3655_ = v_reuseFailAlloc_3659_;
goto v_reusejp_3654_;
}
v_reusejp_3654_:
{
lean_object* v___x_3656_; lean_object* v___x_3657_; lean_object* v___x_3658_; 
v___x_3656_ = lean_st_ref_set(v___y_3633_, v___x_3655_);
v___x_3657_ = lean_box(0);
v___x_3658_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3658_, 0, v___x_3657_);
return v___x_3658_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg___boxed(lean_object* v_wi_3661_, lean_object* v___y_3662_, lean_object* v___y_3663_){
_start:
{
lean_object* v_res_3664_; 
v_res_3664_ = lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg(v_wi_3661_, v___y_3662_);
lean_dec(v___y_3662_);
return v_res_3664_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0(void){
_start:
{
lean_object* v___x_3665_; lean_object* v___x_3666_; 
v___x_3665_ = lean_box(0);
v___x_3666_ = l_Lean_Json_mkObj(v___x_3665_);
return v___x_3666_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1(void){
_start:
{
lean_object* v___x_3667_; lean_object* v___f_3668_; 
v___x_3667_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__0);
v___f_3668_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___lam__0), 2, 1);
lean_closure_set(v___f_3668_, 0, v___x_3667_);
return v___f_3668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1(lean_object* v_x_3669_, lean_object* v_a_3670_, lean_object* v_a_3671_){
_start:
{
lean_object* v___x_3673_; uint8_t v___x_3674_; 
v___x_3673_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_ClickSuggestions_command_x23click__suggestions___closed__1));
v___x_3674_ = l_Lean_Syntax_isOfKind(v_x_3669_, v___x_3673_);
if (v___x_3674_ == 0)
{
lean_object* v___x_3675_; 
v___x_3675_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__0___redArg();
return v___x_3675_;
}
else
{
lean_object* v___x_3676_; lean_object* v_toModule_3677_; uint64_t v_javascriptHash_3678_; lean_object* v___f_3679_; lean_object* v___x_3680_; lean_object* v___x_3681_; lean_object* v___x_3682_; 
v___x_3676_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent;
v_toModule_3677_ = lean_ctor_get(v___x_3676_, 0);
v_javascriptHash_3678_ = lean_ctor_get_uint64(v_toModule_3677_, sizeof(void*)*1);
v___f_3679_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1, &lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___closed__1);
v___x_3680_ = lean_box_uint64(v_javascriptHash_3678_);
v___x_3681_ = lean_alloc_closure((void*)(l_Lean_Widget_WidgetInstance_ofHash___boxed), 5, 2);
lean_closure_set(v___x_3681_, 0, v___x_3680_);
lean_closure_set(v___x_3681_, 1, v___f_3679_);
v___x_3682_ = l_Lean_Elab_Command_liftCoreM___redArg(v___x_3681_, v_a_3670_, v_a_3671_);
if (lean_obj_tag(v___x_3682_) == 0)
{
lean_object* v_a_3683_; lean_object* v___x_3684_; 
v_a_3683_ = lean_ctor_get(v___x_3682_, 0);
lean_inc(v_a_3683_);
lean_dec_ref_known(v___x_3682_, 1);
v___x_3684_ = lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg(v_a_3683_, v_a_3671_);
return v___x_3684_;
}
else
{
lean_object* v_a_3685_; lean_object* v___x_3687_; uint8_t v_isShared_3688_; uint8_t v_isSharedCheck_3692_; 
v_a_3685_ = lean_ctor_get(v___x_3682_, 0);
v_isSharedCheck_3692_ = !lean_is_exclusive(v___x_3682_);
if (v_isSharedCheck_3692_ == 0)
{
v___x_3687_ = v___x_3682_;
v_isShared_3688_ = v_isSharedCheck_3692_;
goto v_resetjp_3686_;
}
else
{
lean_inc(v_a_3685_);
lean_dec(v___x_3682_);
v___x_3687_ = lean_box(0);
v_isShared_3688_ = v_isSharedCheck_3692_;
goto v_resetjp_3686_;
}
v_resetjp_3686_:
{
lean_object* v___x_3690_; 
if (v_isShared_3688_ == 0)
{
v___x_3690_ = v___x_3687_;
goto v_reusejp_3689_;
}
else
{
lean_object* v_reuseFailAlloc_3691_; 
v_reuseFailAlloc_3691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3691_, 0, v_a_3685_);
v___x_3690_ = v_reuseFailAlloc_3691_;
goto v_reusejp_3689_;
}
v_reusejp_3689_:
{
return v___x_3690_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1___boxed(lean_object* v_x_3693_, lean_object* v_a_3694_, lean_object* v_a_3695_, lean_object* v_a_3696_){
_start:
{
lean_object* v_res_3697_; 
v_res_3697_ = lp_mathlib_Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1(v_x_3693_, v_a_3694_, v_a_3695_);
lean_dec(v_a_3695_);
lean_dec_ref(v_a_3694_);
return v_res_3697_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1(lean_object* v_wi_3698_, lean_object* v___y_3699_, lean_object* v___y_3700_){
_start:
{
lean_object* v___x_3702_; 
v___x_3702_ = lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___redArg(v_wi_3698_, v___y_3700_);
return v___x_3702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1___boxed(lean_object* v_wi_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_, lean_object* v___y_3706_){
_start:
{
lean_object* v_res_3707_; 
v_res_3707_ = lp_mathlib_Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1(v_wi_3703_, v___y_3704_, v___y_3705_);
lean_dec(v___y_3705_);
lean_dec_ref(v___y_3704_);
return v_res_3707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1(lean_object* v_00_u03b4_3708_, lean_object* v_t_3709_, uint64_t v_k_3710_, lean_object* v_fallback_3711_){
_start:
{
lean_object* v___x_3712_; 
v___x_3712_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___redArg(v_t_3709_, v_k_3710_, v_fallback_3711_);
return v___x_3712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1___boxed(lean_object* v_00_u03b4_3713_, lean_object* v_t_3714_, lean_object* v_k_3715_, lean_object* v_fallback_3716_){
_start:
{
uint64_t v_k_boxed_3717_; lean_object* v_res_3718_; 
v_k_boxed_3717_ = lean_unbox_uint64(v_k_3715_);
lean_dec_ref(v_k_3715_);
v_res_3718_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__1(v_00_u03b4_3713_, v_t_3714_, v_k_boxed_3717_, v_fallback_3716_);
lean_dec(v_fallback_3716_);
lean_dec(v_t_3714_);
return v_res_3718_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2(lean_object* v_00_u03b2_3719_, uint64_t v_k_3720_, lean_object* v_v_3721_, lean_object* v_t_3722_, lean_object* v_hl_3723_){
_start:
{
lean_object* v___x_3724_; 
v___x_3724_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___redArg(v_k_3720_, v_v_3721_, v_t_3722_);
return v___x_3724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2___boxed(lean_object* v_00_u03b2_3725_, lean_object* v_k_3726_, lean_object* v_v_3727_, lean_object* v_t_3728_, lean_object* v_hl_3729_){
_start:
{
uint64_t v_k_boxed_3730_; lean_object* v_res_3731_; 
v_k_boxed_3730_ = lean_unbox_uint64(v_k_3726_);
lean_dec_ref(v_k_3726_);
v_res_3731_ = lp_mathlib_Std_DTreeMap_Internal_Impl_insert___at___00Lean_Widget_addPanelWidgetLocal___at___00Mathlib_Tactic_ClickSuggestions___aux__Mathlib__Tactic__ClickSuggestions______elabRules__Mathlib__Tactic__ClickSuggestions__command_x23click__suggestions__1_spec__1_spec__2(v_00_u03b2_3725_, v_k_boxed_3730_, v_v_3727_, v_t_3728_, v_hl_3729_);
return v_res_3731_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_TryPremises(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(uint8_t builtin);
lean_object* runtime_initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
lean_object* runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_TryPremises(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(uint8_t builtin);
lean_object* runtime_initialize_Lean_Server_FileWorker_RequestHandling(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_GoalsLocation(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Server_FileWorker_RequestHandling(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_GoalsLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_rpc___rpc__wrapped);
lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent = _init_lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_ClickSuggestions_clickSuggestionsComponent);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_TryPremises(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(uint8_t builtin);
lean_object* initialize_Lean_Server_FileWorker_RequestHandling(uint8_t builtin);
lean_object* initialize_Lean_Widget_InteractiveGoal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_GoalsLocation(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_ClickSuggestions(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_TryPremises(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ClickSuggestions_Unfold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta_KAbstractPositions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Server_FileWorker_RequestHandling(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Widget_InteractiveGoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_GoalsLocation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_OfRpcMethod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_ClickSuggestions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_ClickSuggestions(builtin);
}
#ifdef __cplusplus
}
#endif
