// Lean compiler output
// Module: Mathlib.Tactic.FunProp.Elab
// Imports: public import Init public meta import Init public import Mathlib.Tactic.FunProp.Core import Mathlib.Tactic.InferParam import Lean.Elab.InfoTree.Main public import Lean.Elab.ConfigEval public meta import Lean.Elab.ConfigEval
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
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* l_Lean_Elab_Command_instInhabitedScope_default;
lean_object* l_List_head_x21___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofConstName(lean_object*, uint8_t);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Command_getScope___redArg(lean_object*);
lean_object* l_Lean_ResolveName_resolveGlobalName(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_isPrivateName(lean_object*);
extern lean_object* l_Lean_ResolveName_backward_privateInPublic_warn;
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* l_Lean_MessageLog_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t l_Lean_MessageData_hasTag(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getTailPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
uint8_t l_Lean_instBEqMessageSeverity_beq(uint8_t, uint8_t);
extern lean_object* l_Lean_warningAsError;
uint8_t l_Lean_MessageData_hasSyntheticSorry(lean_object*);
uint8_t l_Lean_Name_isAnonymous(lean_object*);
lean_object* l_Lean_Environment_setExporting(lean_object*, uint8_t);
uint8_t l_Lean_Environment_contains(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Environment_getModuleIdxFor_x3f(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_note(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_EnvironmentHeader_moduleNames(lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_Origin_name(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_unknownIdentifierMessageTag;
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_getFunProp_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ppExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppFn(lean_object*);
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_string_append(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_instMonadCommandElabM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Command_instMonadCommandElabM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_formatStx(lean_object*, lean_object*, uint8_t);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_dbg_to_string(lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* l_Lean_Syntax_getOptional_x3f(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_discharger;
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_tacticToDischarge(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt;
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt;
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold;
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl___boxed(lean_object*, lean_object*);
lean_object* l_Std_TreeSet_ofArray___redArg(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default;
lean_object* lp_mathlib_Mathlib_Meta_FunProp_funProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_FunProp_tacticToDischarge___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FunProp"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 127, 90, 142, 244, 122, 64, 171)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__11 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__13 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__13_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "maxSteps"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "maxTransitionDepth"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 127, 90, 142, 244, 122, 64, 171)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(206, 105, 205, 229, 2, 176, 149, 13)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__4_value),LEAN_SCALAR_PTR_LITERAL(52, 127, 90, 142, 244, 122, 64, 171)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(20, 5, 96, 13, 236, 246, 94, 251)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "funPropTacStx"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(146, 162, 46, 139, 110, 137, 217, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fun_prop"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "withoutPosition"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 6, 27, 142, 141, 165, 41, 16)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__15_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__16_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__19_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__19_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__20_value),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__14_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__21_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__27_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "first"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3_value),LEAN_SCALAR_PTR_LITERAL(59, 232, 35, 17, 172, 62, 48, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__5_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__6_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "group"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__7_value),LEAN_SCALAR_PTR_LITERAL(206, 113, 20, 57, 188, 177, 187, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__8_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__9_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__10_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__12_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "withReducible"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__14_value),LEAN_SCALAR_PTR_LITERAL(197, 44, 223, 192, 8, 197, 146, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "with_reducible"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferOptParam"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__19_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__19_value),LEAN_SCALAR_PTR_LITERAL(160, 148, 17, 69, 246, 45, 144, 29)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "infer_param"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__21_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "` is not a `fun_prop` goal!"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = " Consider marking `"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "` with `@[fun_prop]`."};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\n  "};
static const lean_object* lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___closed__0 = (const lean_object*)&lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__0_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "`fun_prop` was unable to prove `"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "`\n\n"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Issues:"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "discharger"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___boxed(lean_object**);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__0_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "command#print_fun_prop_theorems__"};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__2_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__3_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 63, 238, 230, 17, 126, 146, 255)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "#print_fun_prop_theorems "};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__2_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__17_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__4_value),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__5_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__6_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__7_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems____ = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "↓ "};
static const lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 4, .m_data = "↓ ← "};
static const lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "← "};
static const lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__0 = (const lean_object*)&lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__0_value;
static const lean_string_object lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__1 = (const lean_object*)&lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__10(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0;
static const lean_closure_object lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Command_instMonadCommandElabM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__1 = (const lean_object*)&lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__1_value;
static const lean_closure_object lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_Command_instMonadCommandElabM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__2 = (const lean_object*)&lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Lean.ResolveName"};
static const lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__0 = (const lean_object*)&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__0_value;
static const lean_string_object lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Lean.ensureNonAmbiguous"};
static const lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__1 = (const lean_object*)&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__1_value;
static const lean_string_object lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__2 = (const lean_object*)&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3;
static const lean_string_object lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "ambiguous identifier `"};
static const lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__4 = (const lean_object*)&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__4_value;
static const lean_string_object lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 30, .m_capacity = 30, .m_length = 29, .m_data = "`, possible interpretations: "};
static const lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__5 = (const lean_object*)&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__2(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", args: "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", form: "};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__3_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "simple"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__5_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "compositional"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0(uint8_t, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__18(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "A private declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "` (from the current module) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "A public declaration `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "` exists but is imported privately; consider adding `public import "};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__6 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "`."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__8 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__8_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "` (from `"};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__10 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11;
static const lean_string_object lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "`) exists but would need to be public to access here."};
static const lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__12 = (const lean_object*)&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__12_value;
static lean_once_cell_t lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13;
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Unknown constant `"};
static const lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__17(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Private declaration `"};
static const lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__0 = (const lean_object*)&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1;
static const lean_string_object lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 167, .m_capacity = 167, .m_length = 166, .m_data = "` accessed publicly; this is allowed only because the `backward.privateInPublic` option is enabled. \n\nDisable `backward.privateInPublic.warn` to silence this warning."};
static const lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__2 = (const lean_object*)&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7_spec__11(lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__0 = (const lean_object*)&lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__0_value;
static const lean_string_object lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "expected identifier"};
static const lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__1 = (const lean_object*)&lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__1_value;
static const lean_ctor_object lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__1_value)}};
static const lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__2 = (const lean_object*)&lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___closed__0 = (const lean_object*)&lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lean_box(0);
v___x_2_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_3_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
lean_ctor_set(v___x_3_, 1, v___x_1_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_6_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_16_, lean_object* v___y_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_16_, v___y_17_, v___y_18_, v___y_19_, v___y_20_);
lean_dec(v___y_20_);
lean_dec_ref(v___y_19_);
lean_dec(v___y_18_);
lean_dec_ref(v___y_17_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(lean_object* v_msgData_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_29_; lean_object* v_env_30_; lean_object* v___x_31_; lean_object* v_mctx_32_; lean_object* v_lctx_33_; lean_object* v_options_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_29_ = lean_st_ref_get(v___y_27_);
v_env_30_ = lean_ctor_get(v___x_29_, 0);
lean_inc_ref(v_env_30_);
lean_dec(v___x_29_);
v___x_31_ = lean_st_ref_get(v___y_25_);
v_mctx_32_ = lean_ctor_get(v___x_31_, 0);
lean_inc_ref(v_mctx_32_);
lean_dec(v___x_31_);
v_lctx_33_ = lean_ctor_get(v___y_24_, 2);
v_options_34_ = lean_ctor_get(v___y_26_, 2);
lean_inc_ref(v_options_34_);
lean_inc_ref(v_lctx_33_);
v___x_35_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_35_, 0, v_env_30_);
lean_ctor_set(v___x_35_, 1, v_mctx_32_);
lean_ctor_set(v___x_35_, 2, v_lctx_33_);
lean_ctor_set(v___x_35_, 3, v_options_34_);
v___x_36_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v_msgData_23_);
v___x_37_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_37_, 0, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msgData_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_);
lean_dec(v___y_42_);
lean_dec_ref(v___y_41_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg(lean_object* v_msg_45_, lean_object* v___y_46_, lean_object* v___y_47_, lean_object* v___y_48_, lean_object* v___y_49_){
_start:
{
lean_object* v_ref_51_; lean_object* v___x_52_; lean_object* v_a_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_61_; 
v_ref_51_ = lean_ctor_get(v___y_48_, 5);
v___x_52_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_45_, v___y_46_, v___y_47_, v___y_48_, v___y_49_);
v_a_53_ = lean_ctor_get(v___x_52_, 0);
v_isSharedCheck_61_ = !lean_is_exclusive(v___x_52_);
if (v_isSharedCheck_61_ == 0)
{
v___x_55_ = v___x_52_;
v_isShared_56_ = v_isSharedCheck_61_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_a_53_);
lean_dec(v___x_52_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_61_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_57_; lean_object* v___x_59_; 
lean_inc(v_ref_51_);
v___x_57_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_57_, 0, v_ref_51_);
lean_ctor_set(v___x_57_, 1, v_a_53_);
if (v_isShared_56_ == 0)
{
lean_ctor_set_tag(v___x_55_, 1);
lean_ctor_set(v___x_55_, 0, v___x_57_);
v___x_59_ = v___x_55_;
goto v_reusejp_58_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v___x_57_);
v___x_59_ = v_reuseFailAlloc_60_;
goto v_reusejp_58_;
}
v_reusejp_58_:
{
return v___x_59_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_);
lean_dec(v___y_66_);
lean_dec_ref(v___y_65_);
lean_dec(v___y_64_);
lean_dec_ref(v___y_63_);
return v_res_68_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_71_; lean_object* v___x_72_; 
v___x_71_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__1));
v___x_72_ = l_Lean_stringToMessageData(v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_73_, lean_object* v_args_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_){
_start:
{
lean_object* v___x_114_; uint8_t v___x_115_; 
v___x_114_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_115_ = lean_string_dec_eq(v_ctor_73_, v___x_114_);
if (v___x_115_ == 0)
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_116_;
}
else
{
lean_object* v___x_117_; lean_object* v___x_118_; uint8_t v___x_119_; 
v___x_117_ = lean_array_get_size(v_args_74_);
v___x_118_ = lean_unsigned_to_nat(2u);
v___x_119_ = lean_nat_dec_eq(v___x_117_, v___x_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v_a_122_; lean_object* v___x_124_; uint8_t v_isShared_125_; uint8_t v_isSharedCheck_129_; 
v___x_120_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___closed__2);
v___x_121_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg(v___x_120_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
v_a_122_ = lean_ctor_get(v___x_121_, 0);
v_isSharedCheck_129_ = !lean_is_exclusive(v___x_121_);
if (v_isSharedCheck_129_ == 0)
{
v___x_124_ = v___x_121_;
v_isShared_125_ = v_isSharedCheck_129_;
goto v_resetjp_123_;
}
else
{
lean_inc(v_a_122_);
lean_dec(v___x_121_);
v___x_124_ = lean_box(0);
v_isShared_125_ = v_isSharedCheck_129_;
goto v_resetjp_123_;
}
v_resetjp_123_:
{
lean_object* v___x_127_; 
if (v_isShared_125_ == 0)
{
v___x_127_ = v___x_124_;
goto v_reusejp_126_;
}
else
{
lean_object* v_reuseFailAlloc_128_; 
v_reuseFailAlloc_128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_128_, 0, v_a_122_);
v___x_127_ = v_reuseFailAlloc_128_;
goto v_reusejp_126_;
}
v_reusejp_126_:
{
return v___x_127_;
}
}
}
else
{
goto v___jp_80_;
}
}
v___jp_80_:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
v___x_81_ = l_Lean_instInhabitedExpr;
v___x_82_ = lean_unsigned_to_nat(0u);
v___x_83_ = lean_array_get_borrowed(v___x_81_, v_args_74_, v___x_82_);
lean_inc(v___x_83_);
v___x_84_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v___x_83_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
if (lean_obj_tag(v___x_84_) == 0)
{
lean_object* v_a_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; 
v_a_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_a_85_);
lean_dec_ref_known(v___x_84_, 1);
v___x_86_ = lean_unsigned_to_nat(1u);
v___x_87_ = lean_array_get_borrowed(v___x_81_, v_args_74_, v___x_86_);
lean_inc(v___x_87_);
v___x_88_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v___x_87_, v___y_75_, v___y_76_, v___y_77_, v___y_78_);
if (lean_obj_tag(v___x_88_) == 0)
{
lean_object* v_a_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_97_; 
v_a_89_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_97_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_97_ == 0)
{
v___x_91_ = v___x_88_;
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_a_89_);
lean_dec(v___x_88_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_97_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_95_; 
v___x_93_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_93_, 0, v_a_85_);
lean_ctor_set(v___x_93_, 1, v_a_89_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 0, v___x_93_);
v___x_95_ = v___x_91_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_96_; 
v_reuseFailAlloc_96_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_96_, 0, v___x_93_);
v___x_95_ = v_reuseFailAlloc_96_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
return v___x_95_;
}
}
}
else
{
lean_object* v_a_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_105_; 
lean_dec(v_a_85_);
v_a_98_ = lean_ctor_get(v___x_88_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_88_);
if (v_isSharedCheck_105_ == 0)
{
v___x_100_ = v___x_88_;
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_a_98_);
lean_dec(v___x_88_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_105_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_104_; 
v_reuseFailAlloc_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_104_, 0, v_a_98_);
v___x_103_ = v_reuseFailAlloc_104_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
return v___x_103_;
}
}
}
}
else
{
lean_object* v_a_106_; lean_object* v___x_108_; uint8_t v_isShared_109_; uint8_t v_isSharedCheck_113_; 
v_a_106_ = lean_ctor_get(v___x_84_, 0);
v_isSharedCheck_113_ = !lean_is_exclusive(v___x_84_);
if (v_isSharedCheck_113_ == 0)
{
v___x_108_ = v___x_84_;
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
else
{
lean_inc(v_a_106_);
lean_dec(v___x_84_);
v___x_108_ = lean_box(0);
v_isShared_109_ = v_isSharedCheck_113_;
goto v_resetjp_107_;
}
v_resetjp_107_:
{
lean_object* v___x_111_; 
if (v_isShared_109_ == 0)
{
v___x_111_ = v___x_108_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v_a_106_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_130_, lean_object* v_args_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_){
_start:
{
lean_object* v_res_137_; 
v_res_137_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___lam__0(v_ctor_130_, v_args_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
lean_dec(v___y_133_);
lean_dec_ref(v___y_132_);
lean_dec_ref(v_args_131_);
lean_dec_ref(v_ctor_130_);
return v_res_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr(lean_object* v_a_148_, lean_object* v_a_149_, lean_object* v_a_150_, lean_object* v_a_151_, lean_object* v_a_152_){
_start:
{
lean_object* v___f_154_; lean_object* v___x_155_; lean_object* v___x_156_; 
v___f_154_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__0));
v___x_155_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5));
v___x_156_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_155_, v___f_154_, v_a_148_, v_a_149_, v_a_150_, v_a_151_, v_a_152_);
return v___x_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr(v_a_157_, v_a_158_, v_a_159_, v_a_160_, v_a_161_);
lean_dec(v_a_161_);
lean_dec_ref(v_a_160_);
lean_dec(v_a_159_);
lean_dec_ref(v_a_158_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1(lean_object* v_00_u03b1_164_, lean_object* v_msg_165_, lean_object* v___y_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___redArg(v_msg_165_, v___y_166_, v___y_167_, v___y_168_, v___y_169_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_172_, lean_object* v_msg_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1(v_00_u03b1_172_, v_msg_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
return v_res_179_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; 
v___x_181_ = lean_box(0);
v___x_182_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5));
v___x_183_ = l_Lean_Expr_const___override(v___x_182_, v___x_181_);
return v___x_183_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1);
v___x_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
return v___x_185_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_186_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2);
v___x_187_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__0));
v___x_188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v___x_186_);
return v___x_188_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig(void){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__3);
return v___x_189_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0(void){
_start:
{
lean_object* v___x_190_; lean_object* v___x_191_; 
v___x_190_ = lean_box(1);
v___x_191_ = l_Lean_MessageData_ofFormat(v___x_190_);
return v___x_191_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3(void){
_start:
{
lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_195_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__2));
v___x_196_ = l_Lean_MessageData_ofFormat(v___x_195_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(lean_object* v_x_197_, lean_object* v_x_198_){
_start:
{
if (lean_obj_tag(v_x_198_) == 0)
{
return v_x_197_;
}
else
{
lean_object* v_head_199_; lean_object* v_tail_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_222_; 
v_head_199_ = lean_ctor_get(v_x_198_, 0);
v_tail_200_ = lean_ctor_get(v_x_198_, 1);
v_isSharedCheck_222_ = !lean_is_exclusive(v_x_198_);
if (v_isSharedCheck_222_ == 0)
{
v___x_202_ = v_x_198_;
v_isShared_203_ = v_isSharedCheck_222_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_tail_200_);
lean_inc(v_head_199_);
lean_dec(v_x_198_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_222_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v_before_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_220_; 
v_before_204_ = lean_ctor_get(v_head_199_, 0);
v_isSharedCheck_220_ = !lean_is_exclusive(v_head_199_);
if (v_isSharedCheck_220_ == 0)
{
lean_object* v_unused_221_; 
v_unused_221_ = lean_ctor_get(v_head_199_, 1);
lean_dec(v_unused_221_);
v___x_206_ = v_head_199_;
v_isShared_207_ = v_isSharedCheck_220_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_before_204_);
lean_dec(v_head_199_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_220_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_208_; lean_object* v___x_210_; 
v___x_208_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0);
if (v_isShared_207_ == 0)
{
lean_ctor_set_tag(v___x_206_, 7);
lean_ctor_set(v___x_206_, 1, v___x_208_);
lean_ctor_set(v___x_206_, 0, v_x_197_);
v___x_210_ = v___x_206_;
goto v_reusejp_209_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v_x_197_);
lean_ctor_set(v_reuseFailAlloc_219_, 1, v___x_208_);
v___x_210_ = v_reuseFailAlloc_219_;
goto v_reusejp_209_;
}
v_reusejp_209_:
{
lean_object* v___x_211_; lean_object* v___x_213_; 
v___x_211_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__3);
if (v_isShared_203_ == 0)
{
lean_ctor_set_tag(v___x_202_, 7);
lean_ctor_set(v___x_202_, 1, v___x_211_);
lean_ctor_set(v___x_202_, 0, v___x_210_);
v___x_213_ = v___x_202_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v___x_210_);
lean_ctor_set(v_reuseFailAlloc_218_, 1, v___x_211_);
v___x_213_ = v_reuseFailAlloc_218_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_214_ = l_Lean_MessageData_ofSyntax(v_before_204_);
v___x_215_ = l_Lean_indentD(v___x_214_);
v___x_216_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_216_, 0, v___x_213_);
lean_ctor_set(v___x_216_, 1, v___x_215_);
v_x_197_ = v___x_216_;
v_x_198_ = v_tail_200_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(lean_object* v_opts_223_, lean_object* v_opt_224_){
_start:
{
lean_object* v_name_225_; lean_object* v_defValue_226_; lean_object* v_map_227_; lean_object* v___x_228_; 
v_name_225_ = lean_ctor_get(v_opt_224_, 0);
v_defValue_226_ = lean_ctor_get(v_opt_224_, 1);
v_map_227_ = lean_ctor_get(v_opts_223_, 0);
v___x_228_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_227_, v_name_225_);
if (lean_obj_tag(v___x_228_) == 0)
{
uint8_t v___x_229_; 
v___x_229_ = lean_unbox(v_defValue_226_);
return v___x_229_;
}
else
{
lean_object* v_val_230_; 
v_val_230_ = lean_ctor_get(v___x_228_, 0);
lean_inc(v_val_230_);
lean_dec_ref_known(v___x_228_, 1);
if (lean_obj_tag(v_val_230_) == 1)
{
uint8_t v_v_231_; 
v_v_231_ = lean_ctor_get_uint8(v_val_230_, 0);
lean_dec_ref_known(v_val_230_, 0);
return v_v_231_;
}
else
{
uint8_t v___x_232_; 
lean_dec(v_val_230_);
v___x_232_ = lean_unbox(v_defValue_226_);
return v___x_232_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6___boxed(lean_object* v_opts_233_, lean_object* v_opt_234_){
_start:
{
uint8_t v_res_235_; lean_object* v_r_236_; 
v_res_235_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_opts_233_, v_opt_234_);
lean_dec_ref(v_opt_234_);
lean_dec_ref(v_opts_233_);
v_r_236_ = lean_box(v_res_235_);
return v_r_236_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2(void){
_start:
{
lean_object* v___x_240_; lean_object* v___x_241_; 
v___x_240_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__1));
v___x_241_ = l_Lean_MessageData_ofFormat(v___x_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(lean_object* v_msgData_242_, lean_object* v_macroStack_243_, lean_object* v___y_244_){
_start:
{
lean_object* v_options_246_; lean_object* v___x_247_; uint8_t v___x_248_; 
v_options_246_ = lean_ctor_get(v___y_244_, 2);
v___x_247_ = l_Lean_Elab_pp_macroStack;
v___x_248_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_options_246_, v___x_247_);
if (v___x_248_ == 0)
{
lean_object* v___x_249_; 
lean_dec(v_macroStack_243_);
v___x_249_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_249_, 0, v_msgData_242_);
return v___x_249_;
}
else
{
if (lean_obj_tag(v_macroStack_243_) == 0)
{
lean_object* v___x_250_; 
v___x_250_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_250_, 0, v_msgData_242_);
return v___x_250_;
}
else
{
lean_object* v_head_251_; lean_object* v_after_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_267_; 
v_head_251_ = lean_ctor_get(v_macroStack_243_, 0);
lean_inc(v_head_251_);
v_after_252_ = lean_ctor_get(v_head_251_, 1);
v_isSharedCheck_267_ = !lean_is_exclusive(v_head_251_);
if (v_isSharedCheck_267_ == 0)
{
lean_object* v_unused_268_; 
v_unused_268_ = lean_ctor_get(v_head_251_, 0);
lean_dec(v_unused_268_);
v___x_254_ = v_head_251_;
v_isShared_255_ = v_isSharedCheck_267_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_after_252_);
lean_dec(v_head_251_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_267_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; lean_object* v___x_258_; 
v___x_256_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0);
if (v_isShared_255_ == 0)
{
lean_ctor_set_tag(v___x_254_, 7);
lean_ctor_set(v___x_254_, 1, v___x_256_);
lean_ctor_set(v___x_254_, 0, v_msgData_242_);
v___x_258_ = v___x_254_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_266_; 
v_reuseFailAlloc_266_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_266_, 0, v_msgData_242_);
lean_ctor_set(v_reuseFailAlloc_266_, 1, v___x_256_);
v___x_258_ = v_reuseFailAlloc_266_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v_msgData_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v___x_259_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2);
v___x_260_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_260_, 0, v___x_258_);
lean_ctor_set(v___x_260_, 1, v___x_259_);
v___x_261_ = l_Lean_MessageData_ofSyntax(v_after_252_);
v___x_262_ = l_Lean_indentD(v___x_261_);
v_msgData_263_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_263_, 0, v___x_260_);
lean_ctor_set(v_msgData_263_, 1, v___x_262_);
v___x_264_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(v_msgData_263_, v_macroStack_243_);
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v___x_264_);
return v___x_265_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___boxed(lean_object* v_msgData_269_, lean_object* v_macroStack_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_msgData_269_, v_macroStack_270_, v___y_271_);
lean_dec_ref(v___y_271_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(lean_object* v_msg_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_){
_start:
{
lean_object* v_ref_282_; lean_object* v___x_283_; lean_object* v_a_284_; lean_object* v_macroStack_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v_a_288_; lean_object* v___x_290_; uint8_t v_isShared_291_; uint8_t v_isSharedCheck_296_; 
v_ref_282_ = lean_ctor_get(v___y_279_, 5);
v___x_283_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_274_, v___y_277_, v___y_278_, v___y_279_, v___y_280_);
v_a_284_ = lean_ctor_get(v___x_283_, 0);
lean_inc(v_a_284_);
lean_dec_ref(v___x_283_);
v_macroStack_285_ = lean_ctor_get(v___y_275_, 1);
v___x_286_ = l_Lean_Elab_getBetterRef(v_ref_282_, v_macroStack_285_);
lean_inc(v_macroStack_285_);
v___x_287_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_a_284_, v_macroStack_285_, v___y_279_);
v_a_288_ = lean_ctor_get(v___x_287_, 0);
v_isSharedCheck_296_ = !lean_is_exclusive(v___x_287_);
if (v_isSharedCheck_296_ == 0)
{
v___x_290_ = v___x_287_;
v_isShared_291_ = v_isSharedCheck_296_;
goto v_resetjp_289_;
}
else
{
lean_inc(v_a_288_);
lean_dec(v___x_287_);
v___x_290_ = lean_box(0);
v_isShared_291_ = v_isSharedCheck_296_;
goto v_resetjp_289_;
}
v_resetjp_289_:
{
lean_object* v___x_292_; lean_object* v___x_294_; 
v___x_292_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_292_, 0, v___x_286_);
lean_ctor_set(v___x_292_, 1, v_a_288_);
if (v_isShared_291_ == 0)
{
lean_ctor_set_tag(v___x_290_, 1);
lean_ctor_set(v___x_290_, 0, v___x_292_);
v___x_294_ = v___x_290_;
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg___boxed(lean_object* v_msg_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
lean_object* v_res_305_; 
v_res_305_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v_msg_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
return v_res_305_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0(void){
_start:
{
lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; 
v___x_306_ = lean_box(0);
v___x_307_ = l_Lean_Elab_abortTermExceptionId;
v___x_308_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_308_, 0, v___x_307_);
lean_ctor_set(v___x_308_, 1, v___x_306_);
return v___x_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg(){
_start:
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___closed__0);
v___x_311_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg___boxed(lean_object* v___y_312_){
_start:
{
lean_object* v_res_313_; 
v_res_313_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
return v_res_313_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(lean_object* v_e_314_, lean_object* v___y_315_){
_start:
{
uint8_t v___x_317_; 
v___x_317_ = l_Lean_Expr_hasMVar(v_e_314_);
if (v___x_317_ == 0)
{
lean_object* v___x_318_; 
v___x_318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_318_, 0, v_e_314_);
return v___x_318_;
}
else
{
lean_object* v___x_319_; lean_object* v_mctx_320_; lean_object* v___x_321_; lean_object* v_fst_322_; lean_object* v_snd_323_; lean_object* v___x_324_; lean_object* v_cache_325_; lean_object* v_zetaDeltaFVarIds_326_; lean_object* v_postponed_327_; lean_object* v_diag_328_; lean_object* v___x_330_; uint8_t v_isShared_331_; uint8_t v_isSharedCheck_337_; 
v___x_319_ = lean_st_ref_get(v___y_315_);
v_mctx_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc_ref(v_mctx_320_);
lean_dec(v___x_319_);
v___x_321_ = l_Lean_instantiateMVarsCore(v_mctx_320_, v_e_314_);
v_fst_322_ = lean_ctor_get(v___x_321_, 0);
lean_inc(v_fst_322_);
v_snd_323_ = lean_ctor_get(v___x_321_, 1);
lean_inc(v_snd_323_);
lean_dec_ref(v___x_321_);
v___x_324_ = lean_st_ref_take(v___y_315_);
v_cache_325_ = lean_ctor_get(v___x_324_, 1);
v_zetaDeltaFVarIds_326_ = lean_ctor_get(v___x_324_, 2);
v_postponed_327_ = lean_ctor_get(v___x_324_, 3);
v_diag_328_ = lean_ctor_get(v___x_324_, 4);
v_isSharedCheck_337_ = !lean_is_exclusive(v___x_324_);
if (v_isSharedCheck_337_ == 0)
{
lean_object* v_unused_338_; 
v_unused_338_ = lean_ctor_get(v___x_324_, 0);
lean_dec(v_unused_338_);
v___x_330_ = v___x_324_;
v_isShared_331_ = v_isSharedCheck_337_;
goto v_resetjp_329_;
}
else
{
lean_inc(v_diag_328_);
lean_inc(v_postponed_327_);
lean_inc(v_zetaDeltaFVarIds_326_);
lean_inc(v_cache_325_);
lean_dec(v___x_324_);
v___x_330_ = lean_box(0);
v_isShared_331_ = v_isSharedCheck_337_;
goto v_resetjp_329_;
}
v_resetjp_329_:
{
lean_object* v___x_333_; 
if (v_isShared_331_ == 0)
{
lean_ctor_set(v___x_330_, 0, v_snd_323_);
v___x_333_ = v___x_330_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v_snd_323_);
lean_ctor_set(v_reuseFailAlloc_336_, 1, v_cache_325_);
lean_ctor_set(v_reuseFailAlloc_336_, 2, v_zetaDeltaFVarIds_326_);
lean_ctor_set(v_reuseFailAlloc_336_, 3, v_postponed_327_);
lean_ctor_set(v_reuseFailAlloc_336_, 4, v_diag_328_);
v___x_333_ = v_reuseFailAlloc_336_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_334_ = lean_st_ref_set(v___y_315_, v___x_333_);
v___x_335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_335_, 0, v_fst_322_);
return v___x_335_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg___boxed(lean_object* v_e_339_, lean_object* v___y_340_, lean_object* v___y_341_){
_start:
{
lean_object* v_res_342_; 
v_res_342_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(v_e_339_, v___y_340_);
lean_dec(v___y_340_);
return v_res_342_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2(void){
_start:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_346_ = lean_box(0);
v___x_347_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__1));
v___x_348_ = l_Lean_mkConst(v___x_347_, v___x_346_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_349_; lean_object* v_ty_x3f_350_; 
v___x_349_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2);
v_ty_x3f_350_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_ty_x3f_350_, 0, v___x_349_);
return v_ty_x3f_350_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5(void){
_start:
{
lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_352_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__4));
v___x_353_ = l_Lean_stringToMessageData(v___x_352_);
return v___x_353_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6(void){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_354_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__2);
v___x_355_ = l_Lean_MessageData_ofExpr(v___x_354_);
return v___x_355_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_356_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__6);
v___x_357_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_358_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_358_, 0, v___x_357_);
lean_ctor_set(v___x_358_, 1, v___x_356_);
return v___x_358_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; 
v___x_360_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__8));
v___x_361_ = l_Lean_stringToMessageData(v___x_360_);
return v___x_361_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10(void){
_start:
{
lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_362_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_363_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__7);
v___x_364_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v___x_362_);
return v___x_364_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12(void){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; 
v___x_366_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__11));
v___x_367_ = l_Lean_stringToMessageData(v___x_366_);
return v___x_367_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; 
v___x_369_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__13));
v___x_370_ = l_Lean_stringToMessageData(v___x_369_);
return v___x_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0(lean_object* v_stx_371_, lean_object* v_a_372_, lean_object* v_a_373_, lean_object* v_a_374_, lean_object* v_a_375_, lean_object* v_a_376_, lean_object* v_a_377_){
_start:
{
lean_object* v_ty_x3f_379_; uint8_t v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v_fileName_385_; lean_object* v_fileMap_386_; lean_object* v_options_387_; lean_object* v_currRecDepth_388_; lean_object* v_maxRecDepth_389_; lean_object* v_ref_390_; lean_object* v_currNamespace_391_; lean_object* v_openDecls_392_; lean_object* v_initHeartbeats_393_; lean_object* v_maxHeartbeats_394_; lean_object* v_quotContext_395_; lean_object* v_currMacroScope_396_; uint8_t v_diag_397_; lean_object* v_cancelTk_x3f_398_; uint8_t v_suppressElabErrors_399_; lean_object* v_inheritedTraceOptions_400_; uint8_t v___x_401_; lean_object* v_ref_402_; lean_object* v___x_403_; lean_object* v___x_404_; 
v_ty_x3f_379_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__3);
v___x_380_ = 1;
v___x_381_ = lean_box(0);
v___x_382_ = lean_box(v___x_380_);
v___x_383_ = lean_box(v___x_380_);
lean_inc(v_stx_371_);
v___x_384_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_384_, 0, v_stx_371_);
lean_closure_set(v___x_384_, 1, v_ty_x3f_379_);
lean_closure_set(v___x_384_, 2, v___x_382_);
lean_closure_set(v___x_384_, 3, v___x_383_);
lean_closure_set(v___x_384_, 4, v___x_381_);
v_fileName_385_ = lean_ctor_get(v_a_376_, 0);
v_fileMap_386_ = lean_ctor_get(v_a_376_, 1);
v_options_387_ = lean_ctor_get(v_a_376_, 2);
v_currRecDepth_388_ = lean_ctor_get(v_a_376_, 3);
v_maxRecDepth_389_ = lean_ctor_get(v_a_376_, 4);
v_ref_390_ = lean_ctor_get(v_a_376_, 5);
v_currNamespace_391_ = lean_ctor_get(v_a_376_, 6);
v_openDecls_392_ = lean_ctor_get(v_a_376_, 7);
v_initHeartbeats_393_ = lean_ctor_get(v_a_376_, 8);
v_maxHeartbeats_394_ = lean_ctor_get(v_a_376_, 9);
v_quotContext_395_ = lean_ctor_get(v_a_376_, 10);
v_currMacroScope_396_ = lean_ctor_get(v_a_376_, 11);
v_diag_397_ = lean_ctor_get_uint8(v_a_376_, sizeof(void*)*14);
v_cancelTk_x3f_398_ = lean_ctor_get(v_a_376_, 12);
v_suppressElabErrors_399_ = lean_ctor_get_uint8(v_a_376_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_400_ = lean_ctor_get(v_a_376_, 13);
v___x_401_ = 1;
v_ref_402_ = l_Lean_replaceRef(v_stx_371_, v_ref_390_);
lean_dec(v_stx_371_);
lean_inc_ref(v_inheritedTraceOptions_400_);
lean_inc(v_cancelTk_x3f_398_);
lean_inc(v_currMacroScope_396_);
lean_inc(v_quotContext_395_);
lean_inc(v_maxHeartbeats_394_);
lean_inc(v_initHeartbeats_393_);
lean_inc(v_openDecls_392_);
lean_inc(v_currNamespace_391_);
lean_inc(v_maxRecDepth_389_);
lean_inc(v_currRecDepth_388_);
lean_inc_ref(v_options_387_);
lean_inc_ref(v_fileMap_386_);
lean_inc_ref(v_fileName_385_);
v___x_403_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_403_, 0, v_fileName_385_);
lean_ctor_set(v___x_403_, 1, v_fileMap_386_);
lean_ctor_set(v___x_403_, 2, v_options_387_);
lean_ctor_set(v___x_403_, 3, v_currRecDepth_388_);
lean_ctor_set(v___x_403_, 4, v_maxRecDepth_389_);
lean_ctor_set(v___x_403_, 5, v_ref_402_);
lean_ctor_set(v___x_403_, 6, v_currNamespace_391_);
lean_ctor_set(v___x_403_, 7, v_openDecls_392_);
lean_ctor_set(v___x_403_, 8, v_initHeartbeats_393_);
lean_ctor_set(v___x_403_, 9, v_maxHeartbeats_394_);
lean_ctor_set(v___x_403_, 10, v_quotContext_395_);
lean_ctor_set(v___x_403_, 11, v_currMacroScope_396_);
lean_ctor_set(v___x_403_, 12, v_cancelTk_x3f_398_);
lean_ctor_set(v___x_403_, 13, v_inheritedTraceOptions_400_);
lean_ctor_set_uint8(v___x_403_, sizeof(void*)*14, v_diag_397_);
lean_ctor_set_uint8(v___x_403_, sizeof(void*)*14 + 1, v_suppressElabErrors_399_);
v___x_404_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_384_, v___x_401_, v_a_372_, v_a_373_, v_a_374_, v_a_375_, v___x_403_, v_a_377_);
if (lean_obj_tag(v___x_404_) == 0)
{
lean_object* v_a_405_; lean_object* v___x_406_; lean_object* v_a_407_; lean_object* v___y_409_; lean_object* v___y_410_; lean_object* v___y_411_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_415_; lean_object* v___y_416_; lean_object* v___y_417_; uint8_t v___y_418_; lean_object* v___y_435_; lean_object* v___y_436_; lean_object* v___y_437_; lean_object* v___y_438_; lean_object* v___y_439_; lean_object* v___y_440_; lean_object* v___y_447_; lean_object* v___y_448_; lean_object* v___y_449_; lean_object* v___y_450_; lean_object* v___y_451_; lean_object* v___y_452_; lean_object* v___y_484_; lean_object* v___y_485_; lean_object* v___y_486_; lean_object* v___y_487_; lean_object* v___y_488_; lean_object* v___y_489_; uint8_t v___x_502_; 
v_a_405_ = lean_ctor_get(v___x_404_, 0);
lean_inc(v_a_405_);
lean_dec_ref_known(v___x_404_, 1);
v___x_406_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(v_a_405_, v_a_375_);
v_a_407_ = lean_ctor_get(v___x_406_, 0);
lean_inc(v_a_407_);
lean_dec_ref(v___x_406_);
v___x_502_ = l_Lean_Expr_hasSorry(v_a_407_);
if (v___x_502_ == 0)
{
v___y_447_ = v_a_372_;
v___y_448_ = v_a_373_;
v___y_449_ = v_a_374_;
v___y_450_ = v_a_375_;
v___y_451_ = v___x_403_;
v___y_452_ = v_a_377_;
goto v___jp_446_;
}
else
{
uint8_t v___x_503_; 
v___x_503_ = l_Lean_Expr_hasSyntheticSorry(v_a_407_);
if (v___x_503_ == 0)
{
v___y_484_ = v_a_372_;
v___y_485_ = v_a_373_;
v___y_486_ = v_a_374_;
v___y_487_ = v_a_375_;
v___y_488_ = v___x_403_;
v___y_489_ = v_a_377_;
goto v___jp_483_;
}
else
{
lean_object* v___x_504_; lean_object* v_a_505_; lean_object* v___x_507_; uint8_t v_isShared_508_; uint8_t v_isSharedCheck_512_; 
lean_dec(v_a_407_);
lean_dec_ref_known(v___x_403_, 14);
v___x_504_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_505_ = lean_ctor_get(v___x_504_, 0);
v_isSharedCheck_512_ = !lean_is_exclusive(v___x_504_);
if (v_isSharedCheck_512_ == 0)
{
v___x_507_ = v___x_504_;
v_isShared_508_ = v_isSharedCheck_512_;
goto v_resetjp_506_;
}
else
{
lean_inc(v_a_505_);
lean_dec(v___x_504_);
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
v___jp_408_:
{
if (v___y_418_ == 0)
{
if (lean_obj_tag(v___y_413_) == 0)
{
lean_dec_ref_known(v___y_413_, 2);
lean_dec_ref(v___y_417_);
lean_dec(v_a_407_);
return v___y_410_;
}
else
{
lean_object* v_id_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_432_; 
v_id_419_ = lean_ctor_get(v___y_413_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___y_413_);
if (v_isSharedCheck_432_ == 0)
{
lean_object* v_unused_433_; 
v_unused_433_ = lean_ctor_get(v___y_413_, 1);
lean_dec(v_unused_433_);
v___x_421_ = v___y_413_;
v_isShared_422_ = v_isSharedCheck_432_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_id_419_);
lean_dec(v___y_413_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_432_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
uint8_t v___x_423_; 
v___x_423_ = l_Lean_instBEqInternalExceptionId_beq(v___y_416_, v_id_419_);
lean_dec(v_id_419_);
if (v___x_423_ == 0)
{
lean_del_object(v___x_421_);
lean_dec_ref(v___y_417_);
lean_dec(v_a_407_);
return v___y_410_;
}
else
{
lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_428_; 
lean_dec_ref(v___y_410_);
v___x_424_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__10);
v___x_425_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12);
v___x_426_ = l_Lean_indentExpr(v_a_407_);
if (v_isShared_422_ == 0)
{
lean_ctor_set_tag(v___x_421_, 7);
lean_ctor_set(v___x_421_, 1, v___x_426_);
lean_ctor_set(v___x_421_, 0, v___x_425_);
v___x_428_ = v___x_421_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_431_, 1, v___x_426_);
v___x_428_ = v_reuseFailAlloc_431_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_429_, 0, v___x_428_);
lean_ctor_set(v___x_429_, 1, v___x_424_);
v___x_430_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_429_, v___y_414_, v___y_409_, v___y_412_, v___y_411_, v___y_417_, v___y_415_);
lean_dec_ref(v___y_417_);
return v___x_430_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_417_);
lean_dec_ref(v___y_413_);
lean_dec(v_a_407_);
return v___y_410_;
}
}
v___jp_434_:
{
lean_object* v___x_441_; 
lean_inc(v_a_407_);
v___x_441_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v_a_407_, v___y_437_, v___y_438_, v___y_439_, v___y_440_);
if (lean_obj_tag(v___x_441_) == 0)
{
lean_dec_ref(v___y_439_);
lean_dec(v_a_407_);
return v___x_441_;
}
else
{
lean_object* v_a_442_; lean_object* v___x_443_; uint8_t v___x_444_; 
v_a_442_ = lean_ctor_get(v___x_441_, 0);
lean_inc(v_a_442_);
v___x_443_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_444_ = l_Lean_Exception_isInterrupt(v_a_442_);
if (v___x_444_ == 0)
{
uint8_t v___x_445_; 
lean_inc(v_a_442_);
v___x_445_ = l_Lean_Exception_isRuntime(v_a_442_);
v___y_409_ = v___y_436_;
v___y_410_ = v___x_441_;
v___y_411_ = v___y_438_;
v___y_412_ = v___y_437_;
v___y_413_ = v_a_442_;
v___y_414_ = v___y_435_;
v___y_415_ = v___y_440_;
v___y_416_ = v___x_443_;
v___y_417_ = v___y_439_;
v___y_418_ = v___x_445_;
goto v___jp_408_;
}
else
{
v___y_409_ = v___y_436_;
v___y_410_ = v___x_441_;
v___y_411_ = v___y_438_;
v___y_412_ = v___y_437_;
v___y_413_ = v_a_442_;
v___y_414_ = v___y_435_;
v___y_415_ = v___y_440_;
v___y_416_ = v___x_443_;
v___y_417_ = v___y_439_;
v___y_418_ = v___x_444_;
goto v___jp_408_;
}
}
}
v___jp_446_:
{
lean_object* v___x_453_; 
lean_inc(v_a_407_);
v___x_453_ = l_Lean_Meta_getMVars(v_a_407_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
if (lean_obj_tag(v___x_453_) == 0)
{
lean_object* v_a_454_; lean_object* v___x_455_; 
v_a_454_ = lean_ctor_get(v___x_453_, 0);
lean_inc(v_a_454_);
lean_dec_ref_known(v___x_453_, 1);
v___x_455_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_454_, v___x_381_, v___y_447_, v___y_448_, v___y_449_, v___y_450_, v___y_451_, v___y_452_);
lean_dec(v_a_454_);
if (lean_obj_tag(v___x_455_) == 0)
{
lean_object* v_a_456_; uint8_t v___x_457_; 
v_a_456_ = lean_ctor_get(v___x_455_, 0);
lean_inc(v_a_456_);
lean_dec_ref_known(v___x_455_, 1);
v___x_457_ = lean_unbox(v_a_456_);
lean_dec(v_a_456_);
if (v___x_457_ == 0)
{
v___y_435_ = v___y_447_;
v___y_436_ = v___y_448_;
v___y_437_ = v___y_449_;
v___y_438_ = v___y_450_;
v___y_439_ = v___y_451_;
v___y_440_ = v___y_452_;
goto v___jp_434_;
}
else
{
lean_object* v___x_458_; lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_466_; 
lean_dec_ref(v___y_451_);
lean_dec(v_a_407_);
v___x_458_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_459_ = lean_ctor_get(v___x_458_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_458_);
if (v_isSharedCheck_466_ == 0)
{
v___x_461_ = v___x_458_;
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_458_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_464_; 
if (v_isShared_462_ == 0)
{
v___x_464_ = v___x_461_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_459_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
}
else
{
lean_object* v_a_467_; lean_object* v___x_469_; uint8_t v_isShared_470_; uint8_t v_isSharedCheck_474_; 
lean_dec_ref(v___y_451_);
lean_dec(v_a_407_);
v_a_467_ = lean_ctor_get(v___x_455_, 0);
v_isSharedCheck_474_ = !lean_is_exclusive(v___x_455_);
if (v_isSharedCheck_474_ == 0)
{
v___x_469_ = v___x_455_;
v_isShared_470_ = v_isSharedCheck_474_;
goto v_resetjp_468_;
}
else
{
lean_inc(v_a_467_);
lean_dec(v___x_455_);
v___x_469_ = lean_box(0);
v_isShared_470_ = v_isSharedCheck_474_;
goto v_resetjp_468_;
}
v_resetjp_468_:
{
lean_object* v___x_472_; 
if (v_isShared_470_ == 0)
{
v___x_472_ = v___x_469_;
goto v_reusejp_471_;
}
else
{
lean_object* v_reuseFailAlloc_473_; 
v_reuseFailAlloc_473_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_473_, 0, v_a_467_);
v___x_472_ = v_reuseFailAlloc_473_;
goto v_reusejp_471_;
}
v_reusejp_471_:
{
return v___x_472_;
}
}
}
}
else
{
lean_object* v_a_475_; lean_object* v___x_477_; uint8_t v_isShared_478_; uint8_t v_isSharedCheck_482_; 
lean_dec_ref(v___y_451_);
lean_dec(v_a_407_);
v_a_475_ = lean_ctor_get(v___x_453_, 0);
v_isSharedCheck_482_ = !lean_is_exclusive(v___x_453_);
if (v_isSharedCheck_482_ == 0)
{
v___x_477_ = v___x_453_;
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
else
{
lean_inc(v_a_475_);
lean_dec(v___x_453_);
v___x_477_ = lean_box(0);
v_isShared_478_ = v_isSharedCheck_482_;
goto v_resetjp_476_;
}
v_resetjp_476_:
{
lean_object* v___x_480_; 
if (v_isShared_478_ == 0)
{
v___x_480_ = v___x_477_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_481_; 
v_reuseFailAlloc_481_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_481_, 0, v_a_475_);
v___x_480_ = v_reuseFailAlloc_481_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
return v___x_480_;
}
}
}
}
v___jp_483_:
{
lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v_a_494_; lean_object* v___x_496_; uint8_t v_isShared_497_; uint8_t v_isSharedCheck_501_; 
v___x_490_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14);
v___x_491_ = l_Lean_indentExpr(v_a_407_);
v___x_492_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_490_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_492_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_, v___y_489_);
lean_dec_ref(v___y_488_);
v_a_494_ = lean_ctor_get(v___x_493_, 0);
v_isSharedCheck_501_ = !lean_is_exclusive(v___x_493_);
if (v_isSharedCheck_501_ == 0)
{
v___x_496_ = v___x_493_;
v_isShared_497_ = v_isSharedCheck_501_;
goto v_resetjp_495_;
}
else
{
lean_inc(v_a_494_);
lean_dec(v___x_493_);
v___x_496_ = lean_box(0);
v_isShared_497_ = v_isSharedCheck_501_;
goto v_resetjp_495_;
}
v_resetjp_495_:
{
lean_object* v___x_499_; 
if (v_isShared_497_ == 0)
{
v___x_499_ = v___x_496_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v_a_494_);
v___x_499_ = v_reuseFailAlloc_500_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
return v___x_499_;
}
}
}
}
else
{
lean_object* v_a_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_520_; 
lean_dec_ref_known(v___x_403_, 14);
v_a_513_ = lean_ctor_get(v___x_404_, 0);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_404_);
if (v_isSharedCheck_520_ == 0)
{
v___x_515_ = v___x_404_;
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_a_513_);
lean_dec(v___x_404_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_520_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_518_; 
if (v_isShared_516_ == 0)
{
v___x_518_ = v___x_515_;
goto v_reusejp_517_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_a_513_);
v___x_518_ = v_reuseFailAlloc_519_;
goto v_reusejp_517_;
}
v_reusejp_517_:
{
return v___x_518_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_stx_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_, lean_object* v_a_526_, lean_object* v_a_527_, lean_object* v_a_528_){
_start:
{
lean_object* v_res_529_; 
v_res_529_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0(v_stx_521_, v_a_522_, v_a_523_, v_a_524_, v_a_525_, v_a_526_, v_a_527_);
lean_dec(v_a_527_);
lean_dec_ref(v_a_526_);
lean_dec(v_a_525_);
lean_dec_ref(v_a_524_);
lean_dec(v_a_523_);
lean_dec_ref(v_a_522_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0(lean_object* v_stx_530_, lean_object* v_a_531_, lean_object* v_a_532_, lean_object* v_a_533_, lean_object* v_a_534_, lean_object* v_a_535_, lean_object* v_a_536_){
_start:
{
lean_object* v_fileName_538_; lean_object* v_fileMap_539_; lean_object* v_options_540_; lean_object* v_currRecDepth_541_; lean_object* v_maxRecDepth_542_; lean_object* v_ref_543_; lean_object* v_currNamespace_544_; lean_object* v_openDecls_545_; lean_object* v_initHeartbeats_546_; lean_object* v_maxHeartbeats_547_; lean_object* v_quotContext_548_; lean_object* v_currMacroScope_549_; uint8_t v_diag_550_; lean_object* v_cancelTk_x3f_551_; uint8_t v_suppressElabErrors_552_; lean_object* v_inheritedTraceOptions_553_; lean_object* v_ref_554_; lean_object* v___x_555_; lean_object* v___x_556_; 
v_fileName_538_ = lean_ctor_get(v_a_535_, 0);
v_fileMap_539_ = lean_ctor_get(v_a_535_, 1);
v_options_540_ = lean_ctor_get(v_a_535_, 2);
v_currRecDepth_541_ = lean_ctor_get(v_a_535_, 3);
v_maxRecDepth_542_ = lean_ctor_get(v_a_535_, 4);
v_ref_543_ = lean_ctor_get(v_a_535_, 5);
v_currNamespace_544_ = lean_ctor_get(v_a_535_, 6);
v_openDecls_545_ = lean_ctor_get(v_a_535_, 7);
v_initHeartbeats_546_ = lean_ctor_get(v_a_535_, 8);
v_maxHeartbeats_547_ = lean_ctor_get(v_a_535_, 9);
v_quotContext_548_ = lean_ctor_get(v_a_535_, 10);
v_currMacroScope_549_ = lean_ctor_get(v_a_535_, 11);
v_diag_550_ = lean_ctor_get_uint8(v_a_535_, sizeof(void*)*14);
v_cancelTk_x3f_551_ = lean_ctor_get(v_a_535_, 12);
v_suppressElabErrors_552_ = lean_ctor_get_uint8(v_a_535_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_553_ = lean_ctor_get(v_a_535_, 13);
v_ref_554_ = l_Lean_replaceRef(v_stx_530_, v_ref_543_);
lean_inc_ref(v_inheritedTraceOptions_553_);
lean_inc(v_cancelTk_x3f_551_);
lean_inc(v_currMacroScope_549_);
lean_inc(v_quotContext_548_);
lean_inc(v_maxHeartbeats_547_);
lean_inc(v_initHeartbeats_546_);
lean_inc(v_openDecls_545_);
lean_inc(v_currNamespace_544_);
lean_inc(v_maxRecDepth_542_);
lean_inc(v_currRecDepth_541_);
lean_inc_ref(v_options_540_);
lean_inc_ref(v_fileMap_539_);
lean_inc_ref(v_fileName_538_);
v___x_555_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_555_, 0, v_fileName_538_);
lean_ctor_set(v___x_555_, 1, v_fileMap_539_);
lean_ctor_set(v___x_555_, 2, v_options_540_);
lean_ctor_set(v___x_555_, 3, v_currRecDepth_541_);
lean_ctor_set(v___x_555_, 4, v_maxRecDepth_542_);
lean_ctor_set(v___x_555_, 5, v_ref_554_);
lean_ctor_set(v___x_555_, 6, v_currNamespace_544_);
lean_ctor_set(v___x_555_, 7, v_openDecls_545_);
lean_ctor_set(v___x_555_, 8, v_initHeartbeats_546_);
lean_ctor_set(v___x_555_, 9, v_maxHeartbeats_547_);
lean_ctor_set(v___x_555_, 10, v_quotContext_548_);
lean_ctor_set(v___x_555_, 11, v_currMacroScope_549_);
lean_ctor_set(v___x_555_, 12, v_cancelTk_x3f_551_);
lean_ctor_set(v___x_555_, 13, v_inheritedTraceOptions_553_);
lean_ctor_set_uint8(v___x_555_, sizeof(void*)*14, v_diag_550_);
lean_ctor_set_uint8(v___x_555_, sizeof(void*)*14 + 1, v_suppressElabErrors_552_);
lean_inc(v_stx_530_);
v___x_556_ = l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx(v_stx_530_, v_a_531_, v_a_532_, v_a_533_, v_a_534_, v___x_555_, v_a_536_);
if (lean_obj_tag(v___x_556_) == 0)
{
lean_object* v_a_557_; lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_565_; 
lean_dec_ref_known(v___x_555_, 14);
lean_dec(v_stx_530_);
v_a_557_ = lean_ctor_get(v___x_556_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_556_);
if (v_isSharedCheck_565_ == 0)
{
v___x_559_ = v___x_556_;
v_isShared_560_ = v_isSharedCheck_565_;
goto v_resetjp_558_;
}
else
{
lean_inc(v_a_557_);
lean_dec(v___x_556_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_565_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
lean_object* v_fst_561_; lean_object* v___x_563_; 
v_fst_561_ = lean_ctor_get(v_a_557_, 0);
lean_inc(v_fst_561_);
lean_dec(v_a_557_);
if (v_isShared_560_ == 0)
{
lean_ctor_set(v___x_559_, 0, v_fst_561_);
v___x_563_ = v___x_559_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_fst_561_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
else
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_581_; 
v_a_566_ = lean_ctor_get(v___x_556_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_556_);
if (v_isSharedCheck_581_ == 0)
{
v___x_568_ = v___x_556_;
v_isShared_569_ = v_isSharedCheck_581_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_556_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_581_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_570_; lean_object* v___x_572_; 
v___x_570_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_566_);
if (v_isShared_569_ == 0)
{
v___x_572_ = v___x_568_;
goto v_reusejp_571_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v_a_566_);
v___x_572_ = v_reuseFailAlloc_580_;
goto v_reusejp_571_;
}
v_reusejp_571_:
{
uint8_t v___y_574_; uint8_t v___x_578_; 
v___x_578_ = l_Lean_Exception_isInterrupt(v_a_566_);
if (v___x_578_ == 0)
{
uint8_t v___x_579_; 
lean_inc(v_a_566_);
v___x_579_ = l_Lean_Exception_isRuntime(v_a_566_);
v___y_574_ = v___x_579_;
goto v___jp_573_;
}
else
{
v___y_574_ = v___x_578_;
goto v___jp_573_;
}
v___jp_573_:
{
if (v___y_574_ == 0)
{
if (lean_obj_tag(v_a_566_) == 0)
{
lean_dec_ref_known(v_a_566_, 2);
lean_dec_ref_known(v___x_555_, 14);
lean_dec(v_stx_530_);
return v___x_572_;
}
else
{
lean_object* v_id_575_; uint8_t v___x_576_; 
v_id_575_ = lean_ctor_get(v_a_566_, 0);
lean_inc(v_id_575_);
lean_dec_ref_known(v_a_566_, 2);
v___x_576_ = l_Lean_instBEqInternalExceptionId_beq(v___x_570_, v_id_575_);
lean_dec(v_id_575_);
if (v___x_576_ == 0)
{
lean_dec_ref_known(v___x_555_, 14);
lean_dec(v_stx_530_);
return v___x_572_;
}
else
{
lean_object* v___x_577_; 
lean_dec_ref(v___x_572_);
v___x_577_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0(v_stx_530_, v_a_531_, v_a_532_, v_a_533_, v_a_534_, v___x_555_, v_a_536_);
lean_dec_ref_known(v___x_555_, 14);
return v___x_577_;
}
}
}
else
{
lean_dec(v_a_566_);
lean_dec_ref_known(v___x_555_, 14);
lean_dec(v_stx_530_);
return v___x_572_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_, lean_object* v_a_586_, lean_object* v_a_587_, lean_object* v_a_588_, lean_object* v_a_589_){
_start:
{
lean_object* v_res_590_; 
v_res_590_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0(v_stx_582_, v_a_583_, v_a_584_, v_a_585_, v_a_586_, v_a_587_, v_a_588_);
lean_dec(v_a_588_);
lean_dec_ref(v_a_587_);
lean_dec(v_a_586_);
lean_dec_ref(v_a_585_);
lean_dec(v_a_584_);
lean_dec_ref(v_a_583_);
return v_res_590_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0(void){
_start:
{
lean_object* v___x_591_; lean_object* v___x_592_; 
v___x_591_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__1);
v___x_592_ = l_Lean_MessageData_ofExpr(v___x_591_);
return v___x_592_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1(void){
_start:
{
lean_object* v___x_593_; lean_object* v___x_594_; lean_object* v___x_595_; 
v___x_593_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__0);
v___x_594_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__5);
v___x_595_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_595_, 0, v___x_594_);
lean_ctor_set(v___x_595_, 1, v___x_593_);
return v___x_595_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2(void){
_start:
{
lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; 
v___x_596_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_597_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__1);
v___x_598_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_598_, 0, v___x_597_);
lean_ctor_set(v___x_598_, 1, v___x_596_);
return v___x_598_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1(lean_object* v_stx_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v_ty_x3f_607_; uint8_t v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v_fileName_613_; lean_object* v_fileMap_614_; lean_object* v_options_615_; lean_object* v_currRecDepth_616_; lean_object* v_maxRecDepth_617_; lean_object* v_ref_618_; lean_object* v_currNamespace_619_; lean_object* v_openDecls_620_; lean_object* v_initHeartbeats_621_; lean_object* v_maxHeartbeats_622_; lean_object* v_quotContext_623_; lean_object* v_currMacroScope_624_; uint8_t v_diag_625_; lean_object* v_cancelTk_x3f_626_; uint8_t v_suppressElabErrors_627_; lean_object* v_inheritedTraceOptions_628_; uint8_t v___x_629_; lean_object* v_ref_630_; lean_object* v___x_631_; lean_object* v___x_632_; 
v_ty_x3f_607_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig___closed__2);
v___x_608_ = 1;
v___x_609_ = lean_box(0);
v___x_610_ = lean_box(v___x_608_);
v___x_611_ = lean_box(v___x_608_);
lean_inc(v_stx_599_);
v___x_612_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_612_, 0, v_stx_599_);
lean_closure_set(v___x_612_, 1, v_ty_x3f_607_);
lean_closure_set(v___x_612_, 2, v___x_610_);
lean_closure_set(v___x_612_, 3, v___x_611_);
lean_closure_set(v___x_612_, 4, v___x_609_);
v_fileName_613_ = lean_ctor_get(v_a_604_, 0);
v_fileMap_614_ = lean_ctor_get(v_a_604_, 1);
v_options_615_ = lean_ctor_get(v_a_604_, 2);
v_currRecDepth_616_ = lean_ctor_get(v_a_604_, 3);
v_maxRecDepth_617_ = lean_ctor_get(v_a_604_, 4);
v_ref_618_ = lean_ctor_get(v_a_604_, 5);
v_currNamespace_619_ = lean_ctor_get(v_a_604_, 6);
v_openDecls_620_ = lean_ctor_get(v_a_604_, 7);
v_initHeartbeats_621_ = lean_ctor_get(v_a_604_, 8);
v_maxHeartbeats_622_ = lean_ctor_get(v_a_604_, 9);
v_quotContext_623_ = lean_ctor_get(v_a_604_, 10);
v_currMacroScope_624_ = lean_ctor_get(v_a_604_, 11);
v_diag_625_ = lean_ctor_get_uint8(v_a_604_, sizeof(void*)*14);
v_cancelTk_x3f_626_ = lean_ctor_get(v_a_604_, 12);
v_suppressElabErrors_627_ = lean_ctor_get_uint8(v_a_604_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_628_ = lean_ctor_get(v_a_604_, 13);
v___x_629_ = 1;
v_ref_630_ = l_Lean_replaceRef(v_stx_599_, v_ref_618_);
lean_dec(v_stx_599_);
lean_inc_ref(v_inheritedTraceOptions_628_);
lean_inc(v_cancelTk_x3f_626_);
lean_inc(v_currMacroScope_624_);
lean_inc(v_quotContext_623_);
lean_inc(v_maxHeartbeats_622_);
lean_inc(v_initHeartbeats_621_);
lean_inc(v_openDecls_620_);
lean_inc(v_currNamespace_619_);
lean_inc(v_maxRecDepth_617_);
lean_inc(v_currRecDepth_616_);
lean_inc_ref(v_options_615_);
lean_inc_ref(v_fileMap_614_);
lean_inc_ref(v_fileName_613_);
v___x_631_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_631_, 0, v_fileName_613_);
lean_ctor_set(v___x_631_, 1, v_fileMap_614_);
lean_ctor_set(v___x_631_, 2, v_options_615_);
lean_ctor_set(v___x_631_, 3, v_currRecDepth_616_);
lean_ctor_set(v___x_631_, 4, v_maxRecDepth_617_);
lean_ctor_set(v___x_631_, 5, v_ref_630_);
lean_ctor_set(v___x_631_, 6, v_currNamespace_619_);
lean_ctor_set(v___x_631_, 7, v_openDecls_620_);
lean_ctor_set(v___x_631_, 8, v_initHeartbeats_621_);
lean_ctor_set(v___x_631_, 9, v_maxHeartbeats_622_);
lean_ctor_set(v___x_631_, 10, v_quotContext_623_);
lean_ctor_set(v___x_631_, 11, v_currMacroScope_624_);
lean_ctor_set(v___x_631_, 12, v_cancelTk_x3f_626_);
lean_ctor_set(v___x_631_, 13, v_inheritedTraceOptions_628_);
lean_ctor_set_uint8(v___x_631_, sizeof(void*)*14, v_diag_625_);
lean_ctor_set_uint8(v___x_631_, sizeof(void*)*14 + 1, v_suppressElabErrors_627_);
v___x_632_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_612_, v___x_629_, v_a_600_, v_a_601_, v_a_602_, v_a_603_, v___x_631_, v_a_605_);
if (lean_obj_tag(v___x_632_) == 0)
{
lean_object* v_a_633_; lean_object* v___x_634_; lean_object* v_a_635_; lean_object* v___y_637_; lean_object* v___y_638_; lean_object* v___y_639_; lean_object* v___y_640_; lean_object* v___y_641_; lean_object* v___y_642_; lean_object* v___y_643_; lean_object* v___y_644_; lean_object* v___y_645_; uint8_t v___y_646_; lean_object* v___y_663_; lean_object* v___y_664_; lean_object* v___y_665_; lean_object* v___y_666_; lean_object* v___y_667_; lean_object* v___y_668_; lean_object* v___y_675_; lean_object* v___y_676_; lean_object* v___y_677_; lean_object* v___y_678_; lean_object* v___y_679_; lean_object* v___y_680_; lean_object* v___y_712_; lean_object* v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___y_716_; lean_object* v___y_717_; uint8_t v___x_730_; 
v_a_633_ = lean_ctor_get(v___x_632_, 0);
lean_inc(v_a_633_);
lean_dec_ref_known(v___x_632_, 1);
v___x_634_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(v_a_633_, v_a_603_);
v_a_635_ = lean_ctor_get(v___x_634_, 0);
lean_inc(v_a_635_);
lean_dec_ref(v___x_634_);
v___x_730_ = l_Lean_Expr_hasSorry(v_a_635_);
if (v___x_730_ == 0)
{
v___y_675_ = v_a_600_;
v___y_676_ = v_a_601_;
v___y_677_ = v_a_602_;
v___y_678_ = v_a_603_;
v___y_679_ = v___x_631_;
v___y_680_ = v_a_605_;
goto v___jp_674_;
}
else
{
uint8_t v___x_731_; 
v___x_731_ = l_Lean_Expr_hasSyntheticSorry(v_a_635_);
if (v___x_731_ == 0)
{
v___y_712_ = v_a_600_;
v___y_713_ = v_a_601_;
v___y_714_ = v_a_602_;
v___y_715_ = v_a_603_;
v___y_716_ = v___x_631_;
v___y_717_ = v_a_605_;
goto v___jp_711_;
}
else
{
lean_object* v___x_732_; lean_object* v_a_733_; lean_object* v___x_735_; uint8_t v_isShared_736_; uint8_t v_isSharedCheck_740_; 
lean_dec(v_a_635_);
lean_dec_ref_known(v___x_631_, 14);
v___x_732_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_733_ = lean_ctor_get(v___x_732_, 0);
v_isSharedCheck_740_ = !lean_is_exclusive(v___x_732_);
if (v_isSharedCheck_740_ == 0)
{
v___x_735_ = v___x_732_;
v_isShared_736_ = v_isSharedCheck_740_;
goto v_resetjp_734_;
}
else
{
lean_inc(v_a_733_);
lean_dec(v___x_732_);
v___x_735_ = lean_box(0);
v_isShared_736_ = v_isSharedCheck_740_;
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
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v_a_733_);
v___x_738_ = v_reuseFailAlloc_739_;
goto v_reusejp_737_;
}
v_reusejp_737_:
{
return v___x_738_;
}
}
}
}
v___jp_636_:
{
if (v___y_646_ == 0)
{
if (lean_obj_tag(v___y_643_) == 0)
{
lean_dec_ref_known(v___y_643_, 2);
lean_dec_ref(v___y_640_);
lean_dec(v_a_635_);
return v___y_637_;
}
else
{
lean_object* v_id_647_; lean_object* v___x_649_; uint8_t v_isShared_650_; uint8_t v_isSharedCheck_660_; 
v_id_647_ = lean_ctor_get(v___y_643_, 0);
v_isSharedCheck_660_ = !lean_is_exclusive(v___y_643_);
if (v_isSharedCheck_660_ == 0)
{
lean_object* v_unused_661_; 
v_unused_661_ = lean_ctor_get(v___y_643_, 1);
lean_dec(v_unused_661_);
v___x_649_ = v___y_643_;
v_isShared_650_ = v_isSharedCheck_660_;
goto v_resetjp_648_;
}
else
{
lean_inc(v_id_647_);
lean_dec(v___y_643_);
v___x_649_ = lean_box(0);
v_isShared_650_ = v_isSharedCheck_660_;
goto v_resetjp_648_;
}
v_resetjp_648_:
{
uint8_t v___x_651_; 
v___x_651_ = l_Lean_instBEqInternalExceptionId_beq(v___y_641_, v_id_647_);
lean_dec(v_id_647_);
if (v___x_651_ == 0)
{
lean_del_object(v___x_649_);
lean_dec_ref(v___y_640_);
lean_dec(v_a_635_);
return v___y_637_;
}
else
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_656_; 
lean_dec_ref(v___y_637_);
v___x_652_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___closed__2);
v___x_653_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__12);
v___x_654_ = l_Lean_indentExpr(v_a_635_);
if (v_isShared_650_ == 0)
{
lean_ctor_set_tag(v___x_649_, 7);
lean_ctor_set(v___x_649_, 1, v___x_654_);
lean_ctor_set(v___x_649_, 0, v___x_653_);
v___x_656_ = v___x_649_;
goto v_reusejp_655_;
}
else
{
lean_object* v_reuseFailAlloc_659_; 
v_reuseFailAlloc_659_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_659_, 0, v___x_653_);
lean_ctor_set(v_reuseFailAlloc_659_, 1, v___x_654_);
v___x_656_ = v_reuseFailAlloc_659_;
goto v_reusejp_655_;
}
v_reusejp_655_:
{
lean_object* v___x_657_; lean_object* v___x_658_; 
v___x_657_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_656_);
lean_ctor_set(v___x_657_, 1, v___x_652_);
v___x_658_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_657_, v___y_642_, v___y_639_, v___y_644_, v___y_645_, v___y_640_, v___y_638_);
lean_dec_ref(v___y_640_);
return v___x_658_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_643_);
lean_dec_ref(v___y_640_);
lean_dec(v_a_635_);
return v___y_637_;
}
}
v___jp_662_:
{
lean_object* v___x_669_; 
lean_inc(v_a_635_);
v___x_669_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr(v_a_635_, v___y_665_, v___y_666_, v___y_667_, v___y_668_);
if (lean_obj_tag(v___x_669_) == 0)
{
lean_dec_ref(v___y_667_);
lean_dec(v_a_635_);
return v___x_669_;
}
else
{
lean_object* v_a_670_; lean_object* v___x_671_; uint8_t v___x_672_; 
v_a_670_ = lean_ctor_get(v___x_669_, 0);
lean_inc(v_a_670_);
v___x_671_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_672_ = l_Lean_Exception_isInterrupt(v_a_670_);
if (v___x_672_ == 0)
{
uint8_t v___x_673_; 
lean_inc(v_a_670_);
v___x_673_ = l_Lean_Exception_isRuntime(v_a_670_);
v___y_637_ = v___x_669_;
v___y_638_ = v___y_668_;
v___y_639_ = v___y_664_;
v___y_640_ = v___y_667_;
v___y_641_ = v___x_671_;
v___y_642_ = v___y_663_;
v___y_643_ = v_a_670_;
v___y_644_ = v___y_665_;
v___y_645_ = v___y_666_;
v___y_646_ = v___x_673_;
goto v___jp_636_;
}
else
{
v___y_637_ = v___x_669_;
v___y_638_ = v___y_668_;
v___y_639_ = v___y_664_;
v___y_640_ = v___y_667_;
v___y_641_ = v___x_671_;
v___y_642_ = v___y_663_;
v___y_643_ = v_a_670_;
v___y_644_ = v___y_665_;
v___y_645_ = v___y_666_;
v___y_646_ = v___x_672_;
goto v___jp_636_;
}
}
}
v___jp_674_:
{
lean_object* v___x_681_; 
lean_inc(v_a_635_);
v___x_681_ = l_Lean_Meta_getMVars(v_a_635_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
if (lean_obj_tag(v___x_681_) == 0)
{
lean_object* v_a_682_; lean_object* v___x_683_; 
v_a_682_ = lean_ctor_get(v___x_681_, 0);
lean_inc(v_a_682_);
lean_dec_ref_known(v___x_681_, 1);
v___x_683_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_682_, v___x_609_, v___y_675_, v___y_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
lean_dec(v_a_682_);
if (lean_obj_tag(v___x_683_) == 0)
{
lean_object* v_a_684_; uint8_t v___x_685_; 
v_a_684_ = lean_ctor_get(v___x_683_, 0);
lean_inc(v_a_684_);
lean_dec_ref_known(v___x_683_, 1);
v___x_685_ = lean_unbox(v_a_684_);
lean_dec(v_a_684_);
if (v___x_685_ == 0)
{
v___y_663_ = v___y_675_;
v___y_664_ = v___y_676_;
v___y_665_ = v___y_677_;
v___y_666_ = v___y_678_;
v___y_667_ = v___y_679_;
v___y_668_ = v___y_680_;
goto v___jp_662_;
}
else
{
lean_object* v___x_686_; lean_object* v_a_687_; lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_694_; 
lean_dec_ref(v___y_679_);
lean_dec(v_a_635_);
v___x_686_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
v_a_687_ = lean_ctor_get(v___x_686_, 0);
v_isSharedCheck_694_ = !lean_is_exclusive(v___x_686_);
if (v_isSharedCheck_694_ == 0)
{
v___x_689_ = v___x_686_;
v_isShared_690_ = v_isSharedCheck_694_;
goto v_resetjp_688_;
}
else
{
lean_inc(v_a_687_);
lean_dec(v___x_686_);
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
lean_object* v_a_695_; lean_object* v___x_697_; uint8_t v_isShared_698_; uint8_t v_isSharedCheck_702_; 
lean_dec_ref(v___y_679_);
lean_dec(v_a_635_);
v_a_695_ = lean_ctor_get(v___x_683_, 0);
v_isSharedCheck_702_ = !lean_is_exclusive(v___x_683_);
if (v_isSharedCheck_702_ == 0)
{
v___x_697_ = v___x_683_;
v_isShared_698_ = v_isSharedCheck_702_;
goto v_resetjp_696_;
}
else
{
lean_inc(v_a_695_);
lean_dec(v___x_683_);
v___x_697_ = lean_box(0);
v_isShared_698_ = v_isSharedCheck_702_;
goto v_resetjp_696_;
}
v_resetjp_696_:
{
lean_object* v___x_700_; 
if (v_isShared_698_ == 0)
{
v___x_700_ = v___x_697_;
goto v_reusejp_699_;
}
else
{
lean_object* v_reuseFailAlloc_701_; 
v_reuseFailAlloc_701_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_701_, 0, v_a_695_);
v___x_700_ = v_reuseFailAlloc_701_;
goto v_reusejp_699_;
}
v_reusejp_699_:
{
return v___x_700_;
}
}
}
}
else
{
lean_object* v_a_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_710_; 
lean_dec_ref(v___y_679_);
lean_dec(v_a_635_);
v_a_703_ = lean_ctor_get(v___x_681_, 0);
v_isSharedCheck_710_ = !lean_is_exclusive(v___x_681_);
if (v_isSharedCheck_710_ == 0)
{
v___x_705_ = v___x_681_;
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_a_703_);
lean_dec(v___x_681_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_710_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v___x_708_; 
if (v_isShared_706_ == 0)
{
v___x_708_ = v___x_705_;
goto v_reusejp_707_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v_a_703_);
v___x_708_ = v_reuseFailAlloc_709_;
goto v_reusejp_707_;
}
v_reusejp_707_:
{
return v___x_708_;
}
}
}
}
v___jp_711_:
{
lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; lean_object* v_a_722_; lean_object* v___x_724_; uint8_t v_isShared_725_; uint8_t v_isSharedCheck_729_; 
v___x_718_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__14);
v___x_719_ = l_Lean_indentExpr(v_a_635_);
v___x_720_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_720_, 0, v___x_718_);
lean_ctor_set(v___x_720_, 1, v___x_719_);
v___x_721_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v___x_720_, v___y_712_, v___y_713_, v___y_714_, v___y_715_, v___y_716_, v___y_717_);
lean_dec_ref(v___y_716_);
v_a_722_ = lean_ctor_get(v___x_721_, 0);
v_isSharedCheck_729_ = !lean_is_exclusive(v___x_721_);
if (v_isSharedCheck_729_ == 0)
{
v___x_724_ = v___x_721_;
v_isShared_725_ = v_isSharedCheck_729_;
goto v_resetjp_723_;
}
else
{
lean_inc(v_a_722_);
lean_dec(v___x_721_);
v___x_724_ = lean_box(0);
v_isShared_725_ = v_isSharedCheck_729_;
goto v_resetjp_723_;
}
v_resetjp_723_:
{
lean_object* v___x_727_; 
if (v_isShared_725_ == 0)
{
v___x_727_ = v___x_724_;
goto v_reusejp_726_;
}
else
{
lean_object* v_reuseFailAlloc_728_; 
v_reuseFailAlloc_728_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_728_, 0, v_a_722_);
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
else
{
lean_object* v_a_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_748_; 
lean_dec_ref_known(v___x_631_, 14);
v_a_741_ = lean_ctor_get(v___x_632_, 0);
v_isSharedCheck_748_ = !lean_is_exclusive(v___x_632_);
if (v_isSharedCheck_748_ == 0)
{
v___x_743_ = v___x_632_;
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_a_741_);
lean_dec(v___x_632_);
v___x_743_ = lean_box(0);
v_isShared_744_ = v_isSharedCheck_748_;
goto v_resetjp_742_;
}
v_resetjp_742_:
{
lean_object* v___x_746_; 
if (v_isShared_744_ == 0)
{
v___x_746_ = v___x_743_;
goto v_reusejp_745_;
}
else
{
lean_object* v_reuseFailAlloc_747_; 
v_reuseFailAlloc_747_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_747_, 0, v_a_741_);
v___x_746_ = v_reuseFailAlloc_747_;
goto v_reusejp_745_;
}
v_reusejp_745_:
{
return v___x_746_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1___boxed(lean_object* v_stx_749_, lean_object* v_a_750_, lean_object* v_a_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_){
_start:
{
lean_object* v_res_757_; 
v_res_757_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1(v_stx_749_, v_a_750_, v_a_751_, v_a_752_, v_a_753_, v_a_754_, v_a_755_);
lean_dec(v_a_755_);
lean_dec_ref(v_a_754_);
lean_dec(v_a_753_);
lean_dec_ref(v_a_752_);
lean_dec(v_a_751_);
lean_dec_ref(v_a_750_);
return v_res_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0(lean_object* v_config_775_, lean_object* v_item_776_, lean_object* v___y_777_, lean_object* v___y_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_){
_start:
{
lean_object* v_item_785_; lean_object* v___y_786_; lean_object* v___y_787_; lean_object* v___y_788_; lean_object* v___y_789_; lean_object* v___y_790_; lean_object* v___y_791_; lean_object* v___x_794_; lean_object* v___x_795_; 
v___x_794_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5));
v___x_795_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_776_, v___x_794_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_795_) == 0)
{
uint8_t v___x_796_; 
lean_dec_ref_known(v___x_795_, 1);
v___x_796_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_776_);
if (v___x_796_ == 0)
{
lean_object* v___x_797_; lean_object* v___x_798_; lean_object* v___x_799_; uint8_t v___x_800_; 
v___x_797_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_776_);
lean_inc_ref(v_item_776_);
v___x_798_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_776_);
v___x_799_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__1));
v___x_800_ = lean_string_dec_eq(v___x_797_, v___x_799_);
if (v___x_800_ == 0)
{
lean_object* v___x_801_; uint8_t v___x_802_; 
v___x_801_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__2));
v___x_802_ = lean_string_dec_eq(v___x_797_, v___x_801_);
if (v___x_802_ == 0)
{
lean_object* v___x_803_; uint8_t v___x_804_; 
v___x_803_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__3));
v___x_804_ = lean_string_dec_eq(v___x_797_, v___x_803_);
lean_dec_ref(v___x_797_);
if (v___x_804_ == 0)
{
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_item_785_ = v___x_798_;
v___y_786_ = v___y_777_;
v___y_787_ = v___y_778_;
v___y_788_ = v___y_779_;
v___y_789_ = v___y_780_;
v___y_790_ = v___y_781_;
v___y_791_ = v___y_782_;
goto v___jp_784_;
}
else
{
lean_object* v___x_805_; lean_object* v___x_806_; 
v___x_805_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__4));
v___x_806_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_776_, v___x_805_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_806_) == 0)
{
uint8_t v___x_807_; 
lean_dec_ref_known(v___x_806_, 1);
v___x_807_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_798_);
if (v___x_807_ == 0)
{
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_item_785_ = v___x_798_;
v___y_786_ = v___y_777_;
v___y_787_ = v___y_778_;
v___y_788_ = v___y_779_;
v___y_789_ = v___y_780_;
v___y_790_ = v___y_781_;
v___y_791_ = v___y_782_;
goto v___jp_784_;
}
else
{
lean_object* v___x_808_; 
lean_dec_ref(v___x_798_);
lean_inc_ref(v_item_776_);
v___x_808_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_808_) == 0)
{
lean_object* v_value_809_; lean_object* v___x_810_; 
lean_dec_ref_known(v___x_808_, 1);
v_value_809_ = lean_ctor_get(v_item_776_, 2);
lean_inc(v_value_809_);
lean_dec_ref(v_item_776_);
v___x_810_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0(v_value_809_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_810_) == 0)
{
lean_object* v_a_811_; lean_object* v___x_813_; uint8_t v_isShared_814_; uint8_t v_isSharedCheck_827_; 
v_a_811_ = lean_ctor_get(v___x_810_, 0);
v_isSharedCheck_827_ = !lean_is_exclusive(v___x_810_);
if (v_isSharedCheck_827_ == 0)
{
v___x_813_ = v___x_810_;
v_isShared_814_ = v_isSharedCheck_827_;
goto v_resetjp_812_;
}
else
{
lean_inc(v_a_811_);
lean_dec(v___x_810_);
v___x_813_ = lean_box(0);
v_isShared_814_ = v_isSharedCheck_827_;
goto v_resetjp_812_;
}
v_resetjp_812_:
{
lean_object* v_maxSteps_815_; lean_object* v___x_817_; uint8_t v_isShared_818_; uint8_t v_isSharedCheck_825_; 
v_maxSteps_815_ = lean_ctor_get(v_config_775_, 1);
v_isSharedCheck_825_ = !lean_is_exclusive(v_config_775_);
if (v_isSharedCheck_825_ == 0)
{
lean_object* v_unused_826_; 
v_unused_826_ = lean_ctor_get(v_config_775_, 0);
lean_dec(v_unused_826_);
v___x_817_ = v_config_775_;
v_isShared_818_ = v_isSharedCheck_825_;
goto v_resetjp_816_;
}
else
{
lean_inc(v_maxSteps_815_);
lean_dec(v_config_775_);
v___x_817_ = lean_box(0);
v_isShared_818_ = v_isSharedCheck_825_;
goto v_resetjp_816_;
}
v_resetjp_816_:
{
lean_object* v___x_820_; 
if (v_isShared_818_ == 0)
{
lean_ctor_set(v___x_817_, 0, v_a_811_);
v___x_820_ = v___x_817_;
goto v_reusejp_819_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v_a_811_);
lean_ctor_set(v_reuseFailAlloc_824_, 1, v_maxSteps_815_);
v___x_820_ = v_reuseFailAlloc_824_;
goto v_reusejp_819_;
}
v_reusejp_819_:
{
lean_object* v___x_822_; 
if (v_isShared_814_ == 0)
{
lean_ctor_set(v___x_813_, 0, v___x_820_);
v___x_822_ = v___x_813_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v___x_820_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
}
else
{
lean_object* v_a_828_; lean_object* v___x_830_; uint8_t v_isShared_831_; uint8_t v_isSharedCheck_835_; 
lean_dec_ref(v_config_775_);
v_a_828_ = lean_ctor_get(v___x_810_, 0);
v_isSharedCheck_835_ = !lean_is_exclusive(v___x_810_);
if (v_isSharedCheck_835_ == 0)
{
v___x_830_ = v___x_810_;
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
else
{
lean_inc(v_a_828_);
lean_dec(v___x_810_);
v___x_830_ = lean_box(0);
v_isShared_831_ = v_isSharedCheck_835_;
goto v_resetjp_829_;
}
v_resetjp_829_:
{
lean_object* v___x_833_; 
if (v_isShared_831_ == 0)
{
v___x_833_ = v___x_830_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v_a_828_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
}
else
{
lean_object* v_a_836_; lean_object* v___x_838_; uint8_t v_isShared_839_; uint8_t v_isSharedCheck_843_; 
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_a_836_ = lean_ctor_get(v___x_808_, 0);
v_isSharedCheck_843_ = !lean_is_exclusive(v___x_808_);
if (v_isSharedCheck_843_ == 0)
{
v___x_838_ = v___x_808_;
v_isShared_839_ = v_isSharedCheck_843_;
goto v_resetjp_837_;
}
else
{
lean_inc(v_a_836_);
lean_dec(v___x_808_);
v___x_838_ = lean_box(0);
v_isShared_839_ = v_isSharedCheck_843_;
goto v_resetjp_837_;
}
v_resetjp_837_:
{
lean_object* v___x_841_; 
if (v_isShared_839_ == 0)
{
v___x_841_ = v___x_838_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v_a_836_);
v___x_841_ = v_reuseFailAlloc_842_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
return v___x_841_;
}
}
}
}
}
else
{
lean_object* v_a_844_; lean_object* v___x_846_; uint8_t v_isShared_847_; uint8_t v_isSharedCheck_851_; 
lean_dec_ref(v___x_798_);
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_a_844_ = lean_ctor_get(v___x_806_, 0);
v_isSharedCheck_851_ = !lean_is_exclusive(v___x_806_);
if (v_isSharedCheck_851_ == 0)
{
v___x_846_ = v___x_806_;
v_isShared_847_ = v_isSharedCheck_851_;
goto v_resetjp_845_;
}
else
{
lean_inc(v_a_844_);
lean_dec(v___x_806_);
v___x_846_ = lean_box(0);
v_isShared_847_ = v_isSharedCheck_851_;
goto v_resetjp_845_;
}
v_resetjp_845_:
{
lean_object* v___x_849_; 
if (v_isShared_847_ == 0)
{
v___x_849_ = v___x_846_;
goto v_reusejp_848_;
}
else
{
lean_object* v_reuseFailAlloc_850_; 
v_reuseFailAlloc_850_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_850_, 0, v_a_844_);
v___x_849_ = v_reuseFailAlloc_850_;
goto v_reusejp_848_;
}
v_reusejp_848_:
{
return v___x_849_;
}
}
}
}
}
else
{
lean_object* v___x_852_; lean_object* v___x_853_; 
lean_dec_ref(v___x_797_);
v___x_852_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__5));
v___x_853_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_776_, v___x_852_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_853_) == 0)
{
uint8_t v___x_854_; 
lean_dec_ref_known(v___x_853_, 1);
v___x_854_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_798_);
if (v___x_854_ == 0)
{
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_item_785_ = v___x_798_;
v___y_786_ = v___y_777_;
v___y_787_ = v___y_778_;
v___y_788_ = v___y_779_;
v___y_789_ = v___y_780_;
v___y_790_ = v___y_781_;
v___y_791_ = v___y_782_;
goto v___jp_784_;
}
else
{
lean_object* v___x_855_; 
lean_dec_ref(v___x_798_);
lean_inc_ref(v_item_776_);
v___x_855_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_776_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_855_) == 0)
{
lean_object* v_value_856_; lean_object* v___x_857_; 
lean_dec_ref_known(v___x_855_, 1);
v_value_856_ = lean_ctor_get(v_item_776_, 2);
lean_inc(v_value_856_);
lean_dec_ref(v_item_776_);
v___x_857_ = lp_mathlib_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0(v_value_856_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
if (lean_obj_tag(v___x_857_) == 0)
{
lean_object* v_a_858_; lean_object* v___x_860_; uint8_t v_isShared_861_; uint8_t v_isSharedCheck_874_; 
v_a_858_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_874_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_874_ == 0)
{
v___x_860_ = v___x_857_;
v_isShared_861_ = v_isSharedCheck_874_;
goto v_resetjp_859_;
}
else
{
lean_inc(v_a_858_);
lean_dec(v___x_857_);
v___x_860_ = lean_box(0);
v_isShared_861_ = v_isSharedCheck_874_;
goto v_resetjp_859_;
}
v_resetjp_859_:
{
lean_object* v_maxTransitionDepth_862_; lean_object* v___x_864_; uint8_t v_isShared_865_; uint8_t v_isSharedCheck_872_; 
v_maxTransitionDepth_862_ = lean_ctor_get(v_config_775_, 0);
v_isSharedCheck_872_ = !lean_is_exclusive(v_config_775_);
if (v_isSharedCheck_872_ == 0)
{
lean_object* v_unused_873_; 
v_unused_873_ = lean_ctor_get(v_config_775_, 1);
lean_dec(v_unused_873_);
v___x_864_ = v_config_775_;
v_isShared_865_ = v_isSharedCheck_872_;
goto v_resetjp_863_;
}
else
{
lean_inc(v_maxTransitionDepth_862_);
lean_dec(v_config_775_);
v___x_864_ = lean_box(0);
v_isShared_865_ = v_isSharedCheck_872_;
goto v_resetjp_863_;
}
v_resetjp_863_:
{
lean_object* v___x_867_; 
if (v_isShared_865_ == 0)
{
lean_ctor_set(v___x_864_, 1, v_a_858_);
v___x_867_ = v___x_864_;
goto v_reusejp_866_;
}
else
{
lean_object* v_reuseFailAlloc_871_; 
v_reuseFailAlloc_871_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_871_, 0, v_maxTransitionDepth_862_);
lean_ctor_set(v_reuseFailAlloc_871_, 1, v_a_858_);
v___x_867_ = v_reuseFailAlloc_871_;
goto v_reusejp_866_;
}
v_reusejp_866_:
{
lean_object* v___x_869_; 
if (v_isShared_861_ == 0)
{
lean_ctor_set(v___x_860_, 0, v___x_867_);
v___x_869_ = v___x_860_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v___x_867_);
v___x_869_ = v_reuseFailAlloc_870_;
goto v_reusejp_868_;
}
v_reusejp_868_:
{
return v___x_869_;
}
}
}
}
}
else
{
lean_object* v_a_875_; lean_object* v___x_877_; uint8_t v_isShared_878_; uint8_t v_isSharedCheck_882_; 
lean_dec_ref(v_config_775_);
v_a_875_ = lean_ctor_get(v___x_857_, 0);
v_isSharedCheck_882_ = !lean_is_exclusive(v___x_857_);
if (v_isSharedCheck_882_ == 0)
{
v___x_877_ = v___x_857_;
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
else
{
lean_inc(v_a_875_);
lean_dec(v___x_857_);
v___x_877_ = lean_box(0);
v_isShared_878_ = v_isSharedCheck_882_;
goto v_resetjp_876_;
}
v_resetjp_876_:
{
lean_object* v___x_880_; 
if (v_isShared_878_ == 0)
{
v___x_880_ = v___x_877_;
goto v_reusejp_879_;
}
else
{
lean_object* v_reuseFailAlloc_881_; 
v_reuseFailAlloc_881_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_881_, 0, v_a_875_);
v___x_880_ = v_reuseFailAlloc_881_;
goto v_reusejp_879_;
}
v_reusejp_879_:
{
return v___x_880_;
}
}
}
}
else
{
lean_object* v_a_883_; lean_object* v___x_885_; uint8_t v_isShared_886_; uint8_t v_isSharedCheck_890_; 
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_a_883_ = lean_ctor_get(v___x_855_, 0);
v_isSharedCheck_890_ = !lean_is_exclusive(v___x_855_);
if (v_isSharedCheck_890_ == 0)
{
v___x_885_ = v___x_855_;
v_isShared_886_ = v_isSharedCheck_890_;
goto v_resetjp_884_;
}
else
{
lean_inc(v_a_883_);
lean_dec(v___x_855_);
v___x_885_ = lean_box(0);
v_isShared_886_ = v_isSharedCheck_890_;
goto v_resetjp_884_;
}
v_resetjp_884_:
{
lean_object* v___x_888_; 
if (v_isShared_886_ == 0)
{
v___x_888_ = v___x_885_;
goto v_reusejp_887_;
}
else
{
lean_object* v_reuseFailAlloc_889_; 
v_reuseFailAlloc_889_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_889_, 0, v_a_883_);
v___x_888_ = v_reuseFailAlloc_889_;
goto v_reusejp_887_;
}
v_reusejp_887_:
{
return v___x_888_;
}
}
}
}
}
else
{
lean_object* v_a_891_; lean_object* v___x_893_; uint8_t v_isShared_894_; uint8_t v_isSharedCheck_898_; 
lean_dec_ref(v___x_798_);
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_a_891_ = lean_ctor_get(v___x_853_, 0);
v_isSharedCheck_898_ = !lean_is_exclusive(v___x_853_);
if (v_isSharedCheck_898_ == 0)
{
v___x_893_ = v___x_853_;
v_isShared_894_ = v_isSharedCheck_898_;
goto v_resetjp_892_;
}
else
{
lean_inc(v_a_891_);
lean_dec(v___x_853_);
v___x_893_ = lean_box(0);
v_isShared_894_ = v_isSharedCheck_898_;
goto v_resetjp_892_;
}
v_resetjp_892_:
{
lean_object* v___x_896_; 
if (v_isShared_894_ == 0)
{
v___x_896_ = v___x_893_;
goto v_reusejp_895_;
}
else
{
lean_object* v_reuseFailAlloc_897_; 
v_reuseFailAlloc_897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_897_, 0, v_a_891_);
v___x_896_ = v_reuseFailAlloc_897_;
goto v_reusejp_895_;
}
v_reusejp_895_:
{
return v___x_896_;
}
}
}
}
}
else
{
uint8_t v___x_899_; 
lean_dec_ref(v___x_797_);
lean_dec_ref(v_config_775_);
v___x_899_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_798_);
if (v___x_899_ == 0)
{
lean_dec_ref(v_item_776_);
v_item_785_ = v___x_798_;
v___y_786_ = v___y_777_;
v___y_787_ = v___y_778_;
v___y_788_ = v___y_779_;
v___y_789_ = v___y_780_;
v___y_790_ = v___y_781_;
v___y_791_ = v___y_782_;
goto v___jp_784_;
}
else
{
lean_object* v_value_900_; lean_object* v___x_901_; 
lean_dec_ref(v___x_798_);
v_value_900_ = lean_ctor_get(v_item_776_, 2);
lean_inc(v_value_900_);
lean_dec_ref(v_item_776_);
v___x_901_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1(v_value_900_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
return v___x_901_;
}
}
}
else
{
lean_dec_ref(v_config_775_);
v_item_785_ = v_item_776_;
v___y_786_ = v___y_777_;
v___y_787_ = v___y_778_;
v___y_788_ = v___y_779_;
v___y_789_ = v___y_780_;
v___y_790_ = v___y_781_;
v___y_791_ = v___y_782_;
goto v___jp_784_;
}
}
else
{
lean_object* v_a_902_; lean_object* v___x_904_; uint8_t v_isShared_905_; uint8_t v_isSharedCheck_909_; 
lean_dec_ref(v_item_776_);
lean_dec_ref(v_config_775_);
v_a_902_ = lean_ctor_get(v___x_795_, 0);
v_isSharedCheck_909_ = !lean_is_exclusive(v___x_795_);
if (v_isSharedCheck_909_ == 0)
{
v___x_904_ = v___x_795_;
v_isShared_905_ = v_isSharedCheck_909_;
goto v_resetjp_903_;
}
else
{
lean_inc(v_a_902_);
lean_dec(v___x_795_);
v___x_904_ = lean_box(0);
v_isShared_905_ = v_isSharedCheck_909_;
goto v_resetjp_903_;
}
v_resetjp_903_:
{
lean_object* v___x_907_; 
if (v_isShared_905_ == 0)
{
v___x_907_ = v___x_904_;
goto v_reusejp_906_;
}
else
{
lean_object* v_reuseFailAlloc_908_; 
v_reuseFailAlloc_908_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_908_, 0, v_a_902_);
v___x_907_ = v_reuseFailAlloc_908_;
goto v_reusejp_906_;
}
v_reusejp_906_:
{
return v___x_907_;
}
}
}
v___jp_784_:
{
lean_object* v___x_792_; lean_object* v___x_793_; 
v___x_792_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___closed__0));
v___x_793_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_785_, v___x_792_, v___y_786_, v___y_787_, v___y_788_, v___y_789_, v___y_790_, v___y_791_);
return v___x_793_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_910_, lean_object* v_item_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_){
_start:
{
lean_object* v_res_919_; 
v_res_919_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___lam__0(v_config_910_, v_item_911_, v___y_912_, v___y_913_, v___y_914_, v___y_915_, v___y_916_, v___y_917_);
lean_dec(v___y_917_);
lean_dec_ref(v___y_916_);
lean_dec(v___y_915_);
lean_dec_ref(v___y_914_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
return v_res_919_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2(lean_object* v_e_922_, lean_object* v___y_923_, lean_object* v___y_924_, lean_object* v___y_925_, lean_object* v___y_926_, lean_object* v___y_927_, lean_object* v___y_928_){
_start:
{
lean_object* v___x_930_; 
v___x_930_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___redArg(v_e_922_, v___y_926_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object* v_e_931_, lean_object* v___y_932_, lean_object* v___y_933_, lean_object* v___y_934_, lean_object* v___y_935_, lean_object* v___y_936_, lean_object* v___y_937_, lean_object* v___y_938_){
_start:
{
lean_object* v_res_939_; 
v_res_939_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__2(v_e_931_, v___y_932_, v___y_933_, v___y_934_, v___y_935_, v___y_936_, v___y_937_);
lean_dec(v___y_937_);
lean_dec_ref(v___y_936_);
lean_dec(v___y_935_);
lean_dec_ref(v___y_934_);
lean_dec(v___y_933_);
lean_dec_ref(v___y_932_);
return v_res_939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4(lean_object* v_00_u03b1_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_, lean_object* v___y_946_){
_start:
{
lean_object* v___x_948_; 
v___x_948_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___redArg();
return v___x_948_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4___boxed(lean_object* v_00_u03b1_949_, lean_object* v___y_950_, lean_object* v___y_951_, lean_object* v___y_952_, lean_object* v___y_953_, lean_object* v___y_954_, lean_object* v___y_955_, lean_object* v___y_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__4(v_00_u03b1_949_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_, v___y_955_);
lean_dec(v___y_955_);
lean_dec_ref(v___y_954_);
lean_dec(v___y_953_);
lean_dec_ref(v___y_952_);
lean_dec(v___y_951_);
lean_dec_ref(v___y_950_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3(lean_object* v_00_u03b1_958_, lean_object* v_msg_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_, lean_object* v___y_964_, lean_object* v___y_965_){
_start:
{
lean_object* v___x_967_; 
v___x_967_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___redArg(v_msg_959_, v___y_960_, v___y_961_, v___y_962_, v___y_963_, v___y_964_, v___y_965_);
return v___x_967_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3___boxed(lean_object* v_00_u03b1_968_, lean_object* v_msg_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_){
_start:
{
lean_object* v_res_977_; 
v_res_977_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3(v_00_u03b1_968_, v_msg_969_, v___y_970_, v___y_971_, v___y_972_, v___y_973_, v___y_974_, v___y_975_);
lean_dec(v___y_975_);
lean_dec_ref(v___y_974_);
lean_dec(v___y_973_);
lean_dec_ref(v___y_972_);
lean_dec(v___y_971_);
lean_dec_ref(v___y_970_);
return v_res_977_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4(lean_object* v_msgData_978_, lean_object* v_macroStack_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_, lean_object* v___y_985_){
_start:
{
lean_object* v___x_987_; 
v___x_987_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg(v_msgData_978_, v_macroStack_979_, v___y_984_);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___boxed(lean_object* v_msgData_988_, lean_object* v_macroStack_989_, lean_object* v___y_990_, lean_object* v___y_991_, lean_object* v___y_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
lean_object* v_res_997_; 
v_res_997_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4(v_msgData_988_, v_macroStack_989_, v___y_990_, v___y_991_, v___y_992_, v___y_993_, v___y_994_, v___y_995_);
lean_dec(v___y_995_);
lean_dec_ref(v___y_994_);
lean_dec(v___y_993_);
lean_dec_ref(v___y_992_);
lean_dec(v___y_991_);
lean_dec_ref(v___y_990_);
return v_res_997_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_998_; lean_object* v___x_999_; lean_object* v___x_1000_; 
v___x_998_ = lean_box(0);
v___x_999_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__5));
v___x_1000_ = l_Lean_mkConst(v___x_999_, v___x_998_);
return v___x_1000_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1001_; lean_object* v___x_1002_; 
v___x_1001_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__0);
v___x_1002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1002_, 0, v___x_1001_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0(lean_object* v_cfg_1003_, lean_object* v_cfgItem_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
lean_object* v___x_1012_; lean_object* v___x_1013_; 
v___x_1012_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___closed__1);
v___x_1013_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_1003_, v_cfgItem_1004_, v___x_1012_, v___y_1005_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_, v___y_1010_);
return v___x_1013_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0___boxed(lean_object* v_cfg_1014_, lean_object* v_cfgItem_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_, lean_object* v___y_1021_, lean_object* v___y_1022_){
_start:
{
lean_object* v_res_1023_; 
v_res_1023_ = lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___lam__0(v_cfg_1014_, v_cfgItem_1015_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec(v___y_1021_);
lean_dec_ref(v___y_1020_);
lean_dec(v___y_1019_);
lean_dec_ref(v___y_1018_);
lean_dec(v___y_1017_);
lean_dec_ref(v___y_1016_);
lean_dec(v_cfgItem_1015_);
return v_res_1023_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg(lean_object* v_cfg_1025_, lean_object* v_init_1026_, uint8_t v_logExceptions_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v_onErr_1032_; lean_object* v_eval_1033_; 
v_onErr_1032_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___closed__0));
v_eval_1033_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem___closed__0));
if (v_logExceptions_1027_ == 0)
{
lean_object* v___x_1034_; 
v___x_1034_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_1033_, v_init_1026_, v_cfg_1025_, v_onErr_1032_, v_logExceptions_1027_, v_a_1029_, v_a_1030_);
return v___x_1034_;
}
else
{
uint8_t v_recover_1035_; lean_object* v___x_1036_; 
v_recover_1035_ = lean_ctor_get_uint8(v_a_1028_, sizeof(void*)*1);
v___x_1036_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_1033_, v_init_1026_, v_cfg_1025_, v_onErr_1032_, v_recover_1035_, v_a_1029_, v_a_1030_);
return v___x_1036_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg___boxed(lean_object* v_cfg_1037_, lean_object* v_init_1038_, lean_object* v_logExceptions_1039_, lean_object* v_a_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_, lean_object* v_a_1043_){
_start:
{
uint8_t v_logExceptions_boxed_1044_; lean_object* v_res_1045_; 
v_logExceptions_boxed_1044_ = lean_unbox(v_logExceptions_1039_);
v_res_1045_ = lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg(v_cfg_1037_, v_init_1038_, v_logExceptions_boxed_1044_, v_a_1040_, v_a_1041_, v_a_1042_);
lean_dec(v_a_1042_);
lean_dec_ref(v_a_1041_);
lean_dec_ref(v_a_1040_);
return v_res_1045_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig(lean_object* v_cfg_1046_, lean_object* v_init_1047_, uint8_t v_logExceptions_1048_, lean_object* v_a_1049_, lean_object* v_a_1050_, lean_object* v_a_1051_, lean_object* v_a_1052_, lean_object* v_a_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_){
_start:
{
lean_object* v___x_1058_; 
v___x_1058_ = lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg(v_cfg_1046_, v_init_1047_, v_logExceptions_1048_, v_a_1049_, v_a_1055_, v_a_1056_);
return v___x_1058_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___boxed(lean_object* v_cfg_1059_, lean_object* v_init_1060_, lean_object* v_logExceptions_1061_, lean_object* v_a_1062_, lean_object* v_a_1063_, lean_object* v_a_1064_, lean_object* v_a_1065_, lean_object* v_a_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_, lean_object* v_a_1070_){
_start:
{
uint8_t v_logExceptions_boxed_1071_; lean_object* v_res_1072_; 
v_logExceptions_boxed_1071_ = lean_unbox(v_logExceptions_1061_);
v_res_1072_ = lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig(v_cfg_1059_, v_init_1060_, v_logExceptions_boxed_1071_, v_a_1062_, v_a_1063_, v_a_1064_, v_a_1065_, v_a_1066_, v_a_1067_, v_a_1068_, v_a_1069_);
lean_dec(v_a_1069_);
lean_dec_ref(v_a_1068_);
lean_dec(v_a_1067_);
lean_dec_ref(v_a_1066_);
lean_dec(v_a_1065_);
lean_dec_ref(v_a_1064_);
lean_dec(v_a_1063_);
lean_dec_ref(v_a_1062_);
return v_res_1072_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6(void){
_start:
{
lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; 
v___x_1086_ = l_Lean_Parser_Tactic_optConfig;
v___x_1087_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__5));
v___x_1088_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3));
v___x_1089_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1089_, 0, v___x_1088_);
lean_ctor_set(v___x_1089_, 1, v___x_1087_);
lean_ctor_set(v___x_1089_, 2, v___x_1086_);
return v___x_1089_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9(void){
_start:
{
lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; 
v___x_1093_ = l_Lean_Parser_Tactic_discharger;
v___x_1094_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__8));
v___x_1095_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1095_, 0, v___x_1094_);
lean_ctor_set(v___x_1095_, 1, v___x_1093_);
return v___x_1095_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10(void){
_start:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; 
v___x_1096_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9, &lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__9);
v___x_1097_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__6);
v___x_1098_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3));
v___x_1099_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1098_);
lean_ctor_set(v___x_1099_, 1, v___x_1097_);
lean_ctor_set(v___x_1099_, 2, v___x_1096_);
return v___x_1099_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28(void){
_start:
{
lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; 
v___x_1137_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__27));
v___x_1138_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10, &lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__10);
v___x_1139_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__3));
v___x_1140_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_1140_, 0, v___x_1139_);
lean_ctor_set(v___x_1140_, 1, v___x_1138_);
lean_ctor_set(v___x_1140_, 2, v___x_1137_);
return v___x_1140_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29(void){
_start:
{
lean_object* v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; lean_object* v___x_1144_; 
v___x_1141_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28, &lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__28);
v___x_1142_ = lean_unsigned_to_nat(1022u);
v___x_1143_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1));
v___x_1144_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_1144_, 0, v___x_1143_);
lean_ctor_set(v___x_1144_, 1, v___x_1142_);
lean_ctor_set(v___x_1144_, 2, v___x_1141_);
return v___x_1144_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx(void){
_start:
{
lean_object* v___x_1145_; 
v___x_1145_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29, &lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__29);
return v___x_1145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge(lean_object* v_e_1193_, lean_object* v_a_1194_, lean_object* v_a_1195_, lean_object* v_a_1196_, lean_object* v_a_1197_){
_start:
{
lean_object* v_ref_1199_; uint8_t v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v___x_1207_; lean_object* v___x_1208_; lean_object* v___x_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; lean_object* v___x_1212_; lean_object* v___x_1213_; lean_object* v___x_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; lean_object* v___x_1217_; lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; lean_object* v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1223_; lean_object* v___x_1224_; lean_object* v___x_1225_; lean_object* v___x_1226_; lean_object* v___x_1227_; lean_object* v___x_1228_; lean_object* v___x_1229_; lean_object* v___x_1230_; lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; 
v_ref_1199_ = lean_ctor_get(v_a_1196_, 5);
v___x_1200_ = 0;
v___x_1201_ = l_Lean_SourceInfo_fromRef(v_ref_1199_, v___x_1200_);
v___x_1202_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3));
v___x_1203_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__4));
lean_inc_n(v___x_1201_, 20);
v___x_1204_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1204_, 0, v___x_1201_);
lean_ctor_set(v___x_1204_, 1, v___x_1202_);
v___x_1205_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__6));
v___x_1206_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__8));
v___x_1207_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__9));
v___x_1208_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1208_, 0, v___x_1201_);
lean_ctor_set(v___x_1208_, 1, v___x_1207_);
v___x_1209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__11));
v___x_1210_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__13));
v___x_1211_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__15));
v___x_1212_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__16));
v___x_1213_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1213_, 0, v___x_1201_);
lean_ctor_set(v___x_1213_, 1, v___x_1212_);
v___x_1214_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17));
v___x_1215_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__18));
v___x_1216_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1216_, 0, v___x_1201_);
lean_ctor_set(v___x_1216_, 1, v___x_1214_);
v___x_1217_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1215_, v___x_1216_);
v___x_1218_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1205_, v___x_1217_);
v___x_1219_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1210_, v___x_1218_);
v___x_1220_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1209_, v___x_1219_);
v___x_1221_ = l_Lean_Syntax_node2(v___x_1201_, v___x_1211_, v___x_1213_, v___x_1220_);
v___x_1222_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1205_, v___x_1221_);
v___x_1223_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1210_, v___x_1222_);
v___x_1224_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1209_, v___x_1223_);
lean_inc_ref(v___x_1208_);
v___x_1225_ = l_Lean_Syntax_node2(v___x_1201_, v___x_1206_, v___x_1208_, v___x_1224_);
v___x_1226_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__20));
v___x_1227_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__21));
v___x_1228_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1228_, 0, v___x_1201_);
lean_ctor_set(v___x_1228_, 1, v___x_1227_);
v___x_1229_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1226_, v___x_1228_);
v___x_1230_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1205_, v___x_1229_);
v___x_1231_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1210_, v___x_1230_);
v___x_1232_ = l_Lean_Syntax_node1(v___x_1201_, v___x_1209_, v___x_1231_);
v___x_1233_ = l_Lean_Syntax_node2(v___x_1201_, v___x_1206_, v___x_1208_, v___x_1232_);
v___x_1234_ = l_Lean_Syntax_node2(v___x_1201_, v___x_1205_, v___x_1225_, v___x_1233_);
v___x_1235_ = l_Lean_Syntax_node2(v___x_1201_, v___x_1203_, v___x_1204_, v___x_1234_);
v___x_1236_ = lp_mathlib_Mathlib_Meta_FunProp_tacticToDischarge(v___x_1235_, v_e_1193_, v_a_1194_, v_a_1195_, v_a_1196_, v_a_1197_);
return v___x_1236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___boxed(lean_object* v_e_1237_, lean_object* v_a_1238_, lean_object* v_a_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_, lean_object* v_a_1242_){
_start:
{
lean_object* v_res_1243_; 
v_res_1243_ = lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge(v_e_1237_, v_a_1238_, v_a_1239_, v_a_1240_, v_a_1241_);
lean_dec(v_a_1241_);
lean_dec_ref(v_a_1240_);
lean_dec(v_a_1239_);
lean_dec_ref(v_a_1238_);
return v_res_1243_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1246_; 
v___x_1244_ = lean_box(0);
v___x_1245_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1246_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1246_, 0, v___x_1245_);
lean_ctor_set(v___x_1246_, 1, v___x_1244_);
return v___x_1246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg(){
_start:
{
lean_object* v___x_1248_; lean_object* v___x_1249_; 
v___x_1248_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0);
v___x_1249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1248_);
return v___x_1249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___boxed(lean_object* v___y_1250_){
_start:
{
lean_object* v_res_1251_; 
v_res_1251_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v_res_1251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0(lean_object* v_00_u03b1_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_, lean_object* v___y_1258_, lean_object* v___y_1259_, lean_object* v___y_1260_){
_start:
{
lean_object* v___x_1262_; 
v___x_1262_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v___x_1262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___boxed(lean_object* v_00_u03b1_1263_, lean_object* v___y_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_, lean_object* v___y_1270_, lean_object* v___y_1271_, lean_object* v___y_1272_){
_start:
{
lean_object* v_res_1273_; 
v_res_1273_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0(v_00_u03b1_1263_, v___y_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_, v___y_1269_, v___y_1270_, v___y_1271_);
lean_dec(v___y_1271_);
lean_dec_ref(v___y_1270_);
lean_dec(v___y_1269_);
lean_dec_ref(v___y_1268_);
lean_dec(v___y_1267_);
lean_dec_ref(v___y_1266_);
lean_dec(v___y_1265_);
lean_dec_ref(v___y_1264_);
return v_res_1273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0(lean_object* v_k_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_, lean_object* v___y_1277_, lean_object* v___y_1278_, lean_object* v_b_1279_, lean_object* v_c_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_){
_start:
{
lean_object* v___x_1286_; 
lean_inc(v___y_1284_);
lean_inc_ref(v___y_1283_);
lean_inc(v___y_1282_);
lean_inc_ref(v___y_1281_);
lean_inc(v___y_1278_);
lean_inc_ref(v___y_1277_);
lean_inc(v___y_1276_);
lean_inc_ref(v___y_1275_);
v___x_1286_ = lean_apply_11(v_k_1274_, v_b_1279_, v_c_1280_, v___y_1275_, v___y_1276_, v___y_1277_, v___y_1278_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_, lean_box(0));
return v___x_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0___boxed(lean_object* v_k_1287_, lean_object* v___y_1288_, lean_object* v___y_1289_, lean_object* v___y_1290_, lean_object* v___y_1291_, lean_object* v_b_1292_, lean_object* v_c_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_){
_start:
{
lean_object* v_res_1299_; 
v_res_1299_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0(v_k_1287_, v___y_1288_, v___y_1289_, v___y_1290_, v___y_1291_, v_b_1292_, v_c_1293_, v___y_1294_, v___y_1295_, v___y_1296_, v___y_1297_);
lean_dec(v___y_1297_);
lean_dec_ref(v___y_1296_);
lean_dec(v___y_1295_);
lean_dec_ref(v___y_1294_);
lean_dec(v___y_1291_);
lean_dec_ref(v___y_1290_);
lean_dec(v___y_1289_);
lean_dec_ref(v___y_1288_);
return v_res_1299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg(lean_object* v_type_1300_, lean_object* v_k_1301_, uint8_t v_cleanupAnnotations_1302_, uint8_t v_whnfType_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_, lean_object* v___y_1308_, lean_object* v___y_1309_, lean_object* v___y_1310_, lean_object* v___y_1311_){
_start:
{
lean_object* v___f_1313_; lean_object* v___x_1314_; 
lean_inc(v___y_1307_);
lean_inc_ref(v___y_1306_);
lean_inc(v___y_1305_);
lean_inc_ref(v___y_1304_);
v___f_1313_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___lam__0___boxed), 12, 5);
lean_closure_set(v___f_1313_, 0, v_k_1301_);
lean_closure_set(v___f_1313_, 1, v___y_1304_);
lean_closure_set(v___f_1313_, 2, v___y_1305_);
lean_closure_set(v___f_1313_, 3, v___y_1306_);
lean_closure_set(v___f_1313_, 4, v___y_1307_);
v___x_1314_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingImp(lean_box(0), v_type_1300_, v___f_1313_, v_cleanupAnnotations_1302_, v_whnfType_1303_, v___y_1308_, v___y_1309_, v___y_1310_, v___y_1311_);
if (lean_obj_tag(v___x_1314_) == 0)
{
return v___x_1314_;
}
else
{
lean_object* v_a_1315_; lean_object* v___x_1317_; uint8_t v_isShared_1318_; uint8_t v_isSharedCheck_1322_; 
v_a_1315_ = lean_ctor_get(v___x_1314_, 0);
v_isSharedCheck_1322_ = !lean_is_exclusive(v___x_1314_);
if (v_isSharedCheck_1322_ == 0)
{
v___x_1317_ = v___x_1314_;
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
else
{
lean_inc(v_a_1315_);
lean_dec(v___x_1314_);
v___x_1317_ = lean_box(0);
v_isShared_1318_ = v_isSharedCheck_1322_;
goto v_resetjp_1316_;
}
v_resetjp_1316_:
{
lean_object* v___x_1320_; 
if (v_isShared_1318_ == 0)
{
v___x_1320_ = v___x_1317_;
goto v_reusejp_1319_;
}
else
{
lean_object* v_reuseFailAlloc_1321_; 
v_reuseFailAlloc_1321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1321_, 0, v_a_1315_);
v___x_1320_ = v_reuseFailAlloc_1321_;
goto v_reusejp_1319_;
}
v_reusejp_1319_:
{
return v___x_1320_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg___boxed(lean_object* v_type_1323_, lean_object* v_k_1324_, lean_object* v_cleanupAnnotations_1325_, lean_object* v_whnfType_1326_, lean_object* v___y_1327_, lean_object* v___y_1328_, lean_object* v___y_1329_, lean_object* v___y_1330_, lean_object* v___y_1331_, lean_object* v___y_1332_, lean_object* v___y_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1336_; uint8_t v_whnfType_boxed_1337_; lean_object* v_res_1338_; 
v_cleanupAnnotations_boxed_1336_ = lean_unbox(v_cleanupAnnotations_1325_);
v_whnfType_boxed_1337_ = lean_unbox(v_whnfType_1326_);
v_res_1338_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg(v_type_1323_, v_k_1324_, v_cleanupAnnotations_boxed_1336_, v_whnfType_boxed_1337_, v___y_1327_, v___y_1328_, v___y_1329_, v___y_1330_, v___y_1331_, v___y_1332_, v___y_1333_, v___y_1334_);
lean_dec(v___y_1334_);
lean_dec_ref(v___y_1333_);
lean_dec(v___y_1332_);
lean_dec_ref(v___y_1331_);
lean_dec(v___y_1330_);
lean_dec_ref(v___y_1329_);
lean_dec(v___y_1328_);
lean_dec_ref(v___y_1327_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5(lean_object* v_00_u03b1_1339_, lean_object* v_type_1340_, lean_object* v_k_1341_, uint8_t v_cleanupAnnotations_1342_, uint8_t v_whnfType_1343_, lean_object* v___y_1344_, lean_object* v___y_1345_, lean_object* v___y_1346_, lean_object* v___y_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_){
_start:
{
lean_object* v___x_1353_; 
v___x_1353_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg(v_type_1340_, v_k_1341_, v_cleanupAnnotations_1342_, v_whnfType_1343_, v___y_1344_, v___y_1345_, v___y_1346_, v___y_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_);
return v___x_1353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___boxed(lean_object* v_00_u03b1_1354_, lean_object* v_type_1355_, lean_object* v_k_1356_, lean_object* v_cleanupAnnotations_1357_, lean_object* v_whnfType_1358_, lean_object* v___y_1359_, lean_object* v___y_1360_, lean_object* v___y_1361_, lean_object* v___y_1362_, lean_object* v___y_1363_, lean_object* v___y_1364_, lean_object* v___y_1365_, lean_object* v___y_1366_, lean_object* v___y_1367_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1368_; uint8_t v_whnfType_boxed_1369_; lean_object* v_res_1370_; 
v_cleanupAnnotations_boxed_1368_ = lean_unbox(v_cleanupAnnotations_1357_);
v_whnfType_boxed_1369_ = lean_unbox(v_whnfType_1358_);
v_res_1370_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5(v_00_u03b1_1354_, v_type_1355_, v_k_1356_, v_cleanupAnnotations_boxed_1368_, v_whnfType_boxed_1369_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_, v___y_1363_, v___y_1364_, v___y_1365_, v___y_1366_);
lean_dec(v___y_1366_);
lean_dec_ref(v___y_1365_);
lean_dec(v___y_1364_);
lean_dec_ref(v___y_1363_);
lean_dec(v___y_1362_);
lean_dec_ref(v___y_1361_);
lean_dec(v___y_1360_);
lean_dec_ref(v___y_1359_);
return v_res_1370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0(lean_object* v_x_1371_, lean_object* v___y_1372_, lean_object* v___y_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_, lean_object* v___y_1379_){
_start:
{
lean_object* v___x_1381_; 
lean_inc(v___y_1375_);
lean_inc_ref(v___y_1374_);
lean_inc(v___y_1373_);
lean_inc_ref(v___y_1372_);
v___x_1381_ = lean_apply_9(v_x_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_, v___y_1378_, v___y_1379_, lean_box(0));
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0___boxed(lean_object* v_x_1382_, lean_object* v___y_1383_, lean_object* v___y_1384_, lean_object* v___y_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_){
_start:
{
lean_object* v_res_1392_; 
v_res_1392_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0(v_x_1382_, v___y_1383_, v___y_1384_, v___y_1385_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_, v___y_1390_);
lean_dec(v___y_1386_);
lean_dec_ref(v___y_1385_);
lean_dec(v___y_1384_);
lean_dec_ref(v___y_1383_);
return v_res_1392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg(lean_object* v_mvarId_1393_, lean_object* v_x_1394_, lean_object* v___y_1395_, lean_object* v___y_1396_, lean_object* v___y_1397_, lean_object* v___y_1398_, lean_object* v___y_1399_, lean_object* v___y_1400_, lean_object* v___y_1401_, lean_object* v___y_1402_){
_start:
{
lean_object* v___f_1404_; lean_object* v___x_1405_; 
lean_inc(v___y_1398_);
lean_inc_ref(v___y_1397_);
lean_inc(v___y_1396_);
lean_inc_ref(v___y_1395_);
v___f_1404_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_1404_, 0, v_x_1394_);
lean_closure_set(v___f_1404_, 1, v___y_1395_);
lean_closure_set(v___f_1404_, 2, v___y_1396_);
lean_closure_set(v___f_1404_, 3, v___y_1397_);
lean_closure_set(v___f_1404_, 4, v___y_1398_);
v___x_1405_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1393_, v___f_1404_, v___y_1399_, v___y_1400_, v___y_1401_, v___y_1402_);
if (lean_obj_tag(v___x_1405_) == 0)
{
return v___x_1405_;
}
else
{
lean_object* v_a_1406_; lean_object* v___x_1408_; uint8_t v_isShared_1409_; uint8_t v_isSharedCheck_1413_; 
v_a_1406_ = lean_ctor_get(v___x_1405_, 0);
v_isSharedCheck_1413_ = !lean_is_exclusive(v___x_1405_);
if (v_isSharedCheck_1413_ == 0)
{
v___x_1408_ = v___x_1405_;
v_isShared_1409_ = v_isSharedCheck_1413_;
goto v_resetjp_1407_;
}
else
{
lean_inc(v_a_1406_);
lean_dec(v___x_1405_);
v___x_1408_ = lean_box(0);
v_isShared_1409_ = v_isSharedCheck_1413_;
goto v_resetjp_1407_;
}
v_resetjp_1407_:
{
lean_object* v___x_1411_; 
if (v_isShared_1409_ == 0)
{
v___x_1411_ = v___x_1408_;
goto v_reusejp_1410_;
}
else
{
lean_object* v_reuseFailAlloc_1412_; 
v_reuseFailAlloc_1412_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1412_, 0, v_a_1406_);
v___x_1411_ = v_reuseFailAlloc_1412_;
goto v_reusejp_1410_;
}
v_reusejp_1410_:
{
return v___x_1411_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg___boxed(lean_object* v_mvarId_1414_, lean_object* v_x_1415_, lean_object* v___y_1416_, lean_object* v___y_1417_, lean_object* v___y_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_){
_start:
{
lean_object* v_res_1425_; 
v_res_1425_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg(v_mvarId_1414_, v_x_1415_, v___y_1416_, v___y_1417_, v___y_1418_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_);
lean_dec(v___y_1423_);
lean_dec_ref(v___y_1422_);
lean_dec(v___y_1421_);
lean_dec_ref(v___y_1420_);
lean_dec(v___y_1419_);
lean_dec_ref(v___y_1418_);
lean_dec(v___y_1417_);
lean_dec_ref(v___y_1416_);
return v_res_1425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6(lean_object* v_00_u03b1_1426_, lean_object* v_mvarId_1427_, lean_object* v_x_1428_, lean_object* v___y_1429_, lean_object* v___y_1430_, lean_object* v___y_1431_, lean_object* v___y_1432_, lean_object* v___y_1433_, lean_object* v___y_1434_, lean_object* v___y_1435_, lean_object* v___y_1436_){
_start:
{
lean_object* v___x_1438_; 
v___x_1438_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg(v_mvarId_1427_, v_x_1428_, v___y_1429_, v___y_1430_, v___y_1431_, v___y_1432_, v___y_1433_, v___y_1434_, v___y_1435_, v___y_1436_);
return v___x_1438_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___boxed(lean_object* v_00_u03b1_1439_, lean_object* v_mvarId_1440_, lean_object* v_x_1441_, lean_object* v___y_1442_, lean_object* v___y_1443_, lean_object* v___y_1444_, lean_object* v___y_1445_, lean_object* v___y_1446_, lean_object* v___y_1447_, lean_object* v___y_1448_, lean_object* v___y_1449_, lean_object* v___y_1450_){
_start:
{
lean_object* v_res_1451_; 
v_res_1451_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6(v_00_u03b1_1439_, v_mvarId_1440_, v_x_1441_, v___y_1442_, v___y_1443_, v___y_1444_, v___y_1445_, v___y_1446_, v___y_1447_, v___y_1448_, v___y_1449_);
lean_dec(v___y_1449_);
lean_dec_ref(v___y_1448_);
lean_dec(v___y_1447_);
lean_dec_ref(v___y_1446_);
lean_dec(v___y_1445_);
lean_dec_ref(v___y_1444_);
lean_dec(v___y_1443_);
lean_dec_ref(v___y_1442_);
return v_res_1451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(lean_object* v_msg_1452_, lean_object* v___y_1453_, lean_object* v___y_1454_, lean_object* v___y_1455_, lean_object* v___y_1456_){
_start:
{
lean_object* v_ref_1458_; lean_object* v___x_1459_; lean_object* v_a_1460_; lean_object* v___x_1462_; uint8_t v_isShared_1463_; uint8_t v_isSharedCheck_1468_; 
v_ref_1458_ = lean_ctor_get(v___y_1455_, 5);
v___x_1459_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr_spec__1_spec__1(v_msg_1452_, v___y_1453_, v___y_1454_, v___y_1455_, v___y_1456_);
v_a_1460_ = lean_ctor_get(v___x_1459_, 0);
v_isSharedCheck_1468_ = !lean_is_exclusive(v___x_1459_);
if (v_isSharedCheck_1468_ == 0)
{
v___x_1462_ = v___x_1459_;
v_isShared_1463_ = v_isSharedCheck_1468_;
goto v_resetjp_1461_;
}
else
{
lean_inc(v_a_1460_);
lean_dec(v___x_1459_);
v___x_1462_ = lean_box(0);
v_isShared_1463_ = v_isSharedCheck_1468_;
goto v_resetjp_1461_;
}
v_resetjp_1461_:
{
lean_object* v___x_1464_; lean_object* v___x_1466_; 
lean_inc(v_ref_1458_);
v___x_1464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1464_, 0, v_ref_1458_);
lean_ctor_set(v___x_1464_, 1, v_a_1460_);
if (v_isShared_1463_ == 0)
{
lean_ctor_set_tag(v___x_1462_, 1);
lean_ctor_set(v___x_1462_, 0, v___x_1464_);
v___x_1466_ = v___x_1462_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v___x_1464_);
v___x_1466_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
return v___x_1466_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg___boxed(lean_object* v_msg_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
lean_object* v_res_1475_; 
v_res_1475_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(v_msg_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
lean_dec(v___y_1471_);
lean_dec_ref(v___y_1470_);
return v_res_1475_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1477_; lean_object* v___x_1478_; 
v___x_1477_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__0));
v___x_1478_ = l_Lean_stringToMessageData(v___x_1477_);
return v___x_1478_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6(void){
_start:
{
lean_object* v___x_1484_; lean_object* v___x_1485_; 
v___x_1484_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__5));
v___x_1485_ = l_Lean_MessageData_ofFormat(v___x_1484_);
return v___x_1485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0(uint8_t v___x_1486_, lean_object* v_x_1487_, lean_object* v_type_1488_, lean_object* v___y_1489_, lean_object* v___y_1490_, lean_object* v___y_1491_, lean_object* v___y_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_){
_start:
{
lean_object* v___x_1498_; 
lean_inc_ref(v_type_1488_);
v___x_1498_ = lp_mathlib_Mathlib_Meta_FunProp_getFunProp_x3f(v_type_1488_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_);
if (lean_obj_tag(v___x_1498_) == 0)
{
lean_object* v_a_1499_; lean_object* v___x_1501_; uint8_t v_isShared_1502_; uint8_t v_isSharedCheck_1544_; 
v_a_1499_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1544_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1544_ == 0)
{
v___x_1501_ = v___x_1498_;
v_isShared_1502_ = v_isSharedCheck_1544_;
goto v_resetjp_1500_;
}
else
{
lean_inc(v_a_1499_);
lean_dec(v___x_1498_);
v___x_1501_ = lean_box(0);
v_isShared_1502_ = v_isSharedCheck_1544_;
goto v_resetjp_1500_;
}
v_resetjp_1500_:
{
lean_object* v___y_1504_; 
if (lean_obj_tag(v_a_1499_) == 0)
{
lean_del_object(v___x_1501_);
goto v___jp_1522_;
}
else
{
lean_dec_ref_known(v_a_1499_, 1);
if (v___x_1486_ == 0)
{
lean_del_object(v___x_1501_);
goto v___jp_1522_;
}
else
{
lean_object* v___x_1540_; lean_object* v___x_1542_; 
lean_dec_ref(v_type_1488_);
v___x_1540_ = lean_box(0);
if (v_isShared_1502_ == 0)
{
lean_ctor_set(v___x_1501_, 0, v___x_1540_);
v___x_1542_ = v___x_1501_;
goto v_reusejp_1541_;
}
else
{
lean_object* v_reuseFailAlloc_1543_; 
v_reuseFailAlloc_1543_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1543_, 0, v___x_1540_);
v___x_1542_ = v_reuseFailAlloc_1543_;
goto v_reusejp_1541_;
}
v_reusejp_1541_:
{
return v___x_1542_;
}
}
}
v___jp_1503_:
{
lean_object* v___x_1505_; 
v___x_1505_ = l_Lean_Meta_ppExpr(v_type_1488_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_);
if (lean_obj_tag(v___x_1505_) == 0)
{
lean_object* v_a_1506_; lean_object* v___x_1507_; lean_object* v___x_1508_; lean_object* v___x_1509_; lean_object* v___x_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; 
v_a_1506_ = lean_ctor_get(v___x_1505_, 0);
lean_inc(v_a_1506_);
lean_dec_ref_known(v___x_1505_, 1);
v___x_1507_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_1508_ = l_Lean_MessageData_ofFormat(v_a_1506_);
v___x_1509_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1509_, 0, v___x_1507_);
lean_ctor_set(v___x_1509_, 1, v___x_1508_);
v___x_1510_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__1);
v___x_1511_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1511_, 0, v___x_1509_);
lean_ctor_set(v___x_1511_, 1, v___x_1510_);
v___x_1512_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1512_, 0, v___x_1511_);
lean_ctor_set(v___x_1512_, 1, v___y_1504_);
v___x_1513_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(v___x_1512_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_);
return v___x_1513_;
}
else
{
lean_object* v_a_1514_; lean_object* v___x_1516_; uint8_t v_isShared_1517_; uint8_t v_isSharedCheck_1521_; 
lean_dec_ref(v___y_1504_);
v_a_1514_ = lean_ctor_get(v___x_1505_, 0);
v_isSharedCheck_1521_ = !lean_is_exclusive(v___x_1505_);
if (v_isSharedCheck_1521_ == 0)
{
v___x_1516_ = v___x_1505_;
v_isShared_1517_ = v_isSharedCheck_1521_;
goto v_resetjp_1515_;
}
else
{
lean_inc(v_a_1514_);
lean_dec(v___x_1505_);
v___x_1516_ = lean_box(0);
v_isShared_1517_ = v_isSharedCheck_1521_;
goto v_resetjp_1515_;
}
v_resetjp_1515_:
{
lean_object* v___x_1519_; 
if (v_isShared_1517_ == 0)
{
v___x_1519_ = v___x_1516_;
goto v_reusejp_1518_;
}
else
{
lean_object* v_reuseFailAlloc_1520_; 
v_reuseFailAlloc_1520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1520_, 0, v_a_1514_);
v___x_1519_ = v_reuseFailAlloc_1520_;
goto v_reusejp_1518_;
}
v_reusejp_1518_:
{
return v___x_1519_;
}
}
}
}
v___jp_1522_:
{
lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1523_ = l_Lean_Expr_getAppFn(v_type_1488_);
v___x_1524_ = l_Lean_Expr_constName_x3f(v___x_1523_);
lean_dec_ref(v___x_1523_);
if (lean_obj_tag(v___x_1524_) == 1)
{
lean_object* v_val_1525_; lean_object* v___x_1527_; uint8_t v_isShared_1528_; uint8_t v_isSharedCheck_1538_; 
v_val_1525_ = lean_ctor_get(v___x_1524_, 0);
v_isSharedCheck_1538_ = !lean_is_exclusive(v___x_1524_);
if (v_isSharedCheck_1538_ == 0)
{
v___x_1527_ = v___x_1524_;
v_isShared_1528_ = v_isSharedCheck_1538_;
goto v_resetjp_1526_;
}
else
{
lean_inc(v_val_1525_);
lean_dec(v___x_1524_);
v___x_1527_ = lean_box(0);
v_isShared_1528_ = v_isSharedCheck_1538_;
goto v_resetjp_1526_;
}
v_resetjp_1526_:
{
lean_object* v___x_1529_; lean_object* v___x_1530_; lean_object* v___x_1531_; lean_object* v___x_1532_; lean_object* v___x_1533_; lean_object* v___x_1535_; 
v___x_1529_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__2));
v___x_1530_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_1525_, v___x_1486_);
v___x_1531_ = lean_string_append(v___x_1529_, v___x_1530_);
lean_dec_ref(v___x_1530_);
v___x_1532_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__3));
v___x_1533_ = lean_string_append(v___x_1531_, v___x_1532_);
if (v_isShared_1528_ == 0)
{
lean_ctor_set_tag(v___x_1527_, 3);
lean_ctor_set(v___x_1527_, 0, v___x_1533_);
v___x_1535_ = v___x_1527_;
goto v_reusejp_1534_;
}
else
{
lean_object* v_reuseFailAlloc_1537_; 
v_reuseFailAlloc_1537_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1537_, 0, v___x_1533_);
v___x_1535_ = v_reuseFailAlloc_1537_;
goto v_reusejp_1534_;
}
v_reusejp_1534_:
{
lean_object* v___x_1536_; 
v___x_1536_ = l_Lean_MessageData_ofFormat(v___x_1535_);
v___y_1504_ = v___x_1536_;
goto v___jp_1503_;
}
}
}
else
{
lean_object* v___x_1539_; 
lean_dec(v___x_1524_);
v___x_1539_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6);
v___y_1504_ = v___x_1539_;
goto v___jp_1503_;
}
}
}
}
else
{
lean_object* v_a_1545_; lean_object* v___x_1547_; uint8_t v_isShared_1548_; uint8_t v_isSharedCheck_1552_; 
lean_dec_ref(v_type_1488_);
v_a_1545_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1552_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1552_ == 0)
{
v___x_1547_ = v___x_1498_;
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
else
{
lean_inc(v_a_1545_);
lean_dec(v___x_1498_);
v___x_1547_ = lean_box(0);
v_isShared_1548_ = v_isSharedCheck_1552_;
goto v_resetjp_1546_;
}
v_resetjp_1546_:
{
lean_object* v___x_1550_; 
if (v_isShared_1548_ == 0)
{
v___x_1550_ = v___x_1547_;
goto v_reusejp_1549_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v_a_1545_);
v___x_1550_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1549_;
}
v_reusejp_1549_:
{
return v___x_1550_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___boxed(lean_object* v___x_1553_, lean_object* v_x_1554_, lean_object* v_type_1555_, lean_object* v___y_1556_, lean_object* v___y_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_, lean_object* v___y_1561_, lean_object* v___y_1562_, lean_object* v___y_1563_, lean_object* v___y_1564_){
_start:
{
uint8_t v___x_19009__boxed_1565_; lean_object* v_res_1566_; 
v___x_19009__boxed_1565_ = lean_unbox(v___x_1553_);
v_res_1566_ = lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0(v___x_19009__boxed_1565_, v_x_1554_, v_type_1555_, v___y_1556_, v___y_1557_, v___y_1558_, v___y_1559_, v___y_1560_, v___y_1561_, v___y_1562_, v___y_1563_);
lean_dec(v___y_1563_);
lean_dec_ref(v___y_1562_);
lean_dec(v___y_1561_);
lean_dec_ref(v___y_1560_);
lean_dec(v___y_1559_);
lean_dec_ref(v___y_1558_);
lean_dec(v___y_1557_);
lean_dec_ref(v___y_1556_);
lean_dec_ref(v_x_1554_);
return v_res_1566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4(size_t v_sz_1567_, size_t v_i_1568_, lean_object* v_bs_1569_, lean_object* v___y_1570_, lean_object* v___y_1571_){
_start:
{
uint8_t v___x_1573_; 
v___x_1573_ = lean_usize_dec_lt(v_i_1568_, v_sz_1567_);
if (v___x_1573_ == 0)
{
lean_object* v___x_1574_; 
v___x_1574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1574_, 0, v_bs_1569_);
return v___x_1574_;
}
else
{
lean_object* v_v_1575_; lean_object* v___x_1576_; lean_object* v___x_1577_; 
v_v_1575_ = lean_array_uget_borrowed(v_bs_1569_, v_i_1568_);
v___x_1576_ = lean_box(0);
lean_inc(v_v_1575_);
v___x_1577_ = l_Lean_Elab_realizeGlobalConstNoOverloadWithInfo(v_v_1575_, v___x_1576_, v___y_1570_, v___y_1571_);
if (lean_obj_tag(v___x_1577_) == 0)
{
lean_object* v_a_1578_; lean_object* v___x_1579_; lean_object* v_bs_x27_1580_; size_t v___x_1581_; size_t v___x_1582_; lean_object* v___x_1583_; 
v_a_1578_ = lean_ctor_get(v___x_1577_, 0);
lean_inc(v_a_1578_);
lean_dec_ref_known(v___x_1577_, 1);
v___x_1579_ = lean_unsigned_to_nat(0u);
v_bs_x27_1580_ = lean_array_uset(v_bs_1569_, v_i_1568_, v___x_1579_);
v___x_1581_ = ((size_t)1ULL);
v___x_1582_ = lean_usize_add(v_i_1568_, v___x_1581_);
v___x_1583_ = lean_array_uset(v_bs_x27_1580_, v_i_1568_, v_a_1578_);
v_i_1568_ = v___x_1582_;
v_bs_1569_ = v___x_1583_;
goto _start;
}
else
{
lean_object* v_a_1585_; lean_object* v___x_1587_; uint8_t v_isShared_1588_; uint8_t v_isSharedCheck_1592_; 
lean_dec_ref(v_bs_1569_);
v_a_1585_ = lean_ctor_get(v___x_1577_, 0);
v_isSharedCheck_1592_ = !lean_is_exclusive(v___x_1577_);
if (v_isSharedCheck_1592_ == 0)
{
v___x_1587_ = v___x_1577_;
v_isShared_1588_ = v_isSharedCheck_1592_;
goto v_resetjp_1586_;
}
else
{
lean_inc(v_a_1585_);
lean_dec(v___x_1577_);
v___x_1587_ = lean_box(0);
v_isShared_1588_ = v_isSharedCheck_1592_;
goto v_resetjp_1586_;
}
v_resetjp_1586_:
{
lean_object* v___x_1590_; 
if (v_isShared_1588_ == 0)
{
v___x_1590_ = v___x_1587_;
goto v_reusejp_1589_;
}
else
{
lean_object* v_reuseFailAlloc_1591_; 
v_reuseFailAlloc_1591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1591_, 0, v_a_1585_);
v___x_1590_ = v_reuseFailAlloc_1591_;
goto v_reusejp_1589_;
}
v_reusejp_1589_:
{
return v___x_1590_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4___boxed(lean_object* v_sz_1593_, lean_object* v_i_1594_, lean_object* v_bs_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_, lean_object* v___y_1598_){
_start:
{
size_t v_sz_boxed_1599_; size_t v_i_boxed_1600_; lean_object* v_res_1601_; 
v_sz_boxed_1599_ = lean_unbox_usize(v_sz_1593_);
lean_dec(v_sz_1593_);
v_i_boxed_1600_ = lean_unbox_usize(v_i_1594_);
lean_dec(v_i_1594_);
v_res_1601_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4(v_sz_boxed_1599_, v_i_boxed_1600_, v_bs_1595_, v___y_1596_, v___y_1597_);
lean_dec(v___y_1597_);
lean_dec_ref(v___y_1596_);
return v_res_1601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3(lean_object* v_x_1603_, lean_object* v_x_1604_){
_start:
{
if (lean_obj_tag(v_x_1604_) == 0)
{
return v_x_1603_;
}
else
{
lean_object* v_head_1605_; lean_object* v_tail_1606_; lean_object* v___x_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; 
v_head_1605_ = lean_ctor_get(v_x_1604_, 0);
v_tail_1606_ = lean_ctor_get(v_x_1604_, 1);
v___x_1607_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___closed__0));
v___x_1608_ = lean_string_append(v_x_1603_, v___x_1607_);
v___x_1609_ = lean_string_append(v___x_1608_, v_head_1605_);
v_x_1603_ = v___x_1609_;
v_x_1604_ = v_tail_1606_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___boxed(lean_object* v_x_1611_, lean_object* v_x_1612_){
_start:
{
lean_object* v_res_1613_; 
v_res_1613_ = lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3(v_x_1611_, v_x_1612_);
lean_dec(v_x_1612_);
return v_res_1613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9___redArg(lean_object* v_x_1614_, lean_object* v_x_1615_, lean_object* v_x_1616_, lean_object* v_x_1617_){
_start:
{
lean_object* v_ks_1618_; lean_object* v_vs_1619_; lean_object* v___x_1621_; uint8_t v_isShared_1622_; uint8_t v_isSharedCheck_1643_; 
v_ks_1618_ = lean_ctor_get(v_x_1614_, 0);
v_vs_1619_ = lean_ctor_get(v_x_1614_, 1);
v_isSharedCheck_1643_ = !lean_is_exclusive(v_x_1614_);
if (v_isSharedCheck_1643_ == 0)
{
v___x_1621_ = v_x_1614_;
v_isShared_1622_ = v_isSharedCheck_1643_;
goto v_resetjp_1620_;
}
else
{
lean_inc(v_vs_1619_);
lean_inc(v_ks_1618_);
lean_dec(v_x_1614_);
v___x_1621_ = lean_box(0);
v_isShared_1622_ = v_isSharedCheck_1643_;
goto v_resetjp_1620_;
}
v_resetjp_1620_:
{
lean_object* v___x_1623_; uint8_t v___x_1624_; 
v___x_1623_ = lean_array_get_size(v_ks_1618_);
v___x_1624_ = lean_nat_dec_lt(v_x_1615_, v___x_1623_);
if (v___x_1624_ == 0)
{
lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1628_; 
lean_dec(v_x_1615_);
v___x_1625_ = lean_array_push(v_ks_1618_, v_x_1616_);
v___x_1626_ = lean_array_push(v_vs_1619_, v_x_1617_);
if (v_isShared_1622_ == 0)
{
lean_ctor_set(v___x_1621_, 1, v___x_1626_);
lean_ctor_set(v___x_1621_, 0, v___x_1625_);
v___x_1628_ = v___x_1621_;
goto v_reusejp_1627_;
}
else
{
lean_object* v_reuseFailAlloc_1629_; 
v_reuseFailAlloc_1629_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1629_, 0, v___x_1625_);
lean_ctor_set(v_reuseFailAlloc_1629_, 1, v___x_1626_);
v___x_1628_ = v_reuseFailAlloc_1629_;
goto v_reusejp_1627_;
}
v_reusejp_1627_:
{
return v___x_1628_;
}
}
else
{
lean_object* v_k_x27_1630_; uint8_t v___x_1631_; 
v_k_x27_1630_ = lean_array_fget_borrowed(v_ks_1618_, v_x_1615_);
v___x_1631_ = l_Lean_instBEqMVarId_beq(v_x_1616_, v_k_x27_1630_);
if (v___x_1631_ == 0)
{
lean_object* v___x_1633_; 
if (v_isShared_1622_ == 0)
{
v___x_1633_ = v___x_1621_;
goto v_reusejp_1632_;
}
else
{
lean_object* v_reuseFailAlloc_1637_; 
v_reuseFailAlloc_1637_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1637_, 0, v_ks_1618_);
lean_ctor_set(v_reuseFailAlloc_1637_, 1, v_vs_1619_);
v___x_1633_ = v_reuseFailAlloc_1637_;
goto v_reusejp_1632_;
}
v_reusejp_1632_:
{
lean_object* v___x_1634_; lean_object* v___x_1635_; 
v___x_1634_ = lean_unsigned_to_nat(1u);
v___x_1635_ = lean_nat_add(v_x_1615_, v___x_1634_);
lean_dec(v_x_1615_);
v_x_1614_ = v___x_1633_;
v_x_1615_ = v___x_1635_;
goto _start;
}
}
else
{
lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1641_; 
v___x_1638_ = lean_array_fset(v_ks_1618_, v_x_1615_, v_x_1616_);
v___x_1639_ = lean_array_fset(v_vs_1619_, v_x_1615_, v_x_1617_);
lean_dec(v_x_1615_);
if (v_isShared_1622_ == 0)
{
lean_ctor_set(v___x_1621_, 1, v___x_1639_);
lean_ctor_set(v___x_1621_, 0, v___x_1638_);
v___x_1641_ = v___x_1621_;
goto v_reusejp_1640_;
}
else
{
lean_object* v_reuseFailAlloc_1642_; 
v_reuseFailAlloc_1642_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1642_, 0, v___x_1638_);
lean_ctor_set(v_reuseFailAlloc_1642_, 1, v___x_1639_);
v___x_1641_ = v_reuseFailAlloc_1642_;
goto v_reusejp_1640_;
}
v_reusejp_1640_:
{
return v___x_1641_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8___redArg(lean_object* v_n_1644_, lean_object* v_k_1645_, lean_object* v_v_1646_){
_start:
{
lean_object* v___x_1647_; lean_object* v___x_1648_; 
v___x_1647_ = lean_unsigned_to_nat(0u);
v___x_1648_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9___redArg(v_n_1644_, v___x_1647_, v_k_1645_, v_v_1646_);
return v___x_1648_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_1649_; 
v___x_1649_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(lean_object* v_x_1650_, size_t v_x_1651_, size_t v_x_1652_, lean_object* v_x_1653_, lean_object* v_x_1654_){
_start:
{
if (lean_obj_tag(v_x_1650_) == 0)
{
lean_object* v_es_1655_; size_t v___x_1656_; size_t v___x_1657_; lean_object* v_j_1658_; lean_object* v___x_1659_; uint8_t v___x_1660_; 
v_es_1655_ = lean_ctor_get(v_x_1650_, 0);
v___x_1656_ = ((size_t)31ULL);
v___x_1657_ = lean_usize_land(v_x_1651_, v___x_1656_);
v_j_1658_ = lean_usize_to_nat(v___x_1657_);
v___x_1659_ = lean_array_get_size(v_es_1655_);
v___x_1660_ = lean_nat_dec_lt(v_j_1658_, v___x_1659_);
if (v___x_1660_ == 0)
{
lean_dec(v_j_1658_);
lean_dec(v_x_1654_);
lean_dec(v_x_1653_);
return v_x_1650_;
}
else
{
lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1699_; 
lean_inc_ref(v_es_1655_);
v_isSharedCheck_1699_ = !lean_is_exclusive(v_x_1650_);
if (v_isSharedCheck_1699_ == 0)
{
lean_object* v_unused_1700_; 
v_unused_1700_ = lean_ctor_get(v_x_1650_, 0);
lean_dec(v_unused_1700_);
v___x_1662_ = v_x_1650_;
v_isShared_1663_ = v_isSharedCheck_1699_;
goto v_resetjp_1661_;
}
else
{
lean_dec(v_x_1650_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1699_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v_v_1664_; lean_object* v___x_1665_; lean_object* v_xs_x27_1666_; lean_object* v___y_1668_; 
v_v_1664_ = lean_array_fget(v_es_1655_, v_j_1658_);
v___x_1665_ = lean_box(0);
v_xs_x27_1666_ = lean_array_fset(v_es_1655_, v_j_1658_, v___x_1665_);
switch(lean_obj_tag(v_v_1664_))
{
case 0:
{
lean_object* v_key_1673_; lean_object* v_val_1674_; lean_object* v___x_1676_; uint8_t v_isShared_1677_; uint8_t v_isSharedCheck_1684_; 
v_key_1673_ = lean_ctor_get(v_v_1664_, 0);
v_val_1674_ = lean_ctor_get(v_v_1664_, 1);
v_isSharedCheck_1684_ = !lean_is_exclusive(v_v_1664_);
if (v_isSharedCheck_1684_ == 0)
{
v___x_1676_ = v_v_1664_;
v_isShared_1677_ = v_isSharedCheck_1684_;
goto v_resetjp_1675_;
}
else
{
lean_inc(v_val_1674_);
lean_inc(v_key_1673_);
lean_dec(v_v_1664_);
v___x_1676_ = lean_box(0);
v_isShared_1677_ = v_isSharedCheck_1684_;
goto v_resetjp_1675_;
}
v_resetjp_1675_:
{
uint8_t v___x_1678_; 
v___x_1678_ = l_Lean_instBEqMVarId_beq(v_x_1653_, v_key_1673_);
if (v___x_1678_ == 0)
{
lean_object* v___x_1679_; lean_object* v___x_1680_; 
lean_del_object(v___x_1676_);
v___x_1679_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1673_, v_val_1674_, v_x_1653_, v_x_1654_);
v___x_1680_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1680_, 0, v___x_1679_);
v___y_1668_ = v___x_1680_;
goto v___jp_1667_;
}
else
{
lean_object* v___x_1682_; 
lean_dec(v_val_1674_);
lean_dec(v_key_1673_);
if (v_isShared_1677_ == 0)
{
lean_ctor_set(v___x_1676_, 1, v_x_1654_);
lean_ctor_set(v___x_1676_, 0, v_x_1653_);
v___x_1682_ = v___x_1676_;
goto v_reusejp_1681_;
}
else
{
lean_object* v_reuseFailAlloc_1683_; 
v_reuseFailAlloc_1683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1683_, 0, v_x_1653_);
lean_ctor_set(v_reuseFailAlloc_1683_, 1, v_x_1654_);
v___x_1682_ = v_reuseFailAlloc_1683_;
goto v_reusejp_1681_;
}
v_reusejp_1681_:
{
v___y_1668_ = v___x_1682_;
goto v___jp_1667_;
}
}
}
}
case 1:
{
lean_object* v_node_1685_; lean_object* v___x_1687_; uint8_t v_isShared_1688_; uint8_t v_isSharedCheck_1697_; 
v_node_1685_ = lean_ctor_get(v_v_1664_, 0);
v_isSharedCheck_1697_ = !lean_is_exclusive(v_v_1664_);
if (v_isSharedCheck_1697_ == 0)
{
v___x_1687_ = v_v_1664_;
v_isShared_1688_ = v_isSharedCheck_1697_;
goto v_resetjp_1686_;
}
else
{
lean_inc(v_node_1685_);
lean_dec(v_v_1664_);
v___x_1687_ = lean_box(0);
v_isShared_1688_ = v_isSharedCheck_1697_;
goto v_resetjp_1686_;
}
v_resetjp_1686_:
{
size_t v___x_1689_; size_t v___x_1690_; size_t v___x_1691_; size_t v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1695_; 
v___x_1689_ = ((size_t)5ULL);
v___x_1690_ = lean_usize_shift_right(v_x_1651_, v___x_1689_);
v___x_1691_ = ((size_t)1ULL);
v___x_1692_ = lean_usize_add(v_x_1652_, v___x_1691_);
v___x_1693_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(v_node_1685_, v___x_1690_, v___x_1692_, v_x_1653_, v_x_1654_);
if (v_isShared_1688_ == 0)
{
lean_ctor_set(v___x_1687_, 0, v___x_1693_);
v___x_1695_ = v___x_1687_;
goto v_reusejp_1694_;
}
else
{
lean_object* v_reuseFailAlloc_1696_; 
v_reuseFailAlloc_1696_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1696_, 0, v___x_1693_);
v___x_1695_ = v_reuseFailAlloc_1696_;
goto v_reusejp_1694_;
}
v_reusejp_1694_:
{
v___y_1668_ = v___x_1695_;
goto v___jp_1667_;
}
}
}
default: 
{
lean_object* v___x_1698_; 
v___x_1698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1698_, 0, v_x_1653_);
lean_ctor_set(v___x_1698_, 1, v_x_1654_);
v___y_1668_ = v___x_1698_;
goto v___jp_1667_;
}
}
v___jp_1667_:
{
lean_object* v___x_1669_; lean_object* v___x_1671_; 
v___x_1669_ = lean_array_fset(v_xs_x27_1666_, v_j_1658_, v___y_1668_);
lean_dec(v_j_1658_);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v___x_1669_);
v___x_1671_ = v___x_1662_;
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
else
{
lean_object* v_ks_1701_; lean_object* v_vs_1702_; lean_object* v___x_1704_; uint8_t v_isShared_1705_; uint8_t v_isSharedCheck_1722_; 
v_ks_1701_ = lean_ctor_get(v_x_1650_, 0);
v_vs_1702_ = lean_ctor_get(v_x_1650_, 1);
v_isSharedCheck_1722_ = !lean_is_exclusive(v_x_1650_);
if (v_isSharedCheck_1722_ == 0)
{
v___x_1704_ = v_x_1650_;
v_isShared_1705_ = v_isSharedCheck_1722_;
goto v_resetjp_1703_;
}
else
{
lean_inc(v_vs_1702_);
lean_inc(v_ks_1701_);
lean_dec(v_x_1650_);
v___x_1704_ = lean_box(0);
v_isShared_1705_ = v_isSharedCheck_1722_;
goto v_resetjp_1703_;
}
v_resetjp_1703_:
{
lean_object* v___x_1707_; 
if (v_isShared_1705_ == 0)
{
v___x_1707_ = v___x_1704_;
goto v_reusejp_1706_;
}
else
{
lean_object* v_reuseFailAlloc_1721_; 
v_reuseFailAlloc_1721_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1721_, 0, v_ks_1701_);
lean_ctor_set(v_reuseFailAlloc_1721_, 1, v_vs_1702_);
v___x_1707_ = v_reuseFailAlloc_1721_;
goto v_reusejp_1706_;
}
v_reusejp_1706_:
{
lean_object* v_newNode_1708_; uint8_t v___y_1710_; size_t v___x_1716_; uint8_t v___x_1717_; 
v_newNode_1708_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8___redArg(v___x_1707_, v_x_1653_, v_x_1654_);
v___x_1716_ = ((size_t)7ULL);
v___x_1717_ = lean_usize_dec_le(v___x_1716_, v_x_1652_);
if (v___x_1717_ == 0)
{
lean_object* v___x_1718_; lean_object* v___x_1719_; uint8_t v___x_1720_; 
v___x_1718_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1708_);
v___x_1719_ = lean_unsigned_to_nat(4u);
v___x_1720_ = lean_nat_dec_lt(v___x_1718_, v___x_1719_);
lean_dec(v___x_1718_);
v___y_1710_ = v___x_1720_;
goto v___jp_1709_;
}
else
{
v___y_1710_ = v___x_1717_;
goto v___jp_1709_;
}
v___jp_1709_:
{
if (v___y_1710_ == 0)
{
lean_object* v_ks_1711_; lean_object* v_vs_1712_; lean_object* v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; 
v_ks_1711_ = lean_ctor_get(v_newNode_1708_, 0);
lean_inc_ref(v_ks_1711_);
v_vs_1712_ = lean_ctor_get(v_newNode_1708_, 1);
lean_inc_ref(v_vs_1712_);
lean_dec_ref(v_newNode_1708_);
v___x_1713_ = lean_unsigned_to_nat(0u);
v___x_1714_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___closed__0);
v___x_1715_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg(v_x_1652_, v_ks_1711_, v_vs_1712_, v___x_1713_, v___x_1714_);
lean_dec_ref(v_vs_1712_);
lean_dec_ref(v_ks_1711_);
return v___x_1715_;
}
else
{
return v_newNode_1708_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg(size_t v_depth_1723_, lean_object* v_keys_1724_, lean_object* v_vals_1725_, lean_object* v_i_1726_, lean_object* v_entries_1727_){
_start:
{
lean_object* v___x_1728_; uint8_t v___x_1729_; 
v___x_1728_ = lean_array_get_size(v_keys_1724_);
v___x_1729_ = lean_nat_dec_lt(v_i_1726_, v___x_1728_);
if (v___x_1729_ == 0)
{
lean_dec(v_i_1726_);
return v_entries_1727_;
}
else
{
lean_object* v_k_1730_; lean_object* v_v_1731_; uint64_t v___x_1732_; size_t v_h_1733_; size_t v___x_1734_; lean_object* v___x_1735_; size_t v___x_1736_; size_t v___x_1737_; size_t v___x_1738_; size_t v_h_1739_; lean_object* v___x_1740_; lean_object* v___x_1741_; 
v_k_1730_ = lean_array_fget_borrowed(v_keys_1724_, v_i_1726_);
v_v_1731_ = lean_array_fget_borrowed(v_vals_1725_, v_i_1726_);
v___x_1732_ = l_Lean_instHashableMVarId_hash(v_k_1730_);
v_h_1733_ = lean_uint64_to_usize(v___x_1732_);
v___x_1734_ = ((size_t)5ULL);
v___x_1735_ = lean_unsigned_to_nat(1u);
v___x_1736_ = ((size_t)1ULL);
v___x_1737_ = lean_usize_sub(v_depth_1723_, v___x_1736_);
v___x_1738_ = lean_usize_mul(v___x_1734_, v___x_1737_);
v_h_1739_ = lean_usize_shift_right(v_h_1733_, v___x_1738_);
v___x_1740_ = lean_nat_add(v_i_1726_, v___x_1735_);
lean_dec(v_i_1726_);
lean_inc(v_v_1731_);
lean_inc(v_k_1730_);
v___x_1741_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(v_entries_1727_, v_h_1739_, v_depth_1723_, v_k_1730_, v_v_1731_);
v_i_1726_ = v___x_1740_;
v_entries_1727_ = v___x_1741_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg___boxed(lean_object* v_depth_1743_, lean_object* v_keys_1744_, lean_object* v_vals_1745_, lean_object* v_i_1746_, lean_object* v_entries_1747_){
_start:
{
size_t v_depth_boxed_1748_; lean_object* v_res_1749_; 
v_depth_boxed_1748_ = lean_unbox_usize(v_depth_1743_);
lean_dec(v_depth_1743_);
v_res_1749_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg(v_depth_boxed_1748_, v_keys_1744_, v_vals_1745_, v_i_1746_, v_entries_1747_);
lean_dec_ref(v_vals_1745_);
lean_dec_ref(v_keys_1744_);
return v_res_1749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg___boxed(lean_object* v_x_1750_, lean_object* v_x_1751_, lean_object* v_x_1752_, lean_object* v_x_1753_, lean_object* v_x_1754_){
_start:
{
size_t v_x_19305__boxed_1755_; size_t v_x_19306__boxed_1756_; lean_object* v_res_1757_; 
v_x_19305__boxed_1755_ = lean_unbox_usize(v_x_1751_);
lean_dec(v_x_1751_);
v_x_19306__boxed_1756_ = lean_unbox_usize(v_x_1752_);
lean_dec(v_x_1752_);
v_res_1757_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(v_x_1750_, v_x_19305__boxed_1755_, v_x_19306__boxed_1756_, v_x_1753_, v_x_1754_);
return v_res_1757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2___redArg(lean_object* v_x_1758_, lean_object* v_x_1759_, lean_object* v_x_1760_){
_start:
{
uint64_t v___x_1761_; size_t v___x_1762_; size_t v___x_1763_; lean_object* v___x_1764_; 
v___x_1761_ = l_Lean_instHashableMVarId_hash(v_x_1759_);
v___x_1762_ = lean_uint64_to_usize(v___x_1761_);
v___x_1763_ = ((size_t)1ULL);
v___x_1764_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(v_x_1758_, v___x_1762_, v___x_1763_, v_x_1759_, v_x_1760_);
return v___x_1764_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg(lean_object* v_mvarId_1765_, lean_object* v_val_1766_, lean_object* v___y_1767_){
_start:
{
lean_object* v___x_1769_; lean_object* v_mctx_1770_; lean_object* v_cache_1771_; lean_object* v_zetaDeltaFVarIds_1772_; lean_object* v_postponed_1773_; lean_object* v_diag_1774_; lean_object* v___x_1776_; uint8_t v_isShared_1777_; uint8_t v_isSharedCheck_1802_; 
v___x_1769_ = lean_st_ref_take(v___y_1767_);
v_mctx_1770_ = lean_ctor_get(v___x_1769_, 0);
v_cache_1771_ = lean_ctor_get(v___x_1769_, 1);
v_zetaDeltaFVarIds_1772_ = lean_ctor_get(v___x_1769_, 2);
v_postponed_1773_ = lean_ctor_get(v___x_1769_, 3);
v_diag_1774_ = lean_ctor_get(v___x_1769_, 4);
v_isSharedCheck_1802_ = !lean_is_exclusive(v___x_1769_);
if (v_isSharedCheck_1802_ == 0)
{
v___x_1776_ = v___x_1769_;
v_isShared_1777_ = v_isSharedCheck_1802_;
goto v_resetjp_1775_;
}
else
{
lean_inc(v_diag_1774_);
lean_inc(v_postponed_1773_);
lean_inc(v_zetaDeltaFVarIds_1772_);
lean_inc(v_cache_1771_);
lean_inc(v_mctx_1770_);
lean_dec(v___x_1769_);
v___x_1776_ = lean_box(0);
v_isShared_1777_ = v_isSharedCheck_1802_;
goto v_resetjp_1775_;
}
v_resetjp_1775_:
{
lean_object* v_depth_1778_; lean_object* v_levelAssignDepth_1779_; lean_object* v_lmvarCounter_1780_; lean_object* v_mvarCounter_1781_; lean_object* v_lDecls_1782_; lean_object* v_decls_1783_; lean_object* v_userNames_1784_; lean_object* v_lAssignment_1785_; lean_object* v_eAssignment_1786_; lean_object* v_dAssignment_1787_; lean_object* v___x_1789_; uint8_t v_isShared_1790_; uint8_t v_isSharedCheck_1801_; 
v_depth_1778_ = lean_ctor_get(v_mctx_1770_, 0);
v_levelAssignDepth_1779_ = lean_ctor_get(v_mctx_1770_, 1);
v_lmvarCounter_1780_ = lean_ctor_get(v_mctx_1770_, 2);
v_mvarCounter_1781_ = lean_ctor_get(v_mctx_1770_, 3);
v_lDecls_1782_ = lean_ctor_get(v_mctx_1770_, 4);
v_decls_1783_ = lean_ctor_get(v_mctx_1770_, 5);
v_userNames_1784_ = lean_ctor_get(v_mctx_1770_, 6);
v_lAssignment_1785_ = lean_ctor_get(v_mctx_1770_, 7);
v_eAssignment_1786_ = lean_ctor_get(v_mctx_1770_, 8);
v_dAssignment_1787_ = lean_ctor_get(v_mctx_1770_, 9);
v_isSharedCheck_1801_ = !lean_is_exclusive(v_mctx_1770_);
if (v_isSharedCheck_1801_ == 0)
{
v___x_1789_ = v_mctx_1770_;
v_isShared_1790_ = v_isSharedCheck_1801_;
goto v_resetjp_1788_;
}
else
{
lean_inc(v_dAssignment_1787_);
lean_inc(v_eAssignment_1786_);
lean_inc(v_lAssignment_1785_);
lean_inc(v_userNames_1784_);
lean_inc(v_decls_1783_);
lean_inc(v_lDecls_1782_);
lean_inc(v_mvarCounter_1781_);
lean_inc(v_lmvarCounter_1780_);
lean_inc(v_levelAssignDepth_1779_);
lean_inc(v_depth_1778_);
lean_dec(v_mctx_1770_);
v___x_1789_ = lean_box(0);
v_isShared_1790_ = v_isSharedCheck_1801_;
goto v_resetjp_1788_;
}
v_resetjp_1788_:
{
lean_object* v___x_1791_; lean_object* v___x_1793_; 
v___x_1791_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2___redArg(v_eAssignment_1786_, v_mvarId_1765_, v_val_1766_);
if (v_isShared_1790_ == 0)
{
lean_ctor_set(v___x_1789_, 8, v___x_1791_);
v___x_1793_ = v___x_1789_;
goto v_reusejp_1792_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v_depth_1778_);
lean_ctor_set(v_reuseFailAlloc_1800_, 1, v_levelAssignDepth_1779_);
lean_ctor_set(v_reuseFailAlloc_1800_, 2, v_lmvarCounter_1780_);
lean_ctor_set(v_reuseFailAlloc_1800_, 3, v_mvarCounter_1781_);
lean_ctor_set(v_reuseFailAlloc_1800_, 4, v_lDecls_1782_);
lean_ctor_set(v_reuseFailAlloc_1800_, 5, v_decls_1783_);
lean_ctor_set(v_reuseFailAlloc_1800_, 6, v_userNames_1784_);
lean_ctor_set(v_reuseFailAlloc_1800_, 7, v_lAssignment_1785_);
lean_ctor_set(v_reuseFailAlloc_1800_, 8, v___x_1791_);
lean_ctor_set(v_reuseFailAlloc_1800_, 9, v_dAssignment_1787_);
v___x_1793_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1792_;
}
v_reusejp_1792_:
{
lean_object* v___x_1795_; 
if (v_isShared_1777_ == 0)
{
lean_ctor_set(v___x_1776_, 0, v___x_1793_);
v___x_1795_ = v___x_1776_;
goto v_reusejp_1794_;
}
else
{
lean_object* v_reuseFailAlloc_1799_; 
v_reuseFailAlloc_1799_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1799_, 0, v___x_1793_);
lean_ctor_set(v_reuseFailAlloc_1799_, 1, v_cache_1771_);
lean_ctor_set(v_reuseFailAlloc_1799_, 2, v_zetaDeltaFVarIds_1772_);
lean_ctor_set(v_reuseFailAlloc_1799_, 3, v_postponed_1773_);
lean_ctor_set(v_reuseFailAlloc_1799_, 4, v_diag_1774_);
v___x_1795_ = v_reuseFailAlloc_1799_;
goto v_reusejp_1794_;
}
v_reusejp_1794_:
{
lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; 
v___x_1796_ = lean_st_ref_set(v___y_1767_, v___x_1795_);
v___x_1797_ = lean_box(0);
v___x_1798_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1798_, 0, v___x_1797_);
return v___x_1798_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg___boxed(lean_object* v_mvarId_1803_, lean_object* v_val_1804_, lean_object* v___y_1805_, lean_object* v___y_1806_){
_start:
{
lean_object* v_res_1807_; 
v_res_1807_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg(v_mvarId_1803_, v_val_1804_, v___y_1805_);
lean_dec(v___y_1805_);
return v_res_1807_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1809_; lean_object* v___x_1810_; lean_object* v___x_1811_; 
v___x_1809_ = lean_box(0);
v___x_1810_ = lean_unsigned_to_nat(16u);
v___x_1811_ = lean_mk_array(v___x_1810_, v___x_1809_);
return v___x_1811_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2(void){
_start:
{
lean_object* v___x_1812_; 
v___x_1812_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1812_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1813_; lean_object* v___x_1814_; 
v___x_1813_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__2);
v___x_1814_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1814_, 0, v___x_1813_);
return v___x_1814_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1(lean_object* v_a_1823_, lean_object* v___x_1824_, uint8_t v___x_1825_, lean_object* v_names_1826_, lean_object* v___x_1827_, lean_object* v___x_1828_, lean_object* v_d_1829_, lean_object* v___x_1830_, lean_object* v___x_1831_, lean_object* v___x_1832_, lean_object* v___x_1833_, lean_object* v___x_1834_, lean_object* v___f_1835_, lean_object* v___y_1836_, lean_object* v___y_1837_, lean_object* v___y_1838_, lean_object* v___y_1839_, lean_object* v___y_1840_, lean_object* v___y_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_){
_start:
{
lean_object* v___x_1845_; 
lean_inc(v_a_1823_);
v___x_1845_ = l_Lean_MVarId_getType(v_a_1823_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_);
if (lean_obj_tag(v___x_1845_) == 0)
{
lean_object* v_a_1846_; lean_object* v___y_1848_; lean_object* v___y_1849_; lean_object* v_a_1850_; lean_object* v___y_1913_; lean_object* v_a_1914_; lean_object* v___x_2004_; 
v_a_1846_ = lean_ctor_get(v___x_1845_, 0);
lean_inc_n(v_a_1846_, 2);
lean_dec_ref_known(v___x_1845_, 1);
v___x_2004_ = l_Lean_Meta_whnfR(v_a_1846_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_);
if (lean_obj_tag(v___x_2004_) == 0)
{
lean_object* v_a_2005_; lean_object* v_keyedConfig_2006_; uint8_t v_trackZetaDelta_2007_; lean_object* v_zetaDeltaSet_2008_; lean_object* v_lctx_2009_; lean_object* v_localInstances_2010_; lean_object* v_defEqCtx_x3f_2011_; lean_object* v_synthPendingDepth_2012_; lean_object* v_customCanUnfoldPredicate_x3f_2013_; uint8_t v_univApprox_2014_; uint8_t v_inTypeClassResolution_2015_; uint8_t v_cacheInferType_2016_; uint8_t v___x_2017_; uint8_t v___x_2018_; lean_object* v___x_2019_; lean_object* v___x_2020_; lean_object* v___x_2021_; 
v_a_2005_ = lean_ctor_get(v___x_2004_, 0);
lean_inc(v_a_2005_);
lean_dec_ref_known(v___x_2004_, 1);
v_keyedConfig_2006_ = lean_ctor_get(v___y_1840_, 0);
v_trackZetaDelta_2007_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7);
v_zetaDeltaSet_2008_ = lean_ctor_get(v___y_1840_, 1);
v_lctx_2009_ = lean_ctor_get(v___y_1840_, 2);
v_localInstances_2010_ = lean_ctor_get(v___y_1840_, 3);
v_defEqCtx_x3f_2011_ = lean_ctor_get(v___y_1840_, 4);
v_synthPendingDepth_2012_ = lean_ctor_get(v___y_1840_, 5);
v_customCanUnfoldPredicate_x3f_2013_ = lean_ctor_get(v___y_1840_, 6);
v_univApprox_2014_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2015_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 2);
v_cacheInferType_2016_ = lean_ctor_get_uint8(v___y_1840_, sizeof(void*)*7 + 3);
v___x_2017_ = 0;
v___x_2018_ = 2;
lean_inc_ref(v_keyedConfig_2006_);
v___x_2019_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2018_, v_keyedConfig_2006_);
lean_inc(v_customCanUnfoldPredicate_x3f_2013_);
lean_inc(v_synthPendingDepth_2012_);
lean_inc(v_defEqCtx_x3f_2011_);
lean_inc_ref(v_localInstances_2010_);
lean_inc_ref(v_lctx_2009_);
lean_inc(v_zetaDeltaSet_2008_);
v___x_2020_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2020_, 0, v___x_2019_);
lean_ctor_set(v___x_2020_, 1, v_zetaDeltaSet_2008_);
lean_ctor_set(v___x_2020_, 2, v_lctx_2009_);
lean_ctor_set(v___x_2020_, 3, v_localInstances_2010_);
lean_ctor_set(v___x_2020_, 4, v_defEqCtx_x3f_2011_);
lean_ctor_set(v___x_2020_, 5, v_synthPendingDepth_2012_);
lean_ctor_set(v___x_2020_, 6, v_customCanUnfoldPredicate_x3f_2013_);
lean_ctor_set_uint8(v___x_2020_, sizeof(void*)*7, v_trackZetaDelta_2007_);
lean_ctor_set_uint8(v___x_2020_, sizeof(void*)*7 + 1, v_univApprox_2014_);
lean_ctor_set_uint8(v___x_2020_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2015_);
lean_ctor_set_uint8(v___x_2020_, sizeof(void*)*7 + 3, v_cacheInferType_2016_);
v___x_2021_ = lp_mathlib_Lean_Meta_forallTelescopeReducing___at___00Mathlib_Meta_FunProp_funPropTac_spec__5___redArg(v_a_2005_, v___f_1835_, v___x_2017_, v___x_2017_, v___y_1836_, v___y_1837_, v___y_1838_, v___y_1839_, v___x_2020_, v___y_1841_, v___y_1842_, v___y_1843_);
lean_dec_ref_known(v___x_2020_, 7);
if (lean_obj_tag(v___x_2021_) == 0)
{
lean_dec_ref_known(v___x_2021_, 1);
goto v___jp_1930_;
}
else
{
if (lean_obj_tag(v___x_2021_) == 0)
{
lean_dec_ref_known(v___x_2021_, 1);
goto v___jp_1930_;
}
else
{
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
lean_dec(v_d_1829_);
lean_dec(v___x_1828_);
lean_dec(v___x_1827_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
return v___x_2021_;
}
}
}
else
{
lean_object* v_a_2022_; lean_object* v___x_2024_; uint8_t v_isShared_2025_; uint8_t v_isSharedCheck_2029_; 
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec_ref(v___f_1835_);
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
lean_dec(v_d_1829_);
lean_dec(v___x_1828_);
lean_dec(v___x_1827_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
v_a_2022_ = lean_ctor_get(v___x_2004_, 0);
v_isSharedCheck_2029_ = !lean_is_exclusive(v___x_2004_);
if (v_isSharedCheck_2029_ == 0)
{
v___x_2024_ = v___x_2004_;
v_isShared_2025_ = v_isSharedCheck_2029_;
goto v_resetjp_2023_;
}
else
{
lean_inc(v_a_2022_);
lean_dec(v___x_2004_);
v___x_2024_ = lean_box(0);
v_isShared_2025_ = v_isSharedCheck_2029_;
goto v_resetjp_2023_;
}
v_resetjp_2023_:
{
lean_object* v___x_2027_; 
if (v_isShared_2025_ == 0)
{
v___x_2027_ = v___x_2024_;
goto v_reusejp_2026_;
}
else
{
lean_object* v_reuseFailAlloc_2028_; 
v_reuseFailAlloc_2028_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2028_, 0, v_a_2022_);
v___x_2027_ = v_reuseFailAlloc_2028_;
goto v_reusejp_2026_;
}
v_reusejp_2026_:
{
return v___x_2027_;
}
}
}
v___jp_1847_:
{
lean_object* v___x_1851_; lean_object* v_env_1852_; lean_object* v___x_1853_; lean_object* v_ext_1854_; lean_object* v_toEnvExtension_1855_; lean_object* v_asyncMode_1856_; lean_object* v___x_1857_; lean_object* v_ext_1858_; lean_object* v_toEnvExtension_1859_; lean_object* v_asyncMode_1860_; lean_object* v___x_1861_; lean_object* v___x_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; lean_object* v___x_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; 
v___x_1851_ = lean_st_ref_get(v___y_1843_);
v_env_1852_ = lean_ctor_get(v___x_1851_, 0);
lean_inc_ref_n(v_env_1852_, 2);
lean_dec(v___x_1851_);
v___x_1853_ = lp_mathlib_Mathlib_Meta_FunProp_morTheoremsExt;
v_ext_1854_ = lean_ctor_get(v___x_1853_, 1);
v_toEnvExtension_1855_ = lean_ctor_get(v_ext_1854_, 0);
v_asyncMode_1856_ = lean_ctor_get(v_toEnvExtension_1855_, 2);
v___x_1857_ = lp_mathlib_Mathlib_Meta_FunProp_transitionTheoremsExt;
v_ext_1858_ = lean_ctor_get(v___x_1857_, 1);
v_toEnvExtension_1859_ = lean_ctor_get(v_ext_1858_, 0);
v_asyncMode_1860_ = lean_ctor_get(v_toEnvExtension_1859_, 2);
v___x_1861_ = lp_mathlib_Mathlib_Meta_FunProp_defaultNamesToUnfold;
v___x_1862_ = l_Array_append___redArg(v_a_1850_, v___x_1861_);
v___x_1863_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__0));
v___x_1864_ = l_Std_TreeSet_ofArray___redArg(v___x_1862_, v___x_1863_);
lean_dec_ref(v___x_1862_);
lean_inc_n(v___x_1824_, 3);
v___x_1865_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1865_, 0, v___y_1848_);
lean_ctor_set(v___x_1865_, 1, v___x_1864_);
lean_ctor_set(v___x_1865_, 2, v___y_1849_);
lean_ctor_set(v___x_1865_, 3, v___x_1824_);
v___x_1866_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__1);
v___x_1867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1867_, 0, v___x_1824_);
lean_ctor_set(v___x_1867_, 1, v___x_1866_);
v___x_1868_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__3);
lean_inc_ref(v___x_1867_);
v___x_1869_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_1869_, 0, v___x_1867_);
lean_ctor_set(v___x_1869_, 1, v___x_1868_);
lean_ctor_set_uint8(v___x_1869_, sizeof(void*)*2, v___x_1825_);
v___x_1870_ = lean_box(0);
v___x_1871_ = lp_mathlib_Mathlib_Meta_FunProp_instInhabitedGeneralTheorems_default;
v___x_1872_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1871_, v___x_1853_, v_env_1852_, v_asyncMode_1856_);
v___x_1873_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_1871_, v___x_1857_, v_env_1852_, v_asyncMode_1860_);
v___x_1874_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_1874_, 0, v___x_1869_);
lean_ctor_set(v___x_1874_, 1, v___x_1867_);
lean_ctor_set(v___x_1874_, 2, v___x_1824_);
lean_ctor_set(v___x_1874_, 3, v___x_1870_);
lean_ctor_set(v___x_1874_, 4, v___x_1872_);
lean_ctor_set(v___x_1874_, 5, v___x_1873_);
lean_inc(v_a_1846_);
v___x_1875_ = lp_mathlib_Mathlib_Meta_FunProp_funProp(v_a_1846_, v___x_1865_, v___x_1874_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_);
lean_dec_ref_known(v___x_1865_, 4);
if (lean_obj_tag(v___x_1875_) == 0)
{
lean_object* v_a_1876_; lean_object* v_fst_1877_; 
v_a_1876_ = lean_ctor_get(v___x_1875_, 0);
lean_inc(v_a_1876_);
lean_dec_ref_known(v___x_1875_, 1);
v_fst_1877_ = lean_ctor_get(v_a_1876_, 0);
if (lean_obj_tag(v_fst_1877_) == 1)
{
lean_object* v_val_1878_; lean_object* v___x_1879_; 
lean_inc_ref(v_fst_1877_);
lean_dec(v_a_1876_);
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec(v___x_1824_);
v_val_1878_ = lean_ctor_get(v_fst_1877_, 0);
lean_inc(v_val_1878_);
lean_dec_ref_known(v_fst_1877_, 1);
v___x_1879_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg(v_a_1823_, v_val_1878_, v___y_1841_);
return v___x_1879_;
}
else
{
lean_object* v_snd_1880_; lean_object* v___x_1881_; 
lean_dec(v_a_1823_);
v_snd_1880_ = lean_ctor_get(v_a_1876_, 1);
lean_inc(v_snd_1880_);
lean_dec(v_a_1876_);
v___x_1881_ = l_Lean_Meta_ppExpr(v_a_1846_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_object* v_a_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v_msgLog_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; lean_object* v___x_1891_; lean_object* v___x_1892_; lean_object* v___x_1893_; lean_object* v___x_1894_; lean_object* v___x_1895_; 
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
lean_inc(v_a_1882_);
lean_dec_ref_known(v___x_1881_, 1);
v___x_1883_ = l_Std_Format_defWidth;
lean_inc(v___x_1824_);
v___x_1884_ = l_Std_Format_pretty(v_a_1882_, v___x_1883_, v___x_1824_, v___x_1824_);
v_msgLog_1885_ = lean_ctor_get(v_snd_1880_, 3);
lean_inc(v_msgLog_1885_);
lean_dec(v_snd_1880_);
v___x_1886_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__4));
v___x_1887_ = lean_string_append(v___x_1886_, v___x_1884_);
lean_dec_ref(v___x_1884_);
v___x_1888_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__5));
v___x_1889_ = lean_string_append(v___x_1887_, v___x_1888_);
v___x_1890_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__6));
v___x_1891_ = lean_string_append(v___x_1889_, v___x_1890_);
v___x_1892_ = lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3(v___x_1891_, v_msgLog_1885_);
lean_dec(v_msgLog_1885_);
v___x_1893_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1893_, 0, v___x_1892_);
v___x_1894_ = l_Lean_MessageData_ofFormat(v___x_1893_);
v___x_1895_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(v___x_1894_, v___y_1840_, v___y_1841_, v___y_1842_, v___y_1843_);
lean_dec_ref(v___y_1840_);
return v___x_1895_;
}
else
{
lean_object* v_a_1896_; lean_object* v___x_1898_; uint8_t v_isShared_1899_; uint8_t v_isSharedCheck_1903_; 
lean_dec(v_snd_1880_);
lean_dec_ref(v___y_1840_);
lean_dec(v___x_1824_);
v_a_1896_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1903_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1903_ == 0)
{
v___x_1898_ = v___x_1881_;
v_isShared_1899_ = v_isSharedCheck_1903_;
goto v_resetjp_1897_;
}
else
{
lean_inc(v_a_1896_);
lean_dec(v___x_1881_);
v___x_1898_ = lean_box(0);
v_isShared_1899_ = v_isSharedCheck_1903_;
goto v_resetjp_1897_;
}
v_resetjp_1897_:
{
lean_object* v___x_1901_; 
if (v_isShared_1899_ == 0)
{
v___x_1901_ = v___x_1898_;
goto v_reusejp_1900_;
}
else
{
lean_object* v_reuseFailAlloc_1902_; 
v_reuseFailAlloc_1902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1902_, 0, v_a_1896_);
v___x_1901_ = v_reuseFailAlloc_1902_;
goto v_reusejp_1900_;
}
v_reusejp_1900_:
{
return v___x_1901_;
}
}
}
}
}
else
{
lean_object* v_a_1904_; lean_object* v___x_1906_; uint8_t v_isShared_1907_; uint8_t v_isSharedCheck_1911_; 
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
v_a_1904_ = lean_ctor_get(v___x_1875_, 0);
v_isSharedCheck_1911_ = !lean_is_exclusive(v___x_1875_);
if (v_isSharedCheck_1911_ == 0)
{
v___x_1906_ = v___x_1875_;
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
else
{
lean_inc(v_a_1904_);
lean_dec(v___x_1875_);
v___x_1906_ = lean_box(0);
v_isShared_1907_ = v_isSharedCheck_1911_;
goto v_resetjp_1905_;
}
v_resetjp_1905_:
{
lean_object* v___x_1909_; 
if (v_isShared_1907_ == 0)
{
v___x_1909_ = v___x_1906_;
goto v_reusejp_1908_;
}
else
{
lean_object* v_reuseFailAlloc_1910_; 
v_reuseFailAlloc_1910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1910_, 0, v_a_1904_);
v___x_1909_ = v_reuseFailAlloc_1910_;
goto v_reusejp_1908_;
}
v_reusejp_1908_:
{
return v___x_1909_;
}
}
}
}
v___jp_1912_:
{
if (lean_obj_tag(v_names_1826_) == 0)
{
lean_object* v___x_1915_; 
v___x_1915_ = lean_mk_empty_array_with_capacity(v___x_1824_);
v___y_1848_ = v___y_1913_;
v___y_1849_ = v_a_1914_;
v_a_1850_ = v___x_1915_;
goto v___jp_1847_;
}
else
{
lean_object* v_val_1916_; lean_object* v___x_1917_; size_t v_sz_1918_; size_t v___x_1919_; lean_object* v___x_1920_; 
v_val_1916_ = lean_ctor_get(v_names_1826_, 0);
v___x_1917_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_val_1916_);
v_sz_1918_ = lean_array_size(v___x_1917_);
v___x_1919_ = ((size_t)0ULL);
v___x_1920_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Meta_FunProp_funPropTac_spec__4(v_sz_1918_, v___x_1919_, v___x_1917_, v___y_1842_, v___y_1843_);
if (lean_obj_tag(v___x_1920_) == 0)
{
lean_object* v_a_1921_; 
v_a_1921_ = lean_ctor_get(v___x_1920_, 0);
lean_inc(v_a_1921_);
lean_dec_ref_known(v___x_1920_, 1);
v___y_1848_ = v___y_1913_;
v___y_1849_ = v_a_1914_;
v_a_1850_ = v_a_1921_;
goto v___jp_1847_;
}
else
{
lean_object* v_a_1922_; lean_object* v___x_1924_; uint8_t v_isShared_1925_; uint8_t v_isSharedCheck_1929_; 
lean_dec_ref(v_a_1914_);
lean_dec_ref(v___y_1913_);
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
v_a_1922_ = lean_ctor_get(v___x_1920_, 0);
v_isSharedCheck_1929_ = !lean_is_exclusive(v___x_1920_);
if (v_isSharedCheck_1929_ == 0)
{
v___x_1924_ = v___x_1920_;
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
else
{
lean_inc(v_a_1922_);
lean_dec(v___x_1920_);
v___x_1924_ = lean_box(0);
v_isShared_1925_ = v_isSharedCheck_1929_;
goto v_resetjp_1923_;
}
v_resetjp_1923_:
{
lean_object* v___x_1927_; 
if (v_isShared_1925_ == 0)
{
v___x_1927_ = v___x_1924_;
goto v_reusejp_1926_;
}
else
{
lean_object* v_reuseFailAlloc_1928_; 
v_reuseFailAlloc_1928_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1928_, 0, v_a_1922_);
v___x_1927_ = v_reuseFailAlloc_1928_;
goto v_reusejp_1926_;
}
v_reusejp_1926_:
{
return v___x_1927_;
}
}
}
}
}
v___jp_1930_:
{
lean_object* v___x_1931_; lean_object* v___x_1932_; lean_object* v___x_1933_; 
v___x_1931_ = lean_unsigned_to_nat(100000u);
v___x_1932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1932_, 0, v___x_1827_);
lean_ctor_set(v___x_1932_, 1, v___x_1931_);
v___x_1933_ = lp_mathlib_Mathlib_Meta_FunProp_elabFunPropConfig___redArg(v___x_1828_, v___x_1932_, v___x_1825_, v___y_1836_, v___y_1842_, v___y_1843_);
if (lean_obj_tag(v___x_1933_) == 0)
{
if (lean_obj_tag(v_d_1829_) == 0)
{
lean_object* v_a_1934_; lean_object* v___x_1935_; 
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
v_a_1934_ = lean_ctor_get(v___x_1933_, 0);
lean_inc(v_a_1934_);
lean_dec_ref_known(v___x_1933_, 1);
v___x_1935_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__7));
v___y_1913_ = v_a_1934_;
v_a_1914_ = v___x_1935_;
goto v___jp_1912_;
}
else
{
lean_object* v_a_1936_; lean_object* v_val_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; uint8_t v___x_1940_; 
v_a_1936_ = lean_ctor_get(v___x_1933_, 0);
lean_inc(v_a_1936_);
lean_dec_ref_known(v___x_1933_, 1);
v_val_1937_ = lean_ctor_get(v_d_1829_, 0);
lean_inc_n(v_val_1937_, 2);
lean_dec_ref_known(v_d_1829_, 1);
v___x_1938_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__8));
lean_inc_ref(v___x_1832_);
lean_inc_ref(v___x_1831_);
lean_inc_ref(v___x_1830_);
v___x_1939_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1938_);
v___x_1940_ = l_Lean_Syntax_isOfKind(v_val_1937_, v___x_1939_);
lean_dec(v___x_1939_);
if (v___x_1940_ == 0)
{
lean_object* v___x_1941_; 
lean_dec(v_val_1937_);
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
v___x_1941_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__7));
v___y_1913_ = v_a_1936_;
v_a_1914_ = v___x_1941_;
goto v___jp_1912_;
}
else
{
lean_object* v_ref_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; uint8_t v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; lean_object* v___x_1949_; lean_object* v___x_1950_; lean_object* v___x_1951_; lean_object* v___x_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; lean_object* v___x_1959_; lean_object* v___x_1960_; lean_object* v___x_1961_; lean_object* v___x_1962_; lean_object* v___x_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; lean_object* v___x_1986_; lean_object* v___x_1987_; lean_object* v___x_1988_; lean_object* v___x_1989_; lean_object* v___x_1990_; lean_object* v___x_1991_; lean_object* v___x_1992_; lean_object* v___x_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; 
v_ref_1942_ = lean_ctor_get(v___y_1842_, 5);
v___x_1943_ = l_Lean_Syntax_getArg(v_val_1937_, v___x_1833_);
lean_dec(v_val_1937_);
v___x_1944_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__10));
lean_inc_ref_n(v___x_1832_, 6);
lean_inc_ref_n(v___x_1831_, 5);
lean_inc_ref_n(v___x_1830_, 5);
v___x_1945_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1944_);
v___x_1946_ = 0;
v___x_1947_ = l_Lean_SourceInfo_fromRef(v_ref_1942_, v___x_1946_);
v___x_1948_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__3));
v___x_1949_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1948_);
lean_inc_n(v___x_1947_, 27);
v___x_1950_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1950_, 0, v___x_1947_);
lean_ctor_set(v___x_1950_, 1, v___x_1948_);
v___x_1951_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__6));
v___x_1952_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__8));
v___x_1953_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__9));
v___x_1954_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1954_, 0, v___x_1947_);
lean_ctor_set(v___x_1954_, 1, v___x_1953_);
v___x_1955_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__12));
v___x_1956_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1955_);
v___x_1957_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__14));
v___x_1958_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1957_);
v___x_1959_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__16));
v___x_1960_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1960_, 0, v___x_1947_);
lean_ctor_set(v___x_1960_, 1, v___x_1959_);
v___x_1961_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__17));
v___x_1962_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1961_);
v___x_1963_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1963_, 0, v___x_1947_);
lean_ctor_set(v___x_1963_, 1, v___x_1961_);
v___x_1964_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1962_, v___x_1963_);
v___x_1965_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1951_, v___x_1964_);
lean_inc_n(v___x_1956_, 3);
v___x_1966_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1956_, v___x_1965_);
lean_inc_n(v___x_1945_, 3);
v___x_1967_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1945_, v___x_1966_);
v___x_1968_ = l_Lean_Syntax_node2(v___x_1947_, v___x_1958_, v___x_1960_, v___x_1967_);
v___x_1969_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1951_, v___x_1968_);
v___x_1970_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1956_, v___x_1969_);
v___x_1971_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1945_, v___x_1970_);
lean_inc_ref_n(v___x_1954_, 2);
v___x_1972_ = l_Lean_Syntax_node2(v___x_1947_, v___x_1952_, v___x_1954_, v___x_1971_);
v___x_1973_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__19));
v___x_1974_ = l_Lean_Name_mkStr3(v___x_1834_, v___x_1832_, v___x_1973_);
v___x_1975_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__21));
v___x_1976_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1976_, 0, v___x_1947_);
lean_ctor_set(v___x_1976_, 1, v___x_1975_);
v___x_1977_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1974_, v___x_1976_);
v___x_1978_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1951_, v___x_1977_);
v___x_1979_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1956_, v___x_1978_);
v___x_1980_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1945_, v___x_1979_);
v___x_1981_ = l_Lean_Syntax_node2(v___x_1947_, v___x_1952_, v___x_1954_, v___x_1980_);
v___x_1982_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__9));
v___x_1983_ = l_Lean_Name_mkStr4(v___x_1830_, v___x_1831_, v___x_1832_, v___x_1982_);
v___x_1984_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__10));
v___x_1985_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1985_, 0, v___x_1947_);
lean_ctor_set(v___x_1985_, 1, v___x_1984_);
v___x_1986_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___closed__11));
v___x_1987_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1987_, 0, v___x_1947_);
lean_ctor_set(v___x_1987_, 1, v___x_1986_);
v___x_1988_ = l_Lean_Syntax_node3(v___x_1947_, v___x_1983_, v___x_1985_, v___x_1943_, v___x_1987_);
v___x_1989_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1951_, v___x_1988_);
v___x_1990_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1956_, v___x_1989_);
v___x_1991_ = l_Lean_Syntax_node1(v___x_1947_, v___x_1945_, v___x_1990_);
v___x_1992_ = l_Lean_Syntax_node2(v___x_1947_, v___x_1952_, v___x_1954_, v___x_1991_);
v___x_1993_ = l_Lean_Syntax_node3(v___x_1947_, v___x_1951_, v___x_1972_, v___x_1981_, v___x_1992_);
v___x_1994_ = l_Lean_Syntax_node2(v___x_1947_, v___x_1949_, v___x_1950_, v___x_1993_);
v___x_1995_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_tacticToDischarge___boxed), 7, 1);
lean_closure_set(v___x_1995_, 0, v___x_1994_);
v___y_1913_ = v_a_1936_;
v_a_1914_ = v___x_1995_;
goto v___jp_1912_;
}
}
}
else
{
lean_object* v_a_1996_; lean_object* v___x_1998_; uint8_t v_isShared_1999_; uint8_t v_isSharedCheck_2003_; 
lean_dec(v_a_1846_);
lean_dec_ref(v___y_1840_);
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
lean_dec(v_d_1829_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
v_a_1996_ = lean_ctor_get(v___x_1933_, 0);
v_isSharedCheck_2003_ = !lean_is_exclusive(v___x_1933_);
if (v_isSharedCheck_2003_ == 0)
{
v___x_1998_ = v___x_1933_;
v_isShared_1999_ = v_isSharedCheck_2003_;
goto v_resetjp_1997_;
}
else
{
lean_inc(v_a_1996_);
lean_dec(v___x_1933_);
v___x_1998_ = lean_box(0);
v_isShared_1999_ = v_isSharedCheck_2003_;
goto v_resetjp_1997_;
}
v_resetjp_1997_:
{
lean_object* v___x_2001_; 
if (v_isShared_1999_ == 0)
{
v___x_2001_ = v___x_1998_;
goto v_reusejp_2000_;
}
else
{
lean_object* v_reuseFailAlloc_2002_; 
v_reuseFailAlloc_2002_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2002_, 0, v_a_1996_);
v___x_2001_ = v_reuseFailAlloc_2002_;
goto v_reusejp_2000_;
}
v_reusejp_2000_:
{
return v___x_2001_;
}
}
}
}
}
else
{
lean_object* v_a_2030_; lean_object* v___x_2032_; uint8_t v_isShared_2033_; uint8_t v_isSharedCheck_2037_; 
lean_dec_ref(v___y_1840_);
lean_dec_ref(v___f_1835_);
lean_dec_ref(v___x_1834_);
lean_dec_ref(v___x_1832_);
lean_dec_ref(v___x_1831_);
lean_dec_ref(v___x_1830_);
lean_dec(v_d_1829_);
lean_dec(v___x_1828_);
lean_dec(v___x_1827_);
lean_dec(v___x_1824_);
lean_dec(v_a_1823_);
v_a_2030_ = lean_ctor_get(v___x_1845_, 0);
v_isSharedCheck_2037_ = !lean_is_exclusive(v___x_1845_);
if (v_isSharedCheck_2037_ == 0)
{
v___x_2032_ = v___x_1845_;
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
else
{
lean_inc(v_a_2030_);
lean_dec(v___x_1845_);
v___x_2032_ = lean_box(0);
v_isShared_2033_ = v_isSharedCheck_2037_;
goto v_resetjp_2031_;
}
v_resetjp_2031_:
{
lean_object* v___x_2035_; 
if (v_isShared_2033_ == 0)
{
v___x_2035_ = v___x_2032_;
goto v_reusejp_2034_;
}
else
{
lean_object* v_reuseFailAlloc_2036_; 
v_reuseFailAlloc_2036_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2036_, 0, v_a_2030_);
v___x_2035_ = v_reuseFailAlloc_2036_;
goto v_reusejp_2034_;
}
v_reusejp_2034_:
{
return v___x_2035_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___boxed(lean_object** _args){
lean_object* v_a_2038_ = _args[0];
lean_object* v___x_2039_ = _args[1];
lean_object* v___x_2040_ = _args[2];
lean_object* v_names_2041_ = _args[3];
lean_object* v___x_2042_ = _args[4];
lean_object* v___x_2043_ = _args[5];
lean_object* v_d_2044_ = _args[6];
lean_object* v___x_2045_ = _args[7];
lean_object* v___x_2046_ = _args[8];
lean_object* v___x_2047_ = _args[9];
lean_object* v___x_2048_ = _args[10];
lean_object* v___x_2049_ = _args[11];
lean_object* v___f_2050_ = _args[12];
lean_object* v___y_2051_ = _args[13];
lean_object* v___y_2052_ = _args[14];
lean_object* v___y_2053_ = _args[15];
lean_object* v___y_2054_ = _args[16];
lean_object* v___y_2055_ = _args[17];
lean_object* v___y_2056_ = _args[18];
lean_object* v___y_2057_ = _args[19];
lean_object* v___y_2058_ = _args[20];
lean_object* v___y_2059_ = _args[21];
_start:
{
uint8_t v___x_19566__boxed_2060_; lean_object* v_res_2061_; 
v___x_19566__boxed_2060_ = lean_unbox(v___x_2040_);
v_res_2061_ = lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1(v_a_2038_, v___x_2039_, v___x_19566__boxed_2060_, v_names_2041_, v___x_2042_, v___x_2043_, v_d_2044_, v___x_2045_, v___x_2046_, v___x_2047_, v___x_2048_, v___x_2049_, v___f_2050_, v___y_2051_, v___y_2052_, v___y_2053_, v___y_2054_, v___y_2055_, v___y_2056_, v___y_2057_, v___y_2058_);
lean_dec(v___y_2058_);
lean_dec_ref(v___y_2057_);
lean_dec(v___y_2056_);
lean_dec(v___y_2054_);
lean_dec_ref(v___y_2053_);
lean_dec(v___y_2052_);
lean_dec_ref(v___y_2051_);
lean_dec(v___x_2048_);
lean_dec(v_names_2041_);
return v_res_2061_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac(lean_object* v_x_2068_, lean_object* v_a_2069_, lean_object* v_a_2070_, lean_object* v_a_2071_, lean_object* v_a_2072_, lean_object* v_a_2073_, lean_object* v_a_2074_, lean_object* v_a_2075_, lean_object* v_a_2076_){
_start:
{
lean_object* v___x_2078_; lean_object* v___x_2079_; uint8_t v___x_2080_; 
v___x_2078_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig_evalExpr___closed__1));
v___x_2079_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__1));
lean_inc(v_x_2068_);
v___x_2080_ = l_Lean_Syntax_isOfKind(v_x_2068_, v___x_2079_);
if (v___x_2080_ == 0)
{
lean_object* v___x_2081_; 
lean_dec(v_x_2068_);
v___x_2081_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v___x_2081_;
}
else
{
lean_object* v___x_2082_; lean_object* v___x_2083_; lean_object* v___x_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; uint8_t v___x_2088_; 
v___x_2082_ = lean_unsigned_to_nat(1u);
v___x_2083_ = l_Lean_Syntax_getArg(v_x_2068_, v___x_2082_);
v___x_2084_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__0));
v___x_2085_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__1));
v___x_2086_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_assumptionDischarge___closed__2));
v___x_2087_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___closed__1));
lean_inc(v___x_2083_);
v___x_2088_ = l_Lean_Syntax_isOfKind(v___x_2083_, v___x_2087_);
if (v___x_2088_ == 0)
{
lean_object* v___x_2089_; 
lean_dec(v___x_2083_);
lean_dec(v_x_2068_);
v___x_2089_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v___x_2089_;
}
else
{
lean_object* v___x_2090_; lean_object* v___f_2091_; lean_object* v___x_2092_; lean_object* v___y_2094_; lean_object* v___y_2095_; lean_object* v___y_2096_; lean_object* v___y_2097_; lean_object* v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v_names_2104_; lean_object* v_d_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2122_; lean_object* v___y_2123_; lean_object* v___y_2124_; lean_object* v___y_2125_; lean_object* v___y_2126_; lean_object* v___y_2127_; lean_object* v___x_2137_; lean_object* v___x_2138_; uint8_t v___x_2139_; 
v___x_2090_ = lean_box(v___x_2088_);
v___f_2091_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___boxed), 12, 1);
lean_closure_set(v___f_2091_, 0, v___x_2090_);
v___x_2092_ = lean_unsigned_to_nat(0u);
v___x_2137_ = lean_unsigned_to_nat(2u);
v___x_2138_ = l_Lean_Syntax_getArg(v_x_2068_, v___x_2137_);
v___x_2139_ = l_Lean_Syntax_isNone(v___x_2138_);
if (v___x_2139_ == 0)
{
uint8_t v___x_2140_; 
lean_inc(v___x_2138_);
v___x_2140_ = l_Lean_Syntax_matchesNull(v___x_2138_, v___x_2082_);
if (v___x_2140_ == 0)
{
lean_object* v___x_2141_; 
lean_dec(v___x_2138_);
lean_dec_ref(v___f_2091_);
lean_dec(v___x_2083_);
lean_dec(v_x_2068_);
v___x_2141_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v___x_2141_;
}
else
{
lean_object* v_d_2142_; lean_object* v___x_2143_; 
v_d_2142_ = l_Lean_Syntax_getArg(v___x_2138_, v___x_2092_);
lean_dec(v___x_2138_);
v___x_2143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2143_, 0, v_d_2142_);
v_d_2119_ = v___x_2143_;
v___y_2120_ = v_a_2069_;
v___y_2121_ = v_a_2070_;
v___y_2122_ = v_a_2071_;
v___y_2123_ = v_a_2072_;
v___y_2124_ = v_a_2073_;
v___y_2125_ = v_a_2074_;
v___y_2126_ = v_a_2075_;
v___y_2127_ = v_a_2076_;
goto v___jp_2118_;
}
}
else
{
lean_object* v___x_2144_; 
lean_dec(v___x_2138_);
v___x_2144_ = lean_box(0);
v_d_2119_ = v___x_2144_;
v___y_2120_ = v_a_2069_;
v___y_2121_ = v_a_2070_;
v___y_2122_ = v_a_2071_;
v___y_2123_ = v_a_2072_;
v___y_2124_ = v_a_2073_;
v___y_2125_ = v_a_2074_;
v___y_2126_ = v_a_2075_;
v___y_2127_ = v_a_2076_;
goto v___jp_2118_;
}
v___jp_2093_:
{
lean_object* v___x_2105_; 
v___x_2105_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2100_, v___y_2097_, v___y_2099_, v___y_2096_, v___y_2094_);
if (lean_obj_tag(v___x_2105_) == 0)
{
lean_object* v_a_2106_; lean_object* v___x_2107_; lean_object* v___f_2108_; lean_object* v___x_2109_; 
v_a_2106_ = lean_ctor_get(v___x_2105_, 0);
lean_inc_n(v_a_2106_, 2);
lean_dec_ref_known(v___x_2105_, 1);
v___x_2107_ = lean_box(v___x_2088_);
v___f_2108_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__1___boxed), 22, 13);
lean_closure_set(v___f_2108_, 0, v_a_2106_);
lean_closure_set(v___f_2108_, 1, v___x_2092_);
lean_closure_set(v___f_2108_, 2, v___x_2107_);
lean_closure_set(v___f_2108_, 3, v_names_2104_);
lean_closure_set(v___f_2108_, 4, v___x_2082_);
lean_closure_set(v___f_2108_, 5, v___x_2083_);
lean_closure_set(v___f_2108_, 6, v___y_2103_);
lean_closure_set(v___f_2108_, 7, v___x_2084_);
lean_closure_set(v___f_2108_, 8, v___x_2085_);
lean_closure_set(v___f_2108_, 9, v___x_2086_);
lean_closure_set(v___f_2108_, 10, v___y_2098_);
lean_closure_set(v___f_2108_, 11, v___x_2078_);
lean_closure_set(v___f_2108_, 12, v___f_2091_);
v___x_2109_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Meta_FunProp_funPropTac_spec__6___redArg(v_a_2106_, v___f_2108_, v___y_2102_, v___y_2100_, v___y_2101_, v___y_2095_, v___y_2097_, v___y_2099_, v___y_2096_, v___y_2094_);
return v___x_2109_;
}
else
{
lean_object* v_a_2110_; lean_object* v___x_2112_; uint8_t v_isShared_2113_; uint8_t v_isSharedCheck_2117_; 
lean_dec(v_names_2104_);
lean_dec(v___y_2103_);
lean_dec(v___y_2098_);
lean_dec_ref(v___f_2091_);
lean_dec(v___x_2083_);
v_a_2110_ = lean_ctor_get(v___x_2105_, 0);
v_isSharedCheck_2117_ = !lean_is_exclusive(v___x_2105_);
if (v_isSharedCheck_2117_ == 0)
{
v___x_2112_ = v___x_2105_;
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
else
{
lean_inc(v_a_2110_);
lean_dec(v___x_2105_);
v___x_2112_ = lean_box(0);
v_isShared_2113_ = v_isSharedCheck_2117_;
goto v_resetjp_2111_;
}
v_resetjp_2111_:
{
lean_object* v___x_2115_; 
if (v_isShared_2113_ == 0)
{
v___x_2115_ = v___x_2112_;
goto v_reusejp_2114_;
}
else
{
lean_object* v_reuseFailAlloc_2116_; 
v_reuseFailAlloc_2116_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2116_, 0, v_a_2110_);
v___x_2115_ = v_reuseFailAlloc_2116_;
goto v_reusejp_2114_;
}
v_reusejp_2114_:
{
return v___x_2115_;
}
}
}
}
v___jp_2118_:
{
lean_object* v___x_2128_; lean_object* v___x_2129_; uint8_t v___x_2130_; 
v___x_2128_ = lean_unsigned_to_nat(3u);
v___x_2129_ = l_Lean_Syntax_getArg(v_x_2068_, v___x_2128_);
lean_dec(v_x_2068_);
v___x_2130_ = l_Lean_Syntax_isNone(v___x_2129_);
if (v___x_2130_ == 0)
{
uint8_t v___x_2131_; 
lean_inc(v___x_2129_);
v___x_2131_ = l_Lean_Syntax_matchesNull(v___x_2129_, v___x_2128_);
if (v___x_2131_ == 0)
{
lean_object* v___x_2132_; 
lean_dec(v___x_2129_);
lean_dec(v_d_2119_);
lean_dec_ref(v___f_2091_);
lean_dec(v___x_2083_);
v___x_2132_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg();
return v___x_2132_;
}
else
{
lean_object* v___x_2133_; lean_object* v_names_2134_; lean_object* v___x_2135_; 
v___x_2133_ = l_Lean_Syntax_getArg(v___x_2129_, v___x_2082_);
lean_dec(v___x_2129_);
v_names_2134_ = l_Lean_Syntax_getArgs(v___x_2133_);
lean_dec(v___x_2133_);
v___x_2135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2135_, 0, v_names_2134_);
v___y_2094_ = v___y_2127_;
v___y_2095_ = v___y_2123_;
v___y_2096_ = v___y_2126_;
v___y_2097_ = v___y_2124_;
v___y_2098_ = v___x_2128_;
v___y_2099_ = v___y_2125_;
v___y_2100_ = v___y_2121_;
v___y_2101_ = v___y_2122_;
v___y_2102_ = v___y_2120_;
v___y_2103_ = v_d_2119_;
v_names_2104_ = v___x_2135_;
goto v___jp_2093_;
}
}
else
{
lean_object* v___x_2136_; 
lean_dec(v___x_2129_);
v___x_2136_ = lean_box(0);
v___y_2094_ = v___y_2127_;
v___y_2095_ = v___y_2123_;
v___y_2096_ = v___y_2126_;
v___y_2097_ = v___y_2124_;
v___y_2098_ = v___x_2128_;
v___y_2099_ = v___y_2125_;
v___y_2100_ = v___y_2121_;
v___y_2101_ = v___y_2122_;
v___y_2102_ = v___y_2120_;
v___y_2103_ = v_d_2119_;
v_names_2104_ = v___x_2136_;
goto v___jp_2093_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp_funPropTac___boxed(lean_object* v_x_2145_, lean_object* v_a_2146_, lean_object* v_a_2147_, lean_object* v_a_2148_, lean_object* v_a_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_, lean_object* v_a_2152_, lean_object* v_a_2153_, lean_object* v_a_2154_){
_start:
{
lean_object* v_res_2155_; 
v_res_2155_ = lp_mathlib_Mathlib_Meta_FunProp_funPropTac(v_x_2145_, v_a_2146_, v_a_2147_, v_a_2148_, v_a_2149_, v_a_2150_, v_a_2151_, v_a_2152_, v_a_2153_);
lean_dec(v_a_2153_);
lean_dec_ref(v_a_2152_);
lean_dec(v_a_2151_);
lean_dec_ref(v_a_2150_);
lean_dec(v_a_2149_);
lean_dec_ref(v_a_2148_);
lean_dec(v_a_2147_);
lean_dec_ref(v_a_2146_);
return v_res_2155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1(lean_object* v_00_u03b1_2156_, lean_object* v_msg_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_){
_start:
{
lean_object* v___x_2167_; 
v___x_2167_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___redArg(v_msg_2157_, v___y_2162_, v___y_2163_, v___y_2164_, v___y_2165_);
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1___boxed(lean_object* v_00_u03b1_2168_, lean_object* v_msg_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_, lean_object* v___y_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_, lean_object* v___y_2178_){
_start:
{
lean_object* v_res_2179_; 
v_res_2179_ = lp_mathlib_Lean_throwError___at___00Mathlib_Meta_FunProp_funPropTac_spec__1(v_00_u03b1_2168_, v_msg_2169_, v___y_2170_, v___y_2171_, v___y_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_);
lean_dec(v___y_2177_);
lean_dec_ref(v___y_2176_);
lean_dec(v___y_2175_);
lean_dec_ref(v___y_2174_);
lean_dec(v___y_2173_);
lean_dec_ref(v___y_2172_);
lean_dec(v___y_2171_);
lean_dec_ref(v___y_2170_);
return v_res_2179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2(lean_object* v_mvarId_2180_, lean_object* v_val_2181_, lean_object* v___y_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_, lean_object* v___y_2189_){
_start:
{
lean_object* v___x_2191_; 
v___x_2191_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___redArg(v_mvarId_2180_, v_val_2181_, v___y_2187_);
return v___x_2191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2___boxed(lean_object* v_mvarId_2192_, lean_object* v_val_2193_, lean_object* v___y_2194_, lean_object* v___y_2195_, lean_object* v___y_2196_, lean_object* v___y_2197_, lean_object* v___y_2198_, lean_object* v___y_2199_, lean_object* v___y_2200_, lean_object* v___y_2201_, lean_object* v___y_2202_){
_start:
{
lean_object* v_res_2203_; 
v_res_2203_ = lp_mathlib_Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2(v_mvarId_2192_, v_val_2193_, v___y_2194_, v___y_2195_, v___y_2196_, v___y_2197_, v___y_2198_, v___y_2199_, v___y_2200_, v___y_2201_);
lean_dec(v___y_2201_);
lean_dec_ref(v___y_2200_);
lean_dec(v___y_2199_);
lean_dec_ref(v___y_2198_);
lean_dec(v___y_2197_);
lean_dec_ref(v___y_2196_);
lean_dec(v___y_2195_);
lean_dec_ref(v___y_2194_);
return v_res_2203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2(lean_object* v_00_u03b2_2204_, lean_object* v_x_2205_, lean_object* v_x_2206_, lean_object* v_x_2207_){
_start:
{
lean_object* v___x_2208_; 
v___x_2208_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2___redArg(v_x_2205_, v_x_2206_, v_x_2207_);
return v___x_2208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5(lean_object* v_00_u03b2_2209_, lean_object* v_x_2210_, size_t v_x_2211_, size_t v_x_2212_, lean_object* v_x_2213_, lean_object* v_x_2214_){
_start:
{
lean_object* v___x_2215_; 
v___x_2215_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___redArg(v_x_2210_, v_x_2211_, v_x_2212_, v_x_2213_, v_x_2214_);
return v___x_2215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5___boxed(lean_object* v_00_u03b2_2216_, lean_object* v_x_2217_, lean_object* v_x_2218_, lean_object* v_x_2219_, lean_object* v_x_2220_, lean_object* v_x_2221_){
_start:
{
size_t v_x_20226__boxed_2222_; size_t v_x_20227__boxed_2223_; lean_object* v_res_2224_; 
v_x_20226__boxed_2222_ = lean_unbox_usize(v_x_2218_);
lean_dec(v_x_2218_);
v_x_20227__boxed_2223_ = lean_unbox_usize(v_x_2219_);
lean_dec(v_x_2219_);
v_res_2224_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5(v_00_u03b2_2216_, v_x_2217_, v_x_20226__boxed_2222_, v_x_20227__boxed_2223_, v_x_2220_, v_x_2221_);
return v_res_2224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8(lean_object* v_00_u03b2_2225_, lean_object* v_n_2226_, lean_object* v_k_2227_, lean_object* v_v_2228_){
_start:
{
lean_object* v___x_2229_; 
v___x_2229_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8___redArg(v_n_2226_, v_k_2227_, v_v_2228_);
return v___x_2229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9(lean_object* v_00_u03b2_2230_, size_t v_depth_2231_, lean_object* v_keys_2232_, lean_object* v_vals_2233_, lean_object* v_heq_2234_, lean_object* v_i_2235_, lean_object* v_entries_2236_){
_start:
{
lean_object* v___x_2237_; 
v___x_2237_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___redArg(v_depth_2231_, v_keys_2232_, v_vals_2233_, v_i_2235_, v_entries_2236_);
return v___x_2237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9___boxed(lean_object* v_00_u03b2_2238_, lean_object* v_depth_2239_, lean_object* v_keys_2240_, lean_object* v_vals_2241_, lean_object* v_heq_2242_, lean_object* v_i_2243_, lean_object* v_entries_2244_){
_start:
{
size_t v_depth_boxed_2245_; lean_object* v_res_2246_; 
v_depth_boxed_2245_ = lean_unbox_usize(v_depth_2239_);
lean_dec(v_depth_2239_);
v_res_2246_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__9(v_00_u03b2_2238_, v_depth_boxed_2245_, v_keys_2240_, v_vals_2241_, v_heq_2242_, v_i_2243_, v_entries_2244_);
lean_dec_ref(v_vals_2241_);
lean_dec_ref(v_keys_2240_);
return v_res_2246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9(lean_object* v_00_u03b2_2247_, lean_object* v_x_2248_, lean_object* v_x_2249_, lean_object* v_x_2250_, lean_object* v_x_2251_){
_start:
{
lean_object* v___x_2252_; 
v___x_2252_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Mathlib_Meta_FunProp_funPropTac_spec__2_spec__2_spec__5_spec__8_spec__9___redArg(v_x_2248_, v_x_2249_, v_x_2250_, v_x_2251_);
return v___x_2252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg(){
_start:
{
lean_object* v___x_2279_; lean_object* v___x_2280_; 
v___x_2279_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp_funPropTac_spec__0___redArg___closed__0);
v___x_2280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2280_, 0, v___x_2279_);
return v___x_2280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg___boxed(lean_object* v___y_2281_){
_start:
{
lean_object* v_res_2282_; 
v_res_2282_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg();
return v_res_2282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0(lean_object* v_00_u03b1_2283_, lean_object* v___y_2284_, lean_object* v___y_2285_){
_start:
{
lean_object* v___x_2287_; 
v___x_2287_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg();
return v___x_2287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___boxed(lean_object* v_00_u03b1_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_){
_start:
{
lean_object* v_res_2292_; 
v_res_2292_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0(v_00_u03b1_2288_, v___y_2289_, v___y_2290_);
lean_dec(v___y_2290_);
lean_dec_ref(v___y_2289_);
return v_res_2292_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1(void){
_start:
{
lean_object* v___x_2294_; lean_object* v___x_2295_; 
v___x_2294_ = ((lean_object*)(lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__0));
v___x_2295_ = l_Lean_stringToMessageData(v___x_2294_);
return v___x_2295_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3(void){
_start:
{
lean_object* v___x_2297_; lean_object* v___x_2298_; 
v___x_2297_ = ((lean_object*)(lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__2));
v___x_2298_ = l_Lean_stringToMessageData(v___x_2297_);
return v___x_2298_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5(void){
_start:
{
lean_object* v___x_2300_; lean_object* v___x_2301_; 
v___x_2300_ = ((lean_object*)(lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__4));
v___x_2301_ = l_Lean_stringToMessageData(v___x_2300_);
return v___x_2301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(lean_object* v_x_2302_){
_start:
{
switch(lean_obj_tag(v_x_2302_))
{
case 0:
{
lean_object* v_declName_2304_; uint8_t v_post_2305_; uint8_t v_inv_2306_; uint8_t v___x_2307_; lean_object* v_r_2308_; 
v_declName_2304_ = lean_ctor_get(v_x_2302_, 0);
lean_inc(v_declName_2304_);
v_post_2305_ = lean_ctor_get_uint8(v_x_2302_, sizeof(void*)*1);
v_inv_2306_ = lean_ctor_get_uint8(v_x_2302_, sizeof(void*)*1 + 1);
lean_dec_ref_known(v_x_2302_, 1);
v___x_2307_ = 0;
v_r_2308_ = l_Lean_MessageData_ofConstName(v_declName_2304_, v___x_2307_);
if (v_post_2305_ == 0)
{
if (v_inv_2306_ == 0)
{
lean_object* v___x_2309_; lean_object* v___x_2310_; lean_object* v___x_2311_; 
v___x_2309_ = lean_obj_once(&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1, &lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1_once, _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__1);
v___x_2310_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2310_, 0, v___x_2309_);
lean_ctor_set(v___x_2310_, 1, v_r_2308_);
v___x_2311_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2311_, 0, v___x_2310_);
return v___x_2311_;
}
else
{
lean_object* v___x_2312_; lean_object* v___x_2313_; lean_object* v___x_2314_; 
v___x_2312_ = lean_obj_once(&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3, &lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3_once, _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__3);
v___x_2313_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2313_, 0, v___x_2312_);
lean_ctor_set(v___x_2313_, 1, v_r_2308_);
v___x_2314_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2314_, 0, v___x_2313_);
return v___x_2314_;
}
}
else
{
if (v_inv_2306_ == 0)
{
lean_object* v___x_2315_; 
v___x_2315_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2315_, 0, v_r_2308_);
return v___x_2315_;
}
else
{
lean_object* v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; 
v___x_2316_ = lean_obj_once(&lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5, &lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5_once, _init_lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___closed__5);
v___x_2317_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2317_, 0, v___x_2316_);
lean_ctor_set(v___x_2317_, 1, v_r_2308_);
v___x_2318_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2318_, 0, v___x_2317_);
return v___x_2318_;
}
}
}
case 1:
{
lean_object* v_fvarId_2319_; lean_object* v___x_2321_; uint8_t v_isShared_2322_; uint8_t v_isSharedCheck_2328_; 
v_fvarId_2319_ = lean_ctor_get(v_x_2302_, 0);
v_isSharedCheck_2328_ = !lean_is_exclusive(v_x_2302_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2321_ = v_x_2302_;
v_isShared_2322_ = v_isSharedCheck_2328_;
goto v_resetjp_2320_;
}
else
{
lean_inc(v_fvarId_2319_);
lean_dec(v_x_2302_);
v___x_2321_ = lean_box(0);
v_isShared_2322_ = v_isSharedCheck_2328_;
goto v_resetjp_2320_;
}
v_resetjp_2320_:
{
lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2326_; 
v___x_2323_ = l_Lean_mkFVar(v_fvarId_2319_);
v___x_2324_ = l_Lean_MessageData_ofExpr(v___x_2323_);
if (v_isShared_2322_ == 0)
{
lean_ctor_set_tag(v___x_2321_, 0);
lean_ctor_set(v___x_2321_, 0, v___x_2324_);
v___x_2326_ = v___x_2321_;
goto v_reusejp_2325_;
}
else
{
lean_object* v_reuseFailAlloc_2327_; 
v_reuseFailAlloc_2327_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2327_, 0, v___x_2324_);
v___x_2326_ = v_reuseFailAlloc_2327_;
goto v_reusejp_2325_;
}
v_reusejp_2325_:
{
return v___x_2326_;
}
}
}
case 2:
{
lean_object* v_ref_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; 
v_ref_2329_ = lean_ctor_get(v_x_2302_, 1);
lean_inc(v_ref_2329_);
lean_dec_ref_known(v_x_2302_, 2);
v___x_2330_ = l_Lean_MessageData_ofSyntax(v_ref_2329_);
v___x_2331_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2331_, 0, v___x_2330_);
return v___x_2331_;
}
default: 
{
lean_object* v_name_2332_; lean_object* v___x_2334_; uint8_t v_isShared_2335_; uint8_t v_isSharedCheck_2340_; 
v_name_2332_ = lean_ctor_get(v_x_2302_, 0);
v_isSharedCheck_2340_ = !lean_is_exclusive(v_x_2302_);
if (v_isSharedCheck_2340_ == 0)
{
v___x_2334_ = v_x_2302_;
v_isShared_2335_ = v_isSharedCheck_2340_;
goto v_resetjp_2333_;
}
else
{
lean_inc(v_name_2332_);
lean_dec(v_x_2302_);
v___x_2334_ = lean_box(0);
v_isShared_2335_ = v_isSharedCheck_2340_;
goto v_resetjp_2333_;
}
v_resetjp_2333_:
{
lean_object* v___x_2336_; lean_object* v___x_2338_; 
v___x_2336_ = l_Lean_MessageData_ofName(v_name_2332_);
if (v_isShared_2335_ == 0)
{
lean_ctor_set_tag(v___x_2334_, 0);
lean_ctor_set(v___x_2334_, 0, v___x_2336_);
v___x_2338_ = v___x_2334_;
goto v_reusejp_2337_;
}
else
{
lean_object* v_reuseFailAlloc_2339_; 
v_reuseFailAlloc_2339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2339_, 0, v___x_2336_);
v___x_2338_ = v_reuseFailAlloc_2339_;
goto v_reusejp_2337_;
}
v_reusejp_2337_:
{
return v___x_2338_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg___boxed(lean_object* v_x_2341_, lean_object* v___y_2342_){
_start:
{
lean_object* v_res_2343_; 
v_res_2343_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v_x_2341_);
return v_res_2343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1(lean_object* v_x_2344_, lean_object* v___y_2345_, lean_object* v___y_2346_){
_start:
{
lean_object* v___x_2348_; 
v___x_2348_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v_x_2344_);
return v___x_2348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___boxed(lean_object* v_x_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_){
_start:
{
lean_object* v_res_2353_; 
v_res_2353_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1(v_x_2349_, v___y_2350_, v___y_2351_);
lean_dec(v___y_2351_);
lean_dec_ref(v___y_2350_);
return v_res_2353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16(lean_object* v_x_2354_, lean_object* v_x_2355_){
_start:
{
if (lean_obj_tag(v_x_2355_) == 0)
{
return v_x_2354_;
}
else
{
lean_object* v_head_2356_; lean_object* v_tail_2357_; lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; 
v_head_2356_ = lean_ctor_get(v_x_2355_, 0);
v_tail_2357_ = lean_ctor_get(v_x_2355_, 1);
v___x_2358_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__19));
v___x_2359_ = lean_string_append(v_x_2354_, v___x_2358_);
v___x_2360_ = lean_expr_dbg_to_string(v_head_2356_);
v___x_2361_ = lean_string_append(v___x_2359_, v___x_2360_);
lean_dec_ref(v___x_2360_);
v_x_2354_ = v___x_2361_;
v_x_2355_ = v_tail_2357_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16___boxed(lean_object* v_x_2363_, lean_object* v_x_2364_){
_start:
{
lean_object* v_res_2365_; 
v_res_2365_ = lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16(v_x_2363_, v_x_2364_);
lean_dec(v_x_2364_);
return v_res_2365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11(lean_object* v_x_2368_){
_start:
{
if (lean_obj_tag(v_x_2368_) == 0)
{
lean_object* v___x_2369_; 
v___x_2369_ = ((lean_object*)(lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__0));
return v___x_2369_;
}
else
{
lean_object* v_tail_2370_; 
v_tail_2370_ = lean_ctor_get(v_x_2368_, 1);
if (lean_obj_tag(v_tail_2370_) == 0)
{
lean_object* v_head_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2374_; lean_object* v___x_2375_; lean_object* v___x_2376_; 
v_head_2371_ = lean_ctor_get(v_x_2368_, 0);
v___x_2372_ = ((lean_object*)(lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__1));
v___x_2373_ = lean_expr_dbg_to_string(v_head_2371_);
v___x_2374_ = lean_string_append(v___x_2372_, v___x_2373_);
lean_dec_ref(v___x_2373_);
v___x_2375_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx___closed__24));
v___x_2376_ = lean_string_append(v___x_2374_, v___x_2375_);
return v___x_2376_;
}
else
{
lean_object* v_head_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___x_2381_; uint32_t v___x_2382_; lean_object* v___x_2383_; 
v_head_2377_ = lean_ctor_get(v_x_2368_, 0);
v___x_2378_ = ((lean_object*)(lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___closed__1));
v___x_2379_ = lean_expr_dbg_to_string(v_head_2377_);
v___x_2380_ = lean_string_append(v___x_2378_, v___x_2379_);
lean_dec_ref(v___x_2379_);
v___x_2381_ = lp_mathlib_List_foldl___at___00List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11_spec__16(v___x_2380_, v_tail_2370_);
v___x_2382_ = 93;
v___x_2383_ = lean_string_push(v___x_2381_, v___x_2382_);
return v___x_2383_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11___boxed(lean_object* v_x_2384_){
_start:
{
lean_object* v_res_2385_; 
v_res_2385_ = lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11(v_x_2384_);
lean_dec(v_x_2384_);
return v_res_2385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg(lean_object* v_msgData_2386_, lean_object* v_macroStack_2387_, lean_object* v___y_2388_){
_start:
{
lean_object* v___x_2390_; lean_object* v_scopes_2391_; lean_object* v___x_2392_; lean_object* v___x_2393_; lean_object* v_opts_2394_; lean_object* v___x_2395_; uint8_t v___x_2396_; 
v___x_2390_ = lean_st_ref_get(v___y_2388_);
v_scopes_2391_ = lean_ctor_get(v___x_2390_, 2);
lean_inc(v_scopes_2391_);
lean_dec(v___x_2390_);
v___x_2392_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2393_ = l_List_head_x21___redArg(v___x_2392_, v_scopes_2391_);
lean_dec(v_scopes_2391_);
v_opts_2394_ = lean_ctor_get(v___x_2393_, 1);
lean_inc_ref(v_opts_2394_);
lean_dec(v___x_2393_);
v___x_2395_ = l_Lean_Elab_pp_macroStack;
v___x_2396_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_opts_2394_, v___x_2395_);
lean_dec_ref(v_opts_2394_);
if (v___x_2396_ == 0)
{
lean_object* v___x_2397_; 
lean_dec(v_macroStack_2387_);
v___x_2397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2397_, 0, v_msgData_2386_);
return v___x_2397_;
}
else
{
if (lean_obj_tag(v_macroStack_2387_) == 0)
{
lean_object* v___x_2398_; 
v___x_2398_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2398_, 0, v_msgData_2386_);
return v___x_2398_;
}
else
{
lean_object* v_head_2399_; lean_object* v_after_2400_; lean_object* v___x_2402_; uint8_t v_isShared_2403_; uint8_t v_isSharedCheck_2415_; 
v_head_2399_ = lean_ctor_get(v_macroStack_2387_, 0);
lean_inc(v_head_2399_);
v_after_2400_ = lean_ctor_get(v_head_2399_, 1);
v_isSharedCheck_2415_ = !lean_is_exclusive(v_head_2399_);
if (v_isSharedCheck_2415_ == 0)
{
lean_object* v_unused_2416_; 
v_unused_2416_ = lean_ctor_get(v_head_2399_, 0);
lean_dec(v_unused_2416_);
v___x_2402_ = v_head_2399_;
v_isShared_2403_ = v_isSharedCheck_2415_;
goto v_resetjp_2401_;
}
else
{
lean_inc(v_after_2400_);
lean_dec(v_head_2399_);
v___x_2402_ = lean_box(0);
v_isShared_2403_ = v_isSharedCheck_2415_;
goto v_resetjp_2401_;
}
v_resetjp_2401_:
{
lean_object* v___x_2404_; lean_object* v___x_2406_; 
v___x_2404_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7___closed__0);
if (v_isShared_2403_ == 0)
{
lean_ctor_set_tag(v___x_2402_, 7);
lean_ctor_set(v___x_2402_, 1, v___x_2404_);
lean_ctor_set(v___x_2402_, 0, v_msgData_2386_);
v___x_2406_ = v___x_2402_;
goto v_reusejp_2405_;
}
else
{
lean_object* v_reuseFailAlloc_2414_; 
v_reuseFailAlloc_2414_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2414_, 0, v_msgData_2386_);
lean_ctor_set(v_reuseFailAlloc_2414_, 1, v___x_2404_);
v___x_2406_ = v_reuseFailAlloc_2414_;
goto v_reusejp_2405_;
}
v_reusejp_2405_:
{
lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; lean_object* v_msgData_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; 
v___x_2407_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4___redArg___closed__2);
v___x_2408_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2408_, 0, v___x_2406_);
lean_ctor_set(v___x_2408_, 1, v___x_2407_);
v___x_2409_ = l_Lean_MessageData_ofSyntax(v_after_2400_);
v___x_2410_ = l_Lean_indentD(v___x_2409_);
v_msgData_2411_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2411_, 0, v___x_2408_);
lean_ctor_set(v_msgData_2411_, 1, v___x_2410_);
v___x_2412_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__7(v_msgData_2411_, v_macroStack_2387_);
v___x_2413_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2413_, 0, v___x_2412_);
return v___x_2413_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg___boxed(lean_object* v_msgData_2417_, lean_object* v_macroStack_2418_, lean_object* v___y_2419_, lean_object* v___y_2420_){
_start:
{
lean_object* v_res_2421_; 
v_res_2421_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg(v_msgData_2417_, v_macroStack_2418_, v___y_2419_);
lean_dec(v___y_2419_);
return v_res_2421_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_2422_; 
v___x_2422_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_2422_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1(void){
_start:
{
lean_object* v___x_2423_; lean_object* v___x_2424_; 
v___x_2423_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__0);
v___x_2424_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2424_, 0, v___x_2423_);
return v___x_2424_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2(void){
_start:
{
lean_object* v___x_2425_; lean_object* v___x_2426_; lean_object* v___x_2427_; 
v___x_2425_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1);
v___x_2426_ = lean_unsigned_to_nat(0u);
v___x_2427_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_2427_, 0, v___x_2426_);
lean_ctor_set(v___x_2427_, 1, v___x_2426_);
lean_ctor_set(v___x_2427_, 2, v___x_2426_);
lean_ctor_set(v___x_2427_, 3, v___x_2426_);
lean_ctor_set(v___x_2427_, 4, v___x_2425_);
lean_ctor_set(v___x_2427_, 5, v___x_2425_);
lean_ctor_set(v___x_2427_, 6, v___x_2425_);
lean_ctor_set(v___x_2427_, 7, v___x_2425_);
lean_ctor_set(v___x_2427_, 8, v___x_2425_);
lean_ctor_set(v___x_2427_, 9, v___x_2425_);
return v___x_2427_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3(void){
_start:
{
lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___x_2430_; 
v___x_2428_ = lean_unsigned_to_nat(32u);
v___x_2429_ = lean_mk_empty_array_with_capacity(v___x_2428_);
v___x_2430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2430_, 0, v___x_2429_);
return v___x_2430_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4(void){
_start:
{
size_t v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; lean_object* v___x_2434_; lean_object* v___x_2435_; lean_object* v___x_2436_; 
v___x_2431_ = ((size_t)5ULL);
v___x_2432_ = lean_unsigned_to_nat(0u);
v___x_2433_ = lean_unsigned_to_nat(32u);
v___x_2434_ = lean_mk_empty_array_with_capacity(v___x_2433_);
v___x_2435_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__3);
v___x_2436_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_2436_, 0, v___x_2435_);
lean_ctor_set(v___x_2436_, 1, v___x_2434_);
lean_ctor_set(v___x_2436_, 2, v___x_2432_);
lean_ctor_set(v___x_2436_, 3, v___x_2432_);
lean_ctor_set_usize(v___x_2436_, 4, v___x_2431_);
return v___x_2436_;
}
}
static lean_object* _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5(void){
_start:
{
lean_object* v___x_2437_; lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; 
v___x_2437_ = lean_box(1);
v___x_2438_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__4);
v___x_2439_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__1);
v___x_2440_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2440_, 0, v___x_2439_);
lean_ctor_set(v___x_2440_, 1, v___x_2438_);
lean_ctor_set(v___x_2440_, 2, v___x_2437_);
return v___x_2440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(lean_object* v_msgData_2441_, lean_object* v___y_2442_){
_start:
{
lean_object* v___x_2444_; lean_object* v_env_2445_; lean_object* v___x_2446_; lean_object* v_scopes_2447_; lean_object* v___x_2448_; lean_object* v___x_2449_; lean_object* v_opts_2450_; lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2454_; lean_object* v___x_2455_; 
v___x_2444_ = lean_st_ref_get(v___y_2442_);
v_env_2445_ = lean_ctor_get(v___x_2444_, 0);
lean_inc_ref(v_env_2445_);
lean_dec(v___x_2444_);
v___x_2446_ = lean_st_ref_get(v___y_2442_);
v_scopes_2447_ = lean_ctor_get(v___x_2446_, 2);
lean_inc(v_scopes_2447_);
lean_dec(v___x_2446_);
v___x_2448_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2449_ = l_List_head_x21___redArg(v___x_2448_, v_scopes_2447_);
lean_dec(v_scopes_2447_);
v_opts_2450_ = lean_ctor_get(v___x_2449_, 1);
lean_inc_ref(v_opts_2450_);
lean_dec(v___x_2449_);
v___x_2451_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2);
v___x_2452_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5);
v___x_2453_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2453_, 0, v_env_2445_);
lean_ctor_set(v___x_2453_, 1, v___x_2451_);
lean_ctor_set(v___x_2453_, 2, v___x_2452_);
lean_ctor_set(v___x_2453_, 3, v_opts_2450_);
v___x_2454_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_2454_, 0, v___x_2453_);
lean_ctor_set(v___x_2454_, 1, v_msgData_2441_);
v___x_2455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2455_, 0, v___x_2454_);
return v___x_2455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___boxed(lean_object* v_msgData_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_){
_start:
{
lean_object* v_res_2459_; 
v_res_2459_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(v_msgData_2456_, v___y_2457_);
lean_dec(v___y_2457_);
return v_res_2459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg(lean_object* v_msg_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_){
_start:
{
lean_object* v___x_2464_; 
v___x_2464_ = l_Lean_Elab_Command_getRef___redArg(v___y_2461_);
if (lean_obj_tag(v___x_2464_) == 0)
{
lean_object* v_a_2465_; lean_object* v_macroStack_2466_; lean_object* v___x_2467_; lean_object* v_a_2468_; lean_object* v___x_2469_; lean_object* v___x_2470_; lean_object* v_a_2471_; lean_object* v___x_2473_; uint8_t v_isShared_2474_; uint8_t v_isSharedCheck_2479_; 
v_a_2465_ = lean_ctor_get(v___x_2464_, 0);
lean_inc(v_a_2465_);
lean_dec_ref_known(v___x_2464_, 1);
v_macroStack_2466_ = lean_ctor_get(v___y_2461_, 4);
v___x_2467_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(v_msg_2460_, v___y_2462_);
v_a_2468_ = lean_ctor_get(v___x_2467_, 0);
lean_inc(v_a_2468_);
lean_dec_ref(v___x_2467_);
v___x_2469_ = l_Lean_Elab_getBetterRef(v_a_2465_, v_macroStack_2466_);
lean_dec(v_a_2465_);
lean_inc(v_macroStack_2466_);
v___x_2470_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg(v_a_2468_, v_macroStack_2466_, v___y_2462_);
v_a_2471_ = lean_ctor_get(v___x_2470_, 0);
v_isSharedCheck_2479_ = !lean_is_exclusive(v___x_2470_);
if (v_isSharedCheck_2479_ == 0)
{
v___x_2473_ = v___x_2470_;
v_isShared_2474_ = v_isSharedCheck_2479_;
goto v_resetjp_2472_;
}
else
{
lean_inc(v_a_2471_);
lean_dec(v___x_2470_);
v___x_2473_ = lean_box(0);
v_isShared_2474_ = v_isSharedCheck_2479_;
goto v_resetjp_2472_;
}
v_resetjp_2472_:
{
lean_object* v___x_2475_; lean_object* v___x_2477_; 
v___x_2475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2475_, 0, v___x_2469_);
lean_ctor_set(v___x_2475_, 1, v_a_2471_);
if (v_isShared_2474_ == 0)
{
lean_ctor_set_tag(v___x_2473_, 1);
lean_ctor_set(v___x_2473_, 0, v___x_2475_);
v___x_2477_ = v___x_2473_;
goto v_reusejp_2476_;
}
else
{
lean_object* v_reuseFailAlloc_2478_; 
v_reuseFailAlloc_2478_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2478_, 0, v___x_2475_);
v___x_2477_ = v_reuseFailAlloc_2478_;
goto v_reusejp_2476_;
}
v_reusejp_2476_:
{
return v___x_2477_;
}
}
}
else
{
lean_object* v_a_2480_; lean_object* v___x_2482_; uint8_t v_isShared_2483_; uint8_t v_isSharedCheck_2487_; 
lean_dec_ref(v_msg_2460_);
v_a_2480_ = lean_ctor_get(v___x_2464_, 0);
v_isSharedCheck_2487_ = !lean_is_exclusive(v___x_2464_);
if (v_isSharedCheck_2487_ == 0)
{
v___x_2482_ = v___x_2464_;
v_isShared_2483_ = v_isSharedCheck_2487_;
goto v_resetjp_2481_;
}
else
{
lean_inc(v_a_2480_);
lean_dec(v___x_2464_);
v___x_2482_ = lean_box(0);
v_isShared_2483_ = v_isSharedCheck_2487_;
goto v_resetjp_2481_;
}
v_resetjp_2481_:
{
lean_object* v___x_2485_; 
if (v_isShared_2483_ == 0)
{
v___x_2485_ = v___x_2482_;
goto v_reusejp_2484_;
}
else
{
lean_object* v_reuseFailAlloc_2486_; 
v_reuseFailAlloc_2486_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2486_, 0, v_a_2480_);
v___x_2485_ = v_reuseFailAlloc_2486_;
goto v_reusejp_2484_;
}
v_reusejp_2484_:
{
return v___x_2485_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg___boxed(lean_object* v_msg_2488_, lean_object* v___y_2489_, lean_object* v___y_2490_, lean_object* v___y_2491_){
_start:
{
lean_object* v_res_2492_; 
v_res_2492_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg(v_msg_2488_, v___y_2489_, v___y_2490_);
lean_dec(v___y_2490_);
lean_dec_ref(v___y_2489_);
return v_res_2492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(lean_object* v_ref_2493_, lean_object* v_msg_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_){
_start:
{
lean_object* v___x_2498_; 
v___x_2498_ = l_Lean_Elab_Command_getRef___redArg(v___y_2495_);
if (lean_obj_tag(v___x_2498_) == 0)
{
lean_object* v_a_2499_; lean_object* v_fileName_2500_; lean_object* v_fileMap_2501_; lean_object* v_currRecDepth_2502_; lean_object* v_cmdPos_2503_; lean_object* v_macroStack_2504_; lean_object* v_quotContext_x3f_2505_; lean_object* v_currMacroScope_2506_; lean_object* v_snap_x3f_2507_; lean_object* v_cancelTk_x3f_2508_; uint8_t v_suppressElabErrors_2509_; lean_object* v_ref_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; 
v_a_2499_ = lean_ctor_get(v___x_2498_, 0);
lean_inc(v_a_2499_);
lean_dec_ref_known(v___x_2498_, 1);
v_fileName_2500_ = lean_ctor_get(v___y_2495_, 0);
v_fileMap_2501_ = lean_ctor_get(v___y_2495_, 1);
v_currRecDepth_2502_ = lean_ctor_get(v___y_2495_, 2);
v_cmdPos_2503_ = lean_ctor_get(v___y_2495_, 3);
v_macroStack_2504_ = lean_ctor_get(v___y_2495_, 4);
v_quotContext_x3f_2505_ = lean_ctor_get(v___y_2495_, 5);
v_currMacroScope_2506_ = lean_ctor_get(v___y_2495_, 6);
v_snap_x3f_2507_ = lean_ctor_get(v___y_2495_, 8);
v_cancelTk_x3f_2508_ = lean_ctor_get(v___y_2495_, 9);
v_suppressElabErrors_2509_ = lean_ctor_get_uint8(v___y_2495_, sizeof(void*)*10);
v_ref_2510_ = l_Lean_replaceRef(v_ref_2493_, v_a_2499_);
lean_dec(v_a_2499_);
lean_inc(v_cancelTk_x3f_2508_);
lean_inc(v_snap_x3f_2507_);
lean_inc(v_currMacroScope_2506_);
lean_inc(v_quotContext_x3f_2505_);
lean_inc(v_macroStack_2504_);
lean_inc(v_cmdPos_2503_);
lean_inc(v_currRecDepth_2502_);
lean_inc_ref(v_fileMap_2501_);
lean_inc_ref(v_fileName_2500_);
v___x_2511_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_2511_, 0, v_fileName_2500_);
lean_ctor_set(v___x_2511_, 1, v_fileMap_2501_);
lean_ctor_set(v___x_2511_, 2, v_currRecDepth_2502_);
lean_ctor_set(v___x_2511_, 3, v_cmdPos_2503_);
lean_ctor_set(v___x_2511_, 4, v_macroStack_2504_);
lean_ctor_set(v___x_2511_, 5, v_quotContext_x3f_2505_);
lean_ctor_set(v___x_2511_, 6, v_currMacroScope_2506_);
lean_ctor_set(v___x_2511_, 7, v_ref_2510_);
lean_ctor_set(v___x_2511_, 8, v_snap_x3f_2507_);
lean_ctor_set(v___x_2511_, 9, v_cancelTk_x3f_2508_);
lean_ctor_set_uint8(v___x_2511_, sizeof(void*)*10, v_suppressElabErrors_2509_);
v___x_2512_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg(v_msg_2494_, v___x_2511_, v___y_2496_);
lean_dec_ref_known(v___x_2511_, 10);
return v___x_2512_;
}
else
{
lean_object* v_a_2513_; lean_object* v___x_2515_; uint8_t v_isShared_2516_; uint8_t v_isSharedCheck_2520_; 
lean_dec_ref(v_msg_2494_);
v_a_2513_ = lean_ctor_get(v___x_2498_, 0);
v_isSharedCheck_2520_ = !lean_is_exclusive(v___x_2498_);
if (v_isSharedCheck_2520_ == 0)
{
v___x_2515_ = v___x_2498_;
v_isShared_2516_ = v_isSharedCheck_2520_;
goto v_resetjp_2514_;
}
else
{
lean_inc(v_a_2513_);
lean_dec(v___x_2498_);
v___x_2515_ = lean_box(0);
v_isShared_2516_ = v_isSharedCheck_2520_;
goto v_resetjp_2514_;
}
v_resetjp_2514_:
{
lean_object* v___x_2518_; 
if (v_isShared_2516_ == 0)
{
v___x_2518_ = v___x_2515_;
goto v_reusejp_2517_;
}
else
{
lean_object* v_reuseFailAlloc_2519_; 
v_reuseFailAlloc_2519_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2519_, 0, v_a_2513_);
v___x_2518_ = v_reuseFailAlloc_2519_;
goto v_reusejp_2517_;
}
v_reusejp_2517_:
{
return v___x_2518_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg___boxed(lean_object* v_ref_2521_, lean_object* v_msg_2522_, lean_object* v___y_2523_, lean_object* v___y_2524_, lean_object* v___y_2525_){
_start:
{
lean_object* v_res_2526_; 
v_res_2526_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(v_ref_2521_, v_msg_2522_, v___y_2523_, v___y_2524_);
lean_dec(v___y_2524_);
lean_dec_ref(v___y_2523_);
lean_dec(v_ref_2521_);
return v_res_2526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__10(lean_object* v_a_2527_, lean_object* v_a_2528_){
_start:
{
if (lean_obj_tag(v_a_2527_) == 0)
{
lean_object* v___x_2529_; 
v___x_2529_ = l_List_reverse___redArg(v_a_2528_);
return v___x_2529_;
}
else
{
lean_object* v_head_2530_; lean_object* v_tail_2531_; lean_object* v___x_2533_; uint8_t v_isShared_2534_; uint8_t v_isSharedCheck_2541_; 
v_head_2530_ = lean_ctor_get(v_a_2527_, 0);
v_tail_2531_ = lean_ctor_get(v_a_2527_, 1);
v_isSharedCheck_2541_ = !lean_is_exclusive(v_a_2527_);
if (v_isSharedCheck_2541_ == 0)
{
v___x_2533_ = v_a_2527_;
v_isShared_2534_ = v_isSharedCheck_2541_;
goto v_resetjp_2532_;
}
else
{
lean_inc(v_tail_2531_);
lean_inc(v_head_2530_);
lean_dec(v_a_2527_);
v___x_2533_ = lean_box(0);
v_isShared_2534_ = v_isSharedCheck_2541_;
goto v_resetjp_2532_;
}
v_resetjp_2532_:
{
lean_object* v___x_2535_; lean_object* v___x_2536_; lean_object* v___x_2538_; 
v___x_2535_ = lean_box(0);
v___x_2536_ = l_Lean_mkConst(v_head_2530_, v___x_2535_);
if (v_isShared_2534_ == 0)
{
lean_ctor_set(v___x_2533_, 1, v_a_2528_);
lean_ctor_set(v___x_2533_, 0, v___x_2536_);
v___x_2538_ = v___x_2533_;
goto v_reusejp_2537_;
}
else
{
lean_object* v_reuseFailAlloc_2540_; 
v_reuseFailAlloc_2540_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2540_, 0, v___x_2536_);
lean_ctor_set(v_reuseFailAlloc_2540_, 1, v_a_2528_);
v___x_2538_ = v_reuseFailAlloc_2540_;
goto v_reusejp_2537_;
}
v_reusejp_2537_:
{
v_a_2527_ = v_tail_2531_;
v_a_2528_ = v___x_2538_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0(void){
_start:
{
lean_object* v___x_2542_; 
v___x_2542_ = l_instMonadEIO(lean_box(0));
return v___x_2542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9(lean_object* v_msg_2545_, lean_object* v___y_2546_, lean_object* v___y_2547_){
_start:
{
lean_object* v___x_2549_; lean_object* v___x_2550_; lean_object* v_toApplicative_2551_; lean_object* v___x_2553_; uint8_t v_isShared_2554_; uint8_t v_isSharedCheck_2582_; 
v___x_2549_ = lean_obj_once(&lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0, &lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0_once, _init_lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__0);
v___x_2550_ = l_StateRefT_x27_instMonad___redArg(v___x_2549_);
v_toApplicative_2551_ = lean_ctor_get(v___x_2550_, 0);
v_isSharedCheck_2582_ = !lean_is_exclusive(v___x_2550_);
if (v_isSharedCheck_2582_ == 0)
{
lean_object* v_unused_2583_; 
v_unused_2583_ = lean_ctor_get(v___x_2550_, 1);
lean_dec(v_unused_2583_);
v___x_2553_ = v___x_2550_;
v_isShared_2554_ = v_isSharedCheck_2582_;
goto v_resetjp_2552_;
}
else
{
lean_inc(v_toApplicative_2551_);
lean_dec(v___x_2550_);
v___x_2553_ = lean_box(0);
v_isShared_2554_ = v_isSharedCheck_2582_;
goto v_resetjp_2552_;
}
v_resetjp_2552_:
{
lean_object* v_toFunctor_2555_; lean_object* v_toSeq_2556_; lean_object* v_toSeqLeft_2557_; lean_object* v_toSeqRight_2558_; lean_object* v___x_2560_; uint8_t v_isShared_2561_; uint8_t v_isSharedCheck_2580_; 
v_toFunctor_2555_ = lean_ctor_get(v_toApplicative_2551_, 0);
v_toSeq_2556_ = lean_ctor_get(v_toApplicative_2551_, 2);
v_toSeqLeft_2557_ = lean_ctor_get(v_toApplicative_2551_, 3);
v_toSeqRight_2558_ = lean_ctor_get(v_toApplicative_2551_, 4);
v_isSharedCheck_2580_ = !lean_is_exclusive(v_toApplicative_2551_);
if (v_isSharedCheck_2580_ == 0)
{
lean_object* v_unused_2581_; 
v_unused_2581_ = lean_ctor_get(v_toApplicative_2551_, 1);
lean_dec(v_unused_2581_);
v___x_2560_ = v_toApplicative_2551_;
v_isShared_2561_ = v_isSharedCheck_2580_;
goto v_resetjp_2559_;
}
else
{
lean_inc(v_toSeqRight_2558_);
lean_inc(v_toSeqLeft_2557_);
lean_inc(v_toSeq_2556_);
lean_inc(v_toFunctor_2555_);
lean_dec(v_toApplicative_2551_);
v___x_2560_ = lean_box(0);
v_isShared_2561_ = v_isSharedCheck_2580_;
goto v_resetjp_2559_;
}
v_resetjp_2559_:
{
lean_object* v___f_2562_; lean_object* v___f_2563_; lean_object* v___f_2564_; lean_object* v___f_2565_; lean_object* v___x_2566_; lean_object* v___f_2567_; lean_object* v___f_2568_; lean_object* v___f_2569_; lean_object* v___x_2571_; 
v___f_2562_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__1));
v___f_2563_ = ((lean_object*)(lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___closed__2));
lean_inc_ref(v_toFunctor_2555_);
v___f_2564_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_2564_, 0, v_toFunctor_2555_);
v___f_2565_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2565_, 0, v_toFunctor_2555_);
v___x_2566_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2566_, 0, v___f_2564_);
lean_ctor_set(v___x_2566_, 1, v___f_2565_);
v___f_2567_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2567_, 0, v_toSeqRight_2558_);
v___f_2568_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_2568_, 0, v_toSeqLeft_2557_);
v___f_2569_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2569_, 0, v_toSeq_2556_);
if (v_isShared_2561_ == 0)
{
lean_ctor_set(v___x_2560_, 4, v___f_2567_);
lean_ctor_set(v___x_2560_, 3, v___f_2568_);
lean_ctor_set(v___x_2560_, 2, v___f_2569_);
lean_ctor_set(v___x_2560_, 1, v___f_2562_);
lean_ctor_set(v___x_2560_, 0, v___x_2566_);
v___x_2571_ = v___x_2560_;
goto v_reusejp_2570_;
}
else
{
lean_object* v_reuseFailAlloc_2579_; 
v_reuseFailAlloc_2579_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2579_, 0, v___x_2566_);
lean_ctor_set(v_reuseFailAlloc_2579_, 1, v___f_2562_);
lean_ctor_set(v_reuseFailAlloc_2579_, 2, v___f_2569_);
lean_ctor_set(v_reuseFailAlloc_2579_, 3, v___f_2568_);
lean_ctor_set(v_reuseFailAlloc_2579_, 4, v___f_2567_);
v___x_2571_ = v_reuseFailAlloc_2579_;
goto v_reusejp_2570_;
}
v_reusejp_2570_:
{
lean_object* v___x_2573_; 
if (v_isShared_2554_ == 0)
{
lean_ctor_set(v___x_2553_, 1, v___f_2563_);
lean_ctor_set(v___x_2553_, 0, v___x_2571_);
v___x_2573_ = v___x_2553_;
goto v_reusejp_2572_;
}
else
{
lean_object* v_reuseFailAlloc_2578_; 
v_reuseFailAlloc_2578_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2578_, 0, v___x_2571_);
lean_ctor_set(v_reuseFailAlloc_2578_, 1, v___f_2563_);
v___x_2573_ = v_reuseFailAlloc_2578_;
goto v_reusejp_2572_;
}
v_reusejp_2572_:
{
lean_object* v___x_2574_; lean_object* v___x_2575_; lean_object* v___x_5930__overap_2576_; lean_object* v___x_2577_; 
v___x_2574_ = lean_box(0);
v___x_2575_ = l_instInhabitedOfMonad___redArg(v___x_2573_, v___x_2574_);
v___x_5930__overap_2576_ = lean_panic_fn_borrowed(v___x_2575_, v_msg_2545_);
lean_dec(v___x_2575_);
lean_inc(v___y_2547_);
lean_inc_ref(v___y_2546_);
v___x_2577_ = lean_apply_3(v___x_5930__overap_2576_, v___y_2546_, v___y_2547_, lean_box(0));
return v___x_2577_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9___boxed(lean_object* v_msg_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_){
_start:
{
lean_object* v_res_2588_; 
v_res_2588_ = lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9(v_msg_2584_, v___y_2585_, v___y_2586_);
lean_dec(v___y_2586_);
lean_dec_ref(v___y_2585_);
return v_res_2588_;
}
}
static lean_object* _init_lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3(void){
_start:
{
lean_object* v___x_2592_; lean_object* v___x_2593_; lean_object* v___x_2594_; lean_object* v___x_2595_; lean_object* v___x_2596_; lean_object* v___x_2597_; 
v___x_2592_ = ((lean_object*)(lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__2));
v___x_2593_ = lean_unsigned_to_nat(11u);
v___x_2594_ = lean_unsigned_to_nat(429u);
v___x_2595_ = ((lean_object*)(lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__1));
v___x_2596_ = ((lean_object*)(lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__0));
v___x_2597_ = l_mkPanicMessageWithDecl(v___x_2596_, v___x_2595_, v___x_2594_, v___x_2593_, v___x_2592_);
return v___x_2597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6(lean_object* v_id_2600_, lean_object* v_cs_2601_, lean_object* v___y_2602_, lean_object* v___y_2603_){
_start:
{
if (lean_obj_tag(v_cs_2601_) == 0)
{
lean_object* v___x_2605_; lean_object* v___x_2606_; 
lean_dec(v_id_2600_);
v___x_2605_ = lean_obj_once(&lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3, &lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3_once, _init_lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__3);
v___x_2606_ = lp_mathlib_panic___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__9(v___x_2605_, v___y_2602_, v___y_2603_);
return v___x_2606_;
}
else
{
lean_object* v_tail_2607_; 
v_tail_2607_ = lean_ctor_get(v_cs_2601_, 1);
if (lean_obj_tag(v_tail_2607_) == 0)
{
lean_object* v_head_2608_; lean_object* v___x_2609_; 
lean_dec(v_id_2600_);
v_head_2608_ = lean_ctor_get(v_cs_2601_, 0);
lean_inc(v_head_2608_);
lean_dec_ref_known(v_cs_2601_, 2);
v___x_2609_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2609_, 0, v_head_2608_);
return v___x_2609_;
}
else
{
lean_object* v___x_2610_; lean_object* v___x_2611_; uint8_t v___x_2612_; lean_object* v___x_2613_; lean_object* v___x_2614_; lean_object* v___x_2615_; lean_object* v___x_2616_; lean_object* v___x_2617_; lean_object* v___x_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v___x_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; 
v___x_2610_ = ((lean_object*)(lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__4));
v___x_2611_ = lean_box(0);
v___x_2612_ = 0;
lean_inc(v_id_2600_);
v___x_2613_ = l_Lean_Syntax_formatStx(v_id_2600_, v___x_2611_, v___x_2612_);
v___x_2614_ = l_Std_Format_defWidth;
v___x_2615_ = lean_unsigned_to_nat(0u);
v___x_2616_ = l_Std_Format_pretty(v___x_2613_, v___x_2614_, v___x_2615_, v___x_2615_);
v___x_2617_ = lean_string_append(v___x_2610_, v___x_2616_);
lean_dec_ref(v___x_2616_);
v___x_2618_ = ((lean_object*)(lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___closed__5));
v___x_2619_ = lean_string_append(v___x_2617_, v___x_2618_);
v___x_2620_ = lean_box(0);
v___x_2621_ = lp_mathlib_List_mapTR_loop___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__10(v_cs_2601_, v___x_2620_);
v___x_2622_ = lp_mathlib_List_toString___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__11(v___x_2621_);
lean_dec(v___x_2621_);
v___x_2623_ = lean_string_append(v___x_2619_, v___x_2622_);
lean_dec_ref(v___x_2622_);
v___x_2624_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2624_, 0, v___x_2623_);
v___x_2625_ = l_Lean_MessageData_ofFormat(v___x_2624_);
v___x_2626_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(v_id_2600_, v___x_2625_, v___y_2602_, v___y_2603_);
lean_dec(v_id_2600_);
return v___x_2626_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6___boxed(lean_object* v_id_2627_, lean_object* v_cs_2628_, lean_object* v___y_2629_, lean_object* v___y_2630_, lean_object* v___y_2631_){
_start:
{
lean_object* v_res_2632_; 
v_res_2632_ = lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6(v_id_2627_, v_cs_2628_, v___y_2629_, v___y_2630_);
lean_dec(v___y_2630_);
lean_dec_ref(v___y_2629_);
return v_res_2632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__2(lean_object* v_a_2633_, lean_object* v_a_2634_){
_start:
{
if (lean_obj_tag(v_a_2633_) == 0)
{
lean_object* v___x_2635_; 
v___x_2635_ = l_List_reverse___redArg(v_a_2634_);
return v___x_2635_;
}
else
{
lean_object* v_head_2636_; lean_object* v_tail_2637_; lean_object* v___x_2639_; uint8_t v_isShared_2640_; uint8_t v_isSharedCheck_2648_; 
v_head_2636_ = lean_ctor_get(v_a_2633_, 0);
v_tail_2637_ = lean_ctor_get(v_a_2633_, 1);
v_isSharedCheck_2648_ = !lean_is_exclusive(v_a_2633_);
if (v_isSharedCheck_2648_ == 0)
{
v___x_2639_ = v_a_2633_;
v_isShared_2640_ = v_isSharedCheck_2648_;
goto v_resetjp_2638_;
}
else
{
lean_inc(v_tail_2637_);
lean_inc(v_head_2636_);
lean_dec(v_a_2633_);
v___x_2639_ = lean_box(0);
v_isShared_2640_ = v_isSharedCheck_2648_;
goto v_resetjp_2638_;
}
v_resetjp_2638_:
{
lean_object* v___x_2641_; lean_object* v___x_2642_; lean_object* v___x_2643_; lean_object* v___x_2645_; 
v___x_2641_ = l_Nat_reprFast(v_head_2636_);
v___x_2642_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2642_, 0, v___x_2641_);
v___x_2643_ = l_Lean_MessageData_ofFormat(v___x_2642_);
if (v_isShared_2640_ == 0)
{
lean_ctor_set(v___x_2639_, 1, v_a_2634_);
lean_ctor_set(v___x_2639_, 0, v___x_2643_);
v___x_2645_ = v___x_2639_;
goto v_reusejp_2644_;
}
else
{
lean_object* v_reuseFailAlloc_2647_; 
v_reuseFailAlloc_2647_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2647_, 0, v___x_2643_);
lean_ctor_set(v_reuseFailAlloc_2647_, 1, v_a_2634_);
v___x_2645_ = v_reuseFailAlloc_2647_;
goto v_reusejp_2644_;
}
v_reusejp_2644_:
{
v_a_2633_ = v_tail_2637_;
v_a_2634_ = v___x_2645_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0(void){
_start:
{
lean_object* v___x_2649_; lean_object* v___x_2650_; 
v___x_2649_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Mathlib_Meta_FunProp_funPropTac_spec__3___closed__0));
v___x_2650_ = l_Lean_stringToMessageData(v___x_2649_);
return v___x_2650_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2(void){
_start:
{
lean_object* v___x_2652_; lean_object* v___x_2653_; 
v___x_2652_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__1));
v___x_2653_ = l_Lean_stringToMessageData(v___x_2652_);
return v___x_2653_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4(void){
_start:
{
lean_object* v___x_2655_; lean_object* v___x_2656_; 
v___x_2655_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__3));
v___x_2656_ = l_Lean_stringToMessageData(v___x_2655_);
return v___x_2656_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(uint8_t v___x_2659_, lean_object* v_as_2660_, size_t v_sz_2661_, size_t v_i_2662_, lean_object* v_b_2663_, lean_object* v___y_2664_, lean_object* v___y_2665_){
_start:
{
uint8_t v___x_2667_; 
v___x_2667_ = lean_usize_dec_lt(v_i_2662_, v_sz_2661_);
if (v___x_2667_ == 0)
{
lean_object* v___x_2668_; 
v___x_2668_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2668_, 0, v_b_2663_);
return v___x_2668_;
}
else
{
lean_object* v_a_2669_; lean_object* v_thmOrigin_2670_; lean_object* v_mainArgs_2671_; uint8_t v_form_2672_; uint8_t v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; 
v_a_2669_ = lean_array_uget_borrowed(v_as_2660_, v_i_2662_);
v_thmOrigin_2670_ = lean_ctor_get(v_a_2669_, 1);
v_mainArgs_2671_ = lean_ctor_get(v_a_2669_, 3);
v_form_2672_ = lean_ctor_get_uint8(v_a_2669_, sizeof(void*)*6);
v___x_2673_ = 0;
v___x_2674_ = lp_mathlib_Mathlib_Meta_FunProp_Origin_name(v_thmOrigin_2670_);
v___x_2675_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_2675_, 0, v___x_2674_);
lean_ctor_set_uint8(v___x_2675_, sizeof(void*)*1, v___x_2659_);
lean_ctor_set_uint8(v___x_2675_, sizeof(void*)*1 + 1, v___x_2673_);
v___x_2676_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v___x_2675_);
if (lean_obj_tag(v___x_2676_) == 0)
{
lean_object* v_a_2677_; lean_object* v___x_2679_; uint8_t v_isShared_2680_; uint8_t v_isSharedCheck_2705_; 
v_a_2677_ = lean_ctor_get(v___x_2676_, 0);
v_isSharedCheck_2705_ = !lean_is_exclusive(v___x_2676_);
if (v_isSharedCheck_2705_ == 0)
{
v___x_2679_ = v___x_2676_;
v_isShared_2680_ = v_isSharedCheck_2705_;
goto v_resetjp_2678_;
}
else
{
lean_inc(v_a_2677_);
lean_dec(v___x_2676_);
v___x_2679_ = lean_box(0);
v_isShared_2680_ = v_isSharedCheck_2705_;
goto v_resetjp_2678_;
}
v_resetjp_2678_:
{
lean_object* v___x_2681_; lean_object* v___x_2682_; lean_object* v___x_2683_; lean_object* v___x_2684_; lean_object* v___x_2685_; lean_object* v___x_2686_; lean_object* v___x_2687_; lean_object* v___x_2688_; lean_object* v___x_2689_; lean_object* v___x_2690_; lean_object* v___x_2691_; lean_object* v___y_2693_; 
v___x_2681_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__0);
v___x_2682_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2682_, 0, v___x_2681_);
lean_ctor_set(v___x_2682_, 1, v_a_2677_);
v___x_2683_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__2);
v___x_2684_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2684_, 0, v___x_2682_);
lean_ctor_set(v___x_2684_, 1, v___x_2683_);
lean_inc_ref(v_mainArgs_2671_);
v___x_2685_ = lean_array_to_list(v_mainArgs_2671_);
v___x_2686_ = lean_box(0);
v___x_2687_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__2(v___x_2685_, v___x_2686_);
v___x_2688_ = l_Lean_MessageData_ofList(v___x_2687_);
v___x_2689_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2689_, 0, v___x_2684_);
lean_ctor_set(v___x_2689_, 1, v___x_2688_);
v___x_2690_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__4);
v___x_2691_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2691_, 0, v___x_2689_);
lean_ctor_set(v___x_2691_, 1, v___x_2690_);
if (v_form_2672_ == 0)
{
lean_object* v___x_2703_; 
v___x_2703_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__5));
v___y_2693_ = v___x_2703_;
goto v___jp_2692_;
}
else
{
lean_object* v___x_2704_; 
v___x_2704_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___closed__6));
v___y_2693_ = v___x_2704_;
goto v___jp_2692_;
}
v___jp_2692_:
{
lean_object* v___x_2695_; 
lean_inc_ref(v___y_2693_);
if (v_isShared_2680_ == 0)
{
lean_ctor_set_tag(v___x_2679_, 3);
lean_ctor_set(v___x_2679_, 0, v___y_2693_);
v___x_2695_ = v___x_2679_;
goto v_reusejp_2694_;
}
else
{
lean_object* v_reuseFailAlloc_2702_; 
v_reuseFailAlloc_2702_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2702_, 0, v___y_2693_);
v___x_2695_ = v_reuseFailAlloc_2702_;
goto v_reusejp_2694_;
}
v_reusejp_2694_:
{
lean_object* v___x_2696_; lean_object* v___x_2697_; lean_object* v___x_2698_; size_t v___x_2699_; size_t v___x_2700_; 
v___x_2696_ = l_Lean_MessageData_ofFormat(v___x_2695_);
v___x_2697_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2697_, 0, v___x_2691_);
lean_ctor_set(v___x_2697_, 1, v___x_2696_);
v___x_2698_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2698_, 0, v_b_2663_);
lean_ctor_set(v___x_2698_, 1, v___x_2697_);
v___x_2699_ = ((size_t)1ULL);
v___x_2700_ = lean_usize_add(v_i_2662_, v___x_2699_);
v_i_2662_ = v___x_2700_;
v_b_2663_ = v___x_2698_;
goto _start;
}
}
}
}
else
{
lean_dec_ref(v_b_2663_);
return v___x_2676_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3___boxed(lean_object* v___x_2706_, lean_object* v_as_2707_, lean_object* v_sz_2708_, lean_object* v_i_2709_, lean_object* v_b_2710_, lean_object* v___y_2711_, lean_object* v___y_2712_, lean_object* v___y_2713_){
_start:
{
uint8_t v___x_12740__boxed_2714_; size_t v_sz_boxed_2715_; size_t v_i_boxed_2716_; lean_object* v_res_2717_; 
v___x_12740__boxed_2714_ = lean_unbox(v___x_2706_);
v_sz_boxed_2715_ = lean_unbox_usize(v_sz_2708_);
lean_dec(v_sz_2708_);
v_i_boxed_2716_ = lean_unbox_usize(v_i_2709_);
lean_dec(v_i_2709_);
v_res_2717_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(v___x_12740__boxed_2714_, v_as_2707_, v_sz_boxed_2715_, v_i_boxed_2716_, v_b_2710_, v___y_2711_, v___y_2712_);
lean_dec(v___y_2712_);
lean_dec_ref(v___y_2711_);
lean_dec_ref(v_as_2707_);
return v_res_2717_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0(uint8_t v___y_2719_, uint8_t v_suppressElabErrors_2720_, lean_object* v_x_2721_){
_start:
{
if (lean_obj_tag(v_x_2721_) == 1)
{
lean_object* v_pre_2722_; 
v_pre_2722_ = lean_ctor_get(v_x_2721_, 0);
if (lean_obj_tag(v_pre_2722_) == 0)
{
lean_object* v_str_2723_; lean_object* v___x_2724_; uint8_t v___x_2725_; 
v_str_2723_ = lean_ctor_get(v_x_2721_, 1);
v___x_2724_ = ((lean_object*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___closed__0));
v___x_2725_ = lean_string_dec_eq(v_str_2723_, v___x_2724_);
if (v___x_2725_ == 0)
{
return v___y_2719_;
}
else
{
return v_suppressElabErrors_2720_;
}
}
else
{
return v___y_2719_;
}
}
else
{
return v___y_2719_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___boxed(lean_object* v___y_2726_, lean_object* v_suppressElabErrors_2727_, lean_object* v_x_2728_){
_start:
{
uint8_t v___y_12835__boxed_2729_; uint8_t v_suppressElabErrors_boxed_2730_; uint8_t v_res_2731_; lean_object* v_r_2732_; 
v___y_12835__boxed_2729_ = lean_unbox(v___y_2726_);
v_suppressElabErrors_boxed_2730_ = lean_unbox(v_suppressElabErrors_2727_);
v_res_2731_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0(v___y_12835__boxed_2729_, v_suppressElabErrors_boxed_2730_, v_x_2728_);
lean_dec(v_x_2728_);
v_r_2732_ = lean_box(v_res_2731_);
return v_r_2732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5(lean_object* v_ref_2733_, lean_object* v_msgData_2734_, uint8_t v_severity_2735_, uint8_t v_isSilent_2736_, lean_object* v___y_2737_, lean_object* v___y_2738_){
_start:
{
lean_object* v___y_2741_; lean_object* v___y_2742_; lean_object* v___y_2743_; uint8_t v___y_2744_; lean_object* v___y_2745_; lean_object* v___y_2746_; uint8_t v___y_2747_; lean_object* v___y_2748_; uint8_t v___y_2805_; uint8_t v___y_2806_; uint8_t v___y_2807_; lean_object* v___y_2808_; lean_object* v___y_2809_; uint8_t v___y_2833_; uint8_t v___y_2834_; lean_object* v___y_2835_; uint8_t v___y_2836_; lean_object* v___y_2837_; uint8_t v___y_2841_; uint8_t v___y_2842_; uint8_t v___y_2843_; uint8_t v___x_2858_; uint8_t v___y_2860_; uint8_t v___y_2861_; uint8_t v___y_2862_; uint8_t v___y_2864_; uint8_t v___x_2876_; 
v___x_2858_ = 2;
v___x_2876_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2735_, v___x_2858_);
if (v___x_2876_ == 0)
{
v___y_2864_ = v___x_2876_;
goto v___jp_2863_;
}
else
{
uint8_t v___x_2877_; 
lean_inc_ref(v_msgData_2734_);
v___x_2877_ = l_Lean_MessageData_hasSyntheticSorry(v_msgData_2734_);
v___y_2864_ = v___x_2877_;
goto v___jp_2863_;
}
v___jp_2740_:
{
lean_object* v___x_2749_; 
v___x_2749_ = l_Lean_Elab_Command_getScope___redArg(v___y_2748_);
if (lean_obj_tag(v___x_2749_) == 0)
{
lean_object* v_a_2750_; lean_object* v___x_2751_; 
v_a_2750_ = lean_ctor_get(v___x_2749_, 0);
lean_inc(v_a_2750_);
lean_dec_ref_known(v___x_2749_, 1);
v___x_2751_ = l_Lean_Elab_Command_getScope___redArg(v___y_2748_);
if (lean_obj_tag(v___x_2751_) == 0)
{
lean_object* v_a_2752_; lean_object* v___x_2754_; uint8_t v_isShared_2755_; uint8_t v_isSharedCheck_2787_; 
v_a_2752_ = lean_ctor_get(v___x_2751_, 0);
v_isSharedCheck_2787_ = !lean_is_exclusive(v___x_2751_);
if (v_isSharedCheck_2787_ == 0)
{
v___x_2754_ = v___x_2751_;
v_isShared_2755_ = v_isSharedCheck_2787_;
goto v_resetjp_2753_;
}
else
{
lean_inc(v_a_2752_);
lean_dec(v___x_2751_);
v___x_2754_ = lean_box(0);
v_isShared_2755_ = v_isSharedCheck_2787_;
goto v_resetjp_2753_;
}
v_resetjp_2753_:
{
lean_object* v___x_2756_; lean_object* v_currNamespace_2757_; lean_object* v_openDecls_2758_; lean_object* v_env_2759_; lean_object* v_messages_2760_; lean_object* v_scopes_2761_; lean_object* v_usedQuotCtxts_2762_; lean_object* v_nextMacroScope_2763_; lean_object* v_maxRecDepth_2764_; lean_object* v_ngen_2765_; lean_object* v_auxDeclNGen_2766_; lean_object* v_infoState_2767_; lean_object* v_traceState_2768_; lean_object* v_snapshotTasks_2769_; lean_object* v_prevLinterStates_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2786_; 
v___x_2756_ = lean_st_ref_take(v___y_2748_);
v_currNamespace_2757_ = lean_ctor_get(v_a_2750_, 2);
lean_inc(v_currNamespace_2757_);
lean_dec(v_a_2750_);
v_openDecls_2758_ = lean_ctor_get(v_a_2752_, 3);
lean_inc(v_openDecls_2758_);
lean_dec(v_a_2752_);
v_env_2759_ = lean_ctor_get(v___x_2756_, 0);
v_messages_2760_ = lean_ctor_get(v___x_2756_, 1);
v_scopes_2761_ = lean_ctor_get(v___x_2756_, 2);
v_usedQuotCtxts_2762_ = lean_ctor_get(v___x_2756_, 3);
v_nextMacroScope_2763_ = lean_ctor_get(v___x_2756_, 4);
v_maxRecDepth_2764_ = lean_ctor_get(v___x_2756_, 5);
v_ngen_2765_ = lean_ctor_get(v___x_2756_, 6);
v_auxDeclNGen_2766_ = lean_ctor_get(v___x_2756_, 7);
v_infoState_2767_ = lean_ctor_get(v___x_2756_, 8);
v_traceState_2768_ = lean_ctor_get(v___x_2756_, 9);
v_snapshotTasks_2769_ = lean_ctor_get(v___x_2756_, 10);
v_prevLinterStates_2770_ = lean_ctor_get(v___x_2756_, 11);
v_isSharedCheck_2786_ = !lean_is_exclusive(v___x_2756_);
if (v_isSharedCheck_2786_ == 0)
{
v___x_2772_ = v___x_2756_;
v_isShared_2773_ = v_isSharedCheck_2786_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_prevLinterStates_2770_);
lean_inc(v_snapshotTasks_2769_);
lean_inc(v_traceState_2768_);
lean_inc(v_infoState_2767_);
lean_inc(v_auxDeclNGen_2766_);
lean_inc(v_ngen_2765_);
lean_inc(v_maxRecDepth_2764_);
lean_inc(v_nextMacroScope_2763_);
lean_inc(v_usedQuotCtxts_2762_);
lean_inc(v_scopes_2761_);
lean_inc(v_messages_2760_);
lean_inc(v_env_2759_);
lean_dec(v___x_2756_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2786_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v___x_2774_; lean_object* v___x_2775_; lean_object* v___x_2776_; lean_object* v___x_2777_; lean_object* v___x_2779_; 
v___x_2774_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2774_, 0, v_currNamespace_2757_);
lean_ctor_set(v___x_2774_, 1, v_openDecls_2758_);
v___x_2775_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_2775_, 0, v___x_2774_);
lean_ctor_set(v___x_2775_, 1, v___y_2745_);
lean_inc_ref(v___y_2746_);
lean_inc_ref(v___y_2743_);
v___x_2776_ = lean_alloc_ctor(0, 5, 3);
lean_ctor_set(v___x_2776_, 0, v___y_2743_);
lean_ctor_set(v___x_2776_, 1, v___y_2742_);
lean_ctor_set(v___x_2776_, 2, v___y_2741_);
lean_ctor_set(v___x_2776_, 3, v___y_2746_);
lean_ctor_set(v___x_2776_, 4, v___x_2775_);
lean_ctor_set_uint8(v___x_2776_, sizeof(void*)*5, v___y_2744_);
lean_ctor_set_uint8(v___x_2776_, sizeof(void*)*5 + 1, v___y_2747_);
lean_ctor_set_uint8(v___x_2776_, sizeof(void*)*5 + 2, v_isSilent_2736_);
v___x_2777_ = l_Lean_MessageLog_add(v___x_2776_, v_messages_2760_);
if (v_isShared_2773_ == 0)
{
lean_ctor_set(v___x_2772_, 1, v___x_2777_);
v___x_2779_ = v___x_2772_;
goto v_reusejp_2778_;
}
else
{
lean_object* v_reuseFailAlloc_2785_; 
v_reuseFailAlloc_2785_ = lean_alloc_ctor(0, 12, 0);
lean_ctor_set(v_reuseFailAlloc_2785_, 0, v_env_2759_);
lean_ctor_set(v_reuseFailAlloc_2785_, 1, v___x_2777_);
lean_ctor_set(v_reuseFailAlloc_2785_, 2, v_scopes_2761_);
lean_ctor_set(v_reuseFailAlloc_2785_, 3, v_usedQuotCtxts_2762_);
lean_ctor_set(v_reuseFailAlloc_2785_, 4, v_nextMacroScope_2763_);
lean_ctor_set(v_reuseFailAlloc_2785_, 5, v_maxRecDepth_2764_);
lean_ctor_set(v_reuseFailAlloc_2785_, 6, v_ngen_2765_);
lean_ctor_set(v_reuseFailAlloc_2785_, 7, v_auxDeclNGen_2766_);
lean_ctor_set(v_reuseFailAlloc_2785_, 8, v_infoState_2767_);
lean_ctor_set(v_reuseFailAlloc_2785_, 9, v_traceState_2768_);
lean_ctor_set(v_reuseFailAlloc_2785_, 10, v_snapshotTasks_2769_);
lean_ctor_set(v_reuseFailAlloc_2785_, 11, v_prevLinterStates_2770_);
v___x_2779_ = v_reuseFailAlloc_2785_;
goto v_reusejp_2778_;
}
v_reusejp_2778_:
{
lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2783_; 
v___x_2780_ = lean_st_ref_set(v___y_2748_, v___x_2779_);
v___x_2781_ = lean_box(0);
if (v_isShared_2755_ == 0)
{
lean_ctor_set(v___x_2754_, 0, v___x_2781_);
v___x_2783_ = v___x_2754_;
goto v_reusejp_2782_;
}
else
{
lean_object* v_reuseFailAlloc_2784_; 
v_reuseFailAlloc_2784_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2784_, 0, v___x_2781_);
v___x_2783_ = v_reuseFailAlloc_2784_;
goto v_reusejp_2782_;
}
v_reusejp_2782_:
{
return v___x_2783_;
}
}
}
}
}
else
{
lean_object* v_a_2788_; lean_object* v___x_2790_; uint8_t v_isShared_2791_; uint8_t v_isSharedCheck_2795_; 
lean_dec(v_a_2750_);
lean_dec_ref(v___y_2745_);
lean_dec_ref(v___y_2742_);
lean_dec(v___y_2741_);
v_a_2788_ = lean_ctor_get(v___x_2751_, 0);
v_isSharedCheck_2795_ = !lean_is_exclusive(v___x_2751_);
if (v_isSharedCheck_2795_ == 0)
{
v___x_2790_ = v___x_2751_;
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
else
{
lean_inc(v_a_2788_);
lean_dec(v___x_2751_);
v___x_2790_ = lean_box(0);
v_isShared_2791_ = v_isSharedCheck_2795_;
goto v_resetjp_2789_;
}
v_resetjp_2789_:
{
lean_object* v___x_2793_; 
if (v_isShared_2791_ == 0)
{
v___x_2793_ = v___x_2790_;
goto v_reusejp_2792_;
}
else
{
lean_object* v_reuseFailAlloc_2794_; 
v_reuseFailAlloc_2794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2794_, 0, v_a_2788_);
v___x_2793_ = v_reuseFailAlloc_2794_;
goto v_reusejp_2792_;
}
v_reusejp_2792_:
{
return v___x_2793_;
}
}
}
}
else
{
lean_object* v_a_2796_; lean_object* v___x_2798_; uint8_t v_isShared_2799_; uint8_t v_isSharedCheck_2803_; 
lean_dec_ref(v___y_2745_);
lean_dec_ref(v___y_2742_);
lean_dec(v___y_2741_);
v_a_2796_ = lean_ctor_get(v___x_2749_, 0);
v_isSharedCheck_2803_ = !lean_is_exclusive(v___x_2749_);
if (v_isSharedCheck_2803_ == 0)
{
v___x_2798_ = v___x_2749_;
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
else
{
lean_inc(v_a_2796_);
lean_dec(v___x_2749_);
v___x_2798_ = lean_box(0);
v_isShared_2799_ = v_isSharedCheck_2803_;
goto v_resetjp_2797_;
}
v_resetjp_2797_:
{
lean_object* v___x_2801_; 
if (v_isShared_2799_ == 0)
{
v___x_2801_ = v___x_2798_;
goto v_reusejp_2800_;
}
else
{
lean_object* v_reuseFailAlloc_2802_; 
v_reuseFailAlloc_2802_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2802_, 0, v_a_2796_);
v___x_2801_ = v_reuseFailAlloc_2802_;
goto v_reusejp_2800_;
}
v_reusejp_2800_:
{
return v___x_2801_;
}
}
}
}
v___jp_2804_:
{
lean_object* v_fileName_2810_; lean_object* v_fileMap_2811_; uint8_t v_suppressElabErrors_2812_; lean_object* v___x_2813_; lean_object* v___x_2814_; lean_object* v_a_2815_; lean_object* v___x_2817_; uint8_t v_isShared_2818_; uint8_t v_isSharedCheck_2831_; 
v_fileName_2810_ = lean_ctor_get(v___y_2737_, 0);
v_fileMap_2811_ = lean_ctor_get(v___y_2737_, 1);
v_suppressElabErrors_2812_ = lean_ctor_get_uint8(v___y_2737_, sizeof(void*)*10);
v___x_2813_ = l___private_Lean_Log_0__Lean_MessageData_appendDescriptionWidgetIfNamed(v_msgData_2734_);
v___x_2814_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(v___x_2813_, v___y_2738_);
v_a_2815_ = lean_ctor_get(v___x_2814_, 0);
v_isSharedCheck_2831_ = !lean_is_exclusive(v___x_2814_);
if (v_isSharedCheck_2831_ == 0)
{
v___x_2817_ = v___x_2814_;
v_isShared_2818_ = v_isSharedCheck_2831_;
goto v_resetjp_2816_;
}
else
{
lean_inc(v_a_2815_);
lean_dec(v___x_2814_);
v___x_2817_ = lean_box(0);
v_isShared_2818_ = v_isSharedCheck_2831_;
goto v_resetjp_2816_;
}
v_resetjp_2816_:
{
lean_object* v___x_2819_; lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; 
lean_inc_ref_n(v_fileMap_2811_, 2);
v___x_2819_ = l_Lean_FileMap_toPosition(v_fileMap_2811_, v___y_2808_);
lean_dec(v___y_2808_);
v___x_2820_ = l_Lean_FileMap_toPosition(v_fileMap_2811_, v___y_2809_);
lean_dec(v___y_2809_);
v___x_2821_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2821_, 0, v___x_2820_);
v___x_2822_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__4));
if (v_suppressElabErrors_2812_ == 0)
{
lean_del_object(v___x_2817_);
v___y_2741_ = v___x_2821_;
v___y_2742_ = v___x_2819_;
v___y_2743_ = v_fileName_2810_;
v___y_2744_ = v___y_2806_;
v___y_2745_ = v_a_2815_;
v___y_2746_ = v___x_2822_;
v___y_2747_ = v___y_2807_;
v___y_2748_ = v___y_2738_;
goto v___jp_2740_;
}
else
{
lean_object* v___x_2823_; lean_object* v___x_2824_; lean_object* v___f_2825_; uint8_t v___x_2826_; 
v___x_2823_ = lean_box(v___y_2805_);
v___x_2824_ = lean_box(v_suppressElabErrors_2812_);
v___f_2825_ = lean_alloc_closure((void*)(lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2825_, 0, v___x_2823_);
lean_closure_set(v___f_2825_, 1, v___x_2824_);
lean_inc(v_a_2815_);
v___x_2826_ = l_Lean_MessageData_hasTag(v___f_2825_, v_a_2815_);
if (v___x_2826_ == 0)
{
lean_object* v___x_2827_; lean_object* v___x_2829_; 
lean_dec_ref_known(v___x_2821_, 1);
lean_dec_ref(v___x_2819_);
lean_dec(v_a_2815_);
v___x_2827_ = lean_box(0);
if (v_isShared_2818_ == 0)
{
lean_ctor_set(v___x_2817_, 0, v___x_2827_);
v___x_2829_ = v___x_2817_;
goto v_reusejp_2828_;
}
else
{
lean_object* v_reuseFailAlloc_2830_; 
v_reuseFailAlloc_2830_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2830_, 0, v___x_2827_);
v___x_2829_ = v_reuseFailAlloc_2830_;
goto v_reusejp_2828_;
}
v_reusejp_2828_:
{
return v___x_2829_;
}
}
else
{
lean_del_object(v___x_2817_);
v___y_2741_ = v___x_2821_;
v___y_2742_ = v___x_2819_;
v___y_2743_ = v_fileName_2810_;
v___y_2744_ = v___y_2806_;
v___y_2745_ = v_a_2815_;
v___y_2746_ = v___x_2822_;
v___y_2747_ = v___y_2807_;
v___y_2748_ = v___y_2738_;
goto v___jp_2740_;
}
}
}
}
v___jp_2832_:
{
lean_object* v___x_2838_; 
v___x_2838_ = l_Lean_Syntax_getTailPos_x3f(v___y_2835_, v___y_2834_);
lean_dec(v___y_2835_);
if (lean_obj_tag(v___x_2838_) == 0)
{
lean_inc(v___y_2837_);
v___y_2805_ = v___y_2833_;
v___y_2806_ = v___y_2834_;
v___y_2807_ = v___y_2836_;
v___y_2808_ = v___y_2837_;
v___y_2809_ = v___y_2837_;
goto v___jp_2804_;
}
else
{
lean_object* v_val_2839_; 
v_val_2839_ = lean_ctor_get(v___x_2838_, 0);
lean_inc(v_val_2839_);
lean_dec_ref_known(v___x_2838_, 1);
v___y_2805_ = v___y_2833_;
v___y_2806_ = v___y_2834_;
v___y_2807_ = v___y_2836_;
v___y_2808_ = v___y_2837_;
v___y_2809_ = v_val_2839_;
goto v___jp_2804_;
}
}
v___jp_2840_:
{
lean_object* v___x_2844_; 
v___x_2844_ = l_Lean_Elab_Command_getRef___redArg(v___y_2737_);
if (lean_obj_tag(v___x_2844_) == 0)
{
lean_object* v_a_2845_; lean_object* v_ref_2846_; lean_object* v___x_2847_; 
v_a_2845_ = lean_ctor_get(v___x_2844_, 0);
lean_inc(v_a_2845_);
lean_dec_ref_known(v___x_2844_, 1);
v_ref_2846_ = l_Lean_replaceRef(v_ref_2733_, v_a_2845_);
lean_dec(v_a_2845_);
v___x_2847_ = l_Lean_Syntax_getPos_x3f(v_ref_2846_, v___y_2842_);
if (lean_obj_tag(v___x_2847_) == 0)
{
lean_object* v___x_2848_; 
v___x_2848_ = lean_unsigned_to_nat(0u);
v___y_2833_ = v___y_2841_;
v___y_2834_ = v___y_2842_;
v___y_2835_ = v_ref_2846_;
v___y_2836_ = v___y_2843_;
v___y_2837_ = v___x_2848_;
goto v___jp_2832_;
}
else
{
lean_object* v_val_2849_; 
v_val_2849_ = lean_ctor_get(v___x_2847_, 0);
lean_inc(v_val_2849_);
lean_dec_ref_known(v___x_2847_, 1);
v___y_2833_ = v___y_2841_;
v___y_2834_ = v___y_2842_;
v___y_2835_ = v_ref_2846_;
v___y_2836_ = v___y_2843_;
v___y_2837_ = v_val_2849_;
goto v___jp_2832_;
}
}
else
{
lean_object* v_a_2850_; lean_object* v___x_2852_; uint8_t v_isShared_2853_; uint8_t v_isSharedCheck_2857_; 
lean_dec_ref(v_msgData_2734_);
v_a_2850_ = lean_ctor_get(v___x_2844_, 0);
v_isSharedCheck_2857_ = !lean_is_exclusive(v___x_2844_);
if (v_isSharedCheck_2857_ == 0)
{
v___x_2852_ = v___x_2844_;
v_isShared_2853_ = v_isSharedCheck_2857_;
goto v_resetjp_2851_;
}
else
{
lean_inc(v_a_2850_);
lean_dec(v___x_2844_);
v___x_2852_ = lean_box(0);
v_isShared_2853_ = v_isSharedCheck_2857_;
goto v_resetjp_2851_;
}
v_resetjp_2851_:
{
lean_object* v___x_2855_; 
if (v_isShared_2853_ == 0)
{
v___x_2855_ = v___x_2852_;
goto v_reusejp_2854_;
}
else
{
lean_object* v_reuseFailAlloc_2856_; 
v_reuseFailAlloc_2856_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2856_, 0, v_a_2850_);
v___x_2855_ = v_reuseFailAlloc_2856_;
goto v_reusejp_2854_;
}
v_reusejp_2854_:
{
return v___x_2855_;
}
}
}
}
v___jp_2859_:
{
if (v___y_2862_ == 0)
{
v___y_2841_ = v___y_2860_;
v___y_2842_ = v___y_2861_;
v___y_2843_ = v_severity_2735_;
goto v___jp_2840_;
}
else
{
v___y_2841_ = v___y_2860_;
v___y_2842_ = v___y_2861_;
v___y_2843_ = v___x_2858_;
goto v___jp_2840_;
}
}
v___jp_2863_:
{
if (v___y_2864_ == 0)
{
lean_object* v___x_2865_; lean_object* v_scopes_2866_; lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v_opts_2869_; uint8_t v___x_2870_; uint8_t v___x_2871_; 
v___x_2865_ = lean_st_ref_get(v___y_2738_);
v_scopes_2866_ = lean_ctor_get(v___x_2865_, 2);
lean_inc(v_scopes_2866_);
lean_dec(v___x_2865_);
v___x_2867_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_2868_ = l_List_head_x21___redArg(v___x_2867_, v_scopes_2866_);
lean_dec(v_scopes_2866_);
v_opts_2869_ = lean_ctor_get(v___x_2868_, 1);
lean_inc_ref(v_opts_2869_);
lean_dec(v___x_2868_);
v___x_2870_ = 1;
v___x_2871_ = l_Lean_instBEqMessageSeverity_beq(v_severity_2735_, v___x_2870_);
if (v___x_2871_ == 0)
{
lean_dec_ref(v_opts_2869_);
v___y_2860_ = v___y_2864_;
v___y_2861_ = v___y_2864_;
v___y_2862_ = v___x_2871_;
goto v___jp_2859_;
}
else
{
lean_object* v___x_2872_; uint8_t v___x_2873_; 
v___x_2872_ = l_Lean_warningAsError;
v___x_2873_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_opts_2869_, v___x_2872_);
lean_dec_ref(v_opts_2869_);
v___y_2860_ = v___y_2864_;
v___y_2861_ = v___y_2864_;
v___y_2862_ = v___x_2873_;
goto v___jp_2859_;
}
}
else
{
lean_object* v___x_2874_; lean_object* v___x_2875_; 
lean_dec_ref(v_msgData_2734_);
v___x_2874_ = lean_box(0);
v___x_2875_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2875_, 0, v___x_2874_);
return v___x_2875_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5___boxed(lean_object* v_ref_2878_, lean_object* v_msgData_2879_, lean_object* v_severity_2880_, lean_object* v_isSilent_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_){
_start:
{
uint8_t v_severity_boxed_2885_; uint8_t v_isSilent_boxed_2886_; lean_object* v_res_2887_; 
v_severity_boxed_2885_ = lean_unbox(v_severity_2880_);
v_isSilent_boxed_2886_ = lean_unbox(v_isSilent_2881_);
v_res_2887_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5(v_ref_2878_, v_msgData_2879_, v_severity_boxed_2885_, v_isSilent_boxed_2886_, v___y_2882_, v___y_2883_);
lean_dec(v___y_2883_);
lean_dec_ref(v___y_2882_);
lean_dec(v_ref_2878_);
return v_res_2887_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4(lean_object* v_msgData_2888_, uint8_t v_severity_2889_, uint8_t v_isSilent_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_){
_start:
{
lean_object* v___x_2894_; 
v___x_2894_ = l_Lean_Elab_Command_getRef___redArg(v___y_2891_);
if (lean_obj_tag(v___x_2894_) == 0)
{
lean_object* v_a_2895_; lean_object* v___x_2896_; 
v_a_2895_ = lean_ctor_get(v___x_2894_, 0);
lean_inc(v_a_2895_);
lean_dec_ref_known(v___x_2894_, 1);
v___x_2896_ = lp_mathlib_Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5(v_a_2895_, v_msgData_2888_, v_severity_2889_, v_isSilent_2890_, v___y_2891_, v___y_2892_);
lean_dec(v_a_2895_);
return v___x_2896_;
}
else
{
lean_object* v_a_2897_; lean_object* v___x_2899_; uint8_t v_isShared_2900_; uint8_t v_isSharedCheck_2904_; 
lean_dec_ref(v_msgData_2888_);
v_a_2897_ = lean_ctor_get(v___x_2894_, 0);
v_isSharedCheck_2904_ = !lean_is_exclusive(v___x_2894_);
if (v_isSharedCheck_2904_ == 0)
{
v___x_2899_ = v___x_2894_;
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
else
{
lean_inc(v_a_2897_);
lean_dec(v___x_2894_);
v___x_2899_ = lean_box(0);
v_isShared_2900_ = v_isSharedCheck_2904_;
goto v_resetjp_2898_;
}
v_resetjp_2898_:
{
lean_object* v___x_2902_; 
if (v_isShared_2900_ == 0)
{
v___x_2902_ = v___x_2899_;
goto v_reusejp_2901_;
}
else
{
lean_object* v_reuseFailAlloc_2903_; 
v_reuseFailAlloc_2903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2903_, 0, v_a_2897_);
v___x_2902_ = v_reuseFailAlloc_2903_;
goto v_reusejp_2901_;
}
v_reusejp_2901_:
{
return v___x_2902_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4___boxed(lean_object* v_msgData_2905_, lean_object* v_severity_2906_, lean_object* v_isSilent_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_){
_start:
{
uint8_t v_severity_boxed_2911_; uint8_t v_isSilent_boxed_2912_; lean_object* v_res_2913_; 
v_severity_boxed_2911_ = lean_unbox(v_severity_2906_);
v_isSilent_boxed_2912_ = lean_unbox(v_isSilent_2907_);
v_res_2913_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4(v_msgData_2905_, v_severity_boxed_2911_, v_isSilent_boxed_2912_, v___y_2908_, v___y_2909_);
lean_dec(v___y_2909_);
lean_dec_ref(v___y_2908_);
return v_res_2913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(lean_object* v_msgData_2914_, lean_object* v___y_2915_, lean_object* v___y_2916_){
_start:
{
uint8_t v___x_2918_; uint8_t v___x_2919_; lean_object* v___x_2920_; 
v___x_2918_ = 0;
v___x_2919_ = 0;
v___x_2920_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4(v_msgData_2914_, v___x_2918_, v___x_2919_, v___y_2915_, v___y_2916_);
return v___x_2920_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4___boxed(lean_object* v_msgData_2921_, lean_object* v___y_2922_, lean_object* v___y_2923_, lean_object* v___y_2924_){
_start:
{
lean_object* v_res_2925_; 
v_res_2925_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(v_msgData_2921_, v___y_2922_, v___y_2923_);
lean_dec(v___y_2923_);
lean_dec_ref(v___y_2922_);
return v_res_2925_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(lean_object* v_t_2926_, lean_object* v_k_2927_, lean_object* v_fallback_2928_){
_start:
{
if (lean_obj_tag(v_t_2926_) == 0)
{
lean_object* v_k_2929_; lean_object* v_v_2930_; lean_object* v_l_2931_; lean_object* v_r_2932_; uint8_t v___x_2933_; 
v_k_2929_ = lean_ctor_get(v_t_2926_, 1);
v_v_2930_ = lean_ctor_get(v_t_2926_, 2);
v_l_2931_ = lean_ctor_get(v_t_2926_, 3);
v_r_2932_ = lean_ctor_get(v_t_2926_, 4);
v___x_2933_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_2927_, v_k_2929_);
switch(v___x_2933_)
{
case 0:
{
v_t_2926_ = v_l_2931_;
goto _start;
}
case 1:
{
lean_inc(v_v_2930_);
return v_v_2930_;
}
default: 
{
v_t_2926_ = v_r_2932_;
goto _start;
}
}
}
else
{
lean_inc(v_fallback_2928_);
return v_fallback_2928_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg___boxed(lean_object* v_t_2936_, lean_object* v_k_2937_, lean_object* v_fallback_2938_){
_start:
{
lean_object* v_res_2939_; 
v_res_2939_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(v_t_2936_, v_k_2937_, v_fallback_2938_);
lean_dec(v_fallback_2938_);
lean_dec(v_k_2937_);
lean_dec(v_t_2936_);
return v_res_2939_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__18(lean_object* v_a_2940_, lean_object* v_a_2941_){
_start:
{
if (lean_obj_tag(v_a_2940_) == 0)
{
lean_object* v___x_2942_; 
v___x_2942_ = l_List_reverse___redArg(v_a_2941_);
return v___x_2942_;
}
else
{
lean_object* v_head_2943_; lean_object* v_tail_2944_; lean_object* v___x_2946_; uint8_t v_isShared_2947_; uint8_t v_isSharedCheck_2953_; 
v_head_2943_ = lean_ctor_get(v_a_2940_, 0);
v_tail_2944_ = lean_ctor_get(v_a_2940_, 1);
v_isSharedCheck_2953_ = !lean_is_exclusive(v_a_2940_);
if (v_isSharedCheck_2953_ == 0)
{
v___x_2946_ = v_a_2940_;
v_isShared_2947_ = v_isSharedCheck_2953_;
goto v_resetjp_2945_;
}
else
{
lean_inc(v_tail_2944_);
lean_inc(v_head_2943_);
lean_dec(v_a_2940_);
v___x_2946_ = lean_box(0);
v_isShared_2947_ = v_isSharedCheck_2953_;
goto v_resetjp_2945_;
}
v_resetjp_2945_:
{
lean_object* v_fst_2948_; lean_object* v___x_2950_; 
v_fst_2948_ = lean_ctor_get(v_head_2943_, 0);
lean_inc(v_fst_2948_);
lean_dec(v_head_2943_);
if (v_isShared_2947_ == 0)
{
lean_ctor_set(v___x_2946_, 1, v_a_2941_);
lean_ctor_set(v___x_2946_, 0, v_fst_2948_);
v___x_2950_ = v___x_2946_;
goto v_reusejp_2949_;
}
else
{
lean_object* v_reuseFailAlloc_2952_; 
v_reuseFailAlloc_2952_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2952_, 0, v_fst_2948_);
lean_ctor_set(v_reuseFailAlloc_2952_, 1, v_a_2941_);
v___x_2950_ = v_reuseFailAlloc_2952_;
goto v_reusejp_2949_;
}
v_reusejp_2949_:
{
v_a_2940_ = v_tail_2944_;
v_a_2941_ = v___x_2950_;
goto _start;
}
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1(void){
_start:
{
lean_object* v___x_2955_; lean_object* v___x_2956_; 
v___x_2955_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__0));
v___x_2956_ = l_Lean_stringToMessageData(v___x_2955_);
return v___x_2956_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3(void){
_start:
{
lean_object* v___x_2958_; lean_object* v___x_2959_; 
v___x_2958_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__2));
v___x_2959_ = l_Lean_stringToMessageData(v___x_2958_);
return v___x_2959_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5(void){
_start:
{
lean_object* v___x_2961_; lean_object* v___x_2962_; 
v___x_2961_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__4));
v___x_2962_ = l_Lean_stringToMessageData(v___x_2961_);
return v___x_2962_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7(void){
_start:
{
lean_object* v___x_2964_; lean_object* v___x_2965_; 
v___x_2964_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__6));
v___x_2965_ = l_Lean_stringToMessageData(v___x_2964_);
return v___x_2965_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9(void){
_start:
{
lean_object* v___x_2967_; lean_object* v___x_2968_; 
v___x_2967_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__8));
v___x_2968_ = l_Lean_stringToMessageData(v___x_2967_);
return v___x_2968_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11(void){
_start:
{
lean_object* v___x_2970_; lean_object* v___x_2971_; 
v___x_2970_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__10));
v___x_2971_ = l_Lean_stringToMessageData(v___x_2970_);
return v___x_2971_;
}
}
static lean_object* _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13(void){
_start:
{
lean_object* v___x_2973_; lean_object* v___x_2974_; 
v___x_2973_ = ((lean_object*)(lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__12));
v___x_2974_ = l_Lean_stringToMessageData(v___x_2973_);
return v___x_2974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg(lean_object* v_msg_2975_, lean_object* v_declHint_2976_, lean_object* v___y_2977_){
_start:
{
lean_object* v___x_2979_; lean_object* v_env_2980_; uint8_t v___x_2981_; 
v___x_2979_ = lean_st_ref_get(v___y_2977_);
v_env_2980_ = lean_ctor_get(v___x_2979_, 0);
lean_inc_ref(v_env_2980_);
lean_dec(v___x_2979_);
v___x_2981_ = l_Lean_Name_isAnonymous(v_declHint_2976_);
if (v___x_2981_ == 0)
{
uint8_t v_isExporting_2982_; 
v_isExporting_2982_ = lean_ctor_get_uint8(v_env_2980_, sizeof(void*)*8);
if (v_isExporting_2982_ == 0)
{
lean_object* v___x_2983_; 
lean_dec_ref(v_env_2980_);
lean_dec(v_declHint_2976_);
v___x_2983_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2983_, 0, v_msg_2975_);
return v___x_2983_;
}
else
{
lean_object* v___x_2984_; uint8_t v___x_2985_; 
lean_inc_ref(v_env_2980_);
v___x_2984_ = l_Lean_Environment_setExporting(v_env_2980_, v___x_2981_);
lean_inc(v_declHint_2976_);
lean_inc_ref(v___x_2984_);
v___x_2985_ = l_Lean_Environment_contains(v___x_2984_, v_declHint_2976_, v_isExporting_2982_);
if (v___x_2985_ == 0)
{
lean_object* v___x_2986_; 
lean_dec_ref(v___x_2984_);
lean_dec_ref(v_env_2980_);
lean_dec(v_declHint_2976_);
v___x_2986_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2986_, 0, v_msg_2975_);
return v___x_2986_;
}
else
{
lean_object* v___x_2987_; lean_object* v___x_2988_; lean_object* v___x_2989_; lean_object* v___x_2990_; lean_object* v___x_2991_; lean_object* v_c_2992_; lean_object* v___x_2993_; 
v___x_2987_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__2);
v___x_2988_ = lean_obj_once(&lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5, &lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5_once, _init_lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg___closed__5);
v___x_2989_ = l_Lean_Options_empty;
v___x_2990_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2990_, 0, v___x_2984_);
lean_ctor_set(v___x_2990_, 1, v___x_2987_);
lean_ctor_set(v___x_2990_, 2, v___x_2988_);
lean_ctor_set(v___x_2990_, 3, v___x_2989_);
lean_inc(v_declHint_2976_);
v___x_2991_ = l_Lean_MessageData_ofConstName(v_declHint_2976_, v___x_2981_);
v_c_2992_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v_c_2992_, 0, v___x_2990_);
lean_ctor_set(v_c_2992_, 1, v___x_2991_);
v___x_2993_ = l_Lean_Environment_getModuleIdxFor_x3f(v_env_2980_, v_declHint_2976_);
if (lean_obj_tag(v___x_2993_) == 0)
{
lean_object* v___x_2994_; lean_object* v___x_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v___x_2998_; lean_object* v___x_2999_; lean_object* v___x_3000_; 
lean_dec_ref(v_env_2980_);
lean_dec(v_declHint_2976_);
v___x_2994_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1);
v___x_2995_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2995_, 0, v___x_2994_);
lean_ctor_set(v___x_2995_, 1, v_c_2992_);
v___x_2996_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__3);
v___x_2997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2997_, 0, v___x_2995_);
lean_ctor_set(v___x_2997_, 1, v___x_2996_);
v___x_2998_ = l_Lean_MessageData_note(v___x_2997_);
v___x_2999_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2999_, 0, v_msg_2975_);
lean_ctor_set(v___x_2999_, 1, v___x_2998_);
v___x_3000_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3000_, 0, v___x_2999_);
return v___x_3000_;
}
else
{
lean_object* v_val_3001_; lean_object* v___x_3003_; uint8_t v_isShared_3004_; uint8_t v_isSharedCheck_3036_; 
v_val_3001_ = lean_ctor_get(v___x_2993_, 0);
v_isSharedCheck_3036_ = !lean_is_exclusive(v___x_2993_);
if (v_isSharedCheck_3036_ == 0)
{
v___x_3003_ = v___x_2993_;
v_isShared_3004_ = v_isSharedCheck_3036_;
goto v_resetjp_3002_;
}
else
{
lean_inc(v_val_3001_);
lean_dec(v___x_2993_);
v___x_3003_ = lean_box(0);
v_isShared_3004_ = v_isSharedCheck_3036_;
goto v_resetjp_3002_;
}
v_resetjp_3002_:
{
lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v___x_3007_; lean_object* v_mod_3008_; uint8_t v___x_3009_; 
v___x_3005_ = lean_box(0);
v___x_3006_ = l_Lean_Environment_header(v_env_2980_);
lean_dec_ref(v_env_2980_);
v___x_3007_ = l_Lean_EnvironmentHeader_moduleNames(v___x_3006_);
v_mod_3008_ = lean_array_get(v___x_3005_, v___x_3007_, v_val_3001_);
lean_dec(v_val_3001_);
lean_dec_ref(v___x_3007_);
v___x_3009_ = l_Lean_isPrivateName(v_declHint_2976_);
lean_dec(v_declHint_2976_);
if (v___x_3009_ == 0)
{
lean_object* v___x_3010_; lean_object* v___x_3011_; lean_object* v___x_3012_; lean_object* v___x_3013_; lean_object* v___x_3014_; lean_object* v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3021_; 
v___x_3010_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__5);
v___x_3011_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3011_, 0, v___x_3010_);
lean_ctor_set(v___x_3011_, 1, v_c_2992_);
v___x_3012_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__7);
v___x_3013_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3013_, 0, v___x_3011_);
lean_ctor_set(v___x_3013_, 1, v___x_3012_);
v___x_3014_ = l_Lean_MessageData_ofName(v_mod_3008_);
v___x_3015_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3015_, 0, v___x_3013_);
lean_ctor_set(v___x_3015_, 1, v___x_3014_);
v___x_3016_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__9);
v___x_3017_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3017_, 0, v___x_3015_);
lean_ctor_set(v___x_3017_, 1, v___x_3016_);
v___x_3018_ = l_Lean_MessageData_note(v___x_3017_);
v___x_3019_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3019_, 0, v_msg_2975_);
lean_ctor_set(v___x_3019_, 1, v___x_3018_);
if (v_isShared_3004_ == 0)
{
lean_ctor_set_tag(v___x_3003_, 0);
lean_ctor_set(v___x_3003_, 0, v___x_3019_);
v___x_3021_ = v___x_3003_;
goto v_reusejp_3020_;
}
else
{
lean_object* v_reuseFailAlloc_3022_; 
v_reuseFailAlloc_3022_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3022_, 0, v___x_3019_);
v___x_3021_ = v_reuseFailAlloc_3022_;
goto v_reusejp_3020_;
}
v_reusejp_3020_:
{
return v___x_3021_;
}
}
else
{
lean_object* v___x_3023_; lean_object* v___x_3024_; lean_object* v___x_3025_; lean_object* v___x_3026_; lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; lean_object* v___x_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3034_; 
v___x_3023_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__1);
v___x_3024_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3024_, 0, v___x_3023_);
lean_ctor_set(v___x_3024_, 1, v_c_2992_);
v___x_3025_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__11);
v___x_3026_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3026_, 0, v___x_3024_);
lean_ctor_set(v___x_3026_, 1, v___x_3025_);
v___x_3027_ = l_Lean_MessageData_ofName(v_mod_3008_);
v___x_3028_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3028_, 0, v___x_3026_);
lean_ctor_set(v___x_3028_, 1, v___x_3027_);
v___x_3029_ = lean_obj_once(&lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13, &lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13_once, _init_lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___closed__13);
v___x_3030_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3030_, 0, v___x_3028_);
lean_ctor_set(v___x_3030_, 1, v___x_3029_);
v___x_3031_ = l_Lean_MessageData_note(v___x_3030_);
v___x_3032_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3032_, 0, v_msg_2975_);
lean_ctor_set(v___x_3032_, 1, v___x_3031_);
if (v_isShared_3004_ == 0)
{
lean_ctor_set_tag(v___x_3003_, 0);
lean_ctor_set(v___x_3003_, 0, v___x_3032_);
v___x_3034_ = v___x_3003_;
goto v_reusejp_3033_;
}
else
{
lean_object* v_reuseFailAlloc_3035_; 
v_reuseFailAlloc_3035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3035_, 0, v___x_3032_);
v___x_3034_ = v_reuseFailAlloc_3035_;
goto v_reusejp_3033_;
}
v_reusejp_3033_:
{
return v___x_3034_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_3037_; 
lean_dec_ref(v_env_2980_);
lean_dec(v_declHint_2976_);
v___x_3037_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3037_, 0, v_msg_2975_);
return v___x_3037_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg___boxed(lean_object* v_msg_3038_, lean_object* v_declHint_3039_, lean_object* v___y_3040_, lean_object* v___y_3041_){
_start:
{
lean_object* v_res_3042_; 
v_res_3042_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg(v_msg_3038_, v_declHint_3039_, v___y_3040_);
lean_dec(v___y_3040_);
return v_res_3042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31(lean_object* v_msg_3043_, lean_object* v_declHint_3044_, lean_object* v___y_3045_, lean_object* v___y_3046_){
_start:
{
lean_object* v___x_3048_; lean_object* v_a_3049_; lean_object* v___x_3051_; uint8_t v_isShared_3052_; uint8_t v_isSharedCheck_3058_; 
v___x_3048_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg(v_msg_3043_, v_declHint_3044_, v___y_3046_);
v_a_3049_ = lean_ctor_get(v___x_3048_, 0);
v_isSharedCheck_3058_ = !lean_is_exclusive(v___x_3048_);
if (v_isSharedCheck_3058_ == 0)
{
v___x_3051_ = v___x_3048_;
v_isShared_3052_ = v_isSharedCheck_3058_;
goto v_resetjp_3050_;
}
else
{
lean_inc(v_a_3049_);
lean_dec(v___x_3048_);
v___x_3051_ = lean_box(0);
v_isShared_3052_ = v_isSharedCheck_3058_;
goto v_resetjp_3050_;
}
v_resetjp_3050_:
{
lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3056_; 
v___x_3053_ = l_Lean_unknownIdentifierMessageTag;
v___x_3054_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_3054_, 0, v___x_3053_);
lean_ctor_set(v___x_3054_, 1, v_a_3049_);
if (v_isShared_3052_ == 0)
{
lean_ctor_set(v___x_3051_, 0, v___x_3054_);
v___x_3056_ = v___x_3051_;
goto v_reusejp_3055_;
}
else
{
lean_object* v_reuseFailAlloc_3057_; 
v_reuseFailAlloc_3057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3057_, 0, v___x_3054_);
v___x_3056_ = v_reuseFailAlloc_3057_;
goto v_reusejp_3055_;
}
v_reusejp_3055_:
{
return v___x_3056_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31___boxed(lean_object* v_msg_3059_, lean_object* v_declHint_3060_, lean_object* v___y_3061_, lean_object* v___y_3062_, lean_object* v___y_3063_){
_start:
{
lean_object* v_res_3064_; 
v_res_3064_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31(v_msg_3059_, v_declHint_3060_, v___y_3061_, v___y_3062_);
lean_dec(v___y_3062_);
lean_dec_ref(v___y_3061_);
return v_res_3064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg(lean_object* v_ref_3065_, lean_object* v_msg_3066_, lean_object* v_declHint_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_){
_start:
{
lean_object* v___x_3071_; lean_object* v_a_3072_; lean_object* v___x_3073_; 
v___x_3071_ = lp_mathlib_Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31(v_msg_3066_, v_declHint_3067_, v___y_3068_, v___y_3069_);
v_a_3072_ = lean_ctor_get(v___x_3071_, 0);
lean_inc(v_a_3072_);
lean_dec_ref(v___x_3071_);
v___x_3073_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(v_ref_3065_, v_a_3072_, v___y_3068_, v___y_3069_);
return v___x_3073_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg___boxed(lean_object* v_ref_3074_, lean_object* v_msg_3075_, lean_object* v_declHint_3076_, lean_object* v___y_3077_, lean_object* v___y_3078_, lean_object* v___y_3079_){
_start:
{
lean_object* v_res_3080_; 
v_res_3080_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg(v_ref_3074_, v_msg_3075_, v_declHint_3076_, v___y_3077_, v___y_3078_);
lean_dec(v___y_3078_);
lean_dec_ref(v___y_3077_);
lean_dec(v_ref_3074_);
return v_res_3080_;
}
}
static lean_object* _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1(void){
_start:
{
lean_object* v___x_3082_; lean_object* v___x_3083_; 
v___x_3082_ = ((lean_object*)(lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__0));
v___x_3083_ = l_Lean_stringToMessageData(v___x_3082_);
return v___x_3083_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg(lean_object* v_ref_3084_, lean_object* v_constName_3085_, lean_object* v___y_3086_, lean_object* v___y_3087_){
_start:
{
lean_object* v___x_3089_; uint8_t v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3092_; lean_object* v___x_3093_; lean_object* v___x_3094_; lean_object* v___x_3095_; 
v___x_3089_ = lean_obj_once(&lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1, &lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1_once, _init_lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___closed__1);
v___x_3090_ = 0;
lean_inc(v_constName_3085_);
v___x_3091_ = l_Lean_MessageData_ofConstName(v_constName_3085_, v___x_3090_);
v___x_3092_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3092_, 0, v___x_3089_);
lean_ctor_set(v___x_3092_, 1, v___x_3091_);
v___x_3093_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__0_spec__0___closed__9);
v___x_3094_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3094_, 0, v___x_3092_);
lean_ctor_set(v___x_3094_, 1, v___x_3093_);
v___x_3095_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg(v_ref_3084_, v___x_3094_, v_constName_3085_, v___y_3086_, v___y_3087_);
return v___x_3095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg___boxed(lean_object* v_ref_3096_, lean_object* v_constName_3097_, lean_object* v___y_3098_, lean_object* v___y_3099_, lean_object* v___y_3100_){
_start:
{
lean_object* v_res_3101_; 
v_res_3101_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg(v_ref_3096_, v_constName_3097_, v___y_3098_, v___y_3099_);
lean_dec(v___y_3099_);
lean_dec_ref(v___y_3098_);
lean_dec(v_ref_3096_);
return v_res_3101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__17(lean_object* v_a_3102_, lean_object* v_a_3103_){
_start:
{
if (lean_obj_tag(v_a_3102_) == 0)
{
lean_object* v___x_3104_; 
v___x_3104_ = l_List_reverse___redArg(v_a_3103_);
return v___x_3104_;
}
else
{
lean_object* v_head_3105_; lean_object* v_tail_3106_; lean_object* v___x_3108_; uint8_t v_isShared_3109_; uint8_t v_isSharedCheck_3117_; 
v_head_3105_ = lean_ctor_get(v_a_3102_, 0);
v_tail_3106_ = lean_ctor_get(v_a_3102_, 1);
v_isSharedCheck_3117_ = !lean_is_exclusive(v_a_3102_);
if (v_isSharedCheck_3117_ == 0)
{
v___x_3108_ = v_a_3102_;
v_isShared_3109_ = v_isSharedCheck_3117_;
goto v_resetjp_3107_;
}
else
{
lean_inc(v_tail_3106_);
lean_inc(v_head_3105_);
lean_dec(v_a_3102_);
v___x_3108_ = lean_box(0);
v_isShared_3109_ = v_isSharedCheck_3117_;
goto v_resetjp_3107_;
}
v_resetjp_3107_:
{
lean_object* v_snd_3110_; uint8_t v___x_3111_; 
v_snd_3110_ = lean_ctor_get(v_head_3105_, 1);
v___x_3111_ = l_List_isEmpty___redArg(v_snd_3110_);
if (v___x_3111_ == 0)
{
lean_del_object(v___x_3108_);
lean_dec(v_head_3105_);
v_a_3102_ = v_tail_3106_;
goto _start;
}
else
{
lean_object* v___x_3114_; 
if (v_isShared_3109_ == 0)
{
lean_ctor_set(v___x_3108_, 1, v_a_3103_);
v___x_3114_ = v___x_3108_;
goto v_reusejp_3113_;
}
else
{
lean_object* v_reuseFailAlloc_3116_; 
v_reuseFailAlloc_3116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3116_, 0, v_head_3105_);
lean_ctor_set(v_reuseFailAlloc_3116_, 1, v_a_3103_);
v___x_3114_ = v_reuseFailAlloc_3116_;
goto v_reusejp_3113_;
}
v_reusejp_3113_:
{
v_a_3102_ = v_tail_3106_;
v_a_3103_ = v___x_3114_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9(lean_object* v_n_3118_, lean_object* v_cs_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_){
_start:
{
lean_object* v___x_3123_; lean_object* v_cs_3124_; uint8_t v___x_3128_; 
v___x_3123_ = lean_box(0);
v_cs_3124_ = lp_mathlib_List_filterTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__17(v_cs_3119_, v___x_3123_);
v___x_3128_ = l_List_isEmpty___redArg(v_cs_3124_);
if (v___x_3128_ == 0)
{
lean_dec(v_n_3118_);
goto v___jp_3125_;
}
else
{
lean_object* v___x_3129_; 
lean_dec(v_cs_3124_);
v___x_3129_ = l_Lean_Elab_Command_getRef___redArg(v___y_3120_);
if (lean_obj_tag(v___x_3129_) == 0)
{
lean_object* v_a_3130_; lean_object* v___x_3131_; lean_object* v_a_3132_; lean_object* v___x_3134_; uint8_t v_isShared_3135_; uint8_t v_isSharedCheck_3139_; 
v_a_3130_ = lean_ctor_get(v___x_3129_, 0);
lean_inc(v_a_3130_);
lean_dec_ref_known(v___x_3129_, 1);
v___x_3131_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg(v_a_3130_, v_n_3118_, v___y_3120_, v___y_3121_);
lean_dec(v_a_3130_);
v_a_3132_ = lean_ctor_get(v___x_3131_, 0);
v_isSharedCheck_3139_ = !lean_is_exclusive(v___x_3131_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3134_ = v___x_3131_;
v_isShared_3135_ = v_isSharedCheck_3139_;
goto v_resetjp_3133_;
}
else
{
lean_inc(v_a_3132_);
lean_dec(v___x_3131_);
v___x_3134_ = lean_box(0);
v_isShared_3135_ = v_isSharedCheck_3139_;
goto v_resetjp_3133_;
}
v_resetjp_3133_:
{
lean_object* v___x_3137_; 
if (v_isShared_3135_ == 0)
{
v___x_3137_ = v___x_3134_;
goto v_reusejp_3136_;
}
else
{
lean_object* v_reuseFailAlloc_3138_; 
v_reuseFailAlloc_3138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3138_, 0, v_a_3132_);
v___x_3137_ = v_reuseFailAlloc_3138_;
goto v_reusejp_3136_;
}
v_reusejp_3136_:
{
return v___x_3137_;
}
}
}
else
{
lean_object* v_a_3140_; lean_object* v___x_3142_; uint8_t v_isShared_3143_; uint8_t v_isSharedCheck_3147_; 
lean_dec(v_n_3118_);
v_a_3140_ = lean_ctor_get(v___x_3129_, 0);
v_isSharedCheck_3147_ = !lean_is_exclusive(v___x_3129_);
if (v_isSharedCheck_3147_ == 0)
{
v___x_3142_ = v___x_3129_;
v_isShared_3143_ = v_isSharedCheck_3147_;
goto v_resetjp_3141_;
}
else
{
lean_inc(v_a_3140_);
lean_dec(v___x_3129_);
v___x_3142_ = lean_box(0);
v_isShared_3143_ = v_isSharedCheck_3147_;
goto v_resetjp_3141_;
}
v_resetjp_3141_:
{
lean_object* v___x_3145_; 
if (v_isShared_3143_ == 0)
{
v___x_3145_ = v___x_3142_;
goto v_reusejp_3144_;
}
else
{
lean_object* v_reuseFailAlloc_3146_; 
v_reuseFailAlloc_3146_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3146_, 0, v_a_3140_);
v___x_3145_ = v_reuseFailAlloc_3146_;
goto v_reusejp_3144_;
}
v_reusejp_3144_:
{
return v___x_3145_;
}
}
}
}
v___jp_3125_:
{
lean_object* v___x_3126_; lean_object* v___x_3127_; 
v___x_3126_ = lp_mathlib_List_mapTR_loop___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__18(v_cs_3124_, v___x_3123_);
v___x_3127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3127_, 0, v___x_3126_);
return v___x_3127_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9___boxed(lean_object* v_n_3148_, lean_object* v_cs_3149_, lean_object* v___y_3150_, lean_object* v___y_3151_, lean_object* v___y_3152_){
_start:
{
lean_object* v_res_3153_; 
v_res_3153_ = lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9(v_n_3148_, v_cs_3149_, v___y_3150_, v___y_3151_);
lean_dec(v___y_3151_);
lean_dec_ref(v___y_3150_);
return v_res_3153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14(lean_object* v_x_3154_){
_start:
{
if (lean_obj_tag(v_x_3154_) == 0)
{
lean_object* v___x_3155_; 
v___x_3155_ = lean_box(0);
return v___x_3155_;
}
else
{
lean_object* v_head_3156_; lean_object* v_tail_3157_; lean_object* v_fst_3158_; uint8_t v___x_3159_; 
v_head_3156_ = lean_ctor_get(v_x_3154_, 0);
v_tail_3157_ = lean_ctor_get(v_x_3154_, 1);
v_fst_3158_ = lean_ctor_get(v_head_3156_, 0);
v___x_3159_ = l_Lean_isPrivateName(v_fst_3158_);
if (v___x_3159_ == 0)
{
v_x_3154_ = v_tail_3157_;
goto _start;
}
else
{
lean_object* v___x_3161_; 
lean_inc(v_head_3156_);
v___x_3161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3161_, 0, v_head_3156_);
return v___x_3161_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14___boxed(lean_object* v_x_3162_){
_start:
{
lean_object* v_res_3163_; 
v_res_3163_ = lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14(v_x_3162_);
lean_dec(v_x_3162_);
return v_res_3163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22(lean_object* v_msgData_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_){
_start:
{
uint8_t v___x_3168_; uint8_t v___x_3169_; lean_object* v___x_3170_; 
v___x_3168_ = 1;
v___x_3169_ = 0;
v___x_3170_ = lp_mathlib_Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4(v_msgData_3164_, v___x_3168_, v___x_3169_, v___y_3165_, v___y_3166_);
return v___x_3170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22___boxed(lean_object* v_msgData_3171_, lean_object* v___y_3172_, lean_object* v___y_3173_, lean_object* v___y_3174_){
_start:
{
lean_object* v_res_3175_; 
v_res_3175_ = lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22(v_msgData_3171_, v___y_3172_, v___y_3173_);
lean_dec(v___y_3173_);
lean_dec_ref(v___y_3172_);
return v_res_3175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg(lean_object* v_opt_3176_, lean_object* v___y_3177_){
_start:
{
lean_object* v___x_3179_; lean_object* v_scopes_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v_opts_3183_; uint8_t v___x_3184_; lean_object* v___x_3185_; lean_object* v___x_3186_; 
v___x_3179_ = lean_st_ref_get(v___y_3177_);
v_scopes_3180_ = lean_ctor_get(v___x_3179_, 2);
lean_inc(v_scopes_3180_);
lean_dec(v___x_3179_);
v___x_3181_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3182_ = l_List_head_x21___redArg(v___x_3181_, v_scopes_3180_);
lean_dec(v_scopes_3180_);
v_opts_3183_ = lean_ctor_get(v___x_3182_, 1);
lean_inc_ref(v_opts_3183_);
lean_dec(v___x_3182_);
v___x_3184_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_elabFunPropConfig_evalConfigItem_spec__1_spec__3_spec__4_spec__6(v_opts_3183_, v_opt_3176_);
lean_dec_ref(v_opts_3183_);
v___x_3185_ = lean_box(v___x_3184_);
v___x_3186_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3186_, 0, v___x_3185_);
return v___x_3186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg___boxed(lean_object* v_opt_3187_, lean_object* v___y_3188_, lean_object* v___y_3189_){
_start:
{
lean_object* v_res_3190_; 
v_res_3190_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg(v_opt_3187_, v___y_3188_);
lean_dec(v___y_3188_);
lean_dec_ref(v_opt_3187_);
return v_res_3190_;
}
}
static lean_object* _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1(void){
_start:
{
lean_object* v___x_3192_; lean_object* v___x_3193_; 
v___x_3192_ = ((lean_object*)(lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__0));
v___x_3193_ = l_Lean_stringToMessageData(v___x_3192_);
return v___x_3193_;
}
}
static lean_object* _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3(void){
_start:
{
lean_object* v___x_3195_; lean_object* v___x_3196_; 
v___x_3195_ = ((lean_object*)(lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__2));
v___x_3196_ = l_Lean_stringToMessageData(v___x_3195_);
return v___x_3196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15(lean_object* v_id_3197_, lean_object* v___y_3198_, lean_object* v___y_3199_){
_start:
{
lean_object* v___x_3201_; lean_object* v_env_3202_; lean_object* v___x_3203_; lean_object* v___x_3204_; lean_object* v_a_3205_; lean_object* v___x_3207_; uint8_t v_isShared_3208_; uint8_t v_isSharedCheck_3224_; 
v___x_3201_ = lean_st_ref_get(v___y_3199_);
v_env_3202_ = lean_ctor_get(v___x_3201_, 0);
lean_inc_ref(v_env_3202_);
lean_dec(v___x_3201_);
v___x_3203_ = l_Lean_ResolveName_backward_privateInPublic_warn;
v___x_3204_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg(v___x_3203_, v___y_3199_);
v_a_3205_ = lean_ctor_get(v___x_3204_, 0);
v_isSharedCheck_3224_ = !lean_is_exclusive(v___x_3204_);
if (v_isSharedCheck_3224_ == 0)
{
v___x_3207_ = v___x_3204_;
v_isShared_3208_ = v_isSharedCheck_3224_;
goto v_resetjp_3206_;
}
else
{
lean_inc(v_a_3205_);
lean_dec(v___x_3204_);
v___x_3207_ = lean_box(0);
v_isShared_3208_ = v_isSharedCheck_3224_;
goto v_resetjp_3206_;
}
v_resetjp_3206_:
{
uint8_t v_isExporting_3214_; 
v_isExporting_3214_ = lean_ctor_get_uint8(v_env_3202_, sizeof(void*)*8);
lean_dec_ref(v_env_3202_);
if (v_isExporting_3214_ == 0)
{
lean_dec(v_a_3205_);
lean_dec(v_id_3197_);
goto v___jp_3209_;
}
else
{
uint8_t v___x_3215_; 
v___x_3215_ = l_Lean_isPrivateName(v_id_3197_);
if (v___x_3215_ == 0)
{
lean_dec(v_a_3205_);
lean_dec(v_id_3197_);
goto v___jp_3209_;
}
else
{
uint8_t v___x_3216_; 
v___x_3216_ = lean_unbox(v_a_3205_);
lean_dec(v_a_3205_);
if (v___x_3216_ == 0)
{
lean_dec(v_id_3197_);
goto v___jp_3209_;
}
else
{
lean_object* v___x_3217_; uint8_t v___x_3218_; lean_object* v___x_3219_; lean_object* v___x_3220_; lean_object* v___x_3221_; lean_object* v___x_3222_; lean_object* v___x_3223_; 
lean_del_object(v___x_3207_);
v___x_3217_ = lean_obj_once(&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1, &lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1_once, _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__1);
v___x_3218_ = 0;
v___x_3219_ = l_Lean_MessageData_ofConstName(v_id_3197_, v___x_3218_);
v___x_3220_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3220_, 0, v___x_3217_);
lean_ctor_set(v___x_3220_, 1, v___x_3219_);
v___x_3221_ = lean_obj_once(&lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3, &lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3_once, _init_lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___closed__3);
v___x_3222_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3222_, 0, v___x_3220_);
lean_ctor_set(v___x_3222_, 1, v___x_3221_);
v___x_3223_ = lp_mathlib_Lean_logWarning___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__22(v___x_3222_, v___y_3198_, v___y_3199_);
return v___x_3223_;
}
}
}
v___jp_3209_:
{
lean_object* v___x_3210_; lean_object* v___x_3212_; 
v___x_3210_ = lean_box(0);
if (v_isShared_3208_ == 0)
{
lean_ctor_set(v___x_3207_, 0, v___x_3210_);
v___x_3212_ = v___x_3207_;
goto v_reusejp_3211_;
}
else
{
lean_object* v_reuseFailAlloc_3213_; 
v_reuseFailAlloc_3213_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3213_, 0, v___x_3210_);
v___x_3212_ = v_reuseFailAlloc_3213_;
goto v_reusejp_3211_;
}
v_reusejp_3211_:
{
return v___x_3212_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15___boxed(lean_object* v_id_3225_, lean_object* v___y_3226_, lean_object* v___y_3227_, lean_object* v___y_3228_){
_start:
{
lean_object* v_res_3229_; 
v_res_3229_ = lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15(v_id_3225_, v___y_3226_, v___y_3227_);
lean_dec(v___y_3227_);
lean_dec_ref(v___y_3226_);
return v_res_3229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8(lean_object* v_id_3230_, uint8_t v_enableLog_3231_, lean_object* v___y_3232_, lean_object* v___y_3233_){
_start:
{
lean_object* v___x_3235_; lean_object* v_env_3236_; lean_object* v___x_3237_; lean_object* v_scopes_3238_; lean_object* v___x_3239_; lean_object* v___x_3240_; lean_object* v_opts_3241_; lean_object* v___x_3242_; 
v___x_3235_ = lean_st_ref_get(v___y_3233_);
v_env_3236_ = lean_ctor_get(v___x_3235_, 0);
lean_inc_ref(v_env_3236_);
lean_dec(v___x_3235_);
v___x_3237_ = lean_st_ref_get(v___y_3233_);
v_scopes_3238_ = lean_ctor_get(v___x_3237_, 2);
lean_inc(v_scopes_3238_);
lean_dec(v___x_3237_);
v___x_3239_ = l_Lean_Elab_Command_instInhabitedScope_default;
v___x_3240_ = l_List_head_x21___redArg(v___x_3239_, v_scopes_3238_);
lean_dec(v_scopes_3238_);
v_opts_3241_ = lean_ctor_get(v___x_3240_, 1);
lean_inc_ref(v_opts_3241_);
lean_dec(v___x_3240_);
v___x_3242_ = l_Lean_Elab_Command_getScope___redArg(v___y_3233_);
if (lean_obj_tag(v___x_3242_) == 0)
{
lean_object* v_a_3243_; lean_object* v_currNamespace_3244_; lean_object* v___x_3245_; 
v_a_3243_ = lean_ctor_get(v___x_3242_, 0);
lean_inc(v_a_3243_);
lean_dec_ref_known(v___x_3242_, 1);
v_currNamespace_3244_ = lean_ctor_get(v_a_3243_, 2);
lean_inc(v_currNamespace_3244_);
lean_dec(v_a_3243_);
v___x_3245_ = l_Lean_Elab_Command_getScope___redArg(v___y_3233_);
if (lean_obj_tag(v___x_3245_) == 0)
{
lean_object* v_a_3246_; lean_object* v___x_3248_; uint8_t v_isShared_3249_; uint8_t v_isSharedCheck_3284_; 
v_a_3246_ = lean_ctor_get(v___x_3245_, 0);
v_isSharedCheck_3284_ = !lean_is_exclusive(v___x_3245_);
if (v_isSharedCheck_3284_ == 0)
{
v___x_3248_ = v___x_3245_;
v_isShared_3249_ = v_isSharedCheck_3284_;
goto v_resetjp_3247_;
}
else
{
lean_inc(v_a_3246_);
lean_dec(v___x_3245_);
v___x_3248_ = lean_box(0);
v_isShared_3249_ = v_isSharedCheck_3284_;
goto v_resetjp_3247_;
}
v_resetjp_3247_:
{
lean_object* v_openDecls_3250_; lean_object* v___x_3251_; lean_object* v_env_3252_; lean_object* v_res_3253_; 
v_openDecls_3250_ = lean_ctor_get(v_a_3246_, 3);
lean_inc(v_openDecls_3250_);
lean_dec(v_a_3246_);
v___x_3251_ = lean_st_ref_get(v___y_3233_);
v_env_3252_ = lean_ctor_get(v___x_3251_, 0);
lean_inc_ref(v_env_3252_);
lean_dec(v___x_3251_);
v_res_3253_ = l_Lean_ResolveName_resolveGlobalName(v_env_3236_, v_opts_3241_, v_currNamespace_3244_, v_openDecls_3250_, v_id_3230_);
lean_dec_ref(v_opts_3241_);
if (v_enableLog_3231_ == 0)
{
lean_object* v___x_3255_; 
lean_dec_ref(v_env_3252_);
if (v_isShared_3249_ == 0)
{
lean_ctor_set(v___x_3248_, 0, v_res_3253_);
v___x_3255_ = v___x_3248_;
goto v_reusejp_3254_;
}
else
{
lean_object* v_reuseFailAlloc_3256_; 
v_reuseFailAlloc_3256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3256_, 0, v_res_3253_);
v___x_3255_ = v_reuseFailAlloc_3256_;
goto v_reusejp_3254_;
}
v_reusejp_3254_:
{
return v___x_3255_;
}
}
else
{
uint8_t v_isExporting_3257_; 
v_isExporting_3257_ = lean_ctor_get_uint8(v_env_3252_, sizeof(void*)*8);
lean_dec_ref(v_env_3252_);
if (v_isExporting_3257_ == 0)
{
lean_object* v___x_3259_; 
if (v_isShared_3249_ == 0)
{
lean_ctor_set(v___x_3248_, 0, v_res_3253_);
v___x_3259_ = v___x_3248_;
goto v_reusejp_3258_;
}
else
{
lean_object* v_reuseFailAlloc_3260_; 
v_reuseFailAlloc_3260_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3260_, 0, v_res_3253_);
v___x_3259_ = v_reuseFailAlloc_3260_;
goto v_reusejp_3258_;
}
v_reusejp_3258_:
{
return v___x_3259_;
}
}
else
{
lean_object* v___x_3261_; 
v___x_3261_ = lp_mathlib_List_find_x3f___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__14(v_res_3253_);
if (lean_obj_tag(v___x_3261_) == 1)
{
lean_object* v_val_3262_; lean_object* v_fst_3263_; lean_object* v___x_3264_; 
lean_del_object(v___x_3248_);
v_val_3262_ = lean_ctor_get(v___x_3261_, 0);
lean_inc(v_val_3262_);
lean_dec_ref_known(v___x_3261_, 1);
v_fst_3263_ = lean_ctor_get(v_val_3262_, 0);
lean_inc(v_fst_3263_);
lean_dec(v_val_3262_);
v___x_3264_ = lp_mathlib_Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15(v_fst_3263_, v___y_3232_, v___y_3233_);
if (lean_obj_tag(v___x_3264_) == 0)
{
lean_object* v___x_3266_; uint8_t v_isShared_3267_; uint8_t v_isSharedCheck_3271_; 
v_isSharedCheck_3271_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3271_ == 0)
{
lean_object* v_unused_3272_; 
v_unused_3272_ = lean_ctor_get(v___x_3264_, 0);
lean_dec(v_unused_3272_);
v___x_3266_ = v___x_3264_;
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
else
{
lean_dec(v___x_3264_);
v___x_3266_ = lean_box(0);
v_isShared_3267_ = v_isSharedCheck_3271_;
goto v_resetjp_3265_;
}
v_resetjp_3265_:
{
lean_object* v___x_3269_; 
if (v_isShared_3267_ == 0)
{
lean_ctor_set(v___x_3266_, 0, v_res_3253_);
v___x_3269_ = v___x_3266_;
goto v_reusejp_3268_;
}
else
{
lean_object* v_reuseFailAlloc_3270_; 
v_reuseFailAlloc_3270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3270_, 0, v_res_3253_);
v___x_3269_ = v_reuseFailAlloc_3270_;
goto v_reusejp_3268_;
}
v_reusejp_3268_:
{
return v___x_3269_;
}
}
}
else
{
lean_object* v_a_3273_; lean_object* v___x_3275_; uint8_t v_isShared_3276_; uint8_t v_isSharedCheck_3280_; 
lean_dec(v_res_3253_);
v_a_3273_ = lean_ctor_get(v___x_3264_, 0);
v_isSharedCheck_3280_ = !lean_is_exclusive(v___x_3264_);
if (v_isSharedCheck_3280_ == 0)
{
v___x_3275_ = v___x_3264_;
v_isShared_3276_ = v_isSharedCheck_3280_;
goto v_resetjp_3274_;
}
else
{
lean_inc(v_a_3273_);
lean_dec(v___x_3264_);
v___x_3275_ = lean_box(0);
v_isShared_3276_ = v_isSharedCheck_3280_;
goto v_resetjp_3274_;
}
v_resetjp_3274_:
{
lean_object* v___x_3278_; 
if (v_isShared_3276_ == 0)
{
v___x_3278_ = v___x_3275_;
goto v_reusejp_3277_;
}
else
{
lean_object* v_reuseFailAlloc_3279_; 
v_reuseFailAlloc_3279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3279_, 0, v_a_3273_);
v___x_3278_ = v_reuseFailAlloc_3279_;
goto v_reusejp_3277_;
}
v_reusejp_3277_:
{
return v___x_3278_;
}
}
}
}
else
{
lean_object* v___x_3282_; 
lean_dec(v___x_3261_);
if (v_isShared_3249_ == 0)
{
lean_ctor_set(v___x_3248_, 0, v_res_3253_);
v___x_3282_ = v___x_3248_;
goto v_reusejp_3281_;
}
else
{
lean_object* v_reuseFailAlloc_3283_; 
v_reuseFailAlloc_3283_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3283_, 0, v_res_3253_);
v___x_3282_ = v_reuseFailAlloc_3283_;
goto v_reusejp_3281_;
}
v_reusejp_3281_:
{
return v___x_3282_;
}
}
}
}
}
}
else
{
lean_object* v_a_3285_; lean_object* v___x_3287_; uint8_t v_isShared_3288_; uint8_t v_isSharedCheck_3292_; 
lean_dec(v_currNamespace_3244_);
lean_dec_ref(v_opts_3241_);
lean_dec_ref(v_env_3236_);
lean_dec(v_id_3230_);
v_a_3285_ = lean_ctor_get(v___x_3245_, 0);
v_isSharedCheck_3292_ = !lean_is_exclusive(v___x_3245_);
if (v_isSharedCheck_3292_ == 0)
{
v___x_3287_ = v___x_3245_;
v_isShared_3288_ = v_isSharedCheck_3292_;
goto v_resetjp_3286_;
}
else
{
lean_inc(v_a_3285_);
lean_dec(v___x_3245_);
v___x_3287_ = lean_box(0);
v_isShared_3288_ = v_isSharedCheck_3292_;
goto v_resetjp_3286_;
}
v_resetjp_3286_:
{
lean_object* v___x_3290_; 
if (v_isShared_3288_ == 0)
{
v___x_3290_ = v___x_3287_;
goto v_reusejp_3289_;
}
else
{
lean_object* v_reuseFailAlloc_3291_; 
v_reuseFailAlloc_3291_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3291_, 0, v_a_3285_);
v___x_3290_ = v_reuseFailAlloc_3291_;
goto v_reusejp_3289_;
}
v_reusejp_3289_:
{
return v___x_3290_;
}
}
}
}
else
{
lean_object* v_a_3293_; lean_object* v___x_3295_; uint8_t v_isShared_3296_; uint8_t v_isSharedCheck_3300_; 
lean_dec_ref(v_opts_3241_);
lean_dec_ref(v_env_3236_);
lean_dec(v_id_3230_);
v_a_3293_ = lean_ctor_get(v___x_3242_, 0);
v_isSharedCheck_3300_ = !lean_is_exclusive(v___x_3242_);
if (v_isSharedCheck_3300_ == 0)
{
v___x_3295_ = v___x_3242_;
v_isShared_3296_ = v_isSharedCheck_3300_;
goto v_resetjp_3294_;
}
else
{
lean_inc(v_a_3293_);
lean_dec(v___x_3242_);
v___x_3295_ = lean_box(0);
v_isShared_3296_ = v_isSharedCheck_3300_;
goto v_resetjp_3294_;
}
v_resetjp_3294_:
{
lean_object* v___x_3298_; 
if (v_isShared_3296_ == 0)
{
v___x_3298_ = v___x_3295_;
goto v_reusejp_3297_;
}
else
{
lean_object* v_reuseFailAlloc_3299_; 
v_reuseFailAlloc_3299_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3299_, 0, v_a_3293_);
v___x_3298_ = v_reuseFailAlloc_3299_;
goto v_reusejp_3297_;
}
v_reusejp_3297_:
{
return v___x_3298_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8___boxed(lean_object* v_id_3301_, lean_object* v_enableLog_3302_, lean_object* v___y_3303_, lean_object* v___y_3304_, lean_object* v___y_3305_){
_start:
{
uint8_t v_enableLog_boxed_3306_; lean_object* v_res_3307_; 
v_enableLog_boxed_3306_ = lean_unbox(v_enableLog_3302_);
v_res_3307_ = lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8(v_id_3301_, v_enableLog_boxed_3306_, v___y_3303_, v___y_3304_);
lean_dec(v___y_3304_);
lean_dec_ref(v___y_3303_);
return v_res_3307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6(lean_object* v_n_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_){
_start:
{
uint8_t v___x_3312_; lean_object* v___x_3313_; 
v___x_3312_ = 1;
lean_inc(v_n_3308_);
v___x_3313_ = lp_mathlib_Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8(v_n_3308_, v___x_3312_, v___y_3309_, v___y_3310_);
if (lean_obj_tag(v___x_3313_) == 0)
{
lean_object* v_a_3314_; lean_object* v___x_3315_; 
v_a_3314_ = lean_ctor_get(v___x_3313_, 0);
lean_inc(v_a_3314_);
lean_dec_ref_known(v___x_3313_, 1);
v___x_3315_ = lp_mathlib_Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9(v_n_3308_, v_a_3314_, v___y_3309_, v___y_3310_);
return v___x_3315_;
}
else
{
lean_object* v_a_3316_; lean_object* v___x_3318_; uint8_t v_isShared_3319_; uint8_t v_isSharedCheck_3323_; 
lean_dec(v_n_3308_);
v_a_3316_ = lean_ctor_get(v___x_3313_, 0);
v_isSharedCheck_3323_ = !lean_is_exclusive(v___x_3313_);
if (v_isSharedCheck_3323_ == 0)
{
v___x_3318_ = v___x_3313_;
v_isShared_3319_ = v_isSharedCheck_3323_;
goto v_resetjp_3317_;
}
else
{
lean_inc(v_a_3316_);
lean_dec(v___x_3313_);
v___x_3318_ = lean_box(0);
v_isShared_3319_ = v_isSharedCheck_3323_;
goto v_resetjp_3317_;
}
v_resetjp_3317_:
{
lean_object* v___x_3321_; 
if (v_isShared_3319_ == 0)
{
v___x_3321_ = v___x_3318_;
goto v_reusejp_3320_;
}
else
{
lean_object* v_reuseFailAlloc_3322_; 
v_reuseFailAlloc_3322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3322_, 0, v_a_3316_);
v___x_3321_ = v_reuseFailAlloc_3322_;
goto v_reusejp_3320_;
}
v_reusejp_3320_:
{
return v___x_3321_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6___boxed(lean_object* v_n_3324_, lean_object* v___y_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_){
_start:
{
lean_object* v_res_3328_; 
v_res_3328_ = lp_mathlib___private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6(v_n_3324_, v___y_3325_, v___y_3326_);
lean_dec(v___y_3326_);
lean_dec_ref(v___y_3325_);
return v_res_3328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7_spec__11(lean_object* v_a_3329_, lean_object* v_a_3330_){
_start:
{
if (lean_obj_tag(v_a_3329_) == 0)
{
lean_object* v___x_3331_; 
v___x_3331_ = lean_array_to_list(v_a_3330_);
return v___x_3331_;
}
else
{
lean_object* v_head_3332_; 
v_head_3332_ = lean_ctor_get(v_a_3329_, 0);
if (lean_obj_tag(v_head_3332_) == 1)
{
lean_object* v_fields_3333_; 
v_fields_3333_ = lean_ctor_get(v_head_3332_, 1);
if (lean_obj_tag(v_fields_3333_) == 0)
{
lean_object* v_tail_3334_; lean_object* v_n_3335_; lean_object* v___x_3336_; 
lean_inc_ref(v_head_3332_);
v_tail_3334_ = lean_ctor_get(v_a_3329_, 1);
lean_inc(v_tail_3334_);
lean_dec_ref_known(v_a_3329_, 2);
v_n_3335_ = lean_ctor_get(v_head_3332_, 0);
lean_inc(v_n_3335_);
lean_dec_ref_known(v_head_3332_, 2);
v___x_3336_ = lean_array_push(v_a_3330_, v_n_3335_);
v_a_3329_ = v_tail_3334_;
v_a_3330_ = v___x_3336_;
goto _start;
}
else
{
lean_object* v_tail_3338_; 
v_tail_3338_ = lean_ctor_get(v_a_3329_, 1);
lean_inc(v_tail_3338_);
lean_dec_ref_known(v_a_3329_, 2);
v_a_3329_ = v_tail_3338_;
goto _start;
}
}
else
{
lean_object* v_tail_3340_; 
v_tail_3340_ = lean_ctor_get(v_a_3329_, 1);
lean_inc(v_tail_3340_);
lean_dec_ref_known(v_a_3329_, 2);
v_a_3329_ = v_tail_3340_;
goto _start;
}
}
}
}
static lean_object* _init_lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3(void){
_start:
{
lean_object* v___x_3347_; lean_object* v___x_3348_; 
v___x_3347_ = ((lean_object*)(lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__2));
v___x_3348_ = l_Lean_MessageData_ofFormat(v___x_3347_);
return v___x_3348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7(lean_object* v_stx_3349_, lean_object* v_k_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_){
_start:
{
if (lean_obj_tag(v_stx_3349_) == 3)
{
lean_object* v_val_3354_; lean_object* v_preresolved_3355_; lean_object* v___x_3356_; lean_object* v_pre_3357_; uint8_t v___x_3358_; 
v_val_3354_ = lean_ctor_get(v_stx_3349_, 2);
lean_inc(v_val_3354_);
v_preresolved_3355_ = lean_ctor_get(v_stx_3349_, 3);
v___x_3356_ = ((lean_object*)(lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__0));
lean_inc(v_preresolved_3355_);
v_pre_3357_ = lp_mathlib_List_filterMapTR_go___at___00Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7_spec__11(v_preresolved_3355_, v___x_3356_);
v___x_3358_ = l_List_isEmpty___redArg(v_pre_3357_);
if (v___x_3358_ == 0)
{
lean_object* v___x_3359_; 
lean_dec(v_val_3354_);
lean_dec_ref_known(v_stx_3349_, 4);
lean_dec_ref(v_k_3350_);
v___x_3359_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3359_, 0, v_pre_3357_);
return v___x_3359_;
}
else
{
lean_object* v___x_3360_; 
lean_dec(v_pre_3357_);
v___x_3360_ = l_Lean_Elab_Command_getRef___redArg(v___y_3351_);
if (lean_obj_tag(v___x_3360_) == 0)
{
lean_object* v_a_3361_; lean_object* v_fileName_3362_; lean_object* v_fileMap_3363_; lean_object* v_currRecDepth_3364_; lean_object* v_cmdPos_3365_; lean_object* v_macroStack_3366_; lean_object* v_quotContext_x3f_3367_; lean_object* v_currMacroScope_3368_; lean_object* v_snap_x3f_3369_; lean_object* v_cancelTk_x3f_3370_; uint8_t v_suppressElabErrors_3371_; lean_object* v_ref_3372_; lean_object* v___x_3373_; lean_object* v___x_3374_; 
v_a_3361_ = lean_ctor_get(v___x_3360_, 0);
lean_inc(v_a_3361_);
lean_dec_ref_known(v___x_3360_, 1);
v_fileName_3362_ = lean_ctor_get(v___y_3351_, 0);
v_fileMap_3363_ = lean_ctor_get(v___y_3351_, 1);
v_currRecDepth_3364_ = lean_ctor_get(v___y_3351_, 2);
v_cmdPos_3365_ = lean_ctor_get(v___y_3351_, 3);
v_macroStack_3366_ = lean_ctor_get(v___y_3351_, 4);
v_quotContext_x3f_3367_ = lean_ctor_get(v___y_3351_, 5);
v_currMacroScope_3368_ = lean_ctor_get(v___y_3351_, 6);
v_snap_x3f_3369_ = lean_ctor_get(v___y_3351_, 8);
v_cancelTk_x3f_3370_ = lean_ctor_get(v___y_3351_, 9);
v_suppressElabErrors_3371_ = lean_ctor_get_uint8(v___y_3351_, sizeof(void*)*10);
v_ref_3372_ = l_Lean_replaceRef(v_stx_3349_, v_a_3361_);
lean_dec(v_a_3361_);
lean_dec_ref_known(v_stx_3349_, 4);
lean_inc(v_cancelTk_x3f_3370_);
lean_inc(v_snap_x3f_3369_);
lean_inc(v_currMacroScope_3368_);
lean_inc(v_quotContext_x3f_3367_);
lean_inc(v_macroStack_3366_);
lean_inc(v_cmdPos_3365_);
lean_inc(v_currRecDepth_3364_);
lean_inc_ref(v_fileMap_3363_);
lean_inc_ref(v_fileName_3362_);
v___x_3373_ = lean_alloc_ctor(0, 10, 1);
lean_ctor_set(v___x_3373_, 0, v_fileName_3362_);
lean_ctor_set(v___x_3373_, 1, v_fileMap_3363_);
lean_ctor_set(v___x_3373_, 2, v_currRecDepth_3364_);
lean_ctor_set(v___x_3373_, 3, v_cmdPos_3365_);
lean_ctor_set(v___x_3373_, 4, v_macroStack_3366_);
lean_ctor_set(v___x_3373_, 5, v_quotContext_x3f_3367_);
lean_ctor_set(v___x_3373_, 6, v_currMacroScope_3368_);
lean_ctor_set(v___x_3373_, 7, v_ref_3372_);
lean_ctor_set(v___x_3373_, 8, v_snap_x3f_3369_);
lean_ctor_set(v___x_3373_, 9, v_cancelTk_x3f_3370_);
lean_ctor_set_uint8(v___x_3373_, sizeof(void*)*10, v_suppressElabErrors_3371_);
lean_inc(v___y_3352_);
v___x_3374_ = lean_apply_4(v_k_3350_, v_val_3354_, v___x_3373_, v___y_3352_, lean_box(0));
return v___x_3374_;
}
else
{
lean_object* v_a_3375_; lean_object* v___x_3377_; uint8_t v_isShared_3378_; uint8_t v_isSharedCheck_3382_; 
lean_dec(v_val_3354_);
lean_dec_ref_known(v_stx_3349_, 4);
lean_dec_ref(v_k_3350_);
v_a_3375_ = lean_ctor_get(v___x_3360_, 0);
v_isSharedCheck_3382_ = !lean_is_exclusive(v___x_3360_);
if (v_isSharedCheck_3382_ == 0)
{
v___x_3377_ = v___x_3360_;
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
else
{
lean_inc(v_a_3375_);
lean_dec(v___x_3360_);
v___x_3377_ = lean_box(0);
v_isShared_3378_ = v_isSharedCheck_3382_;
goto v_resetjp_3376_;
}
v_resetjp_3376_:
{
lean_object* v___x_3380_; 
if (v_isShared_3378_ == 0)
{
v___x_3380_ = v___x_3377_;
goto v_reusejp_3379_;
}
else
{
lean_object* v_reuseFailAlloc_3381_; 
v_reuseFailAlloc_3381_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3381_, 0, v_a_3375_);
v___x_3380_ = v_reuseFailAlloc_3381_;
goto v_reusejp_3379_;
}
v_reusejp_3379_:
{
return v___x_3380_;
}
}
}
}
}
else
{
lean_object* v___x_3383_; lean_object* v___x_3384_; 
lean_dec_ref(v_k_3350_);
v___x_3383_ = lean_obj_once(&lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3, &lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3_once, _init_lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___closed__3);
v___x_3384_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(v_stx_3349_, v___x_3383_, v___y_3351_, v___y_3352_);
lean_dec(v_stx_3349_);
return v___x_3384_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7___boxed(lean_object* v_stx_3385_, lean_object* v_k_3386_, lean_object* v___y_3387_, lean_object* v___y_3388_, lean_object* v___y_3389_){
_start:
{
lean_object* v_res_3390_; 
v_res_3390_ = lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7(v_stx_3385_, v_k_3386_, v___y_3387_, v___y_3388_);
lean_dec(v___y_3388_);
lean_dec_ref(v___y_3387_);
return v_res_3390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5(lean_object* v_stx_3392_, lean_object* v___y_3393_, lean_object* v___y_3394_){
_start:
{
lean_object* v___x_3396_; lean_object* v___x_3397_; 
v___x_3396_ = ((lean_object*)(lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___closed__0));
v___x_3397_ = lp_mathlib_Lean_preprocessSyntaxAndResolve___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__7(v_stx_3392_, v___x_3396_, v___y_3393_, v___y_3394_);
return v___x_3397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5___boxed(lean_object* v_stx_3398_, lean_object* v___y_3399_, lean_object* v___y_3400_, lean_object* v___y_3401_){
_start:
{
lean_object* v_res_3402_; 
v_res_3402_ = lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5(v_stx_3398_, v___y_3399_, v___y_3400_);
lean_dec(v___y_3400_);
lean_dec_ref(v___y_3399_);
return v_res_3402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(uint8_t v___x_3403_, lean_object* v_init_3404_, lean_object* v_x_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_){
_start:
{
if (lean_obj_tag(v_x_3405_) == 0)
{
lean_object* v_k_3409_; lean_object* v_v_3410_; lean_object* v_l_3411_; lean_object* v_r_3412_; lean_object* v___x_3413_; 
v_k_3409_ = lean_ctor_get(v_x_3405_, 1);
v_v_3410_ = lean_ctor_get(v_x_3405_, 2);
v_l_3411_ = lean_ctor_get(v_x_3405_, 3);
v_r_3412_ = lean_ctor_get(v_x_3405_, 4);
v___x_3413_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(v___x_3403_, v_init_3404_, v_l_3411_, v___y_3406_, v___y_3407_);
if (lean_obj_tag(v___x_3413_) == 0)
{
uint8_t v___x_3414_; lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v_a_3417_; lean_object* v___x_3418_; lean_object* v___x_3419_; size_t v_sz_3420_; size_t v___x_3421_; lean_object* v___x_3422_; 
lean_dec_ref_known(v___x_3413_, 1);
v___x_3414_ = 0;
lean_inc(v_k_3409_);
v___x_3415_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_3415_, 0, v_k_3409_);
lean_ctor_set_uint8(v___x_3415_, sizeof(void*)*1, v___x_3403_);
lean_ctor_set_uint8(v___x_3415_, sizeof(void*)*1 + 1, v___x_3414_);
v___x_3416_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v___x_3415_);
v_a_3417_ = lean_ctor_get(v___x_3416_, 0);
lean_inc(v_a_3417_);
lean_dec_ref(v___x_3416_);
v___x_3418_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6);
v___x_3419_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3419_, 0, v___x_3418_);
lean_ctor_set(v___x_3419_, 1, v_a_3417_);
v_sz_3420_ = lean_array_size(v_v_3410_);
v___x_3421_ = ((size_t)0ULL);
v___x_3422_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(v___x_3403_, v_v_3410_, v_sz_3420_, v___x_3421_, v___x_3419_, v___y_3406_, v___y_3407_);
if (lean_obj_tag(v___x_3422_) == 0)
{
lean_object* v_a_3423_; lean_object* v___x_3424_; 
v_a_3423_ = lean_ctor_get(v___x_3422_, 0);
lean_inc(v_a_3423_);
lean_dec_ref_known(v___x_3422_, 1);
v___x_3424_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(v_a_3423_, v___y_3406_, v___y_3407_);
if (lean_obj_tag(v___x_3424_) == 0)
{
lean_object* v___x_3425_; 
lean_dec_ref_known(v___x_3424_, 1);
v___x_3425_ = lean_box(0);
v_init_3404_ = v___x_3425_;
v_x_3405_ = v_r_3412_;
goto _start;
}
else
{
lean_object* v_a_3427_; lean_object* v___x_3429_; uint8_t v_isShared_3430_; uint8_t v_isSharedCheck_3434_; 
v_a_3427_ = lean_ctor_get(v___x_3424_, 0);
v_isSharedCheck_3434_ = !lean_is_exclusive(v___x_3424_);
if (v_isSharedCheck_3434_ == 0)
{
v___x_3429_ = v___x_3424_;
v_isShared_3430_ = v_isSharedCheck_3434_;
goto v_resetjp_3428_;
}
else
{
lean_inc(v_a_3427_);
lean_dec(v___x_3424_);
v___x_3429_ = lean_box(0);
v_isShared_3430_ = v_isSharedCheck_3434_;
goto v_resetjp_3428_;
}
v_resetjp_3428_:
{
lean_object* v___x_3432_; 
if (v_isShared_3430_ == 0)
{
v___x_3432_ = v___x_3429_;
goto v_reusejp_3431_;
}
else
{
lean_object* v_reuseFailAlloc_3433_; 
v_reuseFailAlloc_3433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3433_, 0, v_a_3427_);
v___x_3432_ = v_reuseFailAlloc_3433_;
goto v_reusejp_3431_;
}
v_reusejp_3431_:
{
return v___x_3432_;
}
}
}
}
else
{
lean_object* v_a_3435_; lean_object* v___x_3437_; uint8_t v_isShared_3438_; uint8_t v_isSharedCheck_3442_; 
v_a_3435_ = lean_ctor_get(v___x_3422_, 0);
v_isSharedCheck_3442_ = !lean_is_exclusive(v___x_3422_);
if (v_isSharedCheck_3442_ == 0)
{
v___x_3437_ = v___x_3422_;
v_isShared_3438_ = v_isSharedCheck_3442_;
goto v_resetjp_3436_;
}
else
{
lean_inc(v_a_3435_);
lean_dec(v___x_3422_);
v___x_3437_ = lean_box(0);
v_isShared_3438_ = v_isSharedCheck_3442_;
goto v_resetjp_3436_;
}
v_resetjp_3436_:
{
lean_object* v___x_3440_; 
if (v_isShared_3438_ == 0)
{
v___x_3440_ = v___x_3437_;
goto v_reusejp_3439_;
}
else
{
lean_object* v_reuseFailAlloc_3441_; 
v_reuseFailAlloc_3441_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3441_, 0, v_a_3435_);
v___x_3440_ = v_reuseFailAlloc_3441_;
goto v_reusejp_3439_;
}
v_reusejp_3439_:
{
return v___x_3440_;
}
}
}
}
else
{
return v___x_3413_;
}
}
else
{
lean_object* v___x_3443_; lean_object* v___x_3444_; 
v___x_3443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3443_, 0, v_init_3404_);
v___x_3444_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3444_, 0, v___x_3443_);
return v___x_3444_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15___boxed(lean_object* v___x_3445_, lean_object* v_init_3446_, lean_object* v_x_3447_, lean_object* v___y_3448_, lean_object* v___y_3449_, lean_object* v___y_3450_){
_start:
{
uint8_t v___x_13966__boxed_3451_; lean_object* v_res_3452_; 
v___x_13966__boxed_3451_ = lean_unbox(v___x_3445_);
v_res_3452_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(v___x_13966__boxed_3451_, v_init_3446_, v_x_3447_, v___y_3448_, v___y_3449_);
lean_dec(v___y_3449_);
lean_dec_ref(v___y_3448_);
lean_dec(v_x_3447_);
return v_res_3452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8(uint8_t v___x_3453_, lean_object* v_init_3454_, lean_object* v_x_3455_, lean_object* v___y_3456_, lean_object* v___y_3457_){
_start:
{
if (lean_obj_tag(v_x_3455_) == 0)
{
lean_object* v_k_3459_; lean_object* v_v_3460_; lean_object* v_l_3461_; lean_object* v_r_3462_; lean_object* v___x_3463_; 
v_k_3459_ = lean_ctor_get(v_x_3455_, 1);
v_v_3460_ = lean_ctor_get(v_x_3455_, 2);
v_l_3461_ = lean_ctor_get(v_x_3455_, 3);
v_r_3462_ = lean_ctor_get(v_x_3455_, 4);
v___x_3463_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(v___x_3453_, v_init_3454_, v_l_3461_, v___y_3456_, v___y_3457_);
if (lean_obj_tag(v___x_3463_) == 0)
{
uint8_t v___x_3464_; lean_object* v___x_3465_; lean_object* v___x_3466_; lean_object* v_a_3467_; lean_object* v___x_3468_; lean_object* v___x_3469_; size_t v_sz_3470_; size_t v___x_3471_; lean_object* v___x_3472_; 
lean_dec_ref_known(v___x_3463_, 1);
v___x_3464_ = 0;
lean_inc(v_k_3459_);
v___x_3465_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_3465_, 0, v_k_3459_);
lean_ctor_set_uint8(v___x_3465_, sizeof(void*)*1, v___x_3453_);
lean_ctor_set_uint8(v___x_3465_, sizeof(void*)*1 + 1, v___x_3464_);
v___x_3466_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v___x_3465_);
v_a_3467_ = lean_ctor_get(v___x_3466_, 0);
lean_inc(v_a_3467_);
lean_dec_ref(v___x_3466_);
v___x_3468_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6);
v___x_3469_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3469_, 0, v___x_3468_);
lean_ctor_set(v___x_3469_, 1, v_a_3467_);
v_sz_3470_ = lean_array_size(v_v_3460_);
v___x_3471_ = ((size_t)0ULL);
v___x_3472_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(v___x_3453_, v_v_3460_, v_sz_3470_, v___x_3471_, v___x_3469_, v___y_3456_, v___y_3457_);
if (lean_obj_tag(v___x_3472_) == 0)
{
lean_object* v_a_3473_; lean_object* v___x_3474_; 
v_a_3473_ = lean_ctor_get(v___x_3472_, 0);
lean_inc(v_a_3473_);
lean_dec_ref_known(v___x_3472_, 1);
v___x_3474_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(v_a_3473_, v___y_3456_, v___y_3457_);
if (lean_obj_tag(v___x_3474_) == 0)
{
lean_object* v___x_3475_; lean_object* v___x_3476_; 
lean_dec_ref_known(v___x_3474_, 1);
v___x_3475_ = lean_box(0);
v___x_3476_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8_spec__15(v___x_3453_, v___x_3475_, v_r_3462_, v___y_3456_, v___y_3457_);
return v___x_3476_;
}
else
{
lean_object* v_a_3477_; lean_object* v___x_3479_; uint8_t v_isShared_3480_; uint8_t v_isSharedCheck_3484_; 
v_a_3477_ = lean_ctor_get(v___x_3474_, 0);
v_isSharedCheck_3484_ = !lean_is_exclusive(v___x_3474_);
if (v_isSharedCheck_3484_ == 0)
{
v___x_3479_ = v___x_3474_;
v_isShared_3480_ = v_isSharedCheck_3484_;
goto v_resetjp_3478_;
}
else
{
lean_inc(v_a_3477_);
lean_dec(v___x_3474_);
v___x_3479_ = lean_box(0);
v_isShared_3480_ = v_isSharedCheck_3484_;
goto v_resetjp_3478_;
}
v_resetjp_3478_:
{
lean_object* v___x_3482_; 
if (v_isShared_3480_ == 0)
{
v___x_3482_ = v___x_3479_;
goto v_reusejp_3481_;
}
else
{
lean_object* v_reuseFailAlloc_3483_; 
v_reuseFailAlloc_3483_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3483_, 0, v_a_3477_);
v___x_3482_ = v_reuseFailAlloc_3483_;
goto v_reusejp_3481_;
}
v_reusejp_3481_:
{
return v___x_3482_;
}
}
}
}
else
{
lean_object* v_a_3485_; lean_object* v___x_3487_; uint8_t v_isShared_3488_; uint8_t v_isSharedCheck_3492_; 
v_a_3485_ = lean_ctor_get(v___x_3472_, 0);
v_isSharedCheck_3492_ = !lean_is_exclusive(v___x_3472_);
if (v_isSharedCheck_3492_ == 0)
{
v___x_3487_ = v___x_3472_;
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
else
{
lean_inc(v_a_3485_);
lean_dec(v___x_3472_);
v___x_3487_ = lean_box(0);
v_isShared_3488_ = v_isSharedCheck_3492_;
goto v_resetjp_3486_;
}
v_resetjp_3486_:
{
lean_object* v___x_3490_; 
if (v_isShared_3488_ == 0)
{
v___x_3490_ = v___x_3487_;
goto v_reusejp_3489_;
}
else
{
lean_object* v_reuseFailAlloc_3491_; 
v_reuseFailAlloc_3491_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3491_, 0, v_a_3485_);
v___x_3490_ = v_reuseFailAlloc_3491_;
goto v_reusejp_3489_;
}
v_reusejp_3489_:
{
return v___x_3490_;
}
}
}
}
else
{
return v___x_3463_;
}
}
else
{
lean_object* v___x_3493_; lean_object* v___x_3494_; 
v___x_3493_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3493_, 0, v_init_3454_);
v___x_3494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3494_, 0, v___x_3493_);
return v___x_3494_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8___boxed(lean_object* v___x_3495_, lean_object* v_init_3496_, lean_object* v_x_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_){
_start:
{
uint8_t v___x_14065__boxed_3501_; lean_object* v_res_3502_; 
v___x_14065__boxed_3501_ = lean_unbox(v___x_3495_);
v_res_3502_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8(v___x_14065__boxed_3501_, v_init_3496_, v_x_3497_, v___y_3498_, v___y_3499_);
lean_dec(v___y_3499_);
lean_dec_ref(v___y_3498_);
lean_dec(v_x_3497_);
return v_res_3502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1(lean_object* v_x_3505_, lean_object* v_a_3506_, lean_object* v_a_3507_){
_start:
{
lean_object* v___x_3509_; uint8_t v___x_3510_; lean_object* v___y_3512_; lean_object* v_a_3513_; 
v___x_3509_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp_command_x23print__fun__prop__theorems_____00__closed__1));
lean_inc(v_x_3505_);
v___x_3510_ = l_Lean_Syntax_isOfKind(v_x_3505_, v___x_3509_);
if (v___x_3510_ == 0)
{
lean_object* v___x_3563_; 
lean_dec(v_x_3505_);
v___x_3563_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__0___redArg();
return v___x_3563_;
}
else
{
lean_object* v___x_3564_; lean_object* v___x_3565_; lean_object* v___y_3567_; lean_object* v___x_3618_; lean_object* v___x_3619_; lean_object* v___x_3620_; 
v___x_3564_ = lean_unsigned_to_nat(1u);
v___x_3565_ = l_Lean_Syntax_getArg(v_x_3505_, v___x_3564_);
v___x_3618_ = lean_unsigned_to_nat(2u);
v___x_3619_ = l_Lean_Syntax_getArg(v_x_3505_, v___x_3618_);
lean_dec(v_x_3505_);
v___x_3620_ = l_Lean_Syntax_getOptional_x3f(v___x_3619_);
lean_dec(v___x_3619_);
if (lean_obj_tag(v___x_3620_) == 0)
{
lean_object* v___x_3621_; 
v___x_3621_ = lean_box(0);
v___y_3567_ = v___x_3621_;
goto v___jp_3566_;
}
else
{
lean_object* v_val_3622_; lean_object* v___x_3624_; uint8_t v_isShared_3625_; uint8_t v_isSharedCheck_3629_; 
v_val_3622_ = lean_ctor_get(v___x_3620_, 0);
v_isSharedCheck_3629_ = !lean_is_exclusive(v___x_3620_);
if (v_isSharedCheck_3629_ == 0)
{
v___x_3624_ = v___x_3620_;
v_isShared_3625_ = v_isSharedCheck_3629_;
goto v_resetjp_3623_;
}
else
{
lean_inc(v_val_3622_);
lean_dec(v___x_3620_);
v___x_3624_ = lean_box(0);
v_isShared_3625_ = v_isSharedCheck_3629_;
goto v_resetjp_3623_;
}
v_resetjp_3623_:
{
lean_object* v___x_3627_; 
if (v_isShared_3625_ == 0)
{
v___x_3627_ = v___x_3624_;
goto v_reusejp_3626_;
}
else
{
lean_object* v_reuseFailAlloc_3628_; 
v_reuseFailAlloc_3628_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3628_, 0, v_val_3622_);
v___x_3627_ = v_reuseFailAlloc_3628_;
goto v_reusejp_3626_;
}
v_reusejp_3626_:
{
v___y_3567_ = v___x_3627_;
goto v___jp_3566_;
}
}
}
v___jp_3566_:
{
lean_object* v___x_3568_; 
lean_inc(v___x_3565_);
v___x_3568_ = lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5(v___x_3565_, v_a_3506_, v_a_3507_);
if (lean_obj_tag(v___x_3568_) == 0)
{
lean_object* v_a_3569_; lean_object* v___x_3570_; 
v_a_3569_ = lean_ctor_get(v___x_3568_, 0);
lean_inc(v_a_3569_);
lean_dec_ref_known(v___x_3568_, 1);
v___x_3570_ = lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6(v___x_3565_, v_a_3569_, v_a_3506_, v_a_3507_);
if (lean_obj_tag(v___x_3570_) == 0)
{
if (lean_obj_tag(v___y_3567_) == 0)
{
lean_object* v_a_3571_; lean_object* v___x_3572_; 
v_a_3571_ = lean_ctor_get(v___x_3570_, 0);
lean_inc(v_a_3571_);
lean_dec_ref_known(v___x_3570_, 1);
v___x_3572_ = lean_box(0);
v___y_3512_ = v_a_3571_;
v_a_3513_ = v___x_3572_;
goto v___jp_3511_;
}
else
{
lean_object* v_a_3573_; lean_object* v_val_3574_; lean_object* v___x_3576_; uint8_t v_isShared_3577_; uint8_t v_isSharedCheck_3601_; 
v_a_3573_ = lean_ctor_get(v___x_3570_, 0);
lean_inc(v_a_3573_);
lean_dec_ref_known(v___x_3570_, 1);
v_val_3574_ = lean_ctor_get(v___y_3567_, 0);
v_isSharedCheck_3601_ = !lean_is_exclusive(v___y_3567_);
if (v_isSharedCheck_3601_ == 0)
{
v___x_3576_ = v___y_3567_;
v_isShared_3577_ = v_isSharedCheck_3601_;
goto v_resetjp_3575_;
}
else
{
lean_inc(v_val_3574_);
lean_dec(v___y_3567_);
v___x_3576_ = lean_box(0);
v_isShared_3577_ = v_isSharedCheck_3601_;
goto v_resetjp_3575_;
}
v_resetjp_3575_:
{
lean_object* v___x_3578_; 
lean_inc(v_val_3574_);
v___x_3578_ = lp_mathlib_Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5(v_val_3574_, v_a_3506_, v_a_3507_);
if (lean_obj_tag(v___x_3578_) == 0)
{
lean_object* v_a_3579_; lean_object* v___x_3580_; 
v_a_3579_ = lean_ctor_get(v___x_3578_, 0);
lean_inc(v_a_3579_);
lean_dec_ref_known(v___x_3578_, 1);
v___x_3580_ = lp_mathlib_Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6(v_val_3574_, v_a_3579_, v_a_3506_, v_a_3507_);
if (lean_obj_tag(v___x_3580_) == 0)
{
lean_object* v_a_3581_; lean_object* v___x_3583_; 
v_a_3581_ = lean_ctor_get(v___x_3580_, 0);
lean_inc(v_a_3581_);
lean_dec_ref_known(v___x_3580_, 1);
if (v_isShared_3577_ == 0)
{
lean_ctor_set(v___x_3576_, 0, v_a_3581_);
v___x_3583_ = v___x_3576_;
goto v_reusejp_3582_;
}
else
{
lean_object* v_reuseFailAlloc_3584_; 
v_reuseFailAlloc_3584_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3584_, 0, v_a_3581_);
v___x_3583_ = v_reuseFailAlloc_3584_;
goto v_reusejp_3582_;
}
v_reusejp_3582_:
{
v___y_3512_ = v_a_3573_;
v_a_3513_ = v___x_3583_;
goto v___jp_3511_;
}
}
else
{
lean_object* v_a_3585_; lean_object* v___x_3587_; uint8_t v_isShared_3588_; uint8_t v_isSharedCheck_3592_; 
lean_del_object(v___x_3576_);
lean_dec(v_a_3573_);
v_a_3585_ = lean_ctor_get(v___x_3580_, 0);
v_isSharedCheck_3592_ = !lean_is_exclusive(v___x_3580_);
if (v_isSharedCheck_3592_ == 0)
{
v___x_3587_ = v___x_3580_;
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
else
{
lean_inc(v_a_3585_);
lean_dec(v___x_3580_);
v___x_3587_ = lean_box(0);
v_isShared_3588_ = v_isSharedCheck_3592_;
goto v_resetjp_3586_;
}
v_resetjp_3586_:
{
lean_object* v___x_3590_; 
if (v_isShared_3588_ == 0)
{
v___x_3590_ = v___x_3587_;
goto v_reusejp_3589_;
}
else
{
lean_object* v_reuseFailAlloc_3591_; 
v_reuseFailAlloc_3591_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3591_, 0, v_a_3585_);
v___x_3590_ = v_reuseFailAlloc_3591_;
goto v_reusejp_3589_;
}
v_reusejp_3589_:
{
return v___x_3590_;
}
}
}
}
else
{
lean_object* v_a_3593_; lean_object* v___x_3595_; uint8_t v_isShared_3596_; uint8_t v_isSharedCheck_3600_; 
lean_del_object(v___x_3576_);
lean_dec(v_val_3574_);
lean_dec(v_a_3573_);
v_a_3593_ = lean_ctor_get(v___x_3578_, 0);
v_isSharedCheck_3600_ = !lean_is_exclusive(v___x_3578_);
if (v_isSharedCheck_3600_ == 0)
{
v___x_3595_ = v___x_3578_;
v_isShared_3596_ = v_isSharedCheck_3600_;
goto v_resetjp_3594_;
}
else
{
lean_inc(v_a_3593_);
lean_dec(v___x_3578_);
v___x_3595_ = lean_box(0);
v_isShared_3596_ = v_isSharedCheck_3600_;
goto v_resetjp_3594_;
}
v_resetjp_3594_:
{
lean_object* v___x_3598_; 
if (v_isShared_3596_ == 0)
{
v___x_3598_ = v___x_3595_;
goto v_reusejp_3597_;
}
else
{
lean_object* v_reuseFailAlloc_3599_; 
v_reuseFailAlloc_3599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3599_, 0, v_a_3593_);
v___x_3598_ = v_reuseFailAlloc_3599_;
goto v_reusejp_3597_;
}
v_reusejp_3597_:
{
return v___x_3598_;
}
}
}
}
}
}
else
{
lean_object* v_a_3602_; lean_object* v___x_3604_; uint8_t v_isShared_3605_; uint8_t v_isSharedCheck_3609_; 
lean_dec(v___y_3567_);
v_a_3602_ = lean_ctor_get(v___x_3570_, 0);
v_isSharedCheck_3609_ = !lean_is_exclusive(v___x_3570_);
if (v_isSharedCheck_3609_ == 0)
{
v___x_3604_ = v___x_3570_;
v_isShared_3605_ = v_isSharedCheck_3609_;
goto v_resetjp_3603_;
}
else
{
lean_inc(v_a_3602_);
lean_dec(v___x_3570_);
v___x_3604_ = lean_box(0);
v_isShared_3605_ = v_isSharedCheck_3609_;
goto v_resetjp_3603_;
}
v_resetjp_3603_:
{
lean_object* v___x_3607_; 
if (v_isShared_3605_ == 0)
{
v___x_3607_ = v___x_3604_;
goto v_reusejp_3606_;
}
else
{
lean_object* v_reuseFailAlloc_3608_; 
v_reuseFailAlloc_3608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3608_, 0, v_a_3602_);
v___x_3607_ = v_reuseFailAlloc_3608_;
goto v_reusejp_3606_;
}
v_reusejp_3606_:
{
return v___x_3607_;
}
}
}
}
else
{
lean_object* v_a_3610_; lean_object* v___x_3612_; uint8_t v_isShared_3613_; uint8_t v_isSharedCheck_3617_; 
lean_dec(v___y_3567_);
lean_dec(v___x_3565_);
v_a_3610_ = lean_ctor_get(v___x_3568_, 0);
v_isSharedCheck_3617_ = !lean_is_exclusive(v___x_3568_);
if (v_isSharedCheck_3617_ == 0)
{
v___x_3612_ = v___x_3568_;
v_isShared_3613_ = v_isSharedCheck_3617_;
goto v_resetjp_3611_;
}
else
{
lean_inc(v_a_3610_);
lean_dec(v___x_3568_);
v___x_3612_ = lean_box(0);
v_isShared_3613_ = v_isSharedCheck_3617_;
goto v_resetjp_3611_;
}
v_resetjp_3611_:
{
lean_object* v___x_3615_; 
if (v_isShared_3613_ == 0)
{
v___x_3615_ = v___x_3612_;
goto v_reusejp_3614_;
}
else
{
lean_object* v_reuseFailAlloc_3616_; 
v_reuseFailAlloc_3616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3616_, 0, v_a_3610_);
v___x_3615_ = v_reuseFailAlloc_3616_;
goto v_reusejp_3614_;
}
v_reusejp_3614_:
{
return v___x_3615_;
}
}
}
}
}
v___jp_3511_:
{
lean_object* v___x_3514_; lean_object* v_env_3515_; lean_object* v___x_3516_; lean_object* v_ext_3517_; lean_object* v_toEnvExtension_3518_; lean_object* v_asyncMode_3519_; lean_object* v___x_3520_; lean_object* v___x_3521_; lean_object* v___x_3522_; 
v___x_3514_ = lean_st_ref_get(v_a_3507_);
v_env_3515_ = lean_ctor_get(v___x_3514_, 0);
lean_inc_ref(v_env_3515_);
lean_dec(v___x_3514_);
v___x_3516_ = lp_mathlib_Mathlib_Meta_FunProp_functionTheoremsExt;
v_ext_3517_ = lean_ctor_get(v___x_3516_, 1);
v_toEnvExtension_3518_ = lean_ctor_get(v_ext_3517_, 0);
v_asyncMode_3519_ = lean_ctor_get(v_toEnvExtension_3518_, 2);
v___x_3520_ = lean_box(1);
v___x_3521_ = l_Lean_ScopedEnvExtension_getState___redArg(v___x_3520_, v___x_3516_, v_env_3515_, v_asyncMode_3519_);
v___x_3522_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(v___x_3521_, v___y_3512_, v___x_3520_);
lean_dec(v___y_3512_);
lean_dec(v___x_3521_);
if (lean_obj_tag(v_a_3513_) == 0)
{
lean_object* v___x_3523_; lean_object* v___x_3524_; 
v___x_3523_ = lean_box(0);
v___x_3524_ = lp_mathlib_Std_DTreeMap_Internal_Impl_forInStep___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__8(v___x_3510_, v___x_3523_, v___x_3522_, v_a_3506_, v_a_3507_);
lean_dec(v___x_3522_);
if (lean_obj_tag(v___x_3524_) == 0)
{
lean_object* v___x_3526_; uint8_t v_isShared_3527_; uint8_t v_isSharedCheck_3531_; 
v_isSharedCheck_3531_ = !lean_is_exclusive(v___x_3524_);
if (v_isSharedCheck_3531_ == 0)
{
lean_object* v_unused_3532_; 
v_unused_3532_ = lean_ctor_get(v___x_3524_, 0);
lean_dec(v_unused_3532_);
v___x_3526_ = v___x_3524_;
v_isShared_3527_ = v_isSharedCheck_3531_;
goto v_resetjp_3525_;
}
else
{
lean_dec(v___x_3524_);
v___x_3526_ = lean_box(0);
v_isShared_3527_ = v_isSharedCheck_3531_;
goto v_resetjp_3525_;
}
v_resetjp_3525_:
{
lean_object* v___x_3529_; 
if (v_isShared_3527_ == 0)
{
lean_ctor_set(v___x_3526_, 0, v___x_3523_);
v___x_3529_ = v___x_3526_;
goto v_reusejp_3528_;
}
else
{
lean_object* v_reuseFailAlloc_3530_; 
v_reuseFailAlloc_3530_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3530_, 0, v___x_3523_);
v___x_3529_ = v_reuseFailAlloc_3530_;
goto v_reusejp_3528_;
}
v_reusejp_3528_:
{
return v___x_3529_;
}
}
}
else
{
lean_object* v_a_3533_; lean_object* v___x_3535_; uint8_t v_isShared_3536_; uint8_t v_isSharedCheck_3540_; 
v_a_3533_ = lean_ctor_get(v___x_3524_, 0);
v_isSharedCheck_3540_ = !lean_is_exclusive(v___x_3524_);
if (v_isSharedCheck_3540_ == 0)
{
v___x_3535_ = v___x_3524_;
v_isShared_3536_ = v_isSharedCheck_3540_;
goto v_resetjp_3534_;
}
else
{
lean_inc(v_a_3533_);
lean_dec(v___x_3524_);
v___x_3535_ = lean_box(0);
v_isShared_3536_ = v_isSharedCheck_3540_;
goto v_resetjp_3534_;
}
v_resetjp_3534_:
{
lean_object* v___x_3538_; 
if (v_isShared_3536_ == 0)
{
v___x_3538_ = v___x_3535_;
goto v_reusejp_3537_;
}
else
{
lean_object* v_reuseFailAlloc_3539_; 
v_reuseFailAlloc_3539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3539_, 0, v_a_3533_);
v___x_3538_ = v_reuseFailAlloc_3539_;
goto v_reusejp_3537_;
}
v_reusejp_3537_:
{
return v___x_3538_;
}
}
}
}
else
{
lean_object* v_val_3541_; lean_object* v___x_3542_; lean_object* v___x_3543_; uint8_t v___x_3544_; lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v_a_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; size_t v_sz_3550_; size_t v___x_3551_; lean_object* v___x_3552_; 
v_val_3541_ = lean_ctor_get(v_a_3513_, 0);
lean_inc(v_val_3541_);
lean_dec_ref_known(v_a_3513_, 1);
v___x_3542_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___closed__0));
v___x_3543_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(v___x_3522_, v_val_3541_, v___x_3542_);
lean_dec(v___x_3522_);
v___x_3544_ = 0;
v___x_3545_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_3545_, 0, v_val_3541_);
lean_ctor_set_uint8(v___x_3545_, sizeof(void*)*1, v___x_3510_);
lean_ctor_set_uint8(v___x_3545_, sizeof(void*)*1 + 1, v___x_3544_);
v___x_3546_ = lp_mathlib_Lean_Meta_ppOrigin___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__1___redArg(v___x_3545_);
v_a_3547_ = lean_ctor_get(v___x_3546_, 0);
lean_inc(v_a_3547_);
lean_dec_ref(v___x_3546_);
v___x_3548_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6, &lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6_once, _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTac___lam__0___closed__6);
v___x_3549_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3549_, 0, v___x_3548_);
lean_ctor_set(v___x_3549_, 1, v_a_3547_);
v_sz_3550_ = lean_array_size(v___x_3543_);
v___x_3551_ = ((size_t)0ULL);
v___x_3552_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__3(v___x_3510_, v___x_3543_, v_sz_3550_, v___x_3551_, v___x_3549_, v_a_3506_, v_a_3507_);
lean_dec(v___x_3543_);
if (lean_obj_tag(v___x_3552_) == 0)
{
lean_object* v_a_3553_; lean_object* v___x_3554_; 
v_a_3553_ = lean_ctor_get(v___x_3552_, 0);
lean_inc(v_a_3553_);
lean_dec_ref_known(v___x_3552_, 1);
v___x_3554_ = lp_mathlib_Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4(v_a_3553_, v_a_3506_, v_a_3507_);
return v___x_3554_;
}
else
{
lean_object* v_a_3555_; lean_object* v___x_3557_; uint8_t v_isShared_3558_; uint8_t v_isSharedCheck_3562_; 
v_a_3555_ = lean_ctor_get(v___x_3552_, 0);
v_isSharedCheck_3562_ = !lean_is_exclusive(v___x_3552_);
if (v_isSharedCheck_3562_ == 0)
{
v___x_3557_ = v___x_3552_;
v_isShared_3558_ = v_isSharedCheck_3562_;
goto v_resetjp_3556_;
}
else
{
lean_inc(v_a_3555_);
lean_dec(v___x_3552_);
v___x_3557_ = lean_box(0);
v_isShared_3558_ = v_isSharedCheck_3562_;
goto v_resetjp_3556_;
}
v_resetjp_3556_:
{
lean_object* v___x_3560_; 
if (v_isShared_3558_ == 0)
{
v___x_3560_ = v___x_3557_;
goto v_reusejp_3559_;
}
else
{
lean_object* v_reuseFailAlloc_3561_; 
v_reuseFailAlloc_3561_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3561_, 0, v_a_3555_);
v___x_3560_ = v_reuseFailAlloc_3561_;
goto v_reusejp_3559_;
}
v_reusejp_3559_:
{
return v___x_3560_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1___boxed(lean_object* v_x_3630_, lean_object* v_a_3631_, lean_object* v_a_3632_, lean_object* v_a_3633_){
_start:
{
lean_object* v_res_3634_; 
v_res_3634_ = lp_mathlib_Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1(v_x_3630_, v_a_3631_, v_a_3632_);
lean_dec(v_a_3632_);
lean_dec_ref(v_a_3631_);
return v_res_3634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7(lean_object* v_00_u03b4_3635_, lean_object* v_t_3636_, lean_object* v_k_3637_, lean_object* v_fallback_3638_){
_start:
{
lean_object* v___x_3639_; 
v___x_3639_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___redArg(v_t_3636_, v_k_3637_, v_fallback_3638_);
return v___x_3639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7___boxed(lean_object* v_00_u03b4_3640_, lean_object* v_t_3641_, lean_object* v_k_3642_, lean_object* v_fallback_3643_){
_start:
{
lean_object* v_res_3644_; 
v_res_3644_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_getD___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__7(v_00_u03b4_3640_, v_t_3641_, v_k_3642_, v_fallback_3643_);
lean_dec(v_fallback_3643_);
lean_dec(v_k_3642_);
lean_dec(v_t_3641_);
return v_res_3644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12(lean_object* v_00_u03b1_3645_, lean_object* v_ref_3646_, lean_object* v_msg_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_){
_start:
{
lean_object* v___x_3651_; 
v___x_3651_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___redArg(v_ref_3646_, v_msg_3647_, v___y_3648_, v___y_3649_);
return v___x_3651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12___boxed(lean_object* v_00_u03b1_3652_, lean_object* v_ref_3653_, lean_object* v_msg_3654_, lean_object* v___y_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_){
_start:
{
lean_object* v_res_3658_; 
v_res_3658_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12(v_00_u03b1_3652_, v_ref_3653_, v_msg_3654_, v___y_3655_, v___y_3656_);
lean_dec(v___y_3656_);
lean_dec_ref(v___y_3655_);
lean_dec(v_ref_3653_);
return v_res_3658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11(lean_object* v_msgData_3659_, lean_object* v___y_3660_, lean_object* v___y_3661_){
_start:
{
lean_object* v___x_3663_; 
v___x_3663_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___redArg(v_msgData_3659_, v___y_3661_);
return v___x_3663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11___boxed(lean_object* v_msgData_3664_, lean_object* v___y_3665_, lean_object* v___y_3666_, lean_object* v___y_3667_){
_start:
{
lean_object* v_res_3668_; 
v_res_3668_ = lp_mathlib_Lean_addMessageContextPartial___at___00Lean_logAt___at___00Lean_log___at___00Lean_logInfo___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__4_spec__4_spec__5_spec__11(v_msgData_3664_, v___y_3665_, v___y_3666_);
lean_dec(v___y_3666_);
lean_dec_ref(v___y_3665_);
return v_res_3668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18(lean_object* v_00_u03b1_3669_, lean_object* v_msg_3670_, lean_object* v___y_3671_, lean_object* v___y_3672_){
_start:
{
lean_object* v___x_3674_; 
v___x_3674_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___redArg(v_msg_3670_, v___y_3671_, v___y_3672_);
return v___x_3674_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18___boxed(lean_object* v_00_u03b1_3675_, lean_object* v_msg_3676_, lean_object* v___y_3677_, lean_object* v___y_3678_, lean_object* v___y_3679_){
_start:
{
lean_object* v_res_3680_; 
v_res_3680_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18(v_00_u03b1_3675_, v_msg_3676_, v___y_3677_, v___y_3678_);
lean_dec(v___y_3678_);
lean_dec_ref(v___y_3677_);
return v_res_3680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19(lean_object* v_00_u03b1_3681_, lean_object* v_ref_3682_, lean_object* v_constName_3683_, lean_object* v___y_3684_, lean_object* v___y_3685_){
_start:
{
lean_object* v___x_3687_; 
v___x_3687_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___redArg(v_ref_3682_, v_constName_3683_, v___y_3684_, v___y_3685_);
return v___x_3687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19___boxed(lean_object* v_00_u03b1_3688_, lean_object* v_ref_3689_, lean_object* v_constName_3690_, lean_object* v___y_3691_, lean_object* v___y_3692_, lean_object* v___y_3693_){
_start:
{
lean_object* v_res_3694_; 
v_res_3694_ = lp_mathlib_Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19(v_00_u03b1_3688_, v_ref_3689_, v_constName_3690_, v___y_3691_, v___y_3692_);
lean_dec(v___y_3692_);
lean_dec_ref(v___y_3691_);
lean_dec(v_ref_3689_);
return v_res_3694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27(lean_object* v_msgData_3695_, lean_object* v_macroStack_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_){
_start:
{
lean_object* v___x_3700_; 
v___x_3700_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___redArg(v_msgData_3695_, v_macroStack_3696_, v___y_3698_);
return v___x_3700_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27___boxed(lean_object* v_msgData_3701_, lean_object* v_macroStack_3702_, lean_object* v___y_3703_, lean_object* v___y_3704_, lean_object* v___y_3705_){
_start:
{
lean_object* v_res_3706_; 
v_res_3706_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_ensureNonAmbiguous___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__6_spec__12_spec__18_spec__27(v_msgData_3701_, v_macroStack_3702_, v___y_3703_, v___y_3704_);
lean_dec(v___y_3704_);
lean_dec_ref(v___y_3703_);
return v_res_3706_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21(lean_object* v_opt_3707_, lean_object* v___y_3708_, lean_object* v___y_3709_){
_start:
{
lean_object* v___x_3711_; 
v___x_3711_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___redArg(v_opt_3707_, v___y_3709_);
return v___x_3711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21___boxed(lean_object* v_opt_3712_, lean_object* v___y_3713_, lean_object* v___y_3714_, lean_object* v___y_3715_){
_start:
{
lean_object* v_res_3716_; 
v_res_3716_ = lp_mathlib_Lean_Option_getM___at___00Lean_checkPrivateInPublic___at___00Lean_resolveGlobalName___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__8_spec__15_spec__21(v_opt_3712_, v___y_3713_, v___y_3714_);
lean_dec(v___y_3714_);
lean_dec_ref(v___y_3713_);
lean_dec_ref(v_opt_3712_);
return v_res_3716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27(lean_object* v_00_u03b1_3717_, lean_object* v_ref_3718_, lean_object* v_msg_3719_, lean_object* v_declHint_3720_, lean_object* v___y_3721_, lean_object* v___y_3722_){
_start:
{
lean_object* v___x_3724_; 
v___x_3724_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___redArg(v_ref_3718_, v_msg_3719_, v_declHint_3720_, v___y_3721_, v___y_3722_);
return v___x_3724_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27___boxed(lean_object* v_00_u03b1_3725_, lean_object* v_ref_3726_, lean_object* v_msg_3727_, lean_object* v_declHint_3728_, lean_object* v___y_3729_, lean_object* v___y_3730_, lean_object* v___y_3731_){
_start:
{
lean_object* v_res_3732_; 
v_res_3732_ = lp_mathlib_Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27(v_00_u03b1_3725_, v_ref_3726_, v_msg_3727_, v_declHint_3728_, v___y_3729_, v___y_3730_);
lean_dec(v___y_3730_);
lean_dec_ref(v___y_3729_);
lean_dec(v_ref_3726_);
return v_res_3732_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33(lean_object* v_msg_3733_, lean_object* v_declHint_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_){
_start:
{
lean_object* v___x_3738_; 
v___x_3738_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___redArg(v_msg_3733_, v_declHint_3734_, v___y_3736_);
return v___x_3738_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33___boxed(lean_object* v_msg_3739_, lean_object* v_declHint_3740_, lean_object* v___y_3741_, lean_object* v___y_3742_, lean_object* v___y_3743_){
_start:
{
lean_object* v_res_3744_; 
v_res_3744_ = lp_mathlib_Lean_mkUnknownIdentifierMessageCore___at___00Lean_mkUnknownIdentifierMessage___at___00Lean_throwUnknownIdentifierAt___at___00Lean_throwUnknownConstantAt___at___00Lean_filterFieldList___at___00__private_Lean_ResolveName_0__Lean_resolveGlobalConstCore___at___00Lean_resolveGlobalConst___at___00Mathlib_Meta_FunProp___aux__Mathlib__Tactic__FunProp__Elab______elabRules__Mathlib__Meta__FunProp__command_x23print__fun__prop__theorems______1_spec__5_spec__6_spec__9_spec__19_spec__27_spec__31_spec__33(v_msg_3739_, v_declHint_3740_, v___y_3741_, v___y_3742_);
lean_dec(v___y_3742_);
lean_dec_ref(v___y_3741_);
return v_res_3744_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Core(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Elab(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_FunProp_Elab(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_FunProp_Elab_0__Mathlib_Meta_FunProp_instEvalExprConfig);
lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx = _init_lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx();
lean_mark_persistent(lp_mathlib_Mathlib_Meta_FunProp_funPropTacStx);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Core(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_InferParam(uint8_t builtin);
lean_object* initialize_Lean_Elab_InfoTree_Main(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_FunProp_Elab(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FunProp_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_InferParam(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_InfoTree_Main(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FunProp_Elab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_FunProp_Elab(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_FunProp_Elab(builtin);
}
#ifdef __cplusplus
}
#endif
