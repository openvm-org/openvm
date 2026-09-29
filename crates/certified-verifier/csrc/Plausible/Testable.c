// Lean compiler output
// Module: Plausible.Testable
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Config public import Lean.CoreM public import Lean.Exception public import Lean.Log public import Plausible.Sampleable
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
lean_object* lean_string_append(lean_object*, lean_object*);
uint8_t l_Lean_ExprStructEq_beq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_ExprStructEq_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Lean_Core_checkSystem(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getFunInfoNArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConst(lean_object*);
size_t lean_ptr_addr(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
extern lean_object* l_Lean_maxRecDepthErrorMessage;
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
lean_object* l_Except_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_pure(lean_object*, lean_object*, lean_object*);
lean_object* l_Except_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Except_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_pure(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_OptionT_instAlternative___redArg(lean_object*);
lean_object* l_List_forIn_x27_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Std_Format_defWidth;
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_dbg_trace(lean_object*, lean_object*);
lean_object* l_List_repr___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
uint8_t lean_string_dec_lt(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_evalBoolItem(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalTerm_evalOptionStx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_EvalExpr_instNat;
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_instOption___redArg(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Gen_resize___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* l_instReprString___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_Gen_run___redArg(lean_object*, lean_object*);
lean_object* lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError(lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_runRandWith___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadLogCoreM;
extern lean_object* l_Lean_Core_instAddMessageContextCoreM;
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_logInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_logWarning___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
extern lean_object* l_Lean_Core_instMonadRefCoreM;
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_error_to_string(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isProp(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
lean_object* l_Lean_mkStrLit(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_expr_abstract(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_mkApp10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_plausible_Plausible_Bool_Arbitrary;
lean_object* l_Bool_repr___boxed(lean_object*, lean_object*);
lean_object* lp_plausible_Plausible_SampleableExt_selfContained___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_closeMainGoal___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_success_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_success_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_gaveUp_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_gaveUp_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_failure_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_failure_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible_Plausible_instInhabitedTestResult_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_instInhabitedTestResult_default___closed__0 = (const lean_object*)&lp_plausible_Plausible_instInhabitedTestResult_default___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instInhabitedTestResult_default(lean_object*);
static lean_once_cell_t lp_plausible_Plausible_instInhabitedTestResult___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instInhabitedTestResult___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instInhabitedTestResult(lean_object*);
static const lean_ctor_object lp_plausible_Plausible_instInhabitedConfiguration_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(10) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_plausible_Plausible_instInhabitedConfiguration_default___closed__0 = (const lean_object*)&lp_plausible_Plausible_instInhabitedConfiguration_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instInhabitedConfiguration_default = (const lean_object*)&lp_plausible_Plausible_instInhabitedConfiguration_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_instInhabitedConfiguration = (const lean_object*)&lp_plausible_Plausible_instInhabitedConfiguration_default___closed__0_value;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Plausible"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Configuration"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__2 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__2_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value_aux_1),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(32, 95, 169, 43, 19, 113, 84, 167)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__5 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__5_value;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__6 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__9 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__9_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Nat"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__12 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__12_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__12_value),LEAN_SCALAR_PTR_LITERAL(155, 221, 223, 104, 58, 13, 204, 158)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__13 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__13_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Option"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__15 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__15_value;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "none"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__16 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__16_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(149, 114, 34, 228, 75, 195, 143, 131)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__18 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__18_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20;
static const lean_string_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "some"};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__21 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__21_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(95, 234, 177, 188, 3, 226, 91, 252)}};
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__21_value),LEAN_SCALAR_PTR_LITERAL(89, 148, 40, 55, 221, 242, 231, 67)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0(lean_object*);
static const lean_closure_object lp_plausible_Plausible_instToExprConfiguration___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_instToExprConfiguration___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___closed__0 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_instToExprConfiguration___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(200, 192, 205, 201, 223, 138, 200, 123)}};
static const lean_object* lp_plausible_Plausible_instToExprConfiguration___closed__1 = (const lean_object*)&lp_plausible_Plausible_instToExprConfiguration___closed__1_value;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___closed__2;
static lean_once_cell_t lp_plausible_Plausible_instToExprConfiguration___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_instToExprConfiguration___closed__3;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instToExprConfiguration;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__1_value;
static lean_once_cell_t lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__0_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__0_value;
static lean_once_cell_t lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1;
static lean_once_cell_t lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2;
static lean_once_cell_t lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration;
static lean_once_cell_t lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0;
static const lean_string_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value;
static const lean_ctor_object lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__1_value)}};
static const lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2 = (const lean_object*)&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2_value;
static lean_once_cell_t lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3;
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value;
static const lean_ctor_object lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__0_value)}};
static const lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1_value;
static lean_once_cell_t lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0;
static const lean_string_object lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__1 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__1_value;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4;
static const lean_string_object lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__5 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__5_value;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7;
static const lean_string_object lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__8 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__8_value;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9;
static const lean_string_object lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__10 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__10_value;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__0 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__0_value;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___closed__0 = (const lean_object*)&lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1;
static lean_once_cell_t lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1_value)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "randomSeed"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__1_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "traceShrink"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__2 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__2_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "traceShrinkCandidates"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__3 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__3_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "traceSuccesses"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__4 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__4_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(67, 153, 156, 237, 74, 121, 247, 63)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(193, 149, 19, 109, 204, 7, 66, 204)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(101, 3, 183, 252, 161, 202, 123, 172)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "sorryIfNoTestable"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__8 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__8_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "traceDiscarded"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__9 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__9_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__9_value),LEAN_SCALAR_PTR_LITERAL(118, 177, 11, 65, 166, 192, 103, 55)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(175, 90, 131, 131, 248, 251, 151, 229)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(74, 0, 39, 5, 176, 226, 44, 202)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__13 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__13_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "maxSize"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__14 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__14_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "numInst"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__15 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__15_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "numRetries"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__16 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__16_value;
static const lean_string_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "quiet"};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__17 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__17_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(21, 95, 60, 160, 240, 154, 255, 228)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__16_value),LEAN_SCALAR_PTR_LITERAL(215, 3, 163, 139, 174, 221, 63, 247)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(180, 96, 41, 64, 88, 43, 237, 101)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20_value;
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value_aux_0),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(184, 53, 79, 246, 225, 140, 106, 156)}};
static const lean_ctor_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value_aux_1),((lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__14_value),LEAN_SCALAR_PTR_LITERAL(46, 194, 223, 33, 133, 54, 234, 100)}};
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21_value;
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem = (const lean_object*)&lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_elabConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_elabConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_elabConfig___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_elabConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_instPrintableProp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⋯"};
static const lean_object* lp_plausible_Plausible_instPrintableProp___closed__0 = (const lean_object*)&lp_plausible_Plausible_instPrintableProp___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_instPrintableProp(lean_object*);
static const lean_string_object lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___closed__0 = (const lean_object*)&lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[]"};
static const lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__0 = (const lean_object*)&lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__0_value;
static const lean_string_object lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__1 = (const lean_object*)&lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__1_value;
static const lean_string_object lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__2 = (const lean_object*)&lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___boxed(lean_object*);
static const lean_string_object lp_plausible_Plausible_TestResult_toString___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "success (no proof)"};
static const lean_object* lp_plausible_Plausible_TestResult_toString___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_TestResult_toString___redArg___closed__0_value;
static const lean_string_object lp_plausible_Plausible_TestResult_toString___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "success (proof)"};
static const lean_object* lp_plausible_Plausible_TestResult_toString___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_TestResult_toString___redArg___closed__1_value;
static const lean_string_object lp_plausible_Plausible_TestResult_toString___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "gave "};
static const lean_object* lp_plausible_Plausible_TestResult_toString___redArg___closed__2 = (const lean_object*)&lp_plausible_Plausible_TestResult_toString___redArg___closed__2_value;
static const lean_string_object lp_plausible_Plausible_TestResult_toString___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " times"};
static const lean_object* lp_plausible_Plausible_TestResult_toString___redArg___closed__3 = (const lean_object*)&lp_plausible_Plausible_TestResult_toString___redArg___closed__3_value;
static const lean_string_object lp_plausible_Plausible_TestResult_toString___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "failed "};
static const lean_object* lp_plausible_Plausible_TestResult_toString___redArg___closed__4 = (const lean_object*)&lp_plausible_Plausible_TestResult_toString___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_toString___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_toString(lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_TestResult_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_TestResult_toString, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_TestResult_instToString___closed__0 = (const lean_object*)&lp_plausible_Plausible_TestResult_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_instToString(lean_object*);
static const lean_ctor_object lp_plausible_Plausible_TestResult_combine___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_TestResult_combine___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_TestResult_combine___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_and___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_and(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_or___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_or(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_iff___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_iff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Plausible_TestResult_isFailure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_isFailure___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Plausible_TestResult_isFailure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_isFailure___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_plausible_Plausible_Configuration_verbose___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(100) << 1) | 1)),((lean_object*)(((size_t)(10) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0)}};
static const lean_object* lp_plausible_Plausible_Configuration_verbose___closed__0 = (const lean_object*)&lp_plausible_Plausible_Configuration_verbose___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Configuration_verbose = (const lean_object*)&lp_plausible_Plausible_Configuration_verbose___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_plausible_Plausible_Testable_runPropE___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_Testable_runPropE___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_runPropE___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_slimTrace___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "[Plausible: "};
static const lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_slimTrace___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__1_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__2_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__3 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__3_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_map, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__4 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__4_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__5 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__5_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_pure, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__6 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__5_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__6_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__1_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__2_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__3_value)}};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__7 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__7_value;
static const lean_closure_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Except_bind, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__8 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__7_value),((lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__8_value)}};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__9 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__9_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10;
static lean_once_cell_t lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11;
static const lean_string_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "discard: Guard "};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__12 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__12_value;
static const lean_string_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = " does not hold"};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__13 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__13_value;
static const lean_string_object lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "guard: "};
static const lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__14 = (const lean_object*)&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__14_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Testable_forallTypesTestable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instReprString___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_forallTypesTestable___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "ULift Int"};
static const lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_formatFailure___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_formatFailure___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "\n==================="};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__1_value;
static const lean_string_object lp_plausible_Plausible_Testable_formatFailure___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__2_value;
static const lean_string_object lp_plausible_Plausible_Testable_formatFailure___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = " shrinks)"};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__3 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__3_value;
static const lean_string_object lp_plausible_Plausible_Testable_formatFailure___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "-------------------"};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__4 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_formatFailure___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__4_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_Testable_formatFailure___closed__5 = (const lean_object*)&lp_plausible_Plausible_Testable_formatFailure___closed__5_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_formatFailure(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_instInhabitedOptionTOfPure___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_instInhabitedOptionTOfPure(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " shrunk to "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = " from "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__1_value;
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Trying "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__2_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18;
static lean_once_cell_t lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19;
static const lean_ctor_object lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__20 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__20_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___boxed(lean_object**);
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "No shrinking possible for "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__21 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__21_value;
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Candidates for "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__22 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__22_value;
static const lean_string_object lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = ":\n  "};
static const lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__23 = (const lean_object*)&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__23_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_minimize___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Shrink"};
static const lean_object* lp_plausible_Plausible_Testable_minimize___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_minimize___redArg___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_minimize___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "Attempting to shrink "};
static const lean_object* lp_plausible_Plausible_Testable_minimize___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_minimize___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = " is a failure"};
static const lean_object* lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Bool_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__0_value;
static const lean_closure_object lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Bool_shrinkable___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__1_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = " is irrelevant (unused)"};
static const lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = " is unused"};
static const lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = " (by construction)"};
static const lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "issue: "};
static const lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Eq_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " = "};
static const lean_object* lp_plausible_Plausible_Eq_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Eq_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Eq_printableProp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Eq_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Ne_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≠ "};
static const lean_object* lp_plausible_Plausible_Ne_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Ne_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Ne_printableProp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Ne_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_LE_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ≤ "};
static const lean_object* lp_plausible_Plausible_LE_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_LE_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_LE_printableProp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_LE_printableProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_LT_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " < "};
static const lean_object* lp_plausible_Plausible_LT_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_LT_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_LT_printableProp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_LT_printableProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_And_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ∧ "};
static const lean_object* lp_plausible_Plausible_And_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_And_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Or_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ∨ "};
static const lean_object* lp_plausible_Plausible_Or_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Or_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Iff_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ↔ "};
static const lean_object* lp_plausible_Plausible_Iff_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Iff_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Imp_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " → "};
static const lean_object* lp_plausible_Plausible_Imp_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Imp_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Not_printableProp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 1, .m_data = "¬"};
static const lean_object* lp_plausible_Plausible_Not_printableProp___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Not_printableProp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_True_printableProp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_plausible_Plausible_True_printableProp___closed__0 = (const lean_object*)&lp_plausible_Plausible_True_printableProp___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_True_printableProp = (const lean_object*)&lp_plausible_Plausible_True_printableProp___closed__0_value;
static const lean_string_object lp_plausible_Plausible_False_printableProp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_plausible_Plausible_False_printableProp___closed__0 = (const lean_object*)&lp_plausible_Plausible_False_printableProp___closed__0_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_False_printableProp = (const lean_object*)&lp_plausible_Plausible_False_printableProp___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_printableProp(uint8_t);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_printableProp___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "New sample"};
static const lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Retrying up to "};
static const lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__1_value;
static const lean_string_object lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = " times until guards hold"};
static const lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Plausible_Testable_checkIO___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "runtime"};
static const lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__0 = (const lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__0_value;
static const lean_string_object lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "maxRecDepth"};
static const lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__1 = (const lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__1_value;
static const lean_ctor_object lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(2, 128, 123, 132, 117, 90, 116, 101)}};
static const lean_ctor_object lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2_value_aux_0),((lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(88, 230, 219, 180, 63, 89, 202, 3)}};
static const lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2 = (const lean_object*)&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2_value;
static lean_once_cell_t lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3;
static lean_once_cell_t lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4;
static lean_once_cell_t lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5;
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "transform"};
static const lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___closed__0 = (const lean_object*)&lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___closed__0_value;
static const lean_array_object lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0 = (const lean_object*)&lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1;
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9(uint8_t, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0;
static lean_once_cell_t lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1;
static lean_once_cell_t lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2;
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_plausible_Plausible_Decorations_addDecorations___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_plausible_Plausible_Decorations_addDecorations___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Decorations_addDecorations___closed__0 = (const lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___closed__0_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "NamedBinder"};
static const lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__0 = (const lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1_value_aux_0),((lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(9, 85, 59, 187, 91, 187, 241, 93)}};
static const lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1 = (const lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 2}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__2 = (const lean_object*)&lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__2_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Decorations"};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__0 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "tacticMk_decorations"};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__1 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__1_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value_aux_0),((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__0_value),LEAN_SCALAR_PTR_LITERAL(62, 152, 114, 249, 205, 70, 85, 33)}};
static const lean_ctor_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value_aux_1),((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__1_value),LEAN_SCALAR_PTR_LITERAL(96, 214, 236, 0, 146, 57, 171, 143)}};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value;
static const lean_string_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "mk_decorations"};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__4 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__4_value)}};
static const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__5 = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__5_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_Decorations_tacticMk__decorations = (const lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__5_value;
static lean_once_cell_t lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "DecorationsOf"};
static const lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__0 = (const lean_object*)&lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3_value),LEAN_SCALAR_PTR_LITERAL(252, 145, 16, 124, 82, 207, 91, 147)}};
static const lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__1 = (const lean_object*)&lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__1_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__0 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__0_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__1_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__2_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__3 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__3_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_0),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_1),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__4_value_aux_2),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__4 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__4_value;
static const lean_array_object lp_plausible_Plausible_Testable_check___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__5 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__5_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__6 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_0),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_1),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__7_value_aux_2),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__7 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__7_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__8 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_check___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__9 = (const lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__9_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__10;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__11;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__12;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__13;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__14;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__15;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__16;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__17;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___auto__1___closed__18;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___auto__1;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__0;
static const lean_closure_object lp_plausible_Plausible_Testable_check___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__1 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__1_value;
static const lean_closure_object lp_plausible_Plausible_Testable_check___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__2 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__2_value;
static const lean_closure_object lp_plausible_Plausible_Testable_check___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__3 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__3_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "Unable to find a counter-example"};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__4 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_check___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__4_value)}};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__5 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__5_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__6;
static const lean_string_object lp_plausible_Plausible_Testable_check___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 73, .m_capacity = 73, .m_length = 72, .m_data = "Gave up after failing to generate values that fulfill the preconditions "};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__7 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__7_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = " times."};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__8 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__8_value;
static const lean_string_object lp_plausible_Plausible_Testable_check___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "Found a counter-example!"};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__9 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__9_value;
static const lean_ctor_object lp_plausible_Plausible_Testable_check___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__9_value)}};
static const lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__10 = (const lean_object*)&lp_plausible_Plausible_Testable_check___redArg___closed__10_value;
static lean_once_cell_t lp_plausible_Plausible_Testable_check___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible_Testable_check___redArg___closed__11;
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_plausible_Plausible_command_x23test___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "command#test_"};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__0 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__0_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__1_value_aux_0),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(10, 255, 199, 241, 136, 241, 234, 179)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__1 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__1_value;
static const lean_string_object lp_plausible_Plausible_command_x23test___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__2 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__2_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__3 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__3_value;
static const lean_string_object lp_plausible_Plausible_command_x23test___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "#test "};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__4 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__4_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__4_value)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__5 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__5_value;
static const lean_string_object lp_plausible_Plausible_command_x23test___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__6 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__6_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__7 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__7_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__8 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__8_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__3_value),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__5_value),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__8_value)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__9 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__9_value;
static const lean_ctor_object lp_plausible_Plausible_command_x23test___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__9_value)}};
static const lean_object* lp_plausible_Plausible_command_x23test___00__closed__10 = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_plausible_Plausible_command_x23test__ = (const lean_object*)&lp_plausible_Plausible_command_x23test___00__closed__10_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Command"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__0 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__0_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "eval"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__1 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__1_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_0),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_1),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(214, 208, 105, 11, 221, 56, 173, 240)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value_aux_2),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(5, 237, 202, 155, 22, 26, 166, 177)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "#eval"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__3 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__3_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__4 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__4_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__5 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__5_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_0),((lean_object*)&lp_plausible_Plausible_Testable_check___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_1),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value_aux_2),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "Testable.check"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__7 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__7_value;
static lean_once_cell_t lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Testable"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__9 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__9_value;
static const lean_string_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "check"};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__10 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__10_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(225, 247, 192, 195, 201, 247, 134, 245)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11_value_aux_0),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(172, 60, 7, 21, 195, 221, 198, 92)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(203, 125, 243, 6, 128, 122, 107, 14)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value_aux_0),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(113, 137, 77, 13, 23, 202, 166, 43)}};
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value_aux_1),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(220, 146, 134, 203, 33, 4, 219, 16)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__13 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__13_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__12_value)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__14 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__14_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__14_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__15 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__15_value;
static const lean_ctor_object lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__13_value),((lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__15_value)}};
static const lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__16 = (const lean_object*)&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__16_value;
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___redArg(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
default: 
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___redArg___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_plausible_Plausible_TestResult_ctorIdx___redArg(v_x_5_);
lean_dec_ref(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx(lean_object* v_p_7_, lean_object* v_x_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_plausible_Plausible_TestResult_ctorIdx___redArg(v_x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorIdx___boxed(lean_object* v_p_10_, lean_object* v_x_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_plausible_Plausible_TestResult_ctorIdx(v_p_10_, v_x_11_);
lean_dec_ref(v_x_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim___redArg(lean_object* v_t_13_, lean_object* v_k_14_){
_start:
{
switch(lean_obj_tag(v_t_13_))
{
case 0:
{
lean_object* v_a_15_; lean_object* v___x_16_; 
v_a_15_ = lean_ctor_get(v_t_13_, 0);
lean_inc_ref(v_a_15_);
lean_dec_ref_known(v_t_13_, 1);
v___x_16_ = lean_apply_1(v_k_14_, v_a_15_);
return v___x_16_;
}
case 1:
{
lean_object* v_a_17_; lean_object* v___x_18_; 
v_a_17_ = lean_ctor_get(v_t_13_, 0);
lean_inc(v_a_17_);
lean_dec_ref_known(v_t_13_, 1);
v___x_18_ = lean_apply_1(v_k_14_, v_a_17_);
return v___x_18_;
}
default: 
{
lean_object* v_a_19_; lean_object* v_a_20_; lean_object* v___x_21_; 
v_a_19_ = lean_ctor_get(v_t_13_, 0);
lean_inc(v_a_19_);
v_a_20_ = lean_ctor_get(v_t_13_, 1);
lean_inc(v_a_20_);
lean_dec_ref_known(v_t_13_, 2);
v___x_21_ = lean_apply_3(v_k_14_, lean_box(0), v_a_19_, v_a_20_);
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim(lean_object* v_p_22_, lean_object* v_motive_23_, lean_object* v_ctorIdx_24_, lean_object* v_t_25_, lean_object* v_h_26_, lean_object* v_k_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_25_, v_k_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_ctorElim___boxed(lean_object* v_p_29_, lean_object* v_motive_30_, lean_object* v_ctorIdx_31_, lean_object* v_t_32_, lean_object* v_h_33_, lean_object* v_k_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_plausible_Plausible_TestResult_ctorElim(v_p_29_, v_motive_30_, v_ctorIdx_31_, v_t_32_, v_h_33_, v_k_34_);
lean_dec(v_ctorIdx_31_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_success_elim___redArg(lean_object* v_t_36_, lean_object* v_success_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_36_, v_success_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_success_elim(lean_object* v_p_39_, lean_object* v_motive_40_, lean_object* v_t_41_, lean_object* v_h_42_, lean_object* v_success_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_41_, v_success_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_gaveUp_elim___redArg(lean_object* v_t_45_, lean_object* v_gaveUp_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_45_, v_gaveUp_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_gaveUp_elim(lean_object* v_p_48_, lean_object* v_motive_49_, lean_object* v_t_50_, lean_object* v_h_51_, lean_object* v_gaveUp_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_50_, v_gaveUp_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_failure_elim___redArg(lean_object* v_t_54_, lean_object* v_failure_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_54_, v_failure_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_failure_elim(lean_object* v_p_57_, lean_object* v_motive_58_, lean_object* v_t_59_, lean_object* v_h_60_, lean_object* v_failure_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = lp_plausible_Plausible_TestResult_ctorElim___redArg(v_t_59_, v_failure_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instInhabitedTestResult_default(lean_object* v_p_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = ((lean_object*)(lp_plausible_Plausible_instInhabitedTestResult_default___closed__0));
return v___x_66_;
}
}
static lean_object* _init_lp_plausible_Plausible_instInhabitedTestResult___closed__0(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_plausible_Plausible_instInhabitedTestResult_default(lean_box(0));
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instInhabitedTestResult(lean_object* v_a_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lean_obj_once(&lp_plausible_Plausible_instInhabitedTestResult___closed__0, &lp_plausible_Plausible_instInhabitedTestResult___closed__0_once, _init_lp_plausible_Plausible_instInhabitedTestResult___closed__0);
return v___x_69_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4(void){
_start:
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v___x_84_ = lean_box(0);
v___x_85_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__3));
v___x_86_ = l_Lean_mkConst(v___x_85_, v___x_84_);
return v___x_86_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8(void){
_start:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
v___x_92_ = lean_box(0);
v___x_93_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__7));
v___x_94_ = l_Lean_mkConst(v___x_93_, v___x_92_);
return v___x_94_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11(void){
_start:
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_99_ = lean_box(0);
v___x_100_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__10));
v___x_101_ = l_Lean_mkConst(v___x_100_, v___x_99_);
return v___x_101_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v_type_107_; 
v___x_105_ = lean_box(0);
v___x_106_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__13));
v_type_107_ = l_Lean_mkConst(v___x_106_, v___x_105_);
return v_type_107_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19(void){
_start:
{
lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_116_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__18));
v___x_117_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__17));
v___x_118_ = l_Lean_mkConst(v___x_117_, v___x_116_);
return v___x_118_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20(void){
_start:
{
lean_object* v_type_119_; lean_object* v___x_120_; lean_object* v___x_121_; 
v_type_119_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14);
v___x_120_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__19);
v___x_121_ = l_Lean_Expr_app___override(v___x_120_, v_type_119_);
return v___x_121_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_126_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__18));
v___x_127_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__22));
v___x_128_ = l_Lean_mkConst(v___x_127_, v___x_126_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instToExprConfiguration___lam__0(lean_object* v_cfg_129_){
_start:
{
lean_object* v_numInst_130_; lean_object* v_maxSize_131_; lean_object* v_numRetries_132_; uint8_t v_traceDiscarded_133_; uint8_t v_traceSuccesses_134_; uint8_t v_traceShrink_135_; uint8_t v_traceShrinkCandidates_136_; lean_object* v_randomSeed_137_; uint8_t v_quiet_138_; uint8_t v_sorryIfNoTestable_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___y_145_; lean_object* v___y_146_; lean_object* v___y_147_; lean_object* v___y_148_; lean_object* v___y_149_; lean_object* v___y_150_; lean_object* v___y_156_; lean_object* v___y_157_; lean_object* v___y_158_; lean_object* v___y_159_; lean_object* v___y_160_; lean_object* v___y_164_; lean_object* v___y_165_; lean_object* v___y_166_; lean_object* v___y_167_; lean_object* v___y_175_; lean_object* v___y_176_; lean_object* v___y_177_; lean_object* v___y_181_; lean_object* v___y_182_; lean_object* v___y_186_; 
v_numInst_130_ = lean_ctor_get(v_cfg_129_, 0);
lean_inc(v_numInst_130_);
v_maxSize_131_ = lean_ctor_get(v_cfg_129_, 1);
lean_inc(v_maxSize_131_);
v_numRetries_132_ = lean_ctor_get(v_cfg_129_, 2);
lean_inc(v_numRetries_132_);
v_traceDiscarded_133_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4);
v_traceSuccesses_134_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4 + 1);
v_traceShrink_135_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_136_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4 + 3);
v_randomSeed_137_ = lean_ctor_get(v_cfg_129_, 3);
lean_inc(v_randomSeed_137_);
v_quiet_138_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_139_ = lean_ctor_get_uint8(v_cfg_129_, sizeof(void*)*4 + 5);
lean_dec_ref(v_cfg_129_);
v___x_140_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__4);
v___x_141_ = l_Lean_mkNatLit(v_numInst_130_);
v___x_142_ = l_Lean_mkNatLit(v_maxSize_131_);
v___x_143_ = l_Lean_mkNatLit(v_numRetries_132_);
if (v_traceDiscarded_133_ == 0)
{
lean_object* v___x_189_; 
v___x_189_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
v___y_186_ = v___x_189_;
goto v___jp_185_;
}
else
{
lean_object* v___x_190_; 
v___x_190_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
v___y_186_ = v___x_190_;
goto v___jp_185_;
}
v___jp_144_:
{
if (v_sorryIfNoTestable_139_ == 0)
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
lean_inc_ref(v___y_150_);
lean_inc_ref(v___y_148_);
lean_inc_ref(v___y_149_);
lean_inc_ref(v___y_146_);
lean_inc_ref(v___y_145_);
v___x_152_ = l_Lean_mkApp10(v___x_140_, v___x_141_, v___x_142_, v___x_143_, v___y_145_, v___y_146_, v___y_149_, v___y_148_, v___y_147_, v___y_150_, v___x_151_);
return v___x_152_;
}
else
{
lean_object* v___x_153_; lean_object* v___x_154_; 
v___x_153_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
lean_inc_ref(v___y_150_);
lean_inc_ref(v___y_148_);
lean_inc_ref(v___y_149_);
lean_inc_ref(v___y_146_);
lean_inc_ref(v___y_145_);
v___x_154_ = l_Lean_mkApp10(v___x_140_, v___x_141_, v___x_142_, v___x_143_, v___y_145_, v___y_146_, v___y_149_, v___y_148_, v___y_147_, v___y_150_, v___x_153_);
return v___x_154_;
}
}
v___jp_155_:
{
if (v_quiet_138_ == 0)
{
lean_object* v___x_161_; 
v___x_161_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
v___y_145_ = v___y_156_;
v___y_146_ = v___y_157_;
v___y_147_ = v___y_160_;
v___y_148_ = v___y_158_;
v___y_149_ = v___y_159_;
v___y_150_ = v___x_161_;
goto v___jp_144_;
}
else
{
lean_object* v___x_162_; 
v___x_162_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
v___y_145_ = v___y_156_;
v___y_146_ = v___y_157_;
v___y_147_ = v___y_160_;
v___y_148_ = v___y_158_;
v___y_149_ = v___y_159_;
v___y_150_ = v___x_162_;
goto v___jp_144_;
}
}
v___jp_163_:
{
lean_object* v_type_168_; 
v_type_168_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14);
if (lean_obj_tag(v_randomSeed_137_) == 0)
{
lean_object* v___x_169_; 
v___x_169_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__20);
v___y_156_ = v___y_164_;
v___y_157_ = v___y_165_;
v___y_158_ = v___y_167_;
v___y_159_ = v___y_166_;
v___y_160_ = v___x_169_;
goto v___jp_155_;
}
else
{
lean_object* v_val_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; 
v_val_170_ = lean_ctor_get(v_randomSeed_137_, 0);
lean_inc(v_val_170_);
lean_dec_ref_known(v_randomSeed_137_, 1);
v___x_171_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__23);
v___x_172_ = l_Lean_mkNatLit(v_val_170_);
v___x_173_ = l_Lean_mkAppB(v___x_171_, v_type_168_, v___x_172_);
v___y_156_ = v___y_164_;
v___y_157_ = v___y_165_;
v___y_158_ = v___y_167_;
v___y_159_ = v___y_166_;
v___y_160_ = v___x_173_;
goto v___jp_155_;
}
}
v___jp_174_:
{
if (v_traceShrinkCandidates_136_ == 0)
{
lean_object* v___x_178_; 
v___x_178_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
v___y_164_ = v___y_175_;
v___y_165_ = v___y_176_;
v___y_166_ = v___y_177_;
v___y_167_ = v___x_178_;
goto v___jp_163_;
}
else
{
lean_object* v___x_179_; 
v___x_179_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
v___y_164_ = v___y_175_;
v___y_165_ = v___y_176_;
v___y_166_ = v___y_177_;
v___y_167_ = v___x_179_;
goto v___jp_163_;
}
}
v___jp_180_:
{
if (v_traceShrink_135_ == 0)
{
lean_object* v___x_183_; 
v___x_183_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
v___y_175_ = v___y_181_;
v___y_176_ = v___y_182_;
v___y_177_ = v___x_183_;
goto v___jp_174_;
}
else
{
lean_object* v___x_184_; 
v___x_184_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
v___y_175_ = v___y_181_;
v___y_176_ = v___y_182_;
v___y_177_ = v___x_184_;
goto v___jp_174_;
}
}
v___jp_185_:
{
if (v_traceSuccesses_134_ == 0)
{
lean_object* v___x_187_; 
v___x_187_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__8);
v___y_181_ = v___y_186_;
v___y_182_ = v___x_187_;
goto v___jp_180_;
}
else
{
lean_object* v___x_188_; 
v___x_188_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__11);
v___y_181_ = v___y_186_;
v___y_182_ = v___x_188_;
goto v___jp_180_;
}
}
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___closed__2(void){
_start:
{
lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v___x_194_ = lean_box(0);
v___x_195_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___closed__1));
v___x_196_ = l_Lean_mkConst(v___x_195_, v___x_194_);
return v___x_196_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration___closed__3(void){
_start:
{
lean_object* v___x_197_; lean_object* v___f_198_; lean_object* v___x_199_; 
v___x_197_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___closed__2, &lp_plausible_Plausible_instToExprConfiguration___closed__2_once, _init_lp_plausible_Plausible_instToExprConfiguration___closed__2);
v___f_198_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___closed__0));
v___x_199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_199_, 0, v___f_198_);
lean_ctor_set(v___x_199_, 1, v___x_197_);
return v___x_199_;
}
}
static lean_object* _init_lp_plausible_Plausible_instToExprConfiguration(void){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___closed__3, &lp_plausible_Plausible_instToExprConfiguration___closed__3_once, _init_lp_plausible_Plausible_instToExprConfiguration___closed__3);
return v___x_200_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_201_ = lean_box(0);
v___x_202_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_203_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
lean_ctor_set(v___x_203_, 1, v___x_201_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
v___x_205_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0, &lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0_once, _init_lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___closed__0);
v___x_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg___boxed(lean_object* v___y_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg();
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0(lean_object* v_00_u03b1_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg();
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0(v_00_u03b1_216_, v___y_217_, v___y_218_, v___y_219_, v___y_220_);
lean_dec(v___y_220_);
lean_dec_ref(v___y_219_);
lean_dec(v___y_218_);
lean_dec_ref(v___y_217_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1(lean_object* v_msgData_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_){
_start:
{
lean_object* v___x_229_; lean_object* v_env_230_; lean_object* v___x_231_; lean_object* v_mctx_232_; lean_object* v_lctx_233_; lean_object* v_options_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v___x_229_ = lean_st_ref_get(v___y_227_);
v_env_230_ = lean_ctor_get(v___x_229_, 0);
lean_inc_ref(v_env_230_);
lean_dec(v___x_229_);
v___x_231_ = lean_st_ref_get(v___y_225_);
v_mctx_232_ = lean_ctor_get(v___x_231_, 0);
lean_inc_ref(v_mctx_232_);
lean_dec(v___x_231_);
v_lctx_233_ = lean_ctor_get(v___y_224_, 2);
v_options_234_ = lean_ctor_get(v___y_226_, 2);
lean_inc_ref(v_options_234_);
lean_inc_ref(v_lctx_233_);
v___x_235_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_235_, 0, v_env_230_);
lean_ctor_set(v___x_235_, 1, v_mctx_232_);
lean_ctor_set(v___x_235_, 2, v_lctx_233_);
lean_ctor_set(v___x_235_, 3, v_options_234_);
v___x_236_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_235_);
lean_ctor_set(v___x_236_, 1, v_msgData_223_);
v___x_237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_237_, 0, v___x_236_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1___boxed(lean_object* v_msgData_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_){
_start:
{
lean_object* v_res_244_; 
v_res_244_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1(v_msgData_238_, v___y_239_, v___y_240_, v___y_241_, v___y_242_);
lean_dec(v___y_242_);
lean_dec_ref(v___y_241_);
lean_dec(v___y_240_);
lean_dec_ref(v___y_239_);
return v_res_244_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg(lean_object* v_msg_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
lean_object* v_ref_251_; lean_object* v___x_252_; lean_object* v_a_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_261_; 
v_ref_251_ = lean_ctor_get(v___y_248_, 5);
v___x_252_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1(v_msg_245_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
v_a_253_ = lean_ctor_get(v___x_252_, 0);
v_isSharedCheck_261_ = !lean_is_exclusive(v___x_252_);
if (v_isSharedCheck_261_ == 0)
{
v___x_255_ = v___x_252_;
v_isShared_256_ = v_isSharedCheck_261_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_a_253_);
lean_dec(v___x_252_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_261_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v___x_259_; 
lean_inc(v_ref_251_);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v_ref_251_);
lean_ctor_set(v___x_257_, 1, v_a_253_);
if (v_isShared_256_ == 0)
{
lean_ctor_set_tag(v___x_255_, 1);
lean_ctor_set(v___x_255_, 0, v___x_257_);
v___x_259_ = v___x_255_;
goto v_reusejp_258_;
}
else
{
lean_object* v_reuseFailAlloc_260_; 
v_reuseFailAlloc_260_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_260_, 0, v___x_257_);
v___x_259_ = v_reuseFailAlloc_260_;
goto v_reusejp_258_;
}
v_reusejp_258_:
{
return v___x_259_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg___boxed(lean_object* v_msg_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_){
_start:
{
lean_object* v_res_268_; 
v_res_268_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg(v_msg_262_, v___y_263_, v___y_264_, v___y_265_, v___y_266_);
lean_dec(v___y_266_);
lean_dec_ref(v___y_265_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
return v_res_268_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0(void){
_start:
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = l_Lean_Elab_ConfigEval_EvalExpr_instNat;
v___x_270_ = l_Lean_Elab_ConfigEval_EvalExpr_instOption___redArg(v___x_269_);
return v___x_270_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__1));
v___x_273_ = l_Lean_stringToMessageData(v___x_272_);
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0(lean_object* v_ctor_274_, lean_object* v_args_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_){
_start:
{
lean_object* v___x_419_; uint8_t v___x_420_; 
v___x_419_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__2));
v___x_420_ = lean_string_dec_eq(v_ctor_274_, v___x_419_);
if (v___x_420_ == 0)
{
lean_object* v___x_421_; 
v___x_421_ = lp_plausible_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__0___redArg();
return v___x_421_;
}
else
{
lean_object* v___x_422_; lean_object* v___x_423_; uint8_t v___x_424_; 
v___x_422_ = lean_array_get_size(v_args_275_);
v___x_423_ = lean_unsigned_to_nat(10u);
v___x_424_ = lean_nat_dec_eq(v___x_422_, v___x_423_);
if (v___x_424_ == 0)
{
lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v_a_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_434_; 
v___x_425_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__2);
v___x_426_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg(v___x_425_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
v_a_427_ = lean_ctor_get(v___x_426_, 0);
v_isSharedCheck_434_ = !lean_is_exclusive(v___x_426_);
if (v_isSharedCheck_434_ == 0)
{
v___x_429_ = v___x_426_;
v_isShared_430_ = v_isSharedCheck_434_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_a_427_);
lean_dec(v___x_426_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_434_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v___x_432_; 
if (v_isShared_430_ == 0)
{
v___x_432_ = v___x_429_;
goto v_reusejp_431_;
}
else
{
lean_object* v_reuseFailAlloc_433_; 
v_reuseFailAlloc_433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_433_, 0, v_a_427_);
v___x_432_ = v_reuseFailAlloc_433_;
goto v_reusejp_431_;
}
v_reusejp_431_:
{
return v___x_432_;
}
}
}
else
{
goto v___jp_281_;
}
}
v___jp_281_:
{
lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; 
v___x_282_ = l_Lean_instInhabitedExpr;
v___x_283_ = lean_unsigned_to_nat(0u);
v___x_284_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_283_);
lean_inc(v___x_284_);
v___x_285_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v___x_284_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_285_) == 0)
{
lean_object* v_a_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v_a_286_ = lean_ctor_get(v___x_285_, 0);
lean_inc(v_a_286_);
lean_dec_ref_known(v___x_285_, 1);
v___x_287_ = lean_unsigned_to_nat(1u);
v___x_288_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_287_);
lean_inc(v___x_288_);
v___x_289_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v___x_288_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_289_) == 0)
{
lean_object* v_a_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; 
v_a_290_ = lean_ctor_get(v___x_289_, 0);
lean_inc(v_a_290_);
lean_dec_ref_known(v___x_289_, 1);
v___x_291_ = lean_unsigned_to_nat(2u);
v___x_292_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_291_);
lean_inc(v___x_292_);
v___x_293_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v___x_292_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_293_) == 0)
{
lean_object* v_a_294_; lean_object* v___x_295_; lean_object* v___x_296_; lean_object* v___x_297_; 
v_a_294_ = lean_ctor_get(v___x_293_, 0);
lean_inc(v_a_294_);
lean_dec_ref_known(v___x_293_, 1);
v___x_295_ = lean_unsigned_to_nat(3u);
v___x_296_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_295_);
lean_inc(v___x_296_);
v___x_297_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_296_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v_a_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v_a_298_ = lean_ctor_get(v___x_297_, 0);
lean_inc(v_a_298_);
lean_dec_ref_known(v___x_297_, 1);
v___x_299_ = lean_unsigned_to_nat(4u);
v___x_300_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_299_);
lean_inc(v___x_300_);
v___x_301_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_300_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_301_) == 0)
{
lean_object* v_a_302_; lean_object* v___x_303_; lean_object* v___x_304_; lean_object* v___x_305_; 
v_a_302_ = lean_ctor_get(v___x_301_, 0);
lean_inc(v_a_302_);
lean_dec_ref_known(v___x_301_, 1);
v___x_303_ = lean_unsigned_to_nat(5u);
v___x_304_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_303_);
lean_inc(v___x_304_);
v___x_305_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_304_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_305_) == 0)
{
lean_object* v_a_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; 
v_a_306_ = lean_ctor_get(v___x_305_, 0);
lean_inc(v_a_306_);
lean_dec_ref_known(v___x_305_, 1);
v___x_307_ = lean_unsigned_to_nat(6u);
v___x_308_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_307_);
lean_inc(v___x_308_);
v___x_309_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_308_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_309_) == 0)
{
lean_object* v_a_310_; lean_object* v___x_311_; lean_object* v_evalExpr_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; 
v_a_310_ = lean_ctor_get(v___x_309_, 0);
lean_inc(v_a_310_);
lean_dec_ref_known(v___x_309_, 1);
v___x_311_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0);
v_evalExpr_312_ = lean_ctor_get(v___x_311_, 0);
v___x_313_ = lean_unsigned_to_nat(7u);
v___x_314_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_313_);
lean_inc_ref(v_evalExpr_312_);
lean_inc(v___y_279_);
lean_inc_ref(v___y_278_);
lean_inc(v___y_277_);
lean_inc_ref(v___y_276_);
lean_inc(v___x_314_);
v___x_315_ = lean_apply_6(v_evalExpr_312_, v___x_314_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, lean_box(0));
if (lean_obj_tag(v___x_315_) == 0)
{
lean_object* v_a_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; 
v_a_316_ = lean_ctor_get(v___x_315_, 0);
lean_inc(v_a_316_);
lean_dec_ref_known(v___x_315_, 1);
v___x_317_ = lean_unsigned_to_nat(8u);
v___x_318_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_317_);
lean_inc(v___x_318_);
v___x_319_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_318_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_319_) == 0)
{
lean_object* v_a_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; 
v_a_320_ = lean_ctor_get(v___x_319_, 0);
lean_inc(v_a_320_);
lean_dec_ref_known(v___x_319_, 1);
v___x_321_ = lean_unsigned_to_nat(9u);
v___x_322_ = lean_array_get_borrowed(v___x_282_, v_args_275_, v___x_321_);
lean_inc(v___x_322_);
v___x_323_ = l_Lean_Elab_ConfigEval_EvalExpr_evalBoolExpr(v___x_322_, v___y_276_, v___y_277_, v___y_278_, v___y_279_);
if (lean_obj_tag(v___x_323_) == 0)
{
lean_object* v_a_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_338_; 
v_a_324_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_338_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_338_ == 0)
{
v___x_326_ = v___x_323_;
v_isShared_327_ = v_isSharedCheck_338_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_a_324_);
lean_dec(v___x_323_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_338_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v___x_328_; uint8_t v___x_329_; uint8_t v___x_330_; uint8_t v___x_331_; uint8_t v___x_332_; uint8_t v___x_333_; uint8_t v___x_334_; lean_object* v___x_336_; 
v___x_328_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v___x_328_, 0, v_a_286_);
lean_ctor_set(v___x_328_, 1, v_a_290_);
lean_ctor_set(v___x_328_, 2, v_a_294_);
lean_ctor_set(v___x_328_, 3, v_a_316_);
v___x_329_ = lean_unbox(v_a_298_);
lean_dec(v_a_298_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4, v___x_329_);
v___x_330_ = lean_unbox(v_a_302_);
lean_dec(v_a_302_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4 + 1, v___x_330_);
v___x_331_ = lean_unbox(v_a_306_);
lean_dec(v_a_306_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4 + 2, v___x_331_);
v___x_332_ = lean_unbox(v_a_310_);
lean_dec(v_a_310_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4 + 3, v___x_332_);
v___x_333_ = lean_unbox(v_a_320_);
lean_dec(v_a_320_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4 + 4, v___x_333_);
v___x_334_ = lean_unbox(v_a_324_);
lean_dec(v_a_324_);
lean_ctor_set_uint8(v___x_328_, sizeof(void*)*4 + 5, v___x_334_);
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_328_);
v___x_336_ = v___x_326_;
goto v_reusejp_335_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_328_);
v___x_336_ = v_reuseFailAlloc_337_;
goto v_reusejp_335_;
}
v_reusejp_335_:
{
return v___x_336_;
}
}
}
else
{
lean_object* v_a_339_; lean_object* v___x_341_; uint8_t v_isShared_342_; uint8_t v_isSharedCheck_346_; 
lean_dec(v_a_320_);
lean_dec(v_a_316_);
lean_dec(v_a_310_);
lean_dec(v_a_306_);
lean_dec(v_a_302_);
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_339_ = lean_ctor_get(v___x_323_, 0);
v_isSharedCheck_346_ = !lean_is_exclusive(v___x_323_);
if (v_isSharedCheck_346_ == 0)
{
v___x_341_ = v___x_323_;
v_isShared_342_ = v_isSharedCheck_346_;
goto v_resetjp_340_;
}
else
{
lean_inc(v_a_339_);
lean_dec(v___x_323_);
v___x_341_ = lean_box(0);
v_isShared_342_ = v_isSharedCheck_346_;
goto v_resetjp_340_;
}
v_resetjp_340_:
{
lean_object* v___x_344_; 
if (v_isShared_342_ == 0)
{
v___x_344_ = v___x_341_;
goto v_reusejp_343_;
}
else
{
lean_object* v_reuseFailAlloc_345_; 
v_reuseFailAlloc_345_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_345_, 0, v_a_339_);
v___x_344_ = v_reuseFailAlloc_345_;
goto v_reusejp_343_;
}
v_reusejp_343_:
{
return v___x_344_;
}
}
}
}
else
{
lean_object* v_a_347_; lean_object* v___x_349_; uint8_t v_isShared_350_; uint8_t v_isSharedCheck_354_; 
lean_dec(v_a_316_);
lean_dec(v_a_310_);
lean_dec(v_a_306_);
lean_dec(v_a_302_);
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_347_ = lean_ctor_get(v___x_319_, 0);
v_isSharedCheck_354_ = !lean_is_exclusive(v___x_319_);
if (v_isSharedCheck_354_ == 0)
{
v___x_349_ = v___x_319_;
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
else
{
lean_inc(v_a_347_);
lean_dec(v___x_319_);
v___x_349_ = lean_box(0);
v_isShared_350_ = v_isSharedCheck_354_;
goto v_resetjp_348_;
}
v_resetjp_348_:
{
lean_object* v___x_352_; 
if (v_isShared_350_ == 0)
{
v___x_352_ = v___x_349_;
goto v_reusejp_351_;
}
else
{
lean_object* v_reuseFailAlloc_353_; 
v_reuseFailAlloc_353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_353_, 0, v_a_347_);
v___x_352_ = v_reuseFailAlloc_353_;
goto v_reusejp_351_;
}
v_reusejp_351_:
{
return v___x_352_;
}
}
}
}
else
{
lean_object* v_a_355_; lean_object* v___x_357_; uint8_t v_isShared_358_; uint8_t v_isSharedCheck_362_; 
lean_dec(v_a_310_);
lean_dec(v_a_306_);
lean_dec(v_a_302_);
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_355_ = lean_ctor_get(v___x_315_, 0);
v_isSharedCheck_362_ = !lean_is_exclusive(v___x_315_);
if (v_isSharedCheck_362_ == 0)
{
v___x_357_ = v___x_315_;
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
else
{
lean_inc(v_a_355_);
lean_dec(v___x_315_);
v___x_357_ = lean_box(0);
v_isShared_358_ = v_isSharedCheck_362_;
goto v_resetjp_356_;
}
v_resetjp_356_:
{
lean_object* v___x_360_; 
if (v_isShared_358_ == 0)
{
v___x_360_ = v___x_357_;
goto v_reusejp_359_;
}
else
{
lean_object* v_reuseFailAlloc_361_; 
v_reuseFailAlloc_361_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_361_, 0, v_a_355_);
v___x_360_ = v_reuseFailAlloc_361_;
goto v_reusejp_359_;
}
v_reusejp_359_:
{
return v___x_360_;
}
}
}
}
else
{
lean_object* v_a_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_370_; 
lean_dec(v_a_306_);
lean_dec(v_a_302_);
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_363_ = lean_ctor_get(v___x_309_, 0);
v_isSharedCheck_370_ = !lean_is_exclusive(v___x_309_);
if (v_isSharedCheck_370_ == 0)
{
v___x_365_ = v___x_309_;
v_isShared_366_ = v_isSharedCheck_370_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_a_363_);
lean_dec(v___x_309_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_370_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
lean_object* v___x_368_; 
if (v_isShared_366_ == 0)
{
v___x_368_ = v___x_365_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_369_; 
v_reuseFailAlloc_369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_369_, 0, v_a_363_);
v___x_368_ = v_reuseFailAlloc_369_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
return v___x_368_;
}
}
}
}
else
{
lean_object* v_a_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_378_; 
lean_dec(v_a_302_);
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_371_ = lean_ctor_get(v___x_305_, 0);
v_isSharedCheck_378_ = !lean_is_exclusive(v___x_305_);
if (v_isSharedCheck_378_ == 0)
{
v___x_373_ = v___x_305_;
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_a_371_);
lean_dec(v___x_305_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_378_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_376_; 
if (v_isShared_374_ == 0)
{
v___x_376_ = v___x_373_;
goto v_reusejp_375_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_a_371_);
v___x_376_ = v_reuseFailAlloc_377_;
goto v_reusejp_375_;
}
v_reusejp_375_:
{
return v___x_376_;
}
}
}
}
else
{
lean_object* v_a_379_; lean_object* v___x_381_; uint8_t v_isShared_382_; uint8_t v_isSharedCheck_386_; 
lean_dec(v_a_298_);
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_379_ = lean_ctor_get(v___x_301_, 0);
v_isSharedCheck_386_ = !lean_is_exclusive(v___x_301_);
if (v_isSharedCheck_386_ == 0)
{
v___x_381_ = v___x_301_;
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
else
{
lean_inc(v_a_379_);
lean_dec(v___x_301_);
v___x_381_ = lean_box(0);
v_isShared_382_ = v_isSharedCheck_386_;
goto v_resetjp_380_;
}
v_resetjp_380_:
{
lean_object* v___x_384_; 
if (v_isShared_382_ == 0)
{
v___x_384_ = v___x_381_;
goto v_reusejp_383_;
}
else
{
lean_object* v_reuseFailAlloc_385_; 
v_reuseFailAlloc_385_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_385_, 0, v_a_379_);
v___x_384_ = v_reuseFailAlloc_385_;
goto v_reusejp_383_;
}
v_reusejp_383_:
{
return v___x_384_;
}
}
}
}
else
{
lean_object* v_a_387_; lean_object* v___x_389_; uint8_t v_isShared_390_; uint8_t v_isSharedCheck_394_; 
lean_dec(v_a_294_);
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_387_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_394_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_394_ == 0)
{
v___x_389_ = v___x_297_;
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
else
{
lean_inc(v_a_387_);
lean_dec(v___x_297_);
v___x_389_ = lean_box(0);
v_isShared_390_ = v_isSharedCheck_394_;
goto v_resetjp_388_;
}
v_resetjp_388_:
{
lean_object* v___x_392_; 
if (v_isShared_390_ == 0)
{
v___x_392_ = v___x_389_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v_a_387_);
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
else
{
lean_object* v_a_395_; lean_object* v___x_397_; uint8_t v_isShared_398_; uint8_t v_isSharedCheck_402_; 
lean_dec(v_a_290_);
lean_dec(v_a_286_);
v_a_395_ = lean_ctor_get(v___x_293_, 0);
v_isSharedCheck_402_ = !lean_is_exclusive(v___x_293_);
if (v_isSharedCheck_402_ == 0)
{
v___x_397_ = v___x_293_;
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
else
{
lean_inc(v_a_395_);
lean_dec(v___x_293_);
v___x_397_ = lean_box(0);
v_isShared_398_ = v_isSharedCheck_402_;
goto v_resetjp_396_;
}
v_resetjp_396_:
{
lean_object* v___x_400_; 
if (v_isShared_398_ == 0)
{
v___x_400_ = v___x_397_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_a_395_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
return v___x_400_;
}
}
}
}
else
{
lean_object* v_a_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_410_; 
lean_dec(v_a_286_);
v_a_403_ = lean_ctor_get(v___x_289_, 0);
v_isSharedCheck_410_ = !lean_is_exclusive(v___x_289_);
if (v_isSharedCheck_410_ == 0)
{
v___x_405_ = v___x_289_;
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_a_403_);
lean_dec(v___x_289_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_410_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
lean_object* v___x_408_; 
if (v_isShared_406_ == 0)
{
v___x_408_ = v___x_405_;
goto v_reusejp_407_;
}
else
{
lean_object* v_reuseFailAlloc_409_; 
v_reuseFailAlloc_409_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_409_, 0, v_a_403_);
v___x_408_ = v_reuseFailAlloc_409_;
goto v_reusejp_407_;
}
v_reusejp_407_:
{
return v___x_408_;
}
}
}
}
else
{
lean_object* v_a_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_418_; 
v_a_411_ = lean_ctor_get(v___x_285_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_418_ == 0)
{
v___x_413_ = v___x_285_;
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_a_411_);
lean_dec(v___x_285_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___x_416_; 
if (v_isShared_414_ == 0)
{
v___x_416_ = v___x_413_;
goto v_reusejp_415_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_a_411_);
v___x_416_ = v_reuseFailAlloc_417_;
goto v_reusejp_415_;
}
v_reusejp_415_:
{
return v___x_416_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___boxed(lean_object* v_ctor_435_, lean_object* v_args_436_, lean_object* v___y_437_, lean_object* v___y_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0(v_ctor_435_, v_args_436_, v___y_437_, v___y_438_, v___y_439_, v___y_440_);
lean_dec(v___y_440_);
lean_dec_ref(v___y_439_);
lean_dec(v___y_438_);
lean_dec_ref(v___y_437_);
lean_dec_ref(v_args_436_);
lean_dec_ref(v_ctor_435_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr(lean_object* v_a_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_){
_start:
{
lean_object* v___f_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v___f_453_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__0));
v___x_454_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1));
v___x_455_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_454_, v___f_453_, v_a_447_, v_a_448_, v_a_449_, v_a_450_, v_a_451_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___boxed(lean_object* v_a_456_, lean_object* v_a_457_, lean_object* v_a_458_, lean_object* v_a_459_, lean_object* v_a_460_, lean_object* v_a_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr(v_a_456_, v_a_457_, v_a_458_, v_a_459_, v_a_460_);
lean_dec(v_a_460_);
lean_dec_ref(v_a_459_);
lean_dec(v_a_458_);
lean_dec_ref(v_a_457_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1(lean_object* v_00_u03b1_463_, lean_object* v_msg_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_, lean_object* v___y_468_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___redArg(v_msg_464_, v___y_465_, v___y_466_, v___y_467_, v___y_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1___boxed(lean_object* v_00_u03b1_471_, lean_object* v_msg_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_plausible_Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1(v_00_u03b1_471_, v_msg_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
return v_res_478_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1(void){
_start:
{
lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; 
v___x_480_ = lean_box(0);
v___x_481_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1));
v___x_482_ = l_Lean_Expr_const___override(v___x_481_, v___x_480_);
return v___x_482_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2(void){
_start:
{
lean_object* v___x_483_; lean_object* v___x_484_; 
v___x_483_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1);
v___x_484_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_484_, 0, v___x_483_);
return v___x_484_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3(void){
_start:
{
lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_485_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2);
v___x_486_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__0));
v___x_487_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_486_);
lean_ctor_set(v___x_487_, 1, v___x_485_);
return v___x_487_;
}
}
static lean_object* _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration(void){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__3);
return v___x_488_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0(void){
_start:
{
lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_489_ = lean_box(0);
v___x_490_ = l_Lean_Elab_abortTermExceptionId;
v___x_491_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_491_, 0, v___x_490_);
lean_ctor_set(v___x_491_, 1, v___x_489_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg(){
_start:
{
lean_object* v___x_493_; lean_object* v___x_494_; 
v___x_493_ = lean_obj_once(&lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0, &lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0_once, _init_lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___closed__0);
v___x_494_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_494_, 0, v___x_493_);
return v___x_494_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg___boxed(lean_object* v___y_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(lean_object* v_e_497_, lean_object* v___y_498_){
_start:
{
uint8_t v___x_500_; 
v___x_500_ = l_Lean_Expr_hasMVar(v_e_497_);
if (v___x_500_ == 0)
{
lean_object* v___x_501_; 
v___x_501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_501_, 0, v_e_497_);
return v___x_501_;
}
else
{
lean_object* v___x_502_; lean_object* v_mctx_503_; lean_object* v___x_504_; lean_object* v_fst_505_; lean_object* v_snd_506_; lean_object* v___x_507_; lean_object* v_cache_508_; lean_object* v_zetaDeltaFVarIds_509_; lean_object* v_postponed_510_; lean_object* v_diag_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_520_; 
v___x_502_ = lean_st_ref_get(v___y_498_);
v_mctx_503_ = lean_ctor_get(v___x_502_, 0);
lean_inc_ref(v_mctx_503_);
lean_dec(v___x_502_);
v___x_504_ = l_Lean_instantiateMVarsCore(v_mctx_503_, v_e_497_);
v_fst_505_ = lean_ctor_get(v___x_504_, 0);
lean_inc(v_fst_505_);
v_snd_506_ = lean_ctor_get(v___x_504_, 1);
lean_inc(v_snd_506_);
lean_dec_ref(v___x_504_);
v___x_507_ = lean_st_ref_take(v___y_498_);
v_cache_508_ = lean_ctor_get(v___x_507_, 1);
v_zetaDeltaFVarIds_509_ = lean_ctor_get(v___x_507_, 2);
v_postponed_510_ = lean_ctor_get(v___x_507_, 3);
v_diag_511_ = lean_ctor_get(v___x_507_, 4);
v_isSharedCheck_520_ = !lean_is_exclusive(v___x_507_);
if (v_isSharedCheck_520_ == 0)
{
lean_object* v_unused_521_; 
v_unused_521_ = lean_ctor_get(v___x_507_, 0);
lean_dec(v_unused_521_);
v___x_513_ = v___x_507_;
v_isShared_514_ = v_isSharedCheck_520_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_diag_511_);
lean_inc(v_postponed_510_);
lean_inc(v_zetaDeltaFVarIds_509_);
lean_inc(v_cache_508_);
lean_dec(v___x_507_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_520_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
lean_ctor_set(v___x_513_, 0, v_snd_506_);
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_snd_506_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v_cache_508_);
lean_ctor_set(v_reuseFailAlloc_519_, 2, v_zetaDeltaFVarIds_509_);
lean_ctor_set(v_reuseFailAlloc_519_, 3, v_postponed_510_);
lean_ctor_set(v_reuseFailAlloc_519_, 4, v_diag_511_);
v___x_516_ = v_reuseFailAlloc_519_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
lean_object* v___x_517_; lean_object* v___x_518_; 
v___x_517_ = lean_st_ref_set(v___y_498_, v___x_516_);
v___x_518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_518_, 0, v_fst_505_);
return v___x_518_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg___boxed(lean_object* v_e_522_, lean_object* v___y_523_, lean_object* v___y_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(v_e_522_, v___y_523_);
lean_dec(v___y_523_);
return v_res_525_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0(void){
_start:
{
lean_object* v___x_526_; lean_object* v___x_527_; 
v___x_526_ = lean_box(1);
v___x_527_ = l_Lean_MessageData_ofFormat(v___x_526_);
return v___x_527_;
}
}
static lean_object* _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3(void){
_start:
{
lean_object* v___x_531_; lean_object* v___x_532_; 
v___x_531_ = ((lean_object*)(lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__2));
v___x_532_ = l_Lean_MessageData_ofFormat(v___x_531_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(lean_object* v_x_533_, lean_object* v_x_534_){
_start:
{
if (lean_obj_tag(v_x_534_) == 0)
{
return v_x_533_;
}
else
{
lean_object* v_head_535_; lean_object* v_tail_536_; lean_object* v___x_538_; uint8_t v_isShared_539_; uint8_t v_isSharedCheck_558_; 
v_head_535_ = lean_ctor_get(v_x_534_, 0);
v_tail_536_ = lean_ctor_get(v_x_534_, 1);
v_isSharedCheck_558_ = !lean_is_exclusive(v_x_534_);
if (v_isSharedCheck_558_ == 0)
{
v___x_538_ = v_x_534_;
v_isShared_539_ = v_isSharedCheck_558_;
goto v_resetjp_537_;
}
else
{
lean_inc(v_tail_536_);
lean_inc(v_head_535_);
lean_dec(v_x_534_);
v___x_538_ = lean_box(0);
v_isShared_539_ = v_isSharedCheck_558_;
goto v_resetjp_537_;
}
v_resetjp_537_:
{
lean_object* v_before_540_; lean_object* v___x_542_; uint8_t v_isShared_543_; uint8_t v_isSharedCheck_556_; 
v_before_540_ = lean_ctor_get(v_head_535_, 0);
v_isSharedCheck_556_ = !lean_is_exclusive(v_head_535_);
if (v_isSharedCheck_556_ == 0)
{
lean_object* v_unused_557_; 
v_unused_557_ = lean_ctor_get(v_head_535_, 1);
lean_dec(v_unused_557_);
v___x_542_ = v_head_535_;
v_isShared_543_ = v_isSharedCheck_556_;
goto v_resetjp_541_;
}
else
{
lean_inc(v_before_540_);
lean_dec(v_head_535_);
v___x_542_ = lean_box(0);
v_isShared_543_ = v_isSharedCheck_556_;
goto v_resetjp_541_;
}
v_resetjp_541_:
{
lean_object* v___x_544_; lean_object* v___x_546_; 
v___x_544_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0);
if (v_isShared_543_ == 0)
{
lean_ctor_set_tag(v___x_542_, 7);
lean_ctor_set(v___x_542_, 1, v___x_544_);
lean_ctor_set(v___x_542_, 0, v_x_533_);
v___x_546_ = v___x_542_;
goto v_reusejp_545_;
}
else
{
lean_object* v_reuseFailAlloc_555_; 
v_reuseFailAlloc_555_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_555_, 0, v_x_533_);
lean_ctor_set(v_reuseFailAlloc_555_, 1, v___x_544_);
v___x_546_ = v_reuseFailAlloc_555_;
goto v_reusejp_545_;
}
v_reusejp_545_:
{
lean_object* v___x_547_; lean_object* v___x_549_; 
v___x_547_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__3);
if (v_isShared_539_ == 0)
{
lean_ctor_set_tag(v___x_538_, 7);
lean_ctor_set(v___x_538_, 1, v___x_547_);
lean_ctor_set(v___x_538_, 0, v___x_546_);
v___x_549_ = v___x_538_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_554_; 
v_reuseFailAlloc_554_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_554_, 0, v___x_546_);
lean_ctor_set(v_reuseFailAlloc_554_, 1, v___x_547_);
v___x_549_ = v_reuseFailAlloc_554_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
lean_object* v___x_550_; lean_object* v___x_551_; lean_object* v___x_552_; 
v___x_550_ = l_Lean_MessageData_ofSyntax(v_before_540_);
v___x_551_ = l_Lean_indentD(v___x_550_);
v___x_552_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_552_, 0, v___x_549_);
lean_ctor_set(v___x_552_, 1, v___x_551_);
v_x_533_ = v___x_552_;
v_x_534_ = v_tail_536_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(lean_object* v_opts_559_, lean_object* v_opt_560_){
_start:
{
lean_object* v_name_561_; lean_object* v_defValue_562_; lean_object* v_map_563_; lean_object* v___x_564_; 
v_name_561_ = lean_ctor_get(v_opt_560_, 0);
v_defValue_562_ = lean_ctor_get(v_opt_560_, 1);
v_map_563_ = lean_ctor_get(v_opts_559_, 0);
v___x_564_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_563_, v_name_561_);
if (lean_obj_tag(v___x_564_) == 0)
{
uint8_t v___x_565_; 
v___x_565_ = lean_unbox(v_defValue_562_);
return v___x_565_;
}
else
{
lean_object* v_val_566_; 
v_val_566_ = lean_ctor_get(v___x_564_, 0);
lean_inc(v_val_566_);
lean_dec_ref_known(v___x_564_, 1);
if (lean_obj_tag(v_val_566_) == 1)
{
uint8_t v_v_567_; 
v_v_567_ = lean_ctor_get_uint8(v_val_566_, 0);
lean_dec_ref_known(v_val_566_, 0);
return v_v_567_;
}
else
{
uint8_t v___x_568_; 
lean_dec(v_val_566_);
v___x_568_ = lean_unbox(v_defValue_562_);
return v___x_568_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8___boxed(lean_object* v_opts_569_, lean_object* v_opt_570_){
_start:
{
uint8_t v_res_571_; lean_object* v_r_572_; 
v_res_571_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(v_opts_569_, v_opt_570_);
lean_dec_ref(v_opt_570_);
lean_dec_ref(v_opts_569_);
v_r_572_ = lean_box(v_res_571_);
return v_r_572_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2(void){
_start:
{
lean_object* v___x_576_; lean_object* v___x_577_; 
v___x_576_ = ((lean_object*)(lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__1));
v___x_577_ = l_Lean_MessageData_ofFormat(v___x_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(lean_object* v_msgData_578_, lean_object* v_macroStack_579_, lean_object* v___y_580_){
_start:
{
lean_object* v_options_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v_options_582_ = lean_ctor_get(v___y_580_, 2);
v___x_583_ = l_Lean_Elab_pp_macroStack;
v___x_584_ = lp_plausible_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__8(v_options_582_, v___x_583_);
if (v___x_584_ == 0)
{
lean_object* v___x_585_; 
lean_dec(v_macroStack_579_);
v___x_585_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_585_, 0, v_msgData_578_);
return v___x_585_;
}
else
{
if (lean_obj_tag(v_macroStack_579_) == 0)
{
lean_object* v___x_586_; 
v___x_586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_586_, 0, v_msgData_578_);
return v___x_586_;
}
else
{
lean_object* v_head_587_; lean_object* v_after_588_; lean_object* v___x_590_; uint8_t v_isShared_591_; uint8_t v_isSharedCheck_603_; 
v_head_587_ = lean_ctor_get(v_macroStack_579_, 0);
lean_inc(v_head_587_);
v_after_588_ = lean_ctor_get(v_head_587_, 1);
v_isSharedCheck_603_ = !lean_is_exclusive(v_head_587_);
if (v_isSharedCheck_603_ == 0)
{
lean_object* v_unused_604_; 
v_unused_604_ = lean_ctor_get(v_head_587_, 0);
lean_dec(v_unused_604_);
v___x_590_ = v_head_587_;
v_isShared_591_ = v_isSharedCheck_603_;
goto v_resetjp_589_;
}
else
{
lean_inc(v_after_588_);
lean_dec(v_head_587_);
v___x_590_ = lean_box(0);
v_isShared_591_ = v_isSharedCheck_603_;
goto v_resetjp_589_;
}
v_resetjp_589_:
{
lean_object* v___x_592_; lean_object* v___x_594_; 
v___x_592_ = lean_obj_once(&lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0, &lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0_once, _init_lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9___closed__0);
if (v_isShared_591_ == 0)
{
lean_ctor_set_tag(v___x_590_, 7);
lean_ctor_set(v___x_590_, 1, v___x_592_);
lean_ctor_set(v___x_590_, 0, v_msgData_578_);
v___x_594_ = v___x_590_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_602_; 
v_reuseFailAlloc_602_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_602_, 0, v_msgData_578_);
lean_ctor_set(v_reuseFailAlloc_602_, 1, v___x_592_);
v___x_594_ = v_reuseFailAlloc_602_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
lean_object* v___x_595_; lean_object* v___x_596_; lean_object* v___x_597_; lean_object* v___x_598_; lean_object* v_msgData_599_; lean_object* v___x_600_; lean_object* v___x_601_; 
v___x_595_ = lean_obj_once(&lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2, &lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2_once, _init_lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___closed__2);
v___x_596_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_596_, 0, v___x_594_);
lean_ctor_set(v___x_596_, 1, v___x_595_);
v___x_597_ = l_Lean_MessageData_ofSyntax(v_after_588_);
v___x_598_ = l_Lean_indentD(v___x_597_);
v_msgData_599_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_599_, 0, v___x_596_);
lean_ctor_set(v_msgData_599_, 1, v___x_598_);
v___x_600_ = lp_plausible_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6_spec__9(v_msgData_599_, v_macroStack_579_);
v___x_601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_601_, 0, v___x_600_);
return v___x_601_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg___boxed(lean_object* v_msgData_605_, lean_object* v_macroStack_606_, lean_object* v___y_607_, lean_object* v___y_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_msgData_605_, v_macroStack_606_, v___y_607_);
lean_dec_ref(v___y_607_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(lean_object* v_msg_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_){
_start:
{
lean_object* v_ref_618_; lean_object* v___x_619_; lean_object* v_a_620_; lean_object* v_macroStack_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v_a_624_; lean_object* v___x_626_; uint8_t v_isShared_627_; uint8_t v_isSharedCheck_632_; 
v_ref_618_ = lean_ctor_get(v___y_615_, 5);
v___x_619_ = lp_plausible_Lean_addMessageContextFull___at___00Lean_throwError___at___00__private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr_spec__1_spec__1(v_msg_610_, v___y_613_, v___y_614_, v___y_615_, v___y_616_);
v_a_620_ = lean_ctor_get(v___x_619_, 0);
lean_inc(v_a_620_);
lean_dec_ref(v___x_619_);
v_macroStack_621_ = lean_ctor_get(v___y_611_, 1);
v___x_622_ = l_Lean_Elab_getBetterRef(v_ref_618_, v_macroStack_621_);
lean_inc(v_macroStack_621_);
v___x_623_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_a_620_, v_macroStack_621_, v___y_615_);
v_a_624_ = lean_ctor_get(v___x_623_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_623_);
if (v_isSharedCheck_632_ == 0)
{
v___x_626_ = v___x_623_;
v_isShared_627_ = v_isSharedCheck_632_;
goto v_resetjp_625_;
}
else
{
lean_inc(v_a_624_);
lean_dec(v___x_623_);
v___x_626_ = lean_box(0);
v_isShared_627_ = v_isSharedCheck_632_;
goto v_resetjp_625_;
}
v_resetjp_625_:
{
lean_object* v___x_628_; lean_object* v___x_630_; 
v___x_628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_628_, 0, v___x_622_);
lean_ctor_set(v___x_628_, 1, v_a_624_);
if (v_isShared_627_ == 0)
{
lean_ctor_set_tag(v___x_626_, 1);
lean_ctor_set(v___x_626_, 0, v___x_628_);
v___x_630_ = v___x_626_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_631_, 0, v___x_628_);
v___x_630_ = v_reuseFailAlloc_631_;
goto v_reusejp_629_;
}
v_reusejp_629_:
{
return v___x_630_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg___boxed(lean_object* v_msg_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v_msg_633_, v___y_634_, v___y_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
lean_dec(v___y_639_);
lean_dec_ref(v___y_638_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
return v_res_641_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0(void){
_start:
{
lean_object* v___x_642_; lean_object* v_ty_x3f_643_; 
v___x_642_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14);
v_ty_x3f_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_ty_x3f_643_, 0, v___x_642_);
return v_ty_x3f_643_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2(void){
_start:
{
lean_object* v___x_645_; lean_object* v___x_646_; 
v___x_645_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__1));
v___x_646_ = l_Lean_stringToMessageData(v___x_645_);
return v___x_646_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3(void){
_start:
{
lean_object* v___x_647_; lean_object* v___x_648_; 
v___x_647_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14);
v___x_648_ = l_Lean_MessageData_ofExpr(v___x_647_);
return v___x_648_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4(void){
_start:
{
lean_object* v___x_649_; lean_object* v___x_650_; lean_object* v___x_651_; 
v___x_649_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__3);
v___x_650_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2);
v___x_651_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_651_, 0, v___x_650_);
lean_ctor_set(v___x_651_, 1, v___x_649_);
return v___x_651_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6(void){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_653_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__5));
v___x_654_ = l_Lean_stringToMessageData(v___x_653_);
return v___x_654_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7(void){
_start:
{
lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_655_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6);
v___x_656_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__4);
v___x_657_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_657_, 0, v___x_656_);
lean_ctor_set(v___x_657_, 1, v___x_655_);
return v___x_657_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9(void){
_start:
{
lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_659_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__8));
v___x_660_ = l_Lean_stringToMessageData(v___x_659_);
return v___x_660_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11(void){
_start:
{
lean_object* v___x_662_; lean_object* v___x_663_; 
v___x_662_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__10));
v___x_663_ = l_Lean_stringToMessageData(v___x_662_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2(lean_object* v_stx_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_){
_start:
{
lean_object* v_ty_x3f_672_; uint8_t v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v_fileName_678_; lean_object* v_fileMap_679_; lean_object* v_options_680_; lean_object* v_currRecDepth_681_; lean_object* v_maxRecDepth_682_; lean_object* v_ref_683_; lean_object* v_currNamespace_684_; lean_object* v_openDecls_685_; lean_object* v_initHeartbeats_686_; lean_object* v_maxHeartbeats_687_; lean_object* v_quotContext_688_; lean_object* v_currMacroScope_689_; uint8_t v_diag_690_; lean_object* v_cancelTk_x3f_691_; uint8_t v_suppressElabErrors_692_; lean_object* v_inheritedTraceOptions_693_; uint8_t v___x_694_; lean_object* v_ref_695_; lean_object* v___x_696_; lean_object* v___x_697_; 
v_ty_x3f_672_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__0);
v___x_673_ = 1;
v___x_674_ = lean_box(0);
v___x_675_ = lean_box(v___x_673_);
v___x_676_ = lean_box(v___x_673_);
lean_inc(v_stx_664_);
v___x_677_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_677_, 0, v_stx_664_);
lean_closure_set(v___x_677_, 1, v_ty_x3f_672_);
lean_closure_set(v___x_677_, 2, v___x_675_);
lean_closure_set(v___x_677_, 3, v___x_676_);
lean_closure_set(v___x_677_, 4, v___x_674_);
v_fileName_678_ = lean_ctor_get(v_a_669_, 0);
v_fileMap_679_ = lean_ctor_get(v_a_669_, 1);
v_options_680_ = lean_ctor_get(v_a_669_, 2);
v_currRecDepth_681_ = lean_ctor_get(v_a_669_, 3);
v_maxRecDepth_682_ = lean_ctor_get(v_a_669_, 4);
v_ref_683_ = lean_ctor_get(v_a_669_, 5);
v_currNamespace_684_ = lean_ctor_get(v_a_669_, 6);
v_openDecls_685_ = lean_ctor_get(v_a_669_, 7);
v_initHeartbeats_686_ = lean_ctor_get(v_a_669_, 8);
v_maxHeartbeats_687_ = lean_ctor_get(v_a_669_, 9);
v_quotContext_688_ = lean_ctor_get(v_a_669_, 10);
v_currMacroScope_689_ = lean_ctor_get(v_a_669_, 11);
v_diag_690_ = lean_ctor_get_uint8(v_a_669_, sizeof(void*)*14);
v_cancelTk_x3f_691_ = lean_ctor_get(v_a_669_, 12);
v_suppressElabErrors_692_ = lean_ctor_get_uint8(v_a_669_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_693_ = lean_ctor_get(v_a_669_, 13);
v___x_694_ = 1;
v_ref_695_ = l_Lean_replaceRef(v_stx_664_, v_ref_683_);
lean_dec(v_stx_664_);
lean_inc_ref(v_inheritedTraceOptions_693_);
lean_inc(v_cancelTk_x3f_691_);
lean_inc(v_currMacroScope_689_);
lean_inc(v_quotContext_688_);
lean_inc(v_maxHeartbeats_687_);
lean_inc(v_initHeartbeats_686_);
lean_inc(v_openDecls_685_);
lean_inc(v_currNamespace_684_);
lean_inc(v_maxRecDepth_682_);
lean_inc(v_currRecDepth_681_);
lean_inc_ref(v_options_680_);
lean_inc_ref(v_fileMap_679_);
lean_inc_ref(v_fileName_678_);
v___x_696_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_696_, 0, v_fileName_678_);
lean_ctor_set(v___x_696_, 1, v_fileMap_679_);
lean_ctor_set(v___x_696_, 2, v_options_680_);
lean_ctor_set(v___x_696_, 3, v_currRecDepth_681_);
lean_ctor_set(v___x_696_, 4, v_maxRecDepth_682_);
lean_ctor_set(v___x_696_, 5, v_ref_695_);
lean_ctor_set(v___x_696_, 6, v_currNamespace_684_);
lean_ctor_set(v___x_696_, 7, v_openDecls_685_);
lean_ctor_set(v___x_696_, 8, v_initHeartbeats_686_);
lean_ctor_set(v___x_696_, 9, v_maxHeartbeats_687_);
lean_ctor_set(v___x_696_, 10, v_quotContext_688_);
lean_ctor_set(v___x_696_, 11, v_currMacroScope_689_);
lean_ctor_set(v___x_696_, 12, v_cancelTk_x3f_691_);
lean_ctor_set(v___x_696_, 13, v_inheritedTraceOptions_693_);
lean_ctor_set_uint8(v___x_696_, sizeof(void*)*14, v_diag_690_);
lean_ctor_set_uint8(v___x_696_, sizeof(void*)*14 + 1, v_suppressElabErrors_692_);
v___x_697_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_677_, v___x_694_, v_a_665_, v_a_666_, v_a_667_, v_a_668_, v___x_696_, v_a_670_);
if (lean_obj_tag(v___x_697_) == 0)
{
lean_object* v_a_698_; lean_object* v___x_699_; lean_object* v_a_700_; lean_object* v___y_702_; lean_object* v___y_703_; lean_object* v___y_704_; lean_object* v___y_705_; lean_object* v___y_706_; lean_object* v___y_707_; lean_object* v___y_708_; lean_object* v___y_709_; lean_object* v___y_710_; uint8_t v___y_711_; lean_object* v___y_728_; lean_object* v___y_729_; lean_object* v___y_730_; lean_object* v___y_731_; lean_object* v___y_732_; lean_object* v___y_733_; lean_object* v___y_740_; lean_object* v___y_741_; lean_object* v___y_742_; lean_object* v___y_743_; lean_object* v___y_744_; lean_object* v___y_745_; lean_object* v___y_777_; lean_object* v___y_778_; lean_object* v___y_779_; lean_object* v___y_780_; lean_object* v___y_781_; lean_object* v___y_782_; uint8_t v___x_795_; 
v_a_698_ = lean_ctor_get(v___x_697_, 0);
lean_inc(v_a_698_);
lean_dec_ref_known(v___x_697_, 1);
v___x_699_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_698_, v_a_668_);
v_a_700_ = lean_ctor_get(v___x_699_, 0);
lean_inc(v_a_700_);
lean_dec_ref(v___x_699_);
v___x_795_ = l_Lean_Expr_hasSorry(v_a_700_);
if (v___x_795_ == 0)
{
v___y_740_ = v_a_665_;
v___y_741_ = v_a_666_;
v___y_742_ = v_a_667_;
v___y_743_ = v_a_668_;
v___y_744_ = v___x_696_;
v___y_745_ = v_a_670_;
goto v___jp_739_;
}
else
{
uint8_t v___x_796_; 
v___x_796_ = l_Lean_Expr_hasSyntheticSorry(v_a_700_);
if (v___x_796_ == 0)
{
v___y_777_ = v_a_665_;
v___y_778_ = v_a_666_;
v___y_779_ = v_a_667_;
v___y_780_ = v_a_668_;
v___y_781_ = v___x_696_;
v___y_782_ = v_a_670_;
goto v___jp_776_;
}
else
{
lean_object* v___x_797_; lean_object* v_a_798_; lean_object* v___x_800_; uint8_t v_isShared_801_; uint8_t v_isSharedCheck_805_; 
lean_dec(v_a_700_);
lean_dec_ref_known(v___x_696_, 14);
v___x_797_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_798_ = lean_ctor_get(v___x_797_, 0);
v_isSharedCheck_805_ = !lean_is_exclusive(v___x_797_);
if (v_isSharedCheck_805_ == 0)
{
v___x_800_ = v___x_797_;
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
else
{
lean_inc(v_a_798_);
lean_dec(v___x_797_);
v___x_800_ = lean_box(0);
v_isShared_801_ = v_isSharedCheck_805_;
goto v_resetjp_799_;
}
v_resetjp_799_:
{
lean_object* v___x_803_; 
if (v_isShared_801_ == 0)
{
v___x_803_ = v___x_800_;
goto v_reusejp_802_;
}
else
{
lean_object* v_reuseFailAlloc_804_; 
v_reuseFailAlloc_804_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_804_, 0, v_a_798_);
v___x_803_ = v_reuseFailAlloc_804_;
goto v_reusejp_802_;
}
v_reusejp_802_:
{
return v___x_803_;
}
}
}
}
v___jp_701_:
{
if (v___y_711_ == 0)
{
if (lean_obj_tag(v___y_704_) == 0)
{
lean_dec_ref_known(v___y_704_, 2);
lean_dec_ref(v___y_707_);
lean_dec(v_a_700_);
return v___y_703_;
}
else
{
lean_object* v_id_712_; lean_object* v___x_714_; uint8_t v_isShared_715_; uint8_t v_isSharedCheck_725_; 
v_id_712_ = lean_ctor_get(v___y_704_, 0);
v_isSharedCheck_725_ = !lean_is_exclusive(v___y_704_);
if (v_isSharedCheck_725_ == 0)
{
lean_object* v_unused_726_; 
v_unused_726_ = lean_ctor_get(v___y_704_, 1);
lean_dec(v_unused_726_);
v___x_714_ = v___y_704_;
v_isShared_715_ = v_isSharedCheck_725_;
goto v_resetjp_713_;
}
else
{
lean_inc(v_id_712_);
lean_dec(v___y_704_);
v___x_714_ = lean_box(0);
v_isShared_715_ = v_isSharedCheck_725_;
goto v_resetjp_713_;
}
v_resetjp_713_:
{
uint8_t v___x_716_; 
v___x_716_ = l_Lean_instBEqInternalExceptionId_beq(v___y_705_, v_id_712_);
lean_dec(v_id_712_);
if (v___x_716_ == 0)
{
lean_del_object(v___x_714_);
lean_dec_ref(v___y_707_);
lean_dec(v_a_700_);
return v___y_703_;
}
else
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_721_; 
lean_dec_ref(v___y_703_);
v___x_717_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__7);
v___x_718_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_719_ = l_Lean_indentExpr(v_a_700_);
if (v_isShared_715_ == 0)
{
lean_ctor_set_tag(v___x_714_, 7);
lean_ctor_set(v___x_714_, 1, v___x_719_);
lean_ctor_set(v___x_714_, 0, v___x_718_);
v___x_721_ = v___x_714_;
goto v_reusejp_720_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v___x_718_);
lean_ctor_set(v_reuseFailAlloc_724_, 1, v___x_719_);
v___x_721_ = v_reuseFailAlloc_724_;
goto v_reusejp_720_;
}
v_reusejp_720_:
{
lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_722_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_722_, 0, v___x_721_);
lean_ctor_set(v___x_722_, 1, v___x_717_);
v___x_723_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_722_, v___y_702_, v___y_709_, v___y_710_, v___y_708_, v___y_707_, v___y_706_);
lean_dec_ref(v___y_707_);
return v___x_723_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_707_);
lean_dec_ref(v___y_704_);
lean_dec(v_a_700_);
return v___y_703_;
}
}
v___jp_727_:
{
lean_object* v___x_734_; 
lean_inc(v_a_700_);
v___x_734_ = l_Lean_Elab_ConfigEval_EvalExpr_evalNatExpr(v_a_700_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
if (lean_obj_tag(v___x_734_) == 0)
{
lean_dec_ref(v___y_732_);
lean_dec(v_a_700_);
return v___x_734_;
}
else
{
lean_object* v_a_735_; lean_object* v___x_736_; uint8_t v___x_737_; 
v_a_735_ = lean_ctor_get(v___x_734_, 0);
lean_inc(v_a_735_);
v___x_736_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_737_ = l_Lean_Exception_isInterrupt(v_a_735_);
if (v___x_737_ == 0)
{
uint8_t v___x_738_; 
lean_inc(v_a_735_);
v___x_738_ = l_Lean_Exception_isRuntime(v_a_735_);
v___y_702_ = v___y_728_;
v___y_703_ = v___x_734_;
v___y_704_ = v_a_735_;
v___y_705_ = v___x_736_;
v___y_706_ = v___y_733_;
v___y_707_ = v___y_732_;
v___y_708_ = v___y_731_;
v___y_709_ = v___y_729_;
v___y_710_ = v___y_730_;
v___y_711_ = v___x_738_;
goto v___jp_701_;
}
else
{
v___y_702_ = v___y_728_;
v___y_703_ = v___x_734_;
v___y_704_ = v_a_735_;
v___y_705_ = v___x_736_;
v___y_706_ = v___y_733_;
v___y_707_ = v___y_732_;
v___y_708_ = v___y_731_;
v___y_709_ = v___y_729_;
v___y_710_ = v___y_730_;
v___y_711_ = v___x_737_;
goto v___jp_701_;
}
}
}
v___jp_739_:
{
lean_object* v___x_746_; 
lean_inc(v_a_700_);
v___x_746_ = l_Lean_Meta_getMVars(v_a_700_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
if (lean_obj_tag(v___x_746_) == 0)
{
lean_object* v_a_747_; lean_object* v___x_748_; 
v_a_747_ = lean_ctor_get(v___x_746_, 0);
lean_inc(v_a_747_);
lean_dec_ref_known(v___x_746_, 1);
v___x_748_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_747_, v___x_674_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, v___y_744_, v___y_745_);
lean_dec(v_a_747_);
if (lean_obj_tag(v___x_748_) == 0)
{
lean_object* v_a_749_; uint8_t v___x_750_; 
v_a_749_ = lean_ctor_get(v___x_748_, 0);
lean_inc(v_a_749_);
lean_dec_ref_known(v___x_748_, 1);
v___x_750_ = lean_unbox(v_a_749_);
lean_dec(v_a_749_);
if (v___x_750_ == 0)
{
v___y_728_ = v___y_740_;
v___y_729_ = v___y_741_;
v___y_730_ = v___y_742_;
v___y_731_ = v___y_743_;
v___y_732_ = v___y_744_;
v___y_733_ = v___y_745_;
goto v___jp_727_;
}
else
{
lean_object* v___x_751_; lean_object* v_a_752_; lean_object* v___x_754_; uint8_t v_isShared_755_; uint8_t v_isSharedCheck_759_; 
lean_dec_ref(v___y_744_);
lean_dec(v_a_700_);
v___x_751_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_752_ = lean_ctor_get(v___x_751_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_751_);
if (v_isSharedCheck_759_ == 0)
{
v___x_754_ = v___x_751_;
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
else
{
lean_inc(v_a_752_);
lean_dec(v___x_751_);
v___x_754_ = lean_box(0);
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
v_resetjp_753_:
{
lean_object* v___x_757_; 
if (v_isShared_755_ == 0)
{
v___x_757_ = v___x_754_;
goto v_reusejp_756_;
}
else
{
lean_object* v_reuseFailAlloc_758_; 
v_reuseFailAlloc_758_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_758_, 0, v_a_752_);
v___x_757_ = v_reuseFailAlloc_758_;
goto v_reusejp_756_;
}
v_reusejp_756_:
{
return v___x_757_;
}
}
}
}
else
{
lean_object* v_a_760_; lean_object* v___x_762_; uint8_t v_isShared_763_; uint8_t v_isSharedCheck_767_; 
lean_dec_ref(v___y_744_);
lean_dec(v_a_700_);
v_a_760_ = lean_ctor_get(v___x_748_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_748_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_748_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_748_);
v___x_762_ = lean_box(0);
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
v_resetjp_761_:
{
lean_object* v___x_765_; 
if (v_isShared_763_ == 0)
{
v___x_765_ = v___x_762_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v_a_760_);
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
else
{
lean_object* v_a_768_; lean_object* v___x_770_; uint8_t v_isShared_771_; uint8_t v_isSharedCheck_775_; 
lean_dec_ref(v___y_744_);
lean_dec(v_a_700_);
v_a_768_ = lean_ctor_get(v___x_746_, 0);
v_isSharedCheck_775_ = !lean_is_exclusive(v___x_746_);
if (v_isSharedCheck_775_ == 0)
{
v___x_770_ = v___x_746_;
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
else
{
lean_inc(v_a_768_);
lean_dec(v___x_746_);
v___x_770_ = lean_box(0);
v_isShared_771_ = v_isSharedCheck_775_;
goto v_resetjp_769_;
}
v_resetjp_769_:
{
lean_object* v___x_773_; 
if (v_isShared_771_ == 0)
{
v___x_773_ = v___x_770_;
goto v_reusejp_772_;
}
else
{
lean_object* v_reuseFailAlloc_774_; 
v_reuseFailAlloc_774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_774_, 0, v_a_768_);
v___x_773_ = v_reuseFailAlloc_774_;
goto v_reusejp_772_;
}
v_reusejp_772_:
{
return v___x_773_;
}
}
}
}
v___jp_776_:
{
lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; lean_object* v___x_786_; lean_object* v_a_787_; lean_object* v___x_789_; uint8_t v_isShared_790_; uint8_t v_isSharedCheck_794_; 
v___x_783_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11);
v___x_784_ = l_Lean_indentExpr(v_a_700_);
v___x_785_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_785_, 0, v___x_783_);
lean_ctor_set(v___x_785_, 1, v___x_784_);
v___x_786_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_785_, v___y_777_, v___y_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
lean_dec_ref(v___y_781_);
v_a_787_ = lean_ctor_get(v___x_786_, 0);
v_isSharedCheck_794_ = !lean_is_exclusive(v___x_786_);
if (v_isSharedCheck_794_ == 0)
{
v___x_789_ = v___x_786_;
v_isShared_790_ = v_isSharedCheck_794_;
goto v_resetjp_788_;
}
else
{
lean_inc(v_a_787_);
lean_dec(v___x_786_);
v___x_789_ = lean_box(0);
v_isShared_790_ = v_isSharedCheck_794_;
goto v_resetjp_788_;
}
v_resetjp_788_:
{
lean_object* v___x_792_; 
if (v_isShared_790_ == 0)
{
v___x_792_ = v___x_789_;
goto v_reusejp_791_;
}
else
{
lean_object* v_reuseFailAlloc_793_; 
v_reuseFailAlloc_793_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_793_, 0, v_a_787_);
v___x_792_ = v_reuseFailAlloc_793_;
goto v_reusejp_791_;
}
v_reusejp_791_:
{
return v___x_792_;
}
}
}
}
else
{
lean_object* v_a_806_; lean_object* v___x_808_; uint8_t v_isShared_809_; uint8_t v_isSharedCheck_813_; 
lean_dec_ref_known(v___x_696_, 14);
v_a_806_ = lean_ctor_get(v___x_697_, 0);
v_isSharedCheck_813_ = !lean_is_exclusive(v___x_697_);
if (v_isSharedCheck_813_ == 0)
{
v___x_808_ = v___x_697_;
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
else
{
lean_inc(v_a_806_);
lean_dec(v___x_697_);
v___x_808_ = lean_box(0);
v_isShared_809_ = v_isSharedCheck_813_;
goto v_resetjp_807_;
}
v_resetjp_807_:
{
lean_object* v___x_811_; 
if (v_isShared_809_ == 0)
{
v___x_811_ = v___x_808_;
goto v_reusejp_810_;
}
else
{
lean_object* v_reuseFailAlloc_812_; 
v_reuseFailAlloc_812_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_812_, 0, v_a_806_);
v___x_811_ = v_reuseFailAlloc_812_;
goto v_reusejp_810_;
}
v_reusejp_810_:
{
return v___x_811_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___boxed(lean_object* v_stx_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_){
_start:
{
lean_object* v_res_822_; 
v_res_822_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2(v_stx_814_, v_a_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_);
lean_dec(v_a_820_);
lean_dec_ref(v_a_819_);
lean_dec(v_a_818_);
lean_dec_ref(v_a_817_);
lean_dec(v_a_816_);
lean_dec_ref(v_a_815_);
return v_res_822_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(lean_object* v_stx_823_, lean_object* v_a_824_, lean_object* v_a_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_){
_start:
{
lean_object* v_fileName_831_; lean_object* v_fileMap_832_; lean_object* v_options_833_; lean_object* v_currRecDepth_834_; lean_object* v_maxRecDepth_835_; lean_object* v_ref_836_; lean_object* v_currNamespace_837_; lean_object* v_openDecls_838_; lean_object* v_initHeartbeats_839_; lean_object* v_maxHeartbeats_840_; lean_object* v_quotContext_841_; lean_object* v_currMacroScope_842_; uint8_t v_diag_843_; lean_object* v_cancelTk_x3f_844_; uint8_t v_suppressElabErrors_845_; lean_object* v_inheritedTraceOptions_846_; lean_object* v_ref_847_; lean_object* v___x_848_; lean_object* v___x_849_; 
v_fileName_831_ = lean_ctor_get(v_a_828_, 0);
v_fileMap_832_ = lean_ctor_get(v_a_828_, 1);
v_options_833_ = lean_ctor_get(v_a_828_, 2);
v_currRecDepth_834_ = lean_ctor_get(v_a_828_, 3);
v_maxRecDepth_835_ = lean_ctor_get(v_a_828_, 4);
v_ref_836_ = lean_ctor_get(v_a_828_, 5);
v_currNamespace_837_ = lean_ctor_get(v_a_828_, 6);
v_openDecls_838_ = lean_ctor_get(v_a_828_, 7);
v_initHeartbeats_839_ = lean_ctor_get(v_a_828_, 8);
v_maxHeartbeats_840_ = lean_ctor_get(v_a_828_, 9);
v_quotContext_841_ = lean_ctor_get(v_a_828_, 10);
v_currMacroScope_842_ = lean_ctor_get(v_a_828_, 11);
v_diag_843_ = lean_ctor_get_uint8(v_a_828_, sizeof(void*)*14);
v_cancelTk_x3f_844_ = lean_ctor_get(v_a_828_, 12);
v_suppressElabErrors_845_ = lean_ctor_get_uint8(v_a_828_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_846_ = lean_ctor_get(v_a_828_, 13);
v_ref_847_ = l_Lean_replaceRef(v_stx_823_, v_ref_836_);
lean_inc_ref(v_inheritedTraceOptions_846_);
lean_inc(v_cancelTk_x3f_844_);
lean_inc(v_currMacroScope_842_);
lean_inc(v_quotContext_841_);
lean_inc(v_maxHeartbeats_840_);
lean_inc(v_initHeartbeats_839_);
lean_inc(v_openDecls_838_);
lean_inc(v_currNamespace_837_);
lean_inc(v_maxRecDepth_835_);
lean_inc(v_currRecDepth_834_);
lean_inc_ref(v_options_833_);
lean_inc_ref(v_fileMap_832_);
lean_inc_ref(v_fileName_831_);
v___x_848_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_848_, 0, v_fileName_831_);
lean_ctor_set(v___x_848_, 1, v_fileMap_832_);
lean_ctor_set(v___x_848_, 2, v_options_833_);
lean_ctor_set(v___x_848_, 3, v_currRecDepth_834_);
lean_ctor_set(v___x_848_, 4, v_maxRecDepth_835_);
lean_ctor_set(v___x_848_, 5, v_ref_847_);
lean_ctor_set(v___x_848_, 6, v_currNamespace_837_);
lean_ctor_set(v___x_848_, 7, v_openDecls_838_);
lean_ctor_set(v___x_848_, 8, v_initHeartbeats_839_);
lean_ctor_set(v___x_848_, 9, v_maxHeartbeats_840_);
lean_ctor_set(v___x_848_, 10, v_quotContext_841_);
lean_ctor_set(v___x_848_, 11, v_currMacroScope_842_);
lean_ctor_set(v___x_848_, 12, v_cancelTk_x3f_844_);
lean_ctor_set(v___x_848_, 13, v_inheritedTraceOptions_846_);
lean_ctor_set_uint8(v___x_848_, sizeof(void*)*14, v_diag_843_);
lean_ctor_set_uint8(v___x_848_, sizeof(void*)*14 + 1, v_suppressElabErrors_845_);
lean_inc(v_stx_823_);
v___x_849_ = l_Lean_Elab_ConfigEval_EvalTerm_evalNatStx(v_stx_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_, v___x_848_, v_a_829_);
if (lean_obj_tag(v___x_849_) == 0)
{
lean_object* v_a_850_; lean_object* v___x_852_; uint8_t v_isShared_853_; uint8_t v_isSharedCheck_858_; 
lean_dec_ref_known(v___x_848_, 14);
lean_dec(v_stx_823_);
v_a_850_ = lean_ctor_get(v___x_849_, 0);
v_isSharedCheck_858_ = !lean_is_exclusive(v___x_849_);
if (v_isSharedCheck_858_ == 0)
{
v___x_852_ = v___x_849_;
v_isShared_853_ = v_isSharedCheck_858_;
goto v_resetjp_851_;
}
else
{
lean_inc(v_a_850_);
lean_dec(v___x_849_);
v___x_852_ = lean_box(0);
v_isShared_853_ = v_isSharedCheck_858_;
goto v_resetjp_851_;
}
v_resetjp_851_:
{
lean_object* v_fst_854_; lean_object* v___x_856_; 
v_fst_854_ = lean_ctor_get(v_a_850_, 0);
lean_inc(v_fst_854_);
lean_dec(v_a_850_);
if (v_isShared_853_ == 0)
{
lean_ctor_set(v___x_852_, 0, v_fst_854_);
v___x_856_ = v___x_852_;
goto v_reusejp_855_;
}
else
{
lean_object* v_reuseFailAlloc_857_; 
v_reuseFailAlloc_857_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_857_, 0, v_fst_854_);
v___x_856_ = v_reuseFailAlloc_857_;
goto v_reusejp_855_;
}
v_reusejp_855_:
{
return v___x_856_;
}
}
}
else
{
lean_object* v_a_859_; lean_object* v___x_861_; uint8_t v_isShared_862_; uint8_t v_isSharedCheck_874_; 
v_a_859_ = lean_ctor_get(v___x_849_, 0);
v_isSharedCheck_874_ = !lean_is_exclusive(v___x_849_);
if (v_isSharedCheck_874_ == 0)
{
v___x_861_ = v___x_849_;
v_isShared_862_ = v_isSharedCheck_874_;
goto v_resetjp_860_;
}
else
{
lean_inc(v_a_859_);
lean_dec(v___x_849_);
v___x_861_ = lean_box(0);
v_isShared_862_ = v_isSharedCheck_874_;
goto v_resetjp_860_;
}
v_resetjp_860_:
{
lean_object* v___x_863_; lean_object* v___x_865_; 
v___x_863_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_859_);
if (v_isShared_862_ == 0)
{
v___x_865_ = v___x_861_;
goto v_reusejp_864_;
}
else
{
lean_object* v_reuseFailAlloc_873_; 
v_reuseFailAlloc_873_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_873_, 0, v_a_859_);
v___x_865_ = v_reuseFailAlloc_873_;
goto v_reusejp_864_;
}
v_reusejp_864_:
{
uint8_t v___y_867_; uint8_t v___x_871_; 
v___x_871_ = l_Lean_Exception_isInterrupt(v_a_859_);
if (v___x_871_ == 0)
{
uint8_t v___x_872_; 
lean_inc(v_a_859_);
v___x_872_ = l_Lean_Exception_isRuntime(v_a_859_);
v___y_867_ = v___x_872_;
goto v___jp_866_;
}
else
{
v___y_867_ = v___x_871_;
goto v___jp_866_;
}
v___jp_866_:
{
if (v___y_867_ == 0)
{
if (lean_obj_tag(v_a_859_) == 0)
{
lean_dec_ref_known(v_a_859_, 2);
lean_dec_ref_known(v___x_848_, 14);
lean_dec(v_stx_823_);
return v___x_865_;
}
else
{
lean_object* v_id_868_; uint8_t v___x_869_; 
v_id_868_ = lean_ctor_get(v_a_859_, 0);
lean_inc(v_id_868_);
lean_dec_ref_known(v_a_859_, 2);
v___x_869_ = l_Lean_instBEqInternalExceptionId_beq(v___x_863_, v_id_868_);
lean_dec(v_id_868_);
if (v___x_869_ == 0)
{
lean_dec_ref_known(v___x_848_, 14);
lean_dec(v_stx_823_);
return v___x_865_;
}
else
{
lean_object* v___x_870_; 
lean_dec_ref(v___x_865_);
v___x_870_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2(v_stx_823_, v_a_824_, v_a_825_, v_a_826_, v_a_827_, v___x_848_, v_a_829_);
lean_dec_ref_known(v___x_848_, 14);
return v___x_870_;
}
}
}
else
{
lean_dec(v_a_859_);
lean_dec_ref_known(v___x_848_, 14);
lean_dec(v_stx_823_);
return v___x_865_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1___boxed(lean_object* v_stx_875_, lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_, lean_object* v_a_879_, lean_object* v_a_880_, lean_object* v_a_881_, lean_object* v_a_882_){
_start:
{
lean_object* v_res_883_; 
v_res_883_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(v_stx_875_, v_a_876_, v_a_877_, v_a_878_, v_a_879_, v_a_880_, v_a_881_);
lean_dec(v_a_881_);
lean_dec_ref(v_a_880_);
lean_dec(v_a_879_);
lean_dec_ref(v_a_878_);
lean_dec(v_a_877_);
lean_dec_ref(v_a_876_);
return v_res_883_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1(void){
_start:
{
lean_object* v___x_885_; lean_object* v___x_886_; 
v___x_885_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__0));
v___x_886_ = l_Lean_stringToMessageData(v___x_885_);
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0(lean_object* v_stx_887_, lean_object* v_a_888_, lean_object* v_a_889_, lean_object* v_a_890_, lean_object* v_a_891_, lean_object* v_a_892_, lean_object* v_a_893_){
_start:
{
lean_object* v___x_895_; lean_object* v_evalExpr_896_; lean_object* v_expectedType_x3f_897_; uint8_t v___x_898_; lean_object* v___x_899_; lean_object* v___x_900_; lean_object* v___x_901_; lean_object* v___x_902_; lean_object* v_fileName_903_; lean_object* v_fileMap_904_; lean_object* v_options_905_; lean_object* v_currRecDepth_906_; lean_object* v_maxRecDepth_907_; lean_object* v_ref_908_; lean_object* v_currNamespace_909_; lean_object* v_openDecls_910_; lean_object* v_initHeartbeats_911_; lean_object* v_maxHeartbeats_912_; lean_object* v_quotContext_913_; lean_object* v_currMacroScope_914_; uint8_t v_diag_915_; lean_object* v_cancelTk_x3f_916_; uint8_t v_suppressElabErrors_917_; lean_object* v_inheritedTraceOptions_918_; uint8_t v___x_919_; lean_object* v_ref_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v___x_895_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___lam__0___closed__0);
v_evalExpr_896_ = lean_ctor_get(v___x_895_, 0);
v_expectedType_x3f_897_ = lean_ctor_get(v___x_895_, 1);
v___x_898_ = 1;
v___x_899_ = lean_box(0);
v___x_900_ = lean_box(v___x_898_);
v___x_901_ = lean_box(v___x_898_);
lean_inc(v_expectedType_x3f_897_);
lean_inc(v_stx_887_);
v___x_902_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_902_, 0, v_stx_887_);
lean_closure_set(v___x_902_, 1, v_expectedType_x3f_897_);
lean_closure_set(v___x_902_, 2, v___x_900_);
lean_closure_set(v___x_902_, 3, v___x_901_);
lean_closure_set(v___x_902_, 4, v___x_899_);
v_fileName_903_ = lean_ctor_get(v_a_892_, 0);
v_fileMap_904_ = lean_ctor_get(v_a_892_, 1);
v_options_905_ = lean_ctor_get(v_a_892_, 2);
v_currRecDepth_906_ = lean_ctor_get(v_a_892_, 3);
v_maxRecDepth_907_ = lean_ctor_get(v_a_892_, 4);
v_ref_908_ = lean_ctor_get(v_a_892_, 5);
v_currNamespace_909_ = lean_ctor_get(v_a_892_, 6);
v_openDecls_910_ = lean_ctor_get(v_a_892_, 7);
v_initHeartbeats_911_ = lean_ctor_get(v_a_892_, 8);
v_maxHeartbeats_912_ = lean_ctor_get(v_a_892_, 9);
v_quotContext_913_ = lean_ctor_get(v_a_892_, 10);
v_currMacroScope_914_ = lean_ctor_get(v_a_892_, 11);
v_diag_915_ = lean_ctor_get_uint8(v_a_892_, sizeof(void*)*14);
v_cancelTk_x3f_916_ = lean_ctor_get(v_a_892_, 12);
v_suppressElabErrors_917_ = lean_ctor_get_uint8(v_a_892_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_918_ = lean_ctor_get(v_a_892_, 13);
v___x_919_ = 1;
v_ref_920_ = l_Lean_replaceRef(v_stx_887_, v_ref_908_);
lean_dec(v_stx_887_);
lean_inc_ref(v_inheritedTraceOptions_918_);
lean_inc(v_cancelTk_x3f_916_);
lean_inc(v_currMacroScope_914_);
lean_inc(v_quotContext_913_);
lean_inc(v_maxHeartbeats_912_);
lean_inc(v_initHeartbeats_911_);
lean_inc(v_openDecls_910_);
lean_inc(v_currNamespace_909_);
lean_inc(v_maxRecDepth_907_);
lean_inc(v_currRecDepth_906_);
lean_inc_ref(v_options_905_);
lean_inc_ref(v_fileMap_904_);
lean_inc_ref(v_fileName_903_);
v___x_921_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_921_, 0, v_fileName_903_);
lean_ctor_set(v___x_921_, 1, v_fileMap_904_);
lean_ctor_set(v___x_921_, 2, v_options_905_);
lean_ctor_set(v___x_921_, 3, v_currRecDepth_906_);
lean_ctor_set(v___x_921_, 4, v_maxRecDepth_907_);
lean_ctor_set(v___x_921_, 5, v_ref_920_);
lean_ctor_set(v___x_921_, 6, v_currNamespace_909_);
lean_ctor_set(v___x_921_, 7, v_openDecls_910_);
lean_ctor_set(v___x_921_, 8, v_initHeartbeats_911_);
lean_ctor_set(v___x_921_, 9, v_maxHeartbeats_912_);
lean_ctor_set(v___x_921_, 10, v_quotContext_913_);
lean_ctor_set(v___x_921_, 11, v_currMacroScope_914_);
lean_ctor_set(v___x_921_, 12, v_cancelTk_x3f_916_);
lean_ctor_set(v___x_921_, 13, v_inheritedTraceOptions_918_);
lean_ctor_set_uint8(v___x_921_, sizeof(void*)*14, v_diag_915_);
lean_ctor_set_uint8(v___x_921_, sizeof(void*)*14 + 1, v_suppressElabErrors_917_);
v___x_922_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_902_, v___x_919_, v_a_888_, v_a_889_, v_a_890_, v_a_891_, v___x_921_, v_a_893_);
if (lean_obj_tag(v___x_922_) == 0)
{
lean_object* v_a_923_; lean_object* v___x_924_; lean_object* v_a_925_; lean_object* v___y_927_; lean_object* v___y_928_; lean_object* v___y_929_; lean_object* v___y_930_; lean_object* v___y_931_; lean_object* v___y_932_; lean_object* v___y_933_; lean_object* v___y_940_; lean_object* v___y_941_; lean_object* v___y_942_; lean_object* v___y_943_; lean_object* v___y_944_; lean_object* v___y_945_; lean_object* v___y_946_; lean_object* v___y_947_; lean_object* v___y_948_; uint8_t v___y_949_; lean_object* v___y_967_; lean_object* v___y_968_; lean_object* v___y_969_; lean_object* v___y_970_; lean_object* v___y_971_; lean_object* v___y_972_; lean_object* v___y_979_; lean_object* v___y_980_; lean_object* v___y_981_; lean_object* v___y_982_; lean_object* v___y_983_; lean_object* v___y_984_; lean_object* v___y_1016_; lean_object* v___y_1017_; lean_object* v___y_1018_; lean_object* v___y_1019_; lean_object* v___y_1020_; lean_object* v___y_1021_; uint8_t v___x_1034_; 
v_a_923_ = lean_ctor_get(v___x_922_, 0);
lean_inc(v_a_923_);
lean_dec_ref_known(v___x_922_, 1);
v___x_924_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_923_, v_a_891_);
v_a_925_ = lean_ctor_get(v___x_924_, 0);
lean_inc(v_a_925_);
lean_dec_ref(v___x_924_);
v___x_1034_ = l_Lean_Expr_hasSorry(v_a_925_);
if (v___x_1034_ == 0)
{
v___y_979_ = v_a_888_;
v___y_980_ = v_a_889_;
v___y_981_ = v_a_890_;
v___y_982_ = v_a_891_;
v___y_983_ = v___x_921_;
v___y_984_ = v_a_893_;
goto v___jp_978_;
}
else
{
uint8_t v___x_1035_; 
v___x_1035_ = l_Lean_Expr_hasSyntheticSorry(v_a_925_);
if (v___x_1035_ == 0)
{
v___y_1016_ = v_a_888_;
v___y_1017_ = v_a_889_;
v___y_1018_ = v_a_890_;
v___y_1019_ = v_a_891_;
v___y_1020_ = v___x_921_;
v___y_1021_ = v_a_893_;
goto v___jp_1015_;
}
else
{
lean_object* v___x_1036_; lean_object* v_a_1037_; lean_object* v___x_1039_; uint8_t v_isShared_1040_; uint8_t v_isSharedCheck_1044_; 
lean_dec(v_a_925_);
lean_dec_ref_known(v___x_921_, 14);
v___x_1036_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_1037_ = lean_ctor_get(v___x_1036_, 0);
v_isSharedCheck_1044_ = !lean_is_exclusive(v___x_1036_);
if (v_isSharedCheck_1044_ == 0)
{
v___x_1039_ = v___x_1036_;
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
else
{
lean_inc(v_a_1037_);
lean_dec(v___x_1036_);
v___x_1039_ = lean_box(0);
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
v_resetjp_1038_:
{
lean_object* v___x_1042_; 
if (v_isShared_1040_ == 0)
{
v___x_1042_ = v___x_1039_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v_a_1037_);
v___x_1042_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
return v___x_1042_;
}
}
}
}
v___jp_926_:
{
lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; 
v___x_934_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_935_ = l_Lean_indentExpr(v_a_925_);
v___x_936_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_936_, 0, v___x_934_);
lean_ctor_set(v___x_936_, 1, v___x_935_);
v___x_937_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_937_, 0, v___x_936_);
lean_ctor_set(v___x_937_, 1, v___y_933_);
v___x_938_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_937_, v___y_928_, v___y_929_, v___y_927_, v___y_932_, v___y_931_, v___y_930_);
lean_dec_ref(v___y_931_);
return v___x_938_;
}
v___jp_939_:
{
if (v___y_949_ == 0)
{
if (lean_obj_tag(v___y_945_) == 0)
{
lean_dec_ref_known(v___y_945_, 2);
lean_dec_ref(v___y_948_);
lean_dec(v_a_925_);
return v___y_944_;
}
else
{
lean_object* v_id_950_; lean_object* v___x_952_; uint8_t v_isShared_953_; uint8_t v_isSharedCheck_964_; 
v_id_950_ = lean_ctor_get(v___y_945_, 0);
v_isSharedCheck_964_ = !lean_is_exclusive(v___y_945_);
if (v_isSharedCheck_964_ == 0)
{
lean_object* v_unused_965_; 
v_unused_965_ = lean_ctor_get(v___y_945_, 1);
lean_dec(v_unused_965_);
v___x_952_ = v___y_945_;
v_isShared_953_ = v_isSharedCheck_964_;
goto v_resetjp_951_;
}
else
{
lean_inc(v_id_950_);
lean_dec(v___y_945_);
v___x_952_ = lean_box(0);
v_isShared_953_ = v_isSharedCheck_964_;
goto v_resetjp_951_;
}
v_resetjp_951_:
{
uint8_t v___x_954_; 
v___x_954_ = l_Lean_instBEqInternalExceptionId_beq(v___y_940_, v_id_950_);
lean_dec(v_id_950_);
if (v___x_954_ == 0)
{
lean_del_object(v___x_952_);
lean_dec_ref(v___y_948_);
lean_dec(v_a_925_);
return v___y_944_;
}
else
{
lean_dec_ref(v___y_944_);
if (lean_obj_tag(v_expectedType_x3f_897_) == 1)
{
lean_object* v_val_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_959_; 
v_val_955_ = lean_ctor_get(v_expectedType_x3f_897_, 0);
v___x_956_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2);
lean_inc(v_val_955_);
v___x_957_ = l_Lean_MessageData_ofExpr(v_val_955_);
if (v_isShared_953_ == 0)
{
lean_ctor_set_tag(v___x_952_, 7);
lean_ctor_set(v___x_952_, 1, v___x_957_);
lean_ctor_set(v___x_952_, 0, v___x_956_);
v___x_959_ = v___x_952_;
goto v_reusejp_958_;
}
else
{
lean_object* v_reuseFailAlloc_962_; 
v_reuseFailAlloc_962_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_962_, 0, v___x_956_);
lean_ctor_set(v_reuseFailAlloc_962_, 1, v___x_957_);
v___x_959_ = v_reuseFailAlloc_962_;
goto v_reusejp_958_;
}
v_reusejp_958_:
{
lean_object* v___x_960_; lean_object* v___x_961_; 
v___x_960_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6);
v___x_961_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_961_, 0, v___x_959_);
lean_ctor_set(v___x_961_, 1, v___x_960_);
v___y_927_ = v___y_942_;
v___y_928_ = v___y_941_;
v___y_929_ = v___y_943_;
v___y_930_ = v___y_946_;
v___y_931_ = v___y_948_;
v___y_932_ = v___y_947_;
v___y_933_ = v___x_961_;
goto v___jp_926_;
}
}
else
{
lean_object* v___x_963_; 
lean_del_object(v___x_952_);
v___x_963_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___closed__1);
v___y_927_ = v___y_942_;
v___y_928_ = v___y_941_;
v___y_929_ = v___y_943_;
v___y_930_ = v___y_946_;
v___y_931_ = v___y_948_;
v___y_932_ = v___y_947_;
v___y_933_ = v___x_963_;
goto v___jp_926_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_948_);
lean_dec_ref(v___y_945_);
lean_dec(v_a_925_);
return v___y_944_;
}
}
v___jp_966_:
{
lean_object* v___x_973_; 
lean_inc_ref(v_evalExpr_896_);
lean_inc(v___y_972_);
lean_inc_ref(v___y_971_);
lean_inc(v___y_970_);
lean_inc_ref(v___y_969_);
lean_inc(v_a_925_);
v___x_973_ = lean_apply_6(v_evalExpr_896_, v_a_925_, v___y_969_, v___y_970_, v___y_971_, v___y_972_, lean_box(0));
if (lean_obj_tag(v___x_973_) == 0)
{
lean_dec_ref(v___y_971_);
lean_dec(v_a_925_);
return v___x_973_;
}
else
{
lean_object* v_a_974_; lean_object* v___x_975_; uint8_t v___x_976_; 
v_a_974_ = lean_ctor_get(v___x_973_, 0);
lean_inc(v_a_974_);
v___x_975_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_976_ = l_Lean_Exception_isInterrupt(v_a_974_);
if (v___x_976_ == 0)
{
uint8_t v___x_977_; 
lean_inc(v_a_974_);
v___x_977_ = l_Lean_Exception_isRuntime(v_a_974_);
v___y_940_ = v___x_975_;
v___y_941_ = v___y_967_;
v___y_942_ = v___y_969_;
v___y_943_ = v___y_968_;
v___y_944_ = v___x_973_;
v___y_945_ = v_a_974_;
v___y_946_ = v___y_972_;
v___y_947_ = v___y_970_;
v___y_948_ = v___y_971_;
v___y_949_ = v___x_977_;
goto v___jp_939_;
}
else
{
v___y_940_ = v___x_975_;
v___y_941_ = v___y_967_;
v___y_942_ = v___y_969_;
v___y_943_ = v___y_968_;
v___y_944_ = v___x_973_;
v___y_945_ = v_a_974_;
v___y_946_ = v___y_972_;
v___y_947_ = v___y_970_;
v___y_948_ = v___y_971_;
v___y_949_ = v___x_976_;
goto v___jp_939_;
}
}
}
v___jp_978_:
{
lean_object* v___x_985_; 
lean_inc(v_a_925_);
v___x_985_ = l_Lean_Meta_getMVars(v_a_925_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
if (lean_obj_tag(v___x_985_) == 0)
{
lean_object* v_a_986_; lean_object* v___x_987_; 
v_a_986_ = lean_ctor_get(v___x_985_, 0);
lean_inc(v_a_986_);
lean_dec_ref_known(v___x_985_, 1);
v___x_987_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_986_, v___x_899_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
lean_dec(v_a_986_);
if (lean_obj_tag(v___x_987_) == 0)
{
lean_object* v_a_988_; uint8_t v___x_989_; 
v_a_988_ = lean_ctor_get(v___x_987_, 0);
lean_inc(v_a_988_);
lean_dec_ref_known(v___x_987_, 1);
v___x_989_ = lean_unbox(v_a_988_);
lean_dec(v_a_988_);
if (v___x_989_ == 0)
{
v___y_967_ = v___y_979_;
v___y_968_ = v___y_980_;
v___y_969_ = v___y_981_;
v___y_970_ = v___y_982_;
v___y_971_ = v___y_983_;
v___y_972_ = v___y_984_;
goto v___jp_966_;
}
else
{
lean_object* v___x_990_; lean_object* v_a_991_; lean_object* v___x_993_; uint8_t v_isShared_994_; uint8_t v_isSharedCheck_998_; 
lean_dec_ref(v___y_983_);
lean_dec(v_a_925_);
v___x_990_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_991_ = lean_ctor_get(v___x_990_, 0);
v_isSharedCheck_998_ = !lean_is_exclusive(v___x_990_);
if (v_isSharedCheck_998_ == 0)
{
v___x_993_ = v___x_990_;
v_isShared_994_ = v_isSharedCheck_998_;
goto v_resetjp_992_;
}
else
{
lean_inc(v_a_991_);
lean_dec(v___x_990_);
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
else
{
lean_object* v_a_999_; lean_object* v___x_1001_; uint8_t v_isShared_1002_; uint8_t v_isSharedCheck_1006_; 
lean_dec_ref(v___y_983_);
lean_dec(v_a_925_);
v_a_999_ = lean_ctor_get(v___x_987_, 0);
v_isSharedCheck_1006_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1006_ == 0)
{
v___x_1001_ = v___x_987_;
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
else
{
lean_inc(v_a_999_);
lean_dec(v___x_987_);
v___x_1001_ = lean_box(0);
v_isShared_1002_ = v_isSharedCheck_1006_;
goto v_resetjp_1000_;
}
v_resetjp_1000_:
{
lean_object* v___x_1004_; 
if (v_isShared_1002_ == 0)
{
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
}
else
{
lean_object* v_a_1007_; lean_object* v___x_1009_; uint8_t v_isShared_1010_; uint8_t v_isSharedCheck_1014_; 
lean_dec_ref(v___y_983_);
lean_dec(v_a_925_);
v_a_1007_ = lean_ctor_get(v___x_985_, 0);
v_isSharedCheck_1014_ = !lean_is_exclusive(v___x_985_);
if (v_isSharedCheck_1014_ == 0)
{
v___x_1009_ = v___x_985_;
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
else
{
lean_inc(v_a_1007_);
lean_dec(v___x_985_);
v___x_1009_ = lean_box(0);
v_isShared_1010_ = v_isSharedCheck_1014_;
goto v_resetjp_1008_;
}
v_resetjp_1008_:
{
lean_object* v___x_1012_; 
if (v_isShared_1010_ == 0)
{
v___x_1012_ = v___x_1009_;
goto v_reusejp_1011_;
}
else
{
lean_object* v_reuseFailAlloc_1013_; 
v_reuseFailAlloc_1013_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1013_, 0, v_a_1007_);
v___x_1012_ = v_reuseFailAlloc_1013_;
goto v_reusejp_1011_;
}
v_reusejp_1011_:
{
return v___x_1012_;
}
}
}
}
v___jp_1015_:
{
lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v_a_1026_; lean_object* v___x_1028_; uint8_t v_isShared_1029_; uint8_t v_isSharedCheck_1033_; 
v___x_1022_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11);
v___x_1023_ = l_Lean_indentExpr(v_a_925_);
v___x_1024_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1024_, 0, v___x_1022_);
lean_ctor_set(v___x_1024_, 1, v___x_1023_);
v___x_1025_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_1024_, v___y_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_, v___y_1021_);
lean_dec_ref(v___y_1020_);
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
return v___x_1031_;
}
}
}
}
else
{
lean_object* v_a_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1052_; 
lean_dec_ref_known(v___x_921_, 14);
v_a_1045_ = lean_ctor_get(v___x_922_, 0);
v_isSharedCheck_1052_ = !lean_is_exclusive(v___x_922_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1047_ = v___x_922_;
v_isShared_1048_ = v_isSharedCheck_1052_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_a_1045_);
lean_dec(v___x_922_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1052_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v___x_1050_; 
if (v_isShared_1048_ == 0)
{
v___x_1050_ = v___x_1047_;
goto v_reusejp_1049_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v_a_1045_);
v___x_1050_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1049_;
}
v_reusejp_1049_:
{
return v___x_1050_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_stx_1053_, lean_object* v_a_1054_, lean_object* v_a_1055_, lean_object* v_a_1056_, lean_object* v_a_1057_, lean_object* v_a_1058_, lean_object* v_a_1059_, lean_object* v_a_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0(v_stx_1053_, v_a_1054_, v_a_1055_, v_a_1056_, v_a_1057_, v_a_1058_, v_a_1059_);
lean_dec(v_a_1059_);
lean_dec_ref(v_a_1058_);
lean_dec(v_a_1057_);
lean_dec_ref(v_a_1056_);
lean_dec(v_a_1055_);
lean_dec_ref(v_a_1054_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0(lean_object* v_stx_1063_, lean_object* v_a_1064_, lean_object* v_a_1065_, lean_object* v_a_1066_, lean_object* v_a_1067_, lean_object* v_a_1068_, lean_object* v_a_1069_){
_start:
{
lean_object* v_fileName_1071_; lean_object* v_fileMap_1072_; lean_object* v_options_1073_; lean_object* v_currRecDepth_1074_; lean_object* v_maxRecDepth_1075_; lean_object* v_ref_1076_; lean_object* v_currNamespace_1077_; lean_object* v_openDecls_1078_; lean_object* v_initHeartbeats_1079_; lean_object* v_maxHeartbeats_1080_; lean_object* v_quotContext_1081_; lean_object* v_currMacroScope_1082_; uint8_t v_diag_1083_; lean_object* v_cancelTk_x3f_1084_; uint8_t v_suppressElabErrors_1085_; lean_object* v_inheritedTraceOptions_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v_ref_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; 
v_fileName_1071_ = lean_ctor_get(v_a_1068_, 0);
v_fileMap_1072_ = lean_ctor_get(v_a_1068_, 1);
v_options_1073_ = lean_ctor_get(v_a_1068_, 2);
v_currRecDepth_1074_ = lean_ctor_get(v_a_1068_, 3);
v_maxRecDepth_1075_ = lean_ctor_get(v_a_1068_, 4);
v_ref_1076_ = lean_ctor_get(v_a_1068_, 5);
v_currNamespace_1077_ = lean_ctor_get(v_a_1068_, 6);
v_openDecls_1078_ = lean_ctor_get(v_a_1068_, 7);
v_initHeartbeats_1079_ = lean_ctor_get(v_a_1068_, 8);
v_maxHeartbeats_1080_ = lean_ctor_get(v_a_1068_, 9);
v_quotContext_1081_ = lean_ctor_get(v_a_1068_, 10);
v_currMacroScope_1082_ = lean_ctor_get(v_a_1068_, 11);
v_diag_1083_ = lean_ctor_get_uint8(v_a_1068_, sizeof(void*)*14);
v_cancelTk_x3f_1084_ = lean_ctor_get(v_a_1068_, 12);
v_suppressElabErrors_1085_ = lean_ctor_get_uint8(v_a_1068_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1086_ = lean_ctor_get(v_a_1068_, 13);
v___x_1087_ = lean_obj_once(&lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14, &lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14_once, _init_lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__14);
v___x_1088_ = ((lean_object*)(lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___closed__0));
v_ref_1089_ = l_Lean_replaceRef(v_stx_1063_, v_ref_1076_);
lean_inc_ref(v_inheritedTraceOptions_1086_);
lean_inc(v_cancelTk_x3f_1084_);
lean_inc(v_currMacroScope_1082_);
lean_inc(v_quotContext_1081_);
lean_inc(v_maxHeartbeats_1080_);
lean_inc(v_initHeartbeats_1079_);
lean_inc(v_openDecls_1078_);
lean_inc(v_currNamespace_1077_);
lean_inc(v_maxRecDepth_1075_);
lean_inc(v_currRecDepth_1074_);
lean_inc_ref(v_options_1073_);
lean_inc_ref(v_fileMap_1072_);
lean_inc_ref(v_fileName_1071_);
v___x_1090_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1090_, 0, v_fileName_1071_);
lean_ctor_set(v___x_1090_, 1, v_fileMap_1072_);
lean_ctor_set(v___x_1090_, 2, v_options_1073_);
lean_ctor_set(v___x_1090_, 3, v_currRecDepth_1074_);
lean_ctor_set(v___x_1090_, 4, v_maxRecDepth_1075_);
lean_ctor_set(v___x_1090_, 5, v_ref_1089_);
lean_ctor_set(v___x_1090_, 6, v_currNamespace_1077_);
lean_ctor_set(v___x_1090_, 7, v_openDecls_1078_);
lean_ctor_set(v___x_1090_, 8, v_initHeartbeats_1079_);
lean_ctor_set(v___x_1090_, 9, v_maxHeartbeats_1080_);
lean_ctor_set(v___x_1090_, 10, v_quotContext_1081_);
lean_ctor_set(v___x_1090_, 11, v_currMacroScope_1082_);
lean_ctor_set(v___x_1090_, 12, v_cancelTk_x3f_1084_);
lean_ctor_set(v___x_1090_, 13, v_inheritedTraceOptions_1086_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*14, v_diag_1083_);
lean_ctor_set_uint8(v___x_1090_, sizeof(void*)*14 + 1, v_suppressElabErrors_1085_);
lean_inc(v_stx_1063_);
v___x_1091_ = l_Lean_Elab_ConfigEval_EvalTerm_evalOptionStx___redArg(v___x_1087_, v___x_1088_, v_stx_1063_, v_a_1064_, v_a_1065_, v_a_1066_, v_a_1067_, v___x_1090_, v_a_1069_);
if (lean_obj_tag(v___x_1091_) == 0)
{
lean_object* v_a_1092_; lean_object* v___x_1094_; uint8_t v_isShared_1095_; uint8_t v_isSharedCheck_1100_; 
lean_dec_ref_known(v___x_1090_, 14);
lean_dec(v_stx_1063_);
v_a_1092_ = lean_ctor_get(v___x_1091_, 0);
v_isSharedCheck_1100_ = !lean_is_exclusive(v___x_1091_);
if (v_isSharedCheck_1100_ == 0)
{
v___x_1094_ = v___x_1091_;
v_isShared_1095_ = v_isSharedCheck_1100_;
goto v_resetjp_1093_;
}
else
{
lean_inc(v_a_1092_);
lean_dec(v___x_1091_);
v___x_1094_ = lean_box(0);
v_isShared_1095_ = v_isSharedCheck_1100_;
goto v_resetjp_1093_;
}
v_resetjp_1093_:
{
lean_object* v_fst_1096_; lean_object* v___x_1098_; 
v_fst_1096_ = lean_ctor_get(v_a_1092_, 0);
lean_inc(v_fst_1096_);
lean_dec(v_a_1092_);
if (v_isShared_1095_ == 0)
{
lean_ctor_set(v___x_1094_, 0, v_fst_1096_);
v___x_1098_ = v___x_1094_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1099_; 
v_reuseFailAlloc_1099_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1099_, 0, v_fst_1096_);
v___x_1098_ = v_reuseFailAlloc_1099_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
return v___x_1098_;
}
}
}
else
{
lean_object* v_a_1101_; lean_object* v___x_1103_; uint8_t v_isShared_1104_; uint8_t v_isSharedCheck_1116_; 
v_a_1101_ = lean_ctor_get(v___x_1091_, 0);
v_isSharedCheck_1116_ = !lean_is_exclusive(v___x_1091_);
if (v_isSharedCheck_1116_ == 0)
{
v___x_1103_ = v___x_1091_;
v_isShared_1104_ = v_isSharedCheck_1116_;
goto v_resetjp_1102_;
}
else
{
lean_inc(v_a_1101_);
lean_dec(v___x_1091_);
v___x_1103_ = lean_box(0);
v_isShared_1104_ = v_isSharedCheck_1116_;
goto v_resetjp_1102_;
}
v_resetjp_1102_:
{
lean_object* v___x_1105_; lean_object* v___x_1107_; 
v___x_1105_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_inc(v_a_1101_);
if (v_isShared_1104_ == 0)
{
v___x_1107_ = v___x_1103_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1115_; 
v_reuseFailAlloc_1115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1115_, 0, v_a_1101_);
v___x_1107_ = v_reuseFailAlloc_1115_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
uint8_t v___y_1109_; uint8_t v___x_1113_; 
v___x_1113_ = l_Lean_Exception_isInterrupt(v_a_1101_);
if (v___x_1113_ == 0)
{
uint8_t v___x_1114_; 
lean_inc(v_a_1101_);
v___x_1114_ = l_Lean_Exception_isRuntime(v_a_1101_);
v___y_1109_ = v___x_1114_;
goto v___jp_1108_;
}
else
{
v___y_1109_ = v___x_1113_;
goto v___jp_1108_;
}
v___jp_1108_:
{
if (v___y_1109_ == 0)
{
if (lean_obj_tag(v_a_1101_) == 0)
{
lean_dec_ref_known(v_a_1101_, 2);
lean_dec_ref_known(v___x_1090_, 14);
lean_dec(v_stx_1063_);
return v___x_1107_;
}
else
{
lean_object* v_id_1110_; uint8_t v___x_1111_; 
v_id_1110_ = lean_ctor_get(v_a_1101_, 0);
lean_inc(v_id_1110_);
lean_dec_ref_known(v_a_1101_, 2);
v___x_1111_ = l_Lean_instBEqInternalExceptionId_beq(v___x_1105_, v_id_1110_);
lean_dec(v_id_1110_);
if (v___x_1111_ == 0)
{
lean_dec_ref_known(v___x_1090_, 14);
lean_dec(v_stx_1063_);
return v___x_1107_;
}
else
{
lean_object* v___x_1112_; 
lean_dec_ref(v___x_1107_);
v___x_1112_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0_spec__0(v_stx_1063_, v_a_1064_, v_a_1065_, v_a_1066_, v_a_1067_, v___x_1090_, v_a_1069_);
lean_dec_ref_known(v___x_1090_, 14);
return v___x_1112_;
}
}
}
else
{
lean_dec(v_a_1101_);
lean_dec_ref_known(v___x_1090_, 14);
lean_dec(v_stx_1063_);
return v___x_1107_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_1117_, lean_object* v_a_1118_, lean_object* v_a_1119_, lean_object* v_a_1120_, lean_object* v_a_1121_, lean_object* v_a_1122_, lean_object* v_a_1123_, lean_object* v_a_1124_){
_start:
{
lean_object* v_res_1125_; 
v_res_1125_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0(v_stx_1117_, v_a_1118_, v_a_1119_, v_a_1120_, v_a_1121_, v_a_1122_, v_a_1123_);
lean_dec(v_a_1123_);
lean_dec_ref(v_a_1122_);
lean_dec(v_a_1121_);
lean_dec_ref(v_a_1120_);
lean_dec(v_a_1119_);
lean_dec_ref(v_a_1118_);
return v_res_1125_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1126_; lean_object* v___x_1127_; 
v___x_1126_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__1);
v___x_1127_ = l_Lean_MessageData_ofExpr(v___x_1126_);
return v___x_1127_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; 
v___x_1128_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__0);
v___x_1129_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__2);
v___x_1130_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1130_, 0, v___x_1129_);
lean_ctor_set(v___x_1130_, 1, v___x_1128_);
return v___x_1130_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2(void){
_start:
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; 
v___x_1131_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__6);
v___x_1132_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__1);
v___x_1133_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1132_);
lean_ctor_set(v___x_1133_, 1, v___x_1131_);
return v___x_1133_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2(lean_object* v_stx_1134_, lean_object* v_a_1135_, lean_object* v_a_1136_, lean_object* v_a_1137_, lean_object* v_a_1138_, lean_object* v_a_1139_, lean_object* v_a_1140_){
_start:
{
lean_object* v_ty_x3f_1142_; uint8_t v___x_1143_; lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1146_; lean_object* v___x_1147_; lean_object* v_fileName_1148_; lean_object* v_fileMap_1149_; lean_object* v_options_1150_; lean_object* v_currRecDepth_1151_; lean_object* v_maxRecDepth_1152_; lean_object* v_ref_1153_; lean_object* v_currNamespace_1154_; lean_object* v_openDecls_1155_; lean_object* v_initHeartbeats_1156_; lean_object* v_maxHeartbeats_1157_; lean_object* v_quotContext_1158_; lean_object* v_currMacroScope_1159_; uint8_t v_diag_1160_; lean_object* v_cancelTk_x3f_1161_; uint8_t v_suppressElabErrors_1162_; lean_object* v_inheritedTraceOptions_1163_; uint8_t v___x_1164_; lean_object* v_ref_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; 
v_ty_x3f_1142_ = lean_obj_once(&lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2, &lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2_once, _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration___closed__2);
v___x_1143_ = 1;
v___x_1144_ = lean_box(0);
v___x_1145_ = lean_box(v___x_1143_);
v___x_1146_ = lean_box(v___x_1143_);
lean_inc(v_stx_1134_);
v___x_1147_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_1147_, 0, v_stx_1134_);
lean_closure_set(v___x_1147_, 1, v_ty_x3f_1142_);
lean_closure_set(v___x_1147_, 2, v___x_1145_);
lean_closure_set(v___x_1147_, 3, v___x_1146_);
lean_closure_set(v___x_1147_, 4, v___x_1144_);
v_fileName_1148_ = lean_ctor_get(v_a_1139_, 0);
v_fileMap_1149_ = lean_ctor_get(v_a_1139_, 1);
v_options_1150_ = lean_ctor_get(v_a_1139_, 2);
v_currRecDepth_1151_ = lean_ctor_get(v_a_1139_, 3);
v_maxRecDepth_1152_ = lean_ctor_get(v_a_1139_, 4);
v_ref_1153_ = lean_ctor_get(v_a_1139_, 5);
v_currNamespace_1154_ = lean_ctor_get(v_a_1139_, 6);
v_openDecls_1155_ = lean_ctor_get(v_a_1139_, 7);
v_initHeartbeats_1156_ = lean_ctor_get(v_a_1139_, 8);
v_maxHeartbeats_1157_ = lean_ctor_get(v_a_1139_, 9);
v_quotContext_1158_ = lean_ctor_get(v_a_1139_, 10);
v_currMacroScope_1159_ = lean_ctor_get(v_a_1139_, 11);
v_diag_1160_ = lean_ctor_get_uint8(v_a_1139_, sizeof(void*)*14);
v_cancelTk_x3f_1161_ = lean_ctor_get(v_a_1139_, 12);
v_suppressElabErrors_1162_ = lean_ctor_get_uint8(v_a_1139_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1163_ = lean_ctor_get(v_a_1139_, 13);
v___x_1164_ = 1;
v_ref_1165_ = l_Lean_replaceRef(v_stx_1134_, v_ref_1153_);
lean_dec(v_stx_1134_);
lean_inc_ref(v_inheritedTraceOptions_1163_);
lean_inc(v_cancelTk_x3f_1161_);
lean_inc(v_currMacroScope_1159_);
lean_inc(v_quotContext_1158_);
lean_inc(v_maxHeartbeats_1157_);
lean_inc(v_initHeartbeats_1156_);
lean_inc(v_openDecls_1155_);
lean_inc(v_currNamespace_1154_);
lean_inc(v_maxRecDepth_1152_);
lean_inc(v_currRecDepth_1151_);
lean_inc_ref(v_options_1150_);
lean_inc_ref(v_fileMap_1149_);
lean_inc_ref(v_fileName_1148_);
v___x_1166_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1166_, 0, v_fileName_1148_);
lean_ctor_set(v___x_1166_, 1, v_fileMap_1149_);
lean_ctor_set(v___x_1166_, 2, v_options_1150_);
lean_ctor_set(v___x_1166_, 3, v_currRecDepth_1151_);
lean_ctor_set(v___x_1166_, 4, v_maxRecDepth_1152_);
lean_ctor_set(v___x_1166_, 5, v_ref_1165_);
lean_ctor_set(v___x_1166_, 6, v_currNamespace_1154_);
lean_ctor_set(v___x_1166_, 7, v_openDecls_1155_);
lean_ctor_set(v___x_1166_, 8, v_initHeartbeats_1156_);
lean_ctor_set(v___x_1166_, 9, v_maxHeartbeats_1157_);
lean_ctor_set(v___x_1166_, 10, v_quotContext_1158_);
lean_ctor_set(v___x_1166_, 11, v_currMacroScope_1159_);
lean_ctor_set(v___x_1166_, 12, v_cancelTk_x3f_1161_);
lean_ctor_set(v___x_1166_, 13, v_inheritedTraceOptions_1163_);
lean_ctor_set_uint8(v___x_1166_, sizeof(void*)*14, v_diag_1160_);
lean_ctor_set_uint8(v___x_1166_, sizeof(void*)*14 + 1, v_suppressElabErrors_1162_);
v___x_1167_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_1147_, v___x_1164_, v_a_1135_, v_a_1136_, v_a_1137_, v_a_1138_, v___x_1166_, v_a_1140_);
if (lean_obj_tag(v___x_1167_) == 0)
{
lean_object* v_a_1168_; lean_object* v___x_1169_; lean_object* v_a_1170_; lean_object* v___y_1172_; lean_object* v___y_1173_; lean_object* v___y_1174_; lean_object* v___y_1175_; lean_object* v___y_1176_; lean_object* v___y_1177_; lean_object* v___y_1178_; lean_object* v___y_1179_; lean_object* v___y_1180_; uint8_t v___y_1181_; lean_object* v___y_1198_; lean_object* v___y_1199_; lean_object* v___y_1200_; lean_object* v___y_1201_; lean_object* v___y_1202_; lean_object* v___y_1203_; lean_object* v___y_1210_; lean_object* v___y_1211_; lean_object* v___y_1212_; lean_object* v___y_1213_; lean_object* v___y_1214_; lean_object* v___y_1215_; lean_object* v___y_1247_; lean_object* v___y_1248_; lean_object* v___y_1249_; lean_object* v___y_1250_; lean_object* v___y_1251_; lean_object* v___y_1252_; uint8_t v___x_1265_; 
v_a_1168_ = lean_ctor_get(v___x_1167_, 0);
lean_inc(v_a_1168_);
lean_dec_ref_known(v___x_1167_, 1);
v___x_1169_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(v_a_1168_, v_a_1138_);
v_a_1170_ = lean_ctor_get(v___x_1169_, 0);
lean_inc(v_a_1170_);
lean_dec_ref(v___x_1169_);
v___x_1265_ = l_Lean_Expr_hasSorry(v_a_1170_);
if (v___x_1265_ == 0)
{
v___y_1210_ = v_a_1135_;
v___y_1211_ = v_a_1136_;
v___y_1212_ = v_a_1137_;
v___y_1213_ = v_a_1138_;
v___y_1214_ = v___x_1166_;
v___y_1215_ = v_a_1140_;
goto v___jp_1209_;
}
else
{
uint8_t v___x_1266_; 
v___x_1266_ = l_Lean_Expr_hasSyntheticSorry(v_a_1170_);
if (v___x_1266_ == 0)
{
v___y_1247_ = v_a_1135_;
v___y_1248_ = v_a_1136_;
v___y_1249_ = v_a_1137_;
v___y_1250_ = v_a_1138_;
v___y_1251_ = v___x_1166_;
v___y_1252_ = v_a_1140_;
goto v___jp_1246_;
}
else
{
lean_object* v___x_1267_; lean_object* v_a_1268_; lean_object* v___x_1270_; uint8_t v_isShared_1271_; uint8_t v_isSharedCheck_1275_; 
lean_dec(v_a_1170_);
lean_dec_ref_known(v___x_1166_, 14);
v___x_1267_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_1268_ = lean_ctor_get(v___x_1267_, 0);
v_isSharedCheck_1275_ = !lean_is_exclusive(v___x_1267_);
if (v_isSharedCheck_1275_ == 0)
{
v___x_1270_ = v___x_1267_;
v_isShared_1271_ = v_isSharedCheck_1275_;
goto v_resetjp_1269_;
}
else
{
lean_inc(v_a_1268_);
lean_dec(v___x_1267_);
v___x_1270_ = lean_box(0);
v_isShared_1271_ = v_isSharedCheck_1275_;
goto v_resetjp_1269_;
}
v_resetjp_1269_:
{
lean_object* v___x_1273_; 
if (v_isShared_1271_ == 0)
{
v___x_1273_ = v___x_1270_;
goto v_reusejp_1272_;
}
else
{
lean_object* v_reuseFailAlloc_1274_; 
v_reuseFailAlloc_1274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1274_, 0, v_a_1268_);
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
v___jp_1171_:
{
if (v___y_1181_ == 0)
{
if (lean_obj_tag(v___y_1180_) == 0)
{
lean_dec_ref_known(v___y_1180_, 2);
lean_dec_ref(v___y_1177_);
lean_dec(v_a_1170_);
return v___y_1176_;
}
else
{
lean_object* v_id_1182_; lean_object* v___x_1184_; uint8_t v_isShared_1185_; uint8_t v_isSharedCheck_1195_; 
v_id_1182_ = lean_ctor_get(v___y_1180_, 0);
v_isSharedCheck_1195_ = !lean_is_exclusive(v___y_1180_);
if (v_isSharedCheck_1195_ == 0)
{
lean_object* v_unused_1196_; 
v_unused_1196_ = lean_ctor_get(v___y_1180_, 1);
lean_dec(v_unused_1196_);
v___x_1184_ = v___y_1180_;
v_isShared_1185_ = v_isSharedCheck_1195_;
goto v_resetjp_1183_;
}
else
{
lean_inc(v_id_1182_);
lean_dec(v___y_1180_);
v___x_1184_ = lean_box(0);
v_isShared_1185_ = v_isSharedCheck_1195_;
goto v_resetjp_1183_;
}
v_resetjp_1183_:
{
uint8_t v___x_1186_; 
v___x_1186_ = l_Lean_instBEqInternalExceptionId_beq(v___y_1174_, v_id_1182_);
lean_dec(v_id_1182_);
if (v___x_1186_ == 0)
{
lean_del_object(v___x_1184_);
lean_dec_ref(v___y_1177_);
lean_dec(v_a_1170_);
return v___y_1176_;
}
else
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1191_; 
lean_dec_ref(v___y_1176_);
v___x_1187_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___closed__2);
v___x_1188_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__9);
v___x_1189_ = l_Lean_indentExpr(v_a_1170_);
if (v_isShared_1185_ == 0)
{
lean_ctor_set_tag(v___x_1184_, 7);
lean_ctor_set(v___x_1184_, 1, v___x_1189_);
lean_ctor_set(v___x_1184_, 0, v___x_1188_);
v___x_1191_ = v___x_1184_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1194_; 
v_reuseFailAlloc_1194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1194_, 0, v___x_1188_);
lean_ctor_set(v_reuseFailAlloc_1194_, 1, v___x_1189_);
v___x_1191_ = v_reuseFailAlloc_1194_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
lean_object* v___x_1192_; lean_object* v___x_1193_; 
v___x_1192_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1192_, 0, v___x_1191_);
lean_ctor_set(v___x_1192_, 1, v___x_1187_);
v___x_1193_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_1192_, v___y_1178_, v___y_1175_, v___y_1173_, v___y_1179_, v___y_1177_, v___y_1172_);
lean_dec_ref(v___y_1177_);
return v___x_1193_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_1180_);
lean_dec_ref(v___y_1177_);
lean_dec(v_a_1170_);
return v___y_1176_;
}
}
v___jp_1197_:
{
lean_object* v___x_1204_; 
lean_inc(v_a_1170_);
v___x_1204_ = lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr(v_a_1170_, v___y_1200_, v___y_1201_, v___y_1202_, v___y_1203_);
if (lean_obj_tag(v___x_1204_) == 0)
{
lean_dec_ref(v___y_1202_);
lean_dec(v_a_1170_);
return v___x_1204_;
}
else
{
lean_object* v_a_1205_; lean_object* v___x_1206_; uint8_t v___x_1207_; 
v_a_1205_ = lean_ctor_get(v___x_1204_, 0);
lean_inc(v_a_1205_);
v___x_1206_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_1207_ = l_Lean_Exception_isInterrupt(v_a_1205_);
if (v___x_1207_ == 0)
{
uint8_t v___x_1208_; 
lean_inc(v_a_1205_);
v___x_1208_ = l_Lean_Exception_isRuntime(v_a_1205_);
v___y_1172_ = v___y_1203_;
v___y_1173_ = v___y_1200_;
v___y_1174_ = v___x_1206_;
v___y_1175_ = v___y_1199_;
v___y_1176_ = v___x_1204_;
v___y_1177_ = v___y_1202_;
v___y_1178_ = v___y_1198_;
v___y_1179_ = v___y_1201_;
v___y_1180_ = v_a_1205_;
v___y_1181_ = v___x_1208_;
goto v___jp_1171_;
}
else
{
v___y_1172_ = v___y_1203_;
v___y_1173_ = v___y_1200_;
v___y_1174_ = v___x_1206_;
v___y_1175_ = v___y_1199_;
v___y_1176_ = v___x_1204_;
v___y_1177_ = v___y_1202_;
v___y_1178_ = v___y_1198_;
v___y_1179_ = v___y_1201_;
v___y_1180_ = v_a_1205_;
v___y_1181_ = v___x_1207_;
goto v___jp_1171_;
}
}
}
v___jp_1209_:
{
lean_object* v___x_1216_; 
lean_inc(v_a_1170_);
v___x_1216_ = l_Lean_Meta_getMVars(v_a_1170_, v___y_1212_, v___y_1213_, v___y_1214_, v___y_1215_);
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_object* v_a_1217_; lean_object* v___x_1218_; 
v_a_1217_ = lean_ctor_get(v___x_1216_, 0);
lean_inc(v_a_1217_);
lean_dec_ref_known(v___x_1216_, 1);
v___x_1218_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_1217_, v___x_1144_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_, v___y_1214_, v___y_1215_);
lean_dec(v_a_1217_);
if (lean_obj_tag(v___x_1218_) == 0)
{
lean_object* v_a_1219_; uint8_t v___x_1220_; 
v_a_1219_ = lean_ctor_get(v___x_1218_, 0);
lean_inc(v_a_1219_);
lean_dec_ref_known(v___x_1218_, 1);
v___x_1220_ = lean_unbox(v_a_1219_);
lean_dec(v_a_1219_);
if (v___x_1220_ == 0)
{
v___y_1198_ = v___y_1210_;
v___y_1199_ = v___y_1211_;
v___y_1200_ = v___y_1212_;
v___y_1201_ = v___y_1213_;
v___y_1202_ = v___y_1214_;
v___y_1203_ = v___y_1215_;
goto v___jp_1197_;
}
else
{
lean_object* v___x_1221_; lean_object* v_a_1222_; lean_object* v___x_1224_; uint8_t v_isShared_1225_; uint8_t v_isSharedCheck_1229_; 
lean_dec_ref(v___y_1214_);
lean_dec(v_a_1170_);
v___x_1221_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
v_a_1222_ = lean_ctor_get(v___x_1221_, 0);
v_isSharedCheck_1229_ = !lean_is_exclusive(v___x_1221_);
if (v_isSharedCheck_1229_ == 0)
{
v___x_1224_ = v___x_1221_;
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
else
{
lean_inc(v_a_1222_);
lean_dec(v___x_1221_);
v___x_1224_ = lean_box(0);
v_isShared_1225_ = v_isSharedCheck_1229_;
goto v_resetjp_1223_;
}
v_resetjp_1223_:
{
lean_object* v___x_1227_; 
if (v_isShared_1225_ == 0)
{
v___x_1227_ = v___x_1224_;
goto v_reusejp_1226_;
}
else
{
lean_object* v_reuseFailAlloc_1228_; 
v_reuseFailAlloc_1228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1228_, 0, v_a_1222_);
v___x_1227_ = v_reuseFailAlloc_1228_;
goto v_reusejp_1226_;
}
v_reusejp_1226_:
{
return v___x_1227_;
}
}
}
}
else
{
lean_object* v_a_1230_; lean_object* v___x_1232_; uint8_t v_isShared_1233_; uint8_t v_isSharedCheck_1237_; 
lean_dec_ref(v___y_1214_);
lean_dec(v_a_1170_);
v_a_1230_ = lean_ctor_get(v___x_1218_, 0);
v_isSharedCheck_1237_ = !lean_is_exclusive(v___x_1218_);
if (v_isSharedCheck_1237_ == 0)
{
v___x_1232_ = v___x_1218_;
v_isShared_1233_ = v_isSharedCheck_1237_;
goto v_resetjp_1231_;
}
else
{
lean_inc(v_a_1230_);
lean_dec(v___x_1218_);
v___x_1232_ = lean_box(0);
v_isShared_1233_ = v_isSharedCheck_1237_;
goto v_resetjp_1231_;
}
v_resetjp_1231_:
{
lean_object* v___x_1235_; 
if (v_isShared_1233_ == 0)
{
v___x_1235_ = v___x_1232_;
goto v_reusejp_1234_;
}
else
{
lean_object* v_reuseFailAlloc_1236_; 
v_reuseFailAlloc_1236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1236_, 0, v_a_1230_);
v___x_1235_ = v_reuseFailAlloc_1236_;
goto v_reusejp_1234_;
}
v_reusejp_1234_:
{
return v___x_1235_;
}
}
}
}
else
{
lean_object* v_a_1238_; lean_object* v___x_1240_; uint8_t v_isShared_1241_; uint8_t v_isSharedCheck_1245_; 
lean_dec_ref(v___y_1214_);
lean_dec(v_a_1170_);
v_a_1238_ = lean_ctor_get(v___x_1216_, 0);
v_isSharedCheck_1245_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1245_ == 0)
{
v___x_1240_ = v___x_1216_;
v_isShared_1241_ = v_isSharedCheck_1245_;
goto v_resetjp_1239_;
}
else
{
lean_inc(v_a_1238_);
lean_dec(v___x_1216_);
v___x_1240_ = lean_box(0);
v_isShared_1241_ = v_isSharedCheck_1245_;
goto v_resetjp_1239_;
}
v_resetjp_1239_:
{
lean_object* v___x_1243_; 
if (v_isShared_1241_ == 0)
{
v___x_1243_ = v___x_1240_;
goto v_reusejp_1242_;
}
else
{
lean_object* v_reuseFailAlloc_1244_; 
v_reuseFailAlloc_1244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1244_, 0, v_a_1238_);
v___x_1243_ = v_reuseFailAlloc_1244_;
goto v_reusejp_1242_;
}
v_reusejp_1242_:
{
return v___x_1243_;
}
}
}
}
v___jp_1246_:
{
lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v_a_1257_; lean_object* v___x_1259_; uint8_t v_isShared_1260_; uint8_t v_isSharedCheck_1264_; 
v___x_1253_ = lean_obj_once(&lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11, &lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11_once, _init_lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1_spec__2___closed__11);
v___x_1254_ = l_Lean_indentExpr(v_a_1170_);
v___x_1255_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1255_, 0, v___x_1253_);
lean_ctor_set(v___x_1255_, 1, v___x_1254_);
v___x_1256_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v___x_1255_, v___y_1247_, v___y_1248_, v___y_1249_, v___y_1250_, v___y_1251_, v___y_1252_);
lean_dec_ref(v___y_1251_);
v_a_1257_ = lean_ctor_get(v___x_1256_, 0);
v_isSharedCheck_1264_ = !lean_is_exclusive(v___x_1256_);
if (v_isSharedCheck_1264_ == 0)
{
v___x_1259_ = v___x_1256_;
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
else
{
lean_inc(v_a_1257_);
lean_dec(v___x_1256_);
v___x_1259_ = lean_box(0);
v_isShared_1260_ = v_isSharedCheck_1264_;
goto v_resetjp_1258_;
}
v_resetjp_1258_:
{
lean_object* v___x_1262_; 
if (v_isShared_1260_ == 0)
{
v___x_1262_ = v___x_1259_;
goto v_reusejp_1261_;
}
else
{
lean_object* v_reuseFailAlloc_1263_; 
v_reuseFailAlloc_1263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1263_, 0, v_a_1257_);
v___x_1262_ = v_reuseFailAlloc_1263_;
goto v_reusejp_1261_;
}
v_reusejp_1261_:
{
return v___x_1262_;
}
}
}
}
else
{
lean_object* v_a_1276_; lean_object* v___x_1278_; uint8_t v_isShared_1279_; uint8_t v_isSharedCheck_1283_; 
lean_dec_ref_known(v___x_1166_, 14);
v_a_1276_ = lean_ctor_get(v___x_1167_, 0);
v_isSharedCheck_1283_ = !lean_is_exclusive(v___x_1167_);
if (v_isSharedCheck_1283_ == 0)
{
v___x_1278_ = v___x_1167_;
v_isShared_1279_ = v_isSharedCheck_1283_;
goto v_resetjp_1277_;
}
else
{
lean_inc(v_a_1276_);
lean_dec(v___x_1167_);
v___x_1278_ = lean_box(0);
v_isShared_1279_ = v_isSharedCheck_1283_;
goto v_resetjp_1277_;
}
v_resetjp_1277_:
{
lean_object* v___x_1281_; 
if (v_isShared_1279_ == 0)
{
v___x_1281_ = v___x_1278_;
goto v_reusejp_1280_;
}
else
{
lean_object* v_reuseFailAlloc_1282_; 
v_reuseFailAlloc_1282_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1282_, 0, v_a_1276_);
v___x_1281_ = v_reuseFailAlloc_1282_;
goto v_reusejp_1280_;
}
v_reusejp_1280_:
{
return v___x_1281_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2___boxed(lean_object* v_stx_1284_, lean_object* v_a_1285_, lean_object* v_a_1286_, lean_object* v_a_1287_, lean_object* v_a_1288_, lean_object* v_a_1289_, lean_object* v_a_1290_, lean_object* v_a_1291_){
_start:
{
lean_object* v_res_1292_; 
v_res_1292_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2(v_stx_1284_, v_a_1285_, v_a_1286_, v_a_1287_, v_a_1288_, v_a_1289_, v_a_1290_);
lean_dec(v_a_1290_);
lean_dec_ref(v_a_1289_);
lean_dec(v_a_1288_);
lean_dec_ref(v_a_1287_);
lean_dec(v_a_1286_);
lean_dec_ref(v_a_1285_);
return v_res_1292_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0(lean_object* v_config_1346_, lean_object* v_item_1347_, lean_object* v___y_1348_, lean_object* v___y_1349_, lean_object* v___y_1350_, lean_object* v___y_1351_, lean_object* v___y_1352_, lean_object* v___y_1353_){
_start:
{
lean_object* v_item_1356_; lean_object* v___y_1357_; lean_object* v___y_1358_; lean_object* v___y_1359_; lean_object* v___y_1360_; lean_object* v___y_1361_; lean_object* v___y_1362_; lean_object* v___x_1365_; lean_object* v___x_1366_; 
v___x_1365_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1));
v___x_1366_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_1347_, v___x_1365_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1366_) == 0)
{
uint8_t v___x_1367_; 
lean_dec_ref_known(v___x_1366_, 1);
v___x_1367_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_1347_);
if (v___x_1367_ == 0)
{
lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; uint8_t v___x_1371_; 
v___x_1368_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_1347_);
lean_inc_ref(v_item_1347_);
v___x_1369_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_1347_);
v___x_1370_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__1));
v___x_1371_ = lean_string_dec_lt(v___x_1368_, v___x_1370_);
if (v___x_1371_ == 0)
{
lean_object* v___x_1372_; uint8_t v___x_1373_; 
v___x_1372_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__2));
v___x_1373_ = lean_string_dec_lt(v___x_1368_, v___x_1372_);
if (v___x_1373_ == 0)
{
uint8_t v___x_1374_; 
v___x_1374_ = lean_string_dec_eq(v___x_1368_, v___x_1372_);
if (v___x_1374_ == 0)
{
lean_object* v___x_1375_; uint8_t v___x_1376_; 
v___x_1375_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__3));
v___x_1376_ = lean_string_dec_eq(v___x_1368_, v___x_1375_);
if (v___x_1376_ == 0)
{
lean_object* v___x_1377_; uint8_t v___x_1378_; 
v___x_1377_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__4));
v___x_1378_ = lean_string_dec_eq(v___x_1368_, v___x_1377_);
lean_dec_ref(v___x_1368_);
if (v___x_1378_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1379_; lean_object* v___x_1380_; 
v___x_1379_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__5));
v___x_1380_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1379_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1380_) == 0)
{
uint8_t v___x_1381_; 
lean_dec_ref_known(v___x_1380_, 1);
v___x_1381_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1381_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1382_; 
lean_dec_ref(v___x_1369_);
v___x_1382_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1382_) == 0)
{
lean_object* v_a_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1407_; 
v_a_1383_ = lean_ctor_get(v___x_1382_, 0);
v_isSharedCheck_1407_ = !lean_is_exclusive(v___x_1382_);
if (v_isSharedCheck_1407_ == 0)
{
v___x_1385_ = v___x_1382_;
v_isShared_1386_ = v_isSharedCheck_1407_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_a_1383_);
lean_dec(v___x_1382_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1407_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v_numInst_1387_; lean_object* v_maxSize_1388_; lean_object* v_numRetries_1389_; uint8_t v_traceDiscarded_1390_; uint8_t v_traceShrink_1391_; uint8_t v_traceShrinkCandidates_1392_; lean_object* v_randomSeed_1393_; uint8_t v_quiet_1394_; uint8_t v_sorryIfNoTestable_1395_; lean_object* v___x_1397_; uint8_t v_isShared_1398_; uint8_t v_isSharedCheck_1406_; 
v_numInst_1387_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1388_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1389_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1390_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceShrink_1391_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1392_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1393_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1394_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1395_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1406_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1406_ == 0)
{
v___x_1397_ = v_config_1346_;
v_isShared_1398_ = v_isSharedCheck_1406_;
goto v_resetjp_1396_;
}
else
{
lean_inc(v_randomSeed_1393_);
lean_inc(v_numRetries_1389_);
lean_inc(v_maxSize_1388_);
lean_inc(v_numInst_1387_);
lean_dec(v_config_1346_);
v___x_1397_ = lean_box(0);
v_isShared_1398_ = v_isSharedCheck_1406_;
goto v_resetjp_1396_;
}
v_resetjp_1396_:
{
lean_object* v___x_1400_; 
if (v_isShared_1398_ == 0)
{
v___x_1400_ = v___x_1397_;
goto v_reusejp_1399_;
}
else
{
lean_object* v_reuseFailAlloc_1405_; 
v_reuseFailAlloc_1405_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1405_, 0, v_numInst_1387_);
lean_ctor_set(v_reuseFailAlloc_1405_, 1, v_maxSize_1388_);
lean_ctor_set(v_reuseFailAlloc_1405_, 2, v_numRetries_1389_);
lean_ctor_set(v_reuseFailAlloc_1405_, 3, v_randomSeed_1393_);
lean_ctor_set_uint8(v_reuseFailAlloc_1405_, sizeof(void*)*4, v_traceDiscarded_1390_);
v___x_1400_ = v_reuseFailAlloc_1405_;
goto v_reusejp_1399_;
}
v_reusejp_1399_:
{
uint8_t v___x_1401_; lean_object* v___x_1403_; 
v___x_1401_ = lean_unbox(v_a_1383_);
lean_dec(v_a_1383_);
lean_ctor_set_uint8(v___x_1400_, sizeof(void*)*4 + 1, v___x_1401_);
lean_ctor_set_uint8(v___x_1400_, sizeof(void*)*4 + 2, v_traceShrink_1391_);
lean_ctor_set_uint8(v___x_1400_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1392_);
lean_ctor_set_uint8(v___x_1400_, sizeof(void*)*4 + 4, v_quiet_1394_);
lean_ctor_set_uint8(v___x_1400_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1395_);
if (v_isShared_1386_ == 0)
{
lean_ctor_set(v___x_1385_, 0, v___x_1400_);
v___x_1403_ = v___x_1385_;
goto v_reusejp_1402_;
}
else
{
lean_object* v_reuseFailAlloc_1404_; 
v_reuseFailAlloc_1404_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1404_, 0, v___x_1400_);
v___x_1403_ = v_reuseFailAlloc_1404_;
goto v_reusejp_1402_;
}
v_reusejp_1402_:
{
return v___x_1403_;
}
}
}
}
}
else
{
lean_object* v_a_1408_; lean_object* v___x_1410_; uint8_t v_isShared_1411_; uint8_t v_isSharedCheck_1415_; 
lean_dec_ref(v_config_1346_);
v_a_1408_ = lean_ctor_get(v___x_1382_, 0);
v_isSharedCheck_1415_ = !lean_is_exclusive(v___x_1382_);
if (v_isSharedCheck_1415_ == 0)
{
v___x_1410_ = v___x_1382_;
v_isShared_1411_ = v_isSharedCheck_1415_;
goto v_resetjp_1409_;
}
else
{
lean_inc(v_a_1408_);
lean_dec(v___x_1382_);
v___x_1410_ = lean_box(0);
v_isShared_1411_ = v_isSharedCheck_1415_;
goto v_resetjp_1409_;
}
v_resetjp_1409_:
{
lean_object* v___x_1413_; 
if (v_isShared_1411_ == 0)
{
v___x_1413_ = v___x_1410_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1414_; 
v_reuseFailAlloc_1414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1414_, 0, v_a_1408_);
v___x_1413_ = v_reuseFailAlloc_1414_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
return v___x_1413_;
}
}
}
}
}
else
{
lean_object* v_a_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1423_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1416_ = lean_ctor_get(v___x_1380_, 0);
v_isSharedCheck_1423_ = !lean_is_exclusive(v___x_1380_);
if (v_isSharedCheck_1423_ == 0)
{
v___x_1418_ = v___x_1380_;
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
else
{
lean_inc(v_a_1416_);
lean_dec(v___x_1380_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1423_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v___x_1421_; 
if (v_isShared_1419_ == 0)
{
v___x_1421_ = v___x_1418_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v_a_1416_);
v___x_1421_ = v_reuseFailAlloc_1422_;
goto v_reusejp_1420_;
}
v_reusejp_1420_:
{
return v___x_1421_;
}
}
}
}
}
else
{
lean_object* v___x_1424_; lean_object* v___x_1425_; 
lean_dec_ref(v___x_1368_);
v___x_1424_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__6));
v___x_1425_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1424_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1425_) == 0)
{
uint8_t v___x_1426_; 
lean_dec_ref_known(v___x_1425_, 1);
v___x_1426_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1426_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1427_; 
lean_dec_ref(v___x_1369_);
v___x_1427_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1427_) == 0)
{
lean_object* v_a_1428_; lean_object* v___x_1430_; uint8_t v_isShared_1431_; uint8_t v_isSharedCheck_1452_; 
v_a_1428_ = lean_ctor_get(v___x_1427_, 0);
v_isSharedCheck_1452_ = !lean_is_exclusive(v___x_1427_);
if (v_isSharedCheck_1452_ == 0)
{
v___x_1430_ = v___x_1427_;
v_isShared_1431_ = v_isSharedCheck_1452_;
goto v_resetjp_1429_;
}
else
{
lean_inc(v_a_1428_);
lean_dec(v___x_1427_);
v___x_1430_ = lean_box(0);
v_isShared_1431_ = v_isSharedCheck_1452_;
goto v_resetjp_1429_;
}
v_resetjp_1429_:
{
lean_object* v_numInst_1432_; lean_object* v_maxSize_1433_; lean_object* v_numRetries_1434_; uint8_t v_traceDiscarded_1435_; uint8_t v_traceSuccesses_1436_; uint8_t v_traceShrink_1437_; lean_object* v_randomSeed_1438_; uint8_t v_quiet_1439_; uint8_t v_sorryIfNoTestable_1440_; lean_object* v___x_1442_; uint8_t v_isShared_1443_; uint8_t v_isSharedCheck_1451_; 
v_numInst_1432_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1433_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1434_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1435_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1436_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1437_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_randomSeed_1438_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1439_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1440_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1451_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1451_ == 0)
{
v___x_1442_ = v_config_1346_;
v_isShared_1443_ = v_isSharedCheck_1451_;
goto v_resetjp_1441_;
}
else
{
lean_inc(v_randomSeed_1438_);
lean_inc(v_numRetries_1434_);
lean_inc(v_maxSize_1433_);
lean_inc(v_numInst_1432_);
lean_dec(v_config_1346_);
v___x_1442_ = lean_box(0);
v_isShared_1443_ = v_isSharedCheck_1451_;
goto v_resetjp_1441_;
}
v_resetjp_1441_:
{
lean_object* v___x_1445_; 
if (v_isShared_1443_ == 0)
{
v___x_1445_ = v___x_1442_;
goto v_reusejp_1444_;
}
else
{
lean_object* v_reuseFailAlloc_1450_; 
v_reuseFailAlloc_1450_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1450_, 0, v_numInst_1432_);
lean_ctor_set(v_reuseFailAlloc_1450_, 1, v_maxSize_1433_);
lean_ctor_set(v_reuseFailAlloc_1450_, 2, v_numRetries_1434_);
lean_ctor_set(v_reuseFailAlloc_1450_, 3, v_randomSeed_1438_);
lean_ctor_set_uint8(v_reuseFailAlloc_1450_, sizeof(void*)*4, v_traceDiscarded_1435_);
lean_ctor_set_uint8(v_reuseFailAlloc_1450_, sizeof(void*)*4 + 1, v_traceSuccesses_1436_);
lean_ctor_set_uint8(v_reuseFailAlloc_1450_, sizeof(void*)*4 + 2, v_traceShrink_1437_);
v___x_1445_ = v_reuseFailAlloc_1450_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
uint8_t v___x_1446_; lean_object* v___x_1448_; 
v___x_1446_ = lean_unbox(v_a_1428_);
lean_dec(v_a_1428_);
lean_ctor_set_uint8(v___x_1445_, sizeof(void*)*4 + 3, v___x_1446_);
lean_ctor_set_uint8(v___x_1445_, sizeof(void*)*4 + 4, v_quiet_1439_);
lean_ctor_set_uint8(v___x_1445_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1440_);
if (v_isShared_1431_ == 0)
{
lean_ctor_set(v___x_1430_, 0, v___x_1445_);
v___x_1448_ = v___x_1430_;
goto v_reusejp_1447_;
}
else
{
lean_object* v_reuseFailAlloc_1449_; 
v_reuseFailAlloc_1449_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1449_, 0, v___x_1445_);
v___x_1448_ = v_reuseFailAlloc_1449_;
goto v_reusejp_1447_;
}
v_reusejp_1447_:
{
return v___x_1448_;
}
}
}
}
}
else
{
lean_object* v_a_1453_; lean_object* v___x_1455_; uint8_t v_isShared_1456_; uint8_t v_isSharedCheck_1460_; 
lean_dec_ref(v_config_1346_);
v_a_1453_ = lean_ctor_get(v___x_1427_, 0);
v_isSharedCheck_1460_ = !lean_is_exclusive(v___x_1427_);
if (v_isSharedCheck_1460_ == 0)
{
v___x_1455_ = v___x_1427_;
v_isShared_1456_ = v_isSharedCheck_1460_;
goto v_resetjp_1454_;
}
else
{
lean_inc(v_a_1453_);
lean_dec(v___x_1427_);
v___x_1455_ = lean_box(0);
v_isShared_1456_ = v_isSharedCheck_1460_;
goto v_resetjp_1454_;
}
v_resetjp_1454_:
{
lean_object* v___x_1458_; 
if (v_isShared_1456_ == 0)
{
v___x_1458_ = v___x_1455_;
goto v_reusejp_1457_;
}
else
{
lean_object* v_reuseFailAlloc_1459_; 
v_reuseFailAlloc_1459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1459_, 0, v_a_1453_);
v___x_1458_ = v_reuseFailAlloc_1459_;
goto v_reusejp_1457_;
}
v_reusejp_1457_:
{
return v___x_1458_;
}
}
}
}
}
else
{
lean_object* v_a_1461_; lean_object* v___x_1463_; uint8_t v_isShared_1464_; uint8_t v_isSharedCheck_1468_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1461_ = lean_ctor_get(v___x_1425_, 0);
v_isSharedCheck_1468_ = !lean_is_exclusive(v___x_1425_);
if (v_isSharedCheck_1468_ == 0)
{
v___x_1463_ = v___x_1425_;
v_isShared_1464_ = v_isSharedCheck_1468_;
goto v_resetjp_1462_;
}
else
{
lean_inc(v_a_1461_);
lean_dec(v___x_1425_);
v___x_1463_ = lean_box(0);
v_isShared_1464_ = v_isSharedCheck_1468_;
goto v_resetjp_1462_;
}
v_resetjp_1462_:
{
lean_object* v___x_1466_; 
if (v_isShared_1464_ == 0)
{
v___x_1466_ = v___x_1463_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v_a_1461_);
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
}
else
{
lean_object* v___x_1469_; lean_object* v___x_1470_; 
lean_dec_ref(v___x_1368_);
v___x_1469_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__7));
v___x_1470_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1469_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1470_) == 0)
{
uint8_t v___x_1471_; 
lean_dec_ref_known(v___x_1470_, 1);
v___x_1471_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1471_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1472_; 
lean_dec_ref(v___x_1369_);
v___x_1472_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1472_) == 0)
{
lean_object* v_a_1473_; lean_object* v___x_1475_; uint8_t v_isShared_1476_; uint8_t v_isSharedCheck_1497_; 
v_a_1473_ = lean_ctor_get(v___x_1472_, 0);
v_isSharedCheck_1497_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1497_ == 0)
{
v___x_1475_ = v___x_1472_;
v_isShared_1476_ = v_isSharedCheck_1497_;
goto v_resetjp_1474_;
}
else
{
lean_inc(v_a_1473_);
lean_dec(v___x_1472_);
v___x_1475_ = lean_box(0);
v_isShared_1476_ = v_isSharedCheck_1497_;
goto v_resetjp_1474_;
}
v_resetjp_1474_:
{
lean_object* v_numInst_1477_; lean_object* v_maxSize_1478_; lean_object* v_numRetries_1479_; uint8_t v_traceDiscarded_1480_; uint8_t v_traceSuccesses_1481_; uint8_t v_traceShrinkCandidates_1482_; lean_object* v_randomSeed_1483_; uint8_t v_quiet_1484_; uint8_t v_sorryIfNoTestable_1485_; lean_object* v___x_1487_; uint8_t v_isShared_1488_; uint8_t v_isSharedCheck_1496_; 
v_numInst_1477_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1478_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1479_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1480_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1481_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrinkCandidates_1482_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1483_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1484_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1485_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1496_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1496_ == 0)
{
v___x_1487_ = v_config_1346_;
v_isShared_1488_ = v_isSharedCheck_1496_;
goto v_resetjp_1486_;
}
else
{
lean_inc(v_randomSeed_1483_);
lean_inc(v_numRetries_1479_);
lean_inc(v_maxSize_1478_);
lean_inc(v_numInst_1477_);
lean_dec(v_config_1346_);
v___x_1487_ = lean_box(0);
v_isShared_1488_ = v_isSharedCheck_1496_;
goto v_resetjp_1486_;
}
v_resetjp_1486_:
{
lean_object* v___x_1490_; 
if (v_isShared_1488_ == 0)
{
v___x_1490_ = v___x_1487_;
goto v_reusejp_1489_;
}
else
{
lean_object* v_reuseFailAlloc_1495_; 
v_reuseFailAlloc_1495_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1495_, 0, v_numInst_1477_);
lean_ctor_set(v_reuseFailAlloc_1495_, 1, v_maxSize_1478_);
lean_ctor_set(v_reuseFailAlloc_1495_, 2, v_numRetries_1479_);
lean_ctor_set(v_reuseFailAlloc_1495_, 3, v_randomSeed_1483_);
lean_ctor_set_uint8(v_reuseFailAlloc_1495_, sizeof(void*)*4, v_traceDiscarded_1480_);
lean_ctor_set_uint8(v_reuseFailAlloc_1495_, sizeof(void*)*4 + 1, v_traceSuccesses_1481_);
v___x_1490_ = v_reuseFailAlloc_1495_;
goto v_reusejp_1489_;
}
v_reusejp_1489_:
{
uint8_t v___x_1491_; lean_object* v___x_1493_; 
v___x_1491_ = lean_unbox(v_a_1473_);
lean_dec(v_a_1473_);
lean_ctor_set_uint8(v___x_1490_, sizeof(void*)*4 + 2, v___x_1491_);
lean_ctor_set_uint8(v___x_1490_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1482_);
lean_ctor_set_uint8(v___x_1490_, sizeof(void*)*4 + 4, v_quiet_1484_);
lean_ctor_set_uint8(v___x_1490_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1485_);
if (v_isShared_1476_ == 0)
{
lean_ctor_set(v___x_1475_, 0, v___x_1490_);
v___x_1493_ = v___x_1475_;
goto v_reusejp_1492_;
}
else
{
lean_object* v_reuseFailAlloc_1494_; 
v_reuseFailAlloc_1494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1494_, 0, v___x_1490_);
v___x_1493_ = v_reuseFailAlloc_1494_;
goto v_reusejp_1492_;
}
v_reusejp_1492_:
{
return v___x_1493_;
}
}
}
}
}
else
{
lean_object* v_a_1498_; lean_object* v___x_1500_; uint8_t v_isShared_1501_; uint8_t v_isSharedCheck_1505_; 
lean_dec_ref(v_config_1346_);
v_a_1498_ = lean_ctor_get(v___x_1472_, 0);
v_isSharedCheck_1505_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1505_ == 0)
{
v___x_1500_ = v___x_1472_;
v_isShared_1501_ = v_isSharedCheck_1505_;
goto v_resetjp_1499_;
}
else
{
lean_inc(v_a_1498_);
lean_dec(v___x_1472_);
v___x_1500_ = lean_box(0);
v_isShared_1501_ = v_isSharedCheck_1505_;
goto v_resetjp_1499_;
}
v_resetjp_1499_:
{
lean_object* v___x_1503_; 
if (v_isShared_1501_ == 0)
{
v___x_1503_ = v___x_1500_;
goto v_reusejp_1502_;
}
else
{
lean_object* v_reuseFailAlloc_1504_; 
v_reuseFailAlloc_1504_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1504_, 0, v_a_1498_);
v___x_1503_ = v_reuseFailAlloc_1504_;
goto v_reusejp_1502_;
}
v_reusejp_1502_:
{
return v___x_1503_;
}
}
}
}
}
else
{
lean_object* v_a_1506_; lean_object* v___x_1508_; uint8_t v_isShared_1509_; uint8_t v_isSharedCheck_1513_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1506_ = lean_ctor_get(v___x_1470_, 0);
v_isSharedCheck_1513_ = !lean_is_exclusive(v___x_1470_);
if (v_isSharedCheck_1513_ == 0)
{
v___x_1508_ = v___x_1470_;
v_isShared_1509_ = v_isSharedCheck_1513_;
goto v_resetjp_1507_;
}
else
{
lean_inc(v_a_1506_);
lean_dec(v___x_1470_);
v___x_1508_ = lean_box(0);
v_isShared_1509_ = v_isSharedCheck_1513_;
goto v_resetjp_1507_;
}
v_resetjp_1507_:
{
lean_object* v___x_1511_; 
if (v_isShared_1509_ == 0)
{
v___x_1511_ = v___x_1508_;
goto v_reusejp_1510_;
}
else
{
lean_object* v_reuseFailAlloc_1512_; 
v_reuseFailAlloc_1512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1512_, 0, v_a_1506_);
v___x_1511_ = v_reuseFailAlloc_1512_;
goto v_reusejp_1510_;
}
v_reusejp_1510_:
{
return v___x_1511_;
}
}
}
}
}
else
{
uint8_t v___x_1514_; 
v___x_1514_ = lean_string_dec_eq(v___x_1368_, v___x_1370_);
if (v___x_1514_ == 0)
{
lean_object* v___x_1515_; uint8_t v___x_1516_; 
v___x_1515_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__8));
v___x_1516_ = lean_string_dec_eq(v___x_1368_, v___x_1515_);
if (v___x_1516_ == 0)
{
lean_object* v___x_1517_; uint8_t v___x_1518_; 
v___x_1517_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__9));
v___x_1518_ = lean_string_dec_eq(v___x_1368_, v___x_1517_);
lean_dec_ref(v___x_1368_);
if (v___x_1518_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1519_; lean_object* v___x_1520_; 
v___x_1519_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__10));
v___x_1520_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1519_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1520_) == 0)
{
uint8_t v___x_1521_; 
lean_dec_ref_known(v___x_1520_, 1);
v___x_1521_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1521_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1522_; 
lean_dec_ref(v___x_1369_);
v___x_1522_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1522_) == 0)
{
lean_object* v_a_1523_; lean_object* v___x_1525_; uint8_t v_isShared_1526_; uint8_t v_isSharedCheck_1547_; 
v_a_1523_ = lean_ctor_get(v___x_1522_, 0);
v_isSharedCheck_1547_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1547_ == 0)
{
v___x_1525_ = v___x_1522_;
v_isShared_1526_ = v_isSharedCheck_1547_;
goto v_resetjp_1524_;
}
else
{
lean_inc(v_a_1523_);
lean_dec(v___x_1522_);
v___x_1525_ = lean_box(0);
v_isShared_1526_ = v_isSharedCheck_1547_;
goto v_resetjp_1524_;
}
v_resetjp_1524_:
{
lean_object* v_numInst_1527_; lean_object* v_maxSize_1528_; lean_object* v_numRetries_1529_; uint8_t v_traceSuccesses_1530_; uint8_t v_traceShrink_1531_; uint8_t v_traceShrinkCandidates_1532_; lean_object* v_randomSeed_1533_; uint8_t v_quiet_1534_; uint8_t v_sorryIfNoTestable_1535_; lean_object* v___x_1537_; uint8_t v_isShared_1538_; uint8_t v_isSharedCheck_1546_; 
v_numInst_1527_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1528_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1529_ = lean_ctor_get(v_config_1346_, 2);
v_traceSuccesses_1530_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1531_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1532_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1533_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1534_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1535_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1546_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1546_ == 0)
{
v___x_1537_ = v_config_1346_;
v_isShared_1538_ = v_isSharedCheck_1546_;
goto v_resetjp_1536_;
}
else
{
lean_inc(v_randomSeed_1533_);
lean_inc(v_numRetries_1529_);
lean_inc(v_maxSize_1528_);
lean_inc(v_numInst_1527_);
lean_dec(v_config_1346_);
v___x_1537_ = lean_box(0);
v_isShared_1538_ = v_isSharedCheck_1546_;
goto v_resetjp_1536_;
}
v_resetjp_1536_:
{
lean_object* v___x_1540_; 
if (v_isShared_1538_ == 0)
{
v___x_1540_ = v___x_1537_;
goto v_reusejp_1539_;
}
else
{
lean_object* v_reuseFailAlloc_1545_; 
v_reuseFailAlloc_1545_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1545_, 0, v_numInst_1527_);
lean_ctor_set(v_reuseFailAlloc_1545_, 1, v_maxSize_1528_);
lean_ctor_set(v_reuseFailAlloc_1545_, 2, v_numRetries_1529_);
lean_ctor_set(v_reuseFailAlloc_1545_, 3, v_randomSeed_1533_);
v___x_1540_ = v_reuseFailAlloc_1545_;
goto v_reusejp_1539_;
}
v_reusejp_1539_:
{
uint8_t v___x_1541_; lean_object* v___x_1543_; 
v___x_1541_ = lean_unbox(v_a_1523_);
lean_dec(v_a_1523_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4, v___x_1541_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4 + 1, v_traceSuccesses_1530_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4 + 2, v_traceShrink_1531_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1532_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4 + 4, v_quiet_1534_);
lean_ctor_set_uint8(v___x_1540_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1535_);
if (v_isShared_1526_ == 0)
{
lean_ctor_set(v___x_1525_, 0, v___x_1540_);
v___x_1543_ = v___x_1525_;
goto v_reusejp_1542_;
}
else
{
lean_object* v_reuseFailAlloc_1544_; 
v_reuseFailAlloc_1544_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1544_, 0, v___x_1540_);
v___x_1543_ = v_reuseFailAlloc_1544_;
goto v_reusejp_1542_;
}
v_reusejp_1542_:
{
return v___x_1543_;
}
}
}
}
}
else
{
lean_object* v_a_1548_; lean_object* v___x_1550_; uint8_t v_isShared_1551_; uint8_t v_isSharedCheck_1555_; 
lean_dec_ref(v_config_1346_);
v_a_1548_ = lean_ctor_get(v___x_1522_, 0);
v_isSharedCheck_1555_ = !lean_is_exclusive(v___x_1522_);
if (v_isSharedCheck_1555_ == 0)
{
v___x_1550_ = v___x_1522_;
v_isShared_1551_ = v_isSharedCheck_1555_;
goto v_resetjp_1549_;
}
else
{
lean_inc(v_a_1548_);
lean_dec(v___x_1522_);
v___x_1550_ = lean_box(0);
v_isShared_1551_ = v_isSharedCheck_1555_;
goto v_resetjp_1549_;
}
v_resetjp_1549_:
{
lean_object* v___x_1553_; 
if (v_isShared_1551_ == 0)
{
v___x_1553_ = v___x_1550_;
goto v_reusejp_1552_;
}
else
{
lean_object* v_reuseFailAlloc_1554_; 
v_reuseFailAlloc_1554_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1554_, 0, v_a_1548_);
v___x_1553_ = v_reuseFailAlloc_1554_;
goto v_reusejp_1552_;
}
v_reusejp_1552_:
{
return v___x_1553_;
}
}
}
}
}
else
{
lean_object* v_a_1556_; lean_object* v___x_1558_; uint8_t v_isShared_1559_; uint8_t v_isSharedCheck_1563_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1556_ = lean_ctor_get(v___x_1520_, 0);
v_isSharedCheck_1563_ = !lean_is_exclusive(v___x_1520_);
if (v_isSharedCheck_1563_ == 0)
{
v___x_1558_ = v___x_1520_;
v_isShared_1559_ = v_isSharedCheck_1563_;
goto v_resetjp_1557_;
}
else
{
lean_inc(v_a_1556_);
lean_dec(v___x_1520_);
v___x_1558_ = lean_box(0);
v_isShared_1559_ = v_isSharedCheck_1563_;
goto v_resetjp_1557_;
}
v_resetjp_1557_:
{
lean_object* v___x_1561_; 
if (v_isShared_1559_ == 0)
{
v___x_1561_ = v___x_1558_;
goto v_reusejp_1560_;
}
else
{
lean_object* v_reuseFailAlloc_1562_; 
v_reuseFailAlloc_1562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1562_, 0, v_a_1556_);
v___x_1561_ = v_reuseFailAlloc_1562_;
goto v_reusejp_1560_;
}
v_reusejp_1560_:
{
return v___x_1561_;
}
}
}
}
}
else
{
lean_object* v___x_1564_; lean_object* v___x_1565_; 
lean_dec_ref(v___x_1368_);
v___x_1564_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__11));
v___x_1565_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1564_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1565_) == 0)
{
uint8_t v___x_1566_; 
lean_dec_ref_known(v___x_1565_, 1);
v___x_1566_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1566_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1567_; 
lean_dec_ref(v___x_1369_);
v___x_1567_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1567_) == 0)
{
lean_object* v_a_1568_; lean_object* v___x_1570_; uint8_t v_isShared_1571_; uint8_t v_isSharedCheck_1592_; 
v_a_1568_ = lean_ctor_get(v___x_1567_, 0);
v_isSharedCheck_1592_ = !lean_is_exclusive(v___x_1567_);
if (v_isSharedCheck_1592_ == 0)
{
v___x_1570_ = v___x_1567_;
v_isShared_1571_ = v_isSharedCheck_1592_;
goto v_resetjp_1569_;
}
else
{
lean_inc(v_a_1568_);
lean_dec(v___x_1567_);
v___x_1570_ = lean_box(0);
v_isShared_1571_ = v_isSharedCheck_1592_;
goto v_resetjp_1569_;
}
v_resetjp_1569_:
{
lean_object* v_numInst_1572_; lean_object* v_maxSize_1573_; lean_object* v_numRetries_1574_; uint8_t v_traceDiscarded_1575_; uint8_t v_traceSuccesses_1576_; uint8_t v_traceShrink_1577_; uint8_t v_traceShrinkCandidates_1578_; lean_object* v_randomSeed_1579_; uint8_t v_quiet_1580_; lean_object* v___x_1582_; uint8_t v_isShared_1583_; uint8_t v_isSharedCheck_1591_; 
v_numInst_1572_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1573_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1574_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1575_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1576_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1577_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1578_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1579_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1580_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_isSharedCheck_1591_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1591_ == 0)
{
v___x_1582_ = v_config_1346_;
v_isShared_1583_ = v_isSharedCheck_1591_;
goto v_resetjp_1581_;
}
else
{
lean_inc(v_randomSeed_1579_);
lean_inc(v_numRetries_1574_);
lean_inc(v_maxSize_1573_);
lean_inc(v_numInst_1572_);
lean_dec(v_config_1346_);
v___x_1582_ = lean_box(0);
v_isShared_1583_ = v_isSharedCheck_1591_;
goto v_resetjp_1581_;
}
v_resetjp_1581_:
{
lean_object* v___x_1585_; 
if (v_isShared_1583_ == 0)
{
v___x_1585_ = v___x_1582_;
goto v_reusejp_1584_;
}
else
{
lean_object* v_reuseFailAlloc_1590_; 
v_reuseFailAlloc_1590_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1590_, 0, v_numInst_1572_);
lean_ctor_set(v_reuseFailAlloc_1590_, 1, v_maxSize_1573_);
lean_ctor_set(v_reuseFailAlloc_1590_, 2, v_numRetries_1574_);
lean_ctor_set(v_reuseFailAlloc_1590_, 3, v_randomSeed_1579_);
lean_ctor_set_uint8(v_reuseFailAlloc_1590_, sizeof(void*)*4, v_traceDiscarded_1575_);
lean_ctor_set_uint8(v_reuseFailAlloc_1590_, sizeof(void*)*4 + 1, v_traceSuccesses_1576_);
lean_ctor_set_uint8(v_reuseFailAlloc_1590_, sizeof(void*)*4 + 2, v_traceShrink_1577_);
lean_ctor_set_uint8(v_reuseFailAlloc_1590_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1578_);
lean_ctor_set_uint8(v_reuseFailAlloc_1590_, sizeof(void*)*4 + 4, v_quiet_1580_);
v___x_1585_ = v_reuseFailAlloc_1590_;
goto v_reusejp_1584_;
}
v_reusejp_1584_:
{
uint8_t v___x_1586_; lean_object* v___x_1588_; 
v___x_1586_ = lean_unbox(v_a_1568_);
lean_dec(v_a_1568_);
lean_ctor_set_uint8(v___x_1585_, sizeof(void*)*4 + 5, v___x_1586_);
if (v_isShared_1571_ == 0)
{
lean_ctor_set(v___x_1570_, 0, v___x_1585_);
v___x_1588_ = v___x_1570_;
goto v_reusejp_1587_;
}
else
{
lean_object* v_reuseFailAlloc_1589_; 
v_reuseFailAlloc_1589_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1589_, 0, v___x_1585_);
v___x_1588_ = v_reuseFailAlloc_1589_;
goto v_reusejp_1587_;
}
v_reusejp_1587_:
{
return v___x_1588_;
}
}
}
}
}
else
{
lean_object* v_a_1593_; lean_object* v___x_1595_; uint8_t v_isShared_1596_; uint8_t v_isSharedCheck_1600_; 
lean_dec_ref(v_config_1346_);
v_a_1593_ = lean_ctor_get(v___x_1567_, 0);
v_isSharedCheck_1600_ = !lean_is_exclusive(v___x_1567_);
if (v_isSharedCheck_1600_ == 0)
{
v___x_1595_ = v___x_1567_;
v_isShared_1596_ = v_isSharedCheck_1600_;
goto v_resetjp_1594_;
}
else
{
lean_inc(v_a_1593_);
lean_dec(v___x_1567_);
v___x_1595_ = lean_box(0);
v_isShared_1596_ = v_isSharedCheck_1600_;
goto v_resetjp_1594_;
}
v_resetjp_1594_:
{
lean_object* v___x_1598_; 
if (v_isShared_1596_ == 0)
{
v___x_1598_ = v___x_1595_;
goto v_reusejp_1597_;
}
else
{
lean_object* v_reuseFailAlloc_1599_; 
v_reuseFailAlloc_1599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1599_, 0, v_a_1593_);
v___x_1598_ = v_reuseFailAlloc_1599_;
goto v_reusejp_1597_;
}
v_reusejp_1597_:
{
return v___x_1598_;
}
}
}
}
}
else
{
lean_object* v_a_1601_; lean_object* v___x_1603_; uint8_t v_isShared_1604_; uint8_t v_isSharedCheck_1608_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1601_ = lean_ctor_get(v___x_1565_, 0);
v_isSharedCheck_1608_ = !lean_is_exclusive(v___x_1565_);
if (v_isSharedCheck_1608_ == 0)
{
v___x_1603_ = v___x_1565_;
v_isShared_1604_ = v_isSharedCheck_1608_;
goto v_resetjp_1602_;
}
else
{
lean_inc(v_a_1601_);
lean_dec(v___x_1565_);
v___x_1603_ = lean_box(0);
v_isShared_1604_ = v_isSharedCheck_1608_;
goto v_resetjp_1602_;
}
v_resetjp_1602_:
{
lean_object* v___x_1606_; 
if (v_isShared_1604_ == 0)
{
v___x_1606_ = v___x_1603_;
goto v_reusejp_1605_;
}
else
{
lean_object* v_reuseFailAlloc_1607_; 
v_reuseFailAlloc_1607_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1607_, 0, v_a_1601_);
v___x_1606_ = v_reuseFailAlloc_1607_;
goto v_reusejp_1605_;
}
v_reusejp_1605_:
{
return v___x_1606_;
}
}
}
}
}
else
{
lean_object* v___x_1609_; lean_object* v___x_1610_; 
lean_dec_ref(v___x_1368_);
v___x_1609_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__12));
v___x_1610_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1609_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1610_) == 0)
{
uint8_t v___x_1611_; 
lean_dec_ref_known(v___x_1610_, 1);
v___x_1611_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1611_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1612_; 
lean_dec_ref(v___x_1369_);
lean_inc_ref(v_item_1347_);
v___x_1612_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1612_) == 0)
{
lean_object* v_value_1613_; lean_object* v___x_1614_; 
lean_dec_ref_known(v___x_1612_, 1);
v_value_1613_ = lean_ctor_get(v_item_1347_, 2);
lean_inc(v_value_1613_);
lean_dec_ref(v_item_1347_);
v___x_1614_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__0(v_value_1613_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1614_) == 0)
{
lean_object* v_a_1615_; lean_object* v___x_1617_; uint8_t v_isShared_1618_; uint8_t v_isSharedCheck_1639_; 
v_a_1615_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1639_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1639_ == 0)
{
v___x_1617_ = v___x_1614_;
v_isShared_1618_ = v_isSharedCheck_1639_;
goto v_resetjp_1616_;
}
else
{
lean_inc(v_a_1615_);
lean_dec(v___x_1614_);
v___x_1617_ = lean_box(0);
v_isShared_1618_ = v_isSharedCheck_1639_;
goto v_resetjp_1616_;
}
v_resetjp_1616_:
{
lean_object* v_numInst_1619_; lean_object* v_maxSize_1620_; lean_object* v_numRetries_1621_; uint8_t v_traceDiscarded_1622_; uint8_t v_traceSuccesses_1623_; uint8_t v_traceShrink_1624_; uint8_t v_traceShrinkCandidates_1625_; uint8_t v_quiet_1626_; uint8_t v_sorryIfNoTestable_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1637_; 
v_numInst_1619_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1620_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1621_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1622_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1623_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1624_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1625_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_quiet_1626_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1627_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1637_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1637_ == 0)
{
lean_object* v_unused_1638_; 
v_unused_1638_ = lean_ctor_get(v_config_1346_, 3);
lean_dec(v_unused_1638_);
v___x_1629_ = v_config_1346_;
v_isShared_1630_ = v_isSharedCheck_1637_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_numRetries_1621_);
lean_inc(v_maxSize_1620_);
lean_inc(v_numInst_1619_);
lean_dec(v_config_1346_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1637_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___x_1632_; 
if (v_isShared_1630_ == 0)
{
lean_ctor_set(v___x_1629_, 3, v_a_1615_);
v___x_1632_ = v___x_1629_;
goto v_reusejp_1631_;
}
else
{
lean_object* v_reuseFailAlloc_1636_; 
v_reuseFailAlloc_1636_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1636_, 0, v_numInst_1619_);
lean_ctor_set(v_reuseFailAlloc_1636_, 1, v_maxSize_1620_);
lean_ctor_set(v_reuseFailAlloc_1636_, 2, v_numRetries_1621_);
lean_ctor_set(v_reuseFailAlloc_1636_, 3, v_a_1615_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4, v_traceDiscarded_1622_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4 + 1, v_traceSuccesses_1623_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4 + 2, v_traceShrink_1624_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1625_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4 + 4, v_quiet_1626_);
lean_ctor_set_uint8(v_reuseFailAlloc_1636_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1627_);
v___x_1632_ = v_reuseFailAlloc_1636_;
goto v_reusejp_1631_;
}
v_reusejp_1631_:
{
lean_object* v___x_1634_; 
if (v_isShared_1618_ == 0)
{
lean_ctor_set(v___x_1617_, 0, v___x_1632_);
v___x_1634_ = v___x_1617_;
goto v_reusejp_1633_;
}
else
{
lean_object* v_reuseFailAlloc_1635_; 
v_reuseFailAlloc_1635_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1635_, 0, v___x_1632_);
v___x_1634_ = v_reuseFailAlloc_1635_;
goto v_reusejp_1633_;
}
v_reusejp_1633_:
{
return v___x_1634_;
}
}
}
}
}
else
{
lean_object* v_a_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_1647_; 
lean_dec_ref(v_config_1346_);
v_a_1640_ = lean_ctor_get(v___x_1614_, 0);
v_isSharedCheck_1647_ = !lean_is_exclusive(v___x_1614_);
if (v_isSharedCheck_1647_ == 0)
{
v___x_1642_ = v___x_1614_;
v_isShared_1643_ = v_isSharedCheck_1647_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_a_1640_);
lean_dec(v___x_1614_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_1647_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
lean_object* v___x_1645_; 
if (v_isShared_1643_ == 0)
{
v___x_1645_ = v___x_1642_;
goto v_reusejp_1644_;
}
else
{
lean_object* v_reuseFailAlloc_1646_; 
v_reuseFailAlloc_1646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1646_, 0, v_a_1640_);
v___x_1645_ = v_reuseFailAlloc_1646_;
goto v_reusejp_1644_;
}
v_reusejp_1644_:
{
return v___x_1645_;
}
}
}
}
else
{
lean_object* v_a_1648_; lean_object* v___x_1650_; uint8_t v_isShared_1651_; uint8_t v_isSharedCheck_1655_; 
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1648_ = lean_ctor_get(v___x_1612_, 0);
v_isSharedCheck_1655_ = !lean_is_exclusive(v___x_1612_);
if (v_isSharedCheck_1655_ == 0)
{
v___x_1650_ = v___x_1612_;
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
else
{
lean_inc(v_a_1648_);
lean_dec(v___x_1612_);
v___x_1650_ = lean_box(0);
v_isShared_1651_ = v_isSharedCheck_1655_;
goto v_resetjp_1649_;
}
v_resetjp_1649_:
{
lean_object* v___x_1653_; 
if (v_isShared_1651_ == 0)
{
v___x_1653_ = v___x_1650_;
goto v_reusejp_1652_;
}
else
{
lean_object* v_reuseFailAlloc_1654_; 
v_reuseFailAlloc_1654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1654_, 0, v_a_1648_);
v___x_1653_ = v_reuseFailAlloc_1654_;
goto v_reusejp_1652_;
}
v_reusejp_1652_:
{
return v___x_1653_;
}
}
}
}
}
else
{
lean_object* v_a_1656_; lean_object* v___x_1658_; uint8_t v_isShared_1659_; uint8_t v_isSharedCheck_1663_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1656_ = lean_ctor_get(v___x_1610_, 0);
v_isSharedCheck_1663_ = !lean_is_exclusive(v___x_1610_);
if (v_isSharedCheck_1663_ == 0)
{
v___x_1658_ = v___x_1610_;
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
else
{
lean_inc(v_a_1656_);
lean_dec(v___x_1610_);
v___x_1658_ = lean_box(0);
v_isShared_1659_ = v_isSharedCheck_1663_;
goto v_resetjp_1657_;
}
v_resetjp_1657_:
{
lean_object* v___x_1661_; 
if (v_isShared_1659_ == 0)
{
v___x_1661_ = v___x_1658_;
goto v_reusejp_1660_;
}
else
{
lean_object* v_reuseFailAlloc_1662_; 
v_reuseFailAlloc_1662_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1662_, 0, v_a_1656_);
v___x_1661_ = v_reuseFailAlloc_1662_;
goto v_reusejp_1660_;
}
v_reusejp_1660_:
{
return v___x_1661_;
}
}
}
}
}
}
else
{
lean_object* v___x_1664_; uint8_t v___x_1665_; 
v___x_1664_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__13));
v___x_1665_ = lean_string_dec_eq(v___x_1368_, v___x_1664_);
if (v___x_1665_ == 0)
{
lean_object* v___x_1666_; uint8_t v___x_1667_; 
v___x_1666_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__14));
v___x_1667_ = lean_string_dec_eq(v___x_1368_, v___x_1666_);
if (v___x_1667_ == 0)
{
lean_object* v___x_1668_; uint8_t v___x_1669_; 
v___x_1668_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__15));
v___x_1669_ = lean_string_dec_eq(v___x_1368_, v___x_1668_);
if (v___x_1669_ == 0)
{
lean_object* v___x_1670_; uint8_t v___x_1671_; 
v___x_1670_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__16));
v___x_1671_ = lean_string_dec_eq(v___x_1368_, v___x_1670_);
if (v___x_1671_ == 0)
{
lean_object* v___x_1672_; uint8_t v___x_1673_; 
v___x_1672_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__17));
v___x_1673_ = lean_string_dec_eq(v___x_1368_, v___x_1672_);
lean_dec_ref(v___x_1368_);
if (v___x_1673_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1674_; lean_object* v___x_1675_; 
v___x_1674_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__18));
v___x_1675_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1674_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1675_) == 0)
{
uint8_t v___x_1676_; 
lean_dec_ref_known(v___x_1675_, 1);
v___x_1676_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1676_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1677_; 
lean_dec_ref(v___x_1369_);
v___x_1677_ = l_Lean_Elab_ConfigEval_evalBoolItem(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1677_) == 0)
{
lean_object* v_a_1678_; lean_object* v___x_1680_; uint8_t v_isShared_1681_; uint8_t v_isSharedCheck_1702_; 
v_a_1678_ = lean_ctor_get(v___x_1677_, 0);
v_isSharedCheck_1702_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1702_ == 0)
{
v___x_1680_ = v___x_1677_;
v_isShared_1681_ = v_isSharedCheck_1702_;
goto v_resetjp_1679_;
}
else
{
lean_inc(v_a_1678_);
lean_dec(v___x_1677_);
v___x_1680_ = lean_box(0);
v_isShared_1681_ = v_isSharedCheck_1702_;
goto v_resetjp_1679_;
}
v_resetjp_1679_:
{
lean_object* v_numInst_1682_; lean_object* v_maxSize_1683_; lean_object* v_numRetries_1684_; uint8_t v_traceDiscarded_1685_; uint8_t v_traceSuccesses_1686_; uint8_t v_traceShrink_1687_; uint8_t v_traceShrinkCandidates_1688_; lean_object* v_randomSeed_1689_; uint8_t v_sorryIfNoTestable_1690_; lean_object* v___x_1692_; uint8_t v_isShared_1693_; uint8_t v_isSharedCheck_1701_; 
v_numInst_1682_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1683_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1684_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1685_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1686_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1687_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1688_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1689_ = lean_ctor_get(v_config_1346_, 3);
v_sorryIfNoTestable_1690_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1701_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1701_ == 0)
{
v___x_1692_ = v_config_1346_;
v_isShared_1693_ = v_isSharedCheck_1701_;
goto v_resetjp_1691_;
}
else
{
lean_inc(v_randomSeed_1689_);
lean_inc(v_numRetries_1684_);
lean_inc(v_maxSize_1683_);
lean_inc(v_numInst_1682_);
lean_dec(v_config_1346_);
v___x_1692_ = lean_box(0);
v_isShared_1693_ = v_isSharedCheck_1701_;
goto v_resetjp_1691_;
}
v_resetjp_1691_:
{
lean_object* v___x_1695_; 
if (v_isShared_1693_ == 0)
{
v___x_1695_ = v___x_1692_;
goto v_reusejp_1694_;
}
else
{
lean_object* v_reuseFailAlloc_1700_; 
v_reuseFailAlloc_1700_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1700_, 0, v_numInst_1682_);
lean_ctor_set(v_reuseFailAlloc_1700_, 1, v_maxSize_1683_);
lean_ctor_set(v_reuseFailAlloc_1700_, 2, v_numRetries_1684_);
lean_ctor_set(v_reuseFailAlloc_1700_, 3, v_randomSeed_1689_);
lean_ctor_set_uint8(v_reuseFailAlloc_1700_, sizeof(void*)*4, v_traceDiscarded_1685_);
lean_ctor_set_uint8(v_reuseFailAlloc_1700_, sizeof(void*)*4 + 1, v_traceSuccesses_1686_);
lean_ctor_set_uint8(v_reuseFailAlloc_1700_, sizeof(void*)*4 + 2, v_traceShrink_1687_);
lean_ctor_set_uint8(v_reuseFailAlloc_1700_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1688_);
v___x_1695_ = v_reuseFailAlloc_1700_;
goto v_reusejp_1694_;
}
v_reusejp_1694_:
{
uint8_t v___x_1696_; lean_object* v___x_1698_; 
v___x_1696_ = lean_unbox(v_a_1678_);
lean_dec(v_a_1678_);
lean_ctor_set_uint8(v___x_1695_, sizeof(void*)*4 + 4, v___x_1696_);
lean_ctor_set_uint8(v___x_1695_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1690_);
if (v_isShared_1681_ == 0)
{
lean_ctor_set(v___x_1680_, 0, v___x_1695_);
v___x_1698_ = v___x_1680_;
goto v_reusejp_1697_;
}
else
{
lean_object* v_reuseFailAlloc_1699_; 
v_reuseFailAlloc_1699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1699_, 0, v___x_1695_);
v___x_1698_ = v_reuseFailAlloc_1699_;
goto v_reusejp_1697_;
}
v_reusejp_1697_:
{
return v___x_1698_;
}
}
}
}
}
else
{
lean_object* v_a_1703_; lean_object* v___x_1705_; uint8_t v_isShared_1706_; uint8_t v_isSharedCheck_1710_; 
lean_dec_ref(v_config_1346_);
v_a_1703_ = lean_ctor_get(v___x_1677_, 0);
v_isSharedCheck_1710_ = !lean_is_exclusive(v___x_1677_);
if (v_isSharedCheck_1710_ == 0)
{
v___x_1705_ = v___x_1677_;
v_isShared_1706_ = v_isSharedCheck_1710_;
goto v_resetjp_1704_;
}
else
{
lean_inc(v_a_1703_);
lean_dec(v___x_1677_);
v___x_1705_ = lean_box(0);
v_isShared_1706_ = v_isSharedCheck_1710_;
goto v_resetjp_1704_;
}
v_resetjp_1704_:
{
lean_object* v___x_1708_; 
if (v_isShared_1706_ == 0)
{
v___x_1708_ = v___x_1705_;
goto v_reusejp_1707_;
}
else
{
lean_object* v_reuseFailAlloc_1709_; 
v_reuseFailAlloc_1709_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1709_, 0, v_a_1703_);
v___x_1708_ = v_reuseFailAlloc_1709_;
goto v_reusejp_1707_;
}
v_reusejp_1707_:
{
return v___x_1708_;
}
}
}
}
}
else
{
lean_object* v_a_1711_; lean_object* v___x_1713_; uint8_t v_isShared_1714_; uint8_t v_isSharedCheck_1718_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1711_ = lean_ctor_get(v___x_1675_, 0);
v_isSharedCheck_1718_ = !lean_is_exclusive(v___x_1675_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1713_ = v___x_1675_;
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
else
{
lean_inc(v_a_1711_);
lean_dec(v___x_1675_);
v___x_1713_ = lean_box(0);
v_isShared_1714_ = v_isSharedCheck_1718_;
goto v_resetjp_1712_;
}
v_resetjp_1712_:
{
lean_object* v___x_1716_; 
if (v_isShared_1714_ == 0)
{
v___x_1716_ = v___x_1713_;
goto v_reusejp_1715_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_a_1711_);
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
}
else
{
lean_object* v___x_1719_; lean_object* v___x_1720_; 
lean_dec_ref(v___x_1368_);
v___x_1719_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__19));
v___x_1720_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1719_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1720_) == 0)
{
uint8_t v___x_1721_; 
lean_dec_ref_known(v___x_1720_, 1);
v___x_1721_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1721_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1722_; 
lean_dec_ref(v___x_1369_);
lean_inc_ref(v_item_1347_);
v___x_1722_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1722_) == 0)
{
lean_object* v_value_1723_; lean_object* v___x_1724_; 
lean_dec_ref_known(v___x_1722_, 1);
v_value_1723_ = lean_ctor_get(v_item_1347_, 2);
lean_inc(v_value_1723_);
lean_dec_ref(v_item_1347_);
v___x_1724_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(v_value_1723_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1724_) == 0)
{
lean_object* v_a_1725_; lean_object* v___x_1727_; uint8_t v_isShared_1728_; uint8_t v_isSharedCheck_1749_; 
v_a_1725_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1749_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1749_ == 0)
{
v___x_1727_ = v___x_1724_;
v_isShared_1728_ = v_isSharedCheck_1749_;
goto v_resetjp_1726_;
}
else
{
lean_inc(v_a_1725_);
lean_dec(v___x_1724_);
v___x_1727_ = lean_box(0);
v_isShared_1728_ = v_isSharedCheck_1749_;
goto v_resetjp_1726_;
}
v_resetjp_1726_:
{
lean_object* v_numInst_1729_; lean_object* v_maxSize_1730_; uint8_t v_traceDiscarded_1731_; uint8_t v_traceSuccesses_1732_; uint8_t v_traceShrink_1733_; uint8_t v_traceShrinkCandidates_1734_; lean_object* v_randomSeed_1735_; uint8_t v_quiet_1736_; uint8_t v_sorryIfNoTestable_1737_; lean_object* v___x_1739_; uint8_t v_isShared_1740_; uint8_t v_isSharedCheck_1747_; 
v_numInst_1729_ = lean_ctor_get(v_config_1346_, 0);
v_maxSize_1730_ = lean_ctor_get(v_config_1346_, 1);
v_traceDiscarded_1731_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1732_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1733_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1734_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1735_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1736_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1737_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1747_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1747_ == 0)
{
lean_object* v_unused_1748_; 
v_unused_1748_ = lean_ctor_get(v_config_1346_, 2);
lean_dec(v_unused_1748_);
v___x_1739_ = v_config_1346_;
v_isShared_1740_ = v_isSharedCheck_1747_;
goto v_resetjp_1738_;
}
else
{
lean_inc(v_randomSeed_1735_);
lean_inc(v_maxSize_1730_);
lean_inc(v_numInst_1729_);
lean_dec(v_config_1346_);
v___x_1739_ = lean_box(0);
v_isShared_1740_ = v_isSharedCheck_1747_;
goto v_resetjp_1738_;
}
v_resetjp_1738_:
{
lean_object* v___x_1742_; 
if (v_isShared_1740_ == 0)
{
lean_ctor_set(v___x_1739_, 2, v_a_1725_);
v___x_1742_ = v___x_1739_;
goto v_reusejp_1741_;
}
else
{
lean_object* v_reuseFailAlloc_1746_; 
v_reuseFailAlloc_1746_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1746_, 0, v_numInst_1729_);
lean_ctor_set(v_reuseFailAlloc_1746_, 1, v_maxSize_1730_);
lean_ctor_set(v_reuseFailAlloc_1746_, 2, v_a_1725_);
lean_ctor_set(v_reuseFailAlloc_1746_, 3, v_randomSeed_1735_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4, v_traceDiscarded_1731_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4 + 1, v_traceSuccesses_1732_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4 + 2, v_traceShrink_1733_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1734_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4 + 4, v_quiet_1736_);
lean_ctor_set_uint8(v_reuseFailAlloc_1746_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1737_);
v___x_1742_ = v_reuseFailAlloc_1746_;
goto v_reusejp_1741_;
}
v_reusejp_1741_:
{
lean_object* v___x_1744_; 
if (v_isShared_1728_ == 0)
{
lean_ctor_set(v___x_1727_, 0, v___x_1742_);
v___x_1744_ = v___x_1727_;
goto v_reusejp_1743_;
}
else
{
lean_object* v_reuseFailAlloc_1745_; 
v_reuseFailAlloc_1745_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1745_, 0, v___x_1742_);
v___x_1744_ = v_reuseFailAlloc_1745_;
goto v_reusejp_1743_;
}
v_reusejp_1743_:
{
return v___x_1744_;
}
}
}
}
}
else
{
lean_object* v_a_1750_; lean_object* v___x_1752_; uint8_t v_isShared_1753_; uint8_t v_isSharedCheck_1757_; 
lean_dec_ref(v_config_1346_);
v_a_1750_ = lean_ctor_get(v___x_1724_, 0);
v_isSharedCheck_1757_ = !lean_is_exclusive(v___x_1724_);
if (v_isSharedCheck_1757_ == 0)
{
v___x_1752_ = v___x_1724_;
v_isShared_1753_ = v_isSharedCheck_1757_;
goto v_resetjp_1751_;
}
else
{
lean_inc(v_a_1750_);
lean_dec(v___x_1724_);
v___x_1752_ = lean_box(0);
v_isShared_1753_ = v_isSharedCheck_1757_;
goto v_resetjp_1751_;
}
v_resetjp_1751_:
{
lean_object* v___x_1755_; 
if (v_isShared_1753_ == 0)
{
v___x_1755_ = v___x_1752_;
goto v_reusejp_1754_;
}
else
{
lean_object* v_reuseFailAlloc_1756_; 
v_reuseFailAlloc_1756_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1756_, 0, v_a_1750_);
v___x_1755_ = v_reuseFailAlloc_1756_;
goto v_reusejp_1754_;
}
v_reusejp_1754_:
{
return v___x_1755_;
}
}
}
}
else
{
lean_object* v_a_1758_; lean_object* v___x_1760_; uint8_t v_isShared_1761_; uint8_t v_isSharedCheck_1765_; 
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1758_ = lean_ctor_get(v___x_1722_, 0);
v_isSharedCheck_1765_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1765_ == 0)
{
v___x_1760_ = v___x_1722_;
v_isShared_1761_ = v_isSharedCheck_1765_;
goto v_resetjp_1759_;
}
else
{
lean_inc(v_a_1758_);
lean_dec(v___x_1722_);
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
else
{
lean_object* v_a_1766_; lean_object* v___x_1768_; uint8_t v_isShared_1769_; uint8_t v_isSharedCheck_1773_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1766_ = lean_ctor_get(v___x_1720_, 0);
v_isSharedCheck_1773_ = !lean_is_exclusive(v___x_1720_);
if (v_isSharedCheck_1773_ == 0)
{
v___x_1768_ = v___x_1720_;
v_isShared_1769_ = v_isSharedCheck_1773_;
goto v_resetjp_1767_;
}
else
{
lean_inc(v_a_1766_);
lean_dec(v___x_1720_);
v___x_1768_ = lean_box(0);
v_isShared_1769_ = v_isSharedCheck_1773_;
goto v_resetjp_1767_;
}
v_resetjp_1767_:
{
lean_object* v___x_1771_; 
if (v_isShared_1769_ == 0)
{
v___x_1771_ = v___x_1768_;
goto v_reusejp_1770_;
}
else
{
lean_object* v_reuseFailAlloc_1772_; 
v_reuseFailAlloc_1772_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1772_, 0, v_a_1766_);
v___x_1771_ = v_reuseFailAlloc_1772_;
goto v_reusejp_1770_;
}
v_reusejp_1770_:
{
return v___x_1771_;
}
}
}
}
}
else
{
lean_object* v___x_1774_; lean_object* v___x_1775_; 
lean_dec_ref(v___x_1368_);
v___x_1774_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__20));
v___x_1775_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1774_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1775_) == 0)
{
uint8_t v___x_1776_; 
lean_dec_ref_known(v___x_1775_, 1);
v___x_1776_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1776_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1777_; 
lean_dec_ref(v___x_1369_);
lean_inc_ref(v_item_1347_);
v___x_1777_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1777_) == 0)
{
lean_object* v_value_1778_; lean_object* v___x_1779_; 
lean_dec_ref_known(v___x_1777_, 1);
v_value_1778_ = lean_ctor_get(v_item_1347_, 2);
lean_inc(v_value_1778_);
lean_dec_ref(v_item_1347_);
v___x_1779_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(v_value_1778_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1779_) == 0)
{
lean_object* v_a_1780_; lean_object* v___x_1782_; uint8_t v_isShared_1783_; uint8_t v_isSharedCheck_1804_; 
v_a_1780_ = lean_ctor_get(v___x_1779_, 0);
v_isSharedCheck_1804_ = !lean_is_exclusive(v___x_1779_);
if (v_isSharedCheck_1804_ == 0)
{
v___x_1782_ = v___x_1779_;
v_isShared_1783_ = v_isSharedCheck_1804_;
goto v_resetjp_1781_;
}
else
{
lean_inc(v_a_1780_);
lean_dec(v___x_1779_);
v___x_1782_ = lean_box(0);
v_isShared_1783_ = v_isSharedCheck_1804_;
goto v_resetjp_1781_;
}
v_resetjp_1781_:
{
lean_object* v_maxSize_1784_; lean_object* v_numRetries_1785_; uint8_t v_traceDiscarded_1786_; uint8_t v_traceSuccesses_1787_; uint8_t v_traceShrink_1788_; uint8_t v_traceShrinkCandidates_1789_; lean_object* v_randomSeed_1790_; uint8_t v_quiet_1791_; uint8_t v_sorryIfNoTestable_1792_; lean_object* v___x_1794_; uint8_t v_isShared_1795_; uint8_t v_isSharedCheck_1802_; 
v_maxSize_1784_ = lean_ctor_get(v_config_1346_, 1);
v_numRetries_1785_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1786_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1787_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1788_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1789_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1790_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1791_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1792_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1802_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1802_ == 0)
{
lean_object* v_unused_1803_; 
v_unused_1803_ = lean_ctor_get(v_config_1346_, 0);
lean_dec(v_unused_1803_);
v___x_1794_ = v_config_1346_;
v_isShared_1795_ = v_isSharedCheck_1802_;
goto v_resetjp_1793_;
}
else
{
lean_inc(v_randomSeed_1790_);
lean_inc(v_numRetries_1785_);
lean_inc(v_maxSize_1784_);
lean_dec(v_config_1346_);
v___x_1794_ = lean_box(0);
v_isShared_1795_ = v_isSharedCheck_1802_;
goto v_resetjp_1793_;
}
v_resetjp_1793_:
{
lean_object* v___x_1797_; 
if (v_isShared_1795_ == 0)
{
lean_ctor_set(v___x_1794_, 0, v_a_1780_);
v___x_1797_ = v___x_1794_;
goto v_reusejp_1796_;
}
else
{
lean_object* v_reuseFailAlloc_1801_; 
v_reuseFailAlloc_1801_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1801_, 0, v_a_1780_);
lean_ctor_set(v_reuseFailAlloc_1801_, 1, v_maxSize_1784_);
lean_ctor_set(v_reuseFailAlloc_1801_, 2, v_numRetries_1785_);
lean_ctor_set(v_reuseFailAlloc_1801_, 3, v_randomSeed_1790_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4, v_traceDiscarded_1786_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4 + 1, v_traceSuccesses_1787_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4 + 2, v_traceShrink_1788_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1789_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4 + 4, v_quiet_1791_);
lean_ctor_set_uint8(v_reuseFailAlloc_1801_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1792_);
v___x_1797_ = v_reuseFailAlloc_1801_;
goto v_reusejp_1796_;
}
v_reusejp_1796_:
{
lean_object* v___x_1799_; 
if (v_isShared_1783_ == 0)
{
lean_ctor_set(v___x_1782_, 0, v___x_1797_);
v___x_1799_ = v___x_1782_;
goto v_reusejp_1798_;
}
else
{
lean_object* v_reuseFailAlloc_1800_; 
v_reuseFailAlloc_1800_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1800_, 0, v___x_1797_);
v___x_1799_ = v_reuseFailAlloc_1800_;
goto v_reusejp_1798_;
}
v_reusejp_1798_:
{
return v___x_1799_;
}
}
}
}
}
else
{
lean_object* v_a_1805_; lean_object* v___x_1807_; uint8_t v_isShared_1808_; uint8_t v_isSharedCheck_1812_; 
lean_dec_ref(v_config_1346_);
v_a_1805_ = lean_ctor_get(v___x_1779_, 0);
v_isSharedCheck_1812_ = !lean_is_exclusive(v___x_1779_);
if (v_isSharedCheck_1812_ == 0)
{
v___x_1807_ = v___x_1779_;
v_isShared_1808_ = v_isSharedCheck_1812_;
goto v_resetjp_1806_;
}
else
{
lean_inc(v_a_1805_);
lean_dec(v___x_1779_);
v___x_1807_ = lean_box(0);
v_isShared_1808_ = v_isSharedCheck_1812_;
goto v_resetjp_1806_;
}
v_resetjp_1806_:
{
lean_object* v___x_1810_; 
if (v_isShared_1808_ == 0)
{
v___x_1810_ = v___x_1807_;
goto v_reusejp_1809_;
}
else
{
lean_object* v_reuseFailAlloc_1811_; 
v_reuseFailAlloc_1811_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1811_, 0, v_a_1805_);
v___x_1810_ = v_reuseFailAlloc_1811_;
goto v_reusejp_1809_;
}
v_reusejp_1809_:
{
return v___x_1810_;
}
}
}
}
else
{
lean_object* v_a_1813_; lean_object* v___x_1815_; uint8_t v_isShared_1816_; uint8_t v_isSharedCheck_1820_; 
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1813_ = lean_ctor_get(v___x_1777_, 0);
v_isSharedCheck_1820_ = !lean_is_exclusive(v___x_1777_);
if (v_isSharedCheck_1820_ == 0)
{
v___x_1815_ = v___x_1777_;
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
else
{
lean_inc(v_a_1813_);
lean_dec(v___x_1777_);
v___x_1815_ = lean_box(0);
v_isShared_1816_ = v_isSharedCheck_1820_;
goto v_resetjp_1814_;
}
v_resetjp_1814_:
{
lean_object* v___x_1818_; 
if (v_isShared_1816_ == 0)
{
v___x_1818_ = v___x_1815_;
goto v_reusejp_1817_;
}
else
{
lean_object* v_reuseFailAlloc_1819_; 
v_reuseFailAlloc_1819_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1819_, 0, v_a_1813_);
v___x_1818_ = v_reuseFailAlloc_1819_;
goto v_reusejp_1817_;
}
v_reusejp_1817_:
{
return v___x_1818_;
}
}
}
}
}
else
{
lean_object* v_a_1821_; lean_object* v___x_1823_; uint8_t v_isShared_1824_; uint8_t v_isSharedCheck_1828_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1821_ = lean_ctor_get(v___x_1775_, 0);
v_isSharedCheck_1828_ = !lean_is_exclusive(v___x_1775_);
if (v_isSharedCheck_1828_ == 0)
{
v___x_1823_ = v___x_1775_;
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
else
{
lean_inc(v_a_1821_);
lean_dec(v___x_1775_);
v___x_1823_ = lean_box(0);
v_isShared_1824_ = v_isSharedCheck_1828_;
goto v_resetjp_1822_;
}
v_resetjp_1822_:
{
lean_object* v___x_1826_; 
if (v_isShared_1824_ == 0)
{
v___x_1826_ = v___x_1823_;
goto v_reusejp_1825_;
}
else
{
lean_object* v_reuseFailAlloc_1827_; 
v_reuseFailAlloc_1827_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1827_, 0, v_a_1821_);
v___x_1826_ = v_reuseFailAlloc_1827_;
goto v_reusejp_1825_;
}
v_reusejp_1825_:
{
return v___x_1826_;
}
}
}
}
}
else
{
lean_object* v___x_1829_; lean_object* v___x_1830_; 
lean_dec_ref(v___x_1368_);
v___x_1829_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__21));
v___x_1830_ = l_Lean_Elab_ConfigEval_ConfigItem_addConstInfo(v_item_1347_, v___x_1829_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1830_) == 0)
{
uint8_t v___x_1831_; 
lean_dec_ref_known(v___x_1830_, 1);
v___x_1831_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1831_ == 0)
{
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v___x_1832_; 
lean_dec_ref(v___x_1369_);
lean_inc_ref(v_item_1347_);
v___x_1832_ = l_Lean_Elab_ConfigEval_ConfigItem_checkNotBool(v_item_1347_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1832_) == 0)
{
lean_object* v_value_1833_; lean_object* v___x_1834_; 
lean_dec_ref_known(v___x_1832_, 1);
v_value_1833_ = lean_ctor_get(v_item_1347_, 2);
lean_inc(v_value_1833_);
lean_dec_ref(v_item_1347_);
v___x_1834_ = lp_plausible_Lean_Elab_ConfigEval_evalTermOrExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__1(v_value_1833_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
if (lean_obj_tag(v___x_1834_) == 0)
{
lean_object* v_a_1835_; lean_object* v___x_1837_; uint8_t v_isShared_1838_; uint8_t v_isSharedCheck_1859_; 
v_a_1835_ = lean_ctor_get(v___x_1834_, 0);
v_isSharedCheck_1859_ = !lean_is_exclusive(v___x_1834_);
if (v_isSharedCheck_1859_ == 0)
{
v___x_1837_ = v___x_1834_;
v_isShared_1838_ = v_isSharedCheck_1859_;
goto v_resetjp_1836_;
}
else
{
lean_inc(v_a_1835_);
lean_dec(v___x_1834_);
v___x_1837_ = lean_box(0);
v_isShared_1838_ = v_isSharedCheck_1859_;
goto v_resetjp_1836_;
}
v_resetjp_1836_:
{
lean_object* v_numInst_1839_; lean_object* v_numRetries_1840_; uint8_t v_traceDiscarded_1841_; uint8_t v_traceSuccesses_1842_; uint8_t v_traceShrink_1843_; uint8_t v_traceShrinkCandidates_1844_; lean_object* v_randomSeed_1845_; uint8_t v_quiet_1846_; uint8_t v_sorryIfNoTestable_1847_; lean_object* v___x_1849_; uint8_t v_isShared_1850_; uint8_t v_isSharedCheck_1857_; 
v_numInst_1839_ = lean_ctor_get(v_config_1346_, 0);
v_numRetries_1840_ = lean_ctor_get(v_config_1346_, 2);
v_traceDiscarded_1841_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4);
v_traceSuccesses_1842_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 1);
v_traceShrink_1843_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_1844_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 3);
v_randomSeed_1845_ = lean_ctor_get(v_config_1346_, 3);
v_quiet_1846_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 4);
v_sorryIfNoTestable_1847_ = lean_ctor_get_uint8(v_config_1346_, sizeof(void*)*4 + 5);
v_isSharedCheck_1857_ = !lean_is_exclusive(v_config_1346_);
if (v_isSharedCheck_1857_ == 0)
{
lean_object* v_unused_1858_; 
v_unused_1858_ = lean_ctor_get(v_config_1346_, 1);
lean_dec(v_unused_1858_);
v___x_1849_ = v_config_1346_;
v_isShared_1850_ = v_isSharedCheck_1857_;
goto v_resetjp_1848_;
}
else
{
lean_inc(v_randomSeed_1845_);
lean_inc(v_numRetries_1840_);
lean_inc(v_numInst_1839_);
lean_dec(v_config_1346_);
v___x_1849_ = lean_box(0);
v_isShared_1850_ = v_isSharedCheck_1857_;
goto v_resetjp_1848_;
}
v_resetjp_1848_:
{
lean_object* v___x_1852_; 
if (v_isShared_1850_ == 0)
{
lean_ctor_set(v___x_1849_, 1, v_a_1835_);
v___x_1852_ = v___x_1849_;
goto v_reusejp_1851_;
}
else
{
lean_object* v_reuseFailAlloc_1856_; 
v_reuseFailAlloc_1856_ = lean_alloc_ctor(0, 4, 6);
lean_ctor_set(v_reuseFailAlloc_1856_, 0, v_numInst_1839_);
lean_ctor_set(v_reuseFailAlloc_1856_, 1, v_a_1835_);
lean_ctor_set(v_reuseFailAlloc_1856_, 2, v_numRetries_1840_);
lean_ctor_set(v_reuseFailAlloc_1856_, 3, v_randomSeed_1845_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4, v_traceDiscarded_1841_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4 + 1, v_traceSuccesses_1842_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4 + 2, v_traceShrink_1843_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4 + 3, v_traceShrinkCandidates_1844_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4 + 4, v_quiet_1846_);
lean_ctor_set_uint8(v_reuseFailAlloc_1856_, sizeof(void*)*4 + 5, v_sorryIfNoTestable_1847_);
v___x_1852_ = v_reuseFailAlloc_1856_;
goto v_reusejp_1851_;
}
v_reusejp_1851_:
{
lean_object* v___x_1854_; 
if (v_isShared_1838_ == 0)
{
lean_ctor_set(v___x_1837_, 0, v___x_1852_);
v___x_1854_ = v___x_1837_;
goto v_reusejp_1853_;
}
else
{
lean_object* v_reuseFailAlloc_1855_; 
v_reuseFailAlloc_1855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1855_, 0, v___x_1852_);
v___x_1854_ = v_reuseFailAlloc_1855_;
goto v_reusejp_1853_;
}
v_reusejp_1853_:
{
return v___x_1854_;
}
}
}
}
}
else
{
lean_object* v_a_1860_; lean_object* v___x_1862_; uint8_t v_isShared_1863_; uint8_t v_isSharedCheck_1867_; 
lean_dec_ref(v_config_1346_);
v_a_1860_ = lean_ctor_get(v___x_1834_, 0);
v_isSharedCheck_1867_ = !lean_is_exclusive(v___x_1834_);
if (v_isSharedCheck_1867_ == 0)
{
v___x_1862_ = v___x_1834_;
v_isShared_1863_ = v_isSharedCheck_1867_;
goto v_resetjp_1861_;
}
else
{
lean_inc(v_a_1860_);
lean_dec(v___x_1834_);
v___x_1862_ = lean_box(0);
v_isShared_1863_ = v_isSharedCheck_1867_;
goto v_resetjp_1861_;
}
v_resetjp_1861_:
{
lean_object* v___x_1865_; 
if (v_isShared_1863_ == 0)
{
v___x_1865_ = v___x_1862_;
goto v_reusejp_1864_;
}
else
{
lean_object* v_reuseFailAlloc_1866_; 
v_reuseFailAlloc_1866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1866_, 0, v_a_1860_);
v___x_1865_ = v_reuseFailAlloc_1866_;
goto v_reusejp_1864_;
}
v_reusejp_1864_:
{
return v___x_1865_;
}
}
}
}
else
{
lean_object* v_a_1868_; lean_object* v___x_1870_; uint8_t v_isShared_1871_; uint8_t v_isSharedCheck_1875_; 
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1868_ = lean_ctor_get(v___x_1832_, 0);
v_isSharedCheck_1875_ = !lean_is_exclusive(v___x_1832_);
if (v_isSharedCheck_1875_ == 0)
{
v___x_1870_ = v___x_1832_;
v_isShared_1871_ = v_isSharedCheck_1875_;
goto v_resetjp_1869_;
}
else
{
lean_inc(v_a_1868_);
lean_dec(v___x_1832_);
v___x_1870_ = lean_box(0);
v_isShared_1871_ = v_isSharedCheck_1875_;
goto v_resetjp_1869_;
}
v_resetjp_1869_:
{
lean_object* v___x_1873_; 
if (v_isShared_1871_ == 0)
{
v___x_1873_ = v___x_1870_;
goto v_reusejp_1872_;
}
else
{
lean_object* v_reuseFailAlloc_1874_; 
v_reuseFailAlloc_1874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1874_, 0, v_a_1868_);
v___x_1873_ = v_reuseFailAlloc_1874_;
goto v_reusejp_1872_;
}
v_reusejp_1872_:
{
return v___x_1873_;
}
}
}
}
}
else
{
lean_object* v_a_1876_; lean_object* v___x_1878_; uint8_t v_isShared_1879_; uint8_t v_isSharedCheck_1883_; 
lean_dec_ref(v___x_1369_);
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1876_ = lean_ctor_get(v___x_1830_, 0);
v_isSharedCheck_1883_ = !lean_is_exclusive(v___x_1830_);
if (v_isSharedCheck_1883_ == 0)
{
v___x_1878_ = v___x_1830_;
v_isShared_1879_ = v_isSharedCheck_1883_;
goto v_resetjp_1877_;
}
else
{
lean_inc(v_a_1876_);
lean_dec(v___x_1830_);
v___x_1878_ = lean_box(0);
v_isShared_1879_ = v_isSharedCheck_1883_;
goto v_resetjp_1877_;
}
v_resetjp_1877_:
{
lean_object* v___x_1881_; 
if (v_isShared_1879_ == 0)
{
v___x_1881_ = v___x_1878_;
goto v_reusejp_1880_;
}
else
{
lean_object* v_reuseFailAlloc_1882_; 
v_reuseFailAlloc_1882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1882_, 0, v_a_1876_);
v___x_1881_ = v_reuseFailAlloc_1882_;
goto v_reusejp_1880_;
}
v_reusejp_1880_:
{
return v___x_1881_;
}
}
}
}
}
else
{
uint8_t v___x_1884_; 
lean_dec_ref(v___x_1368_);
lean_dec_ref(v_config_1346_);
v___x_1884_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_1369_);
if (v___x_1884_ == 0)
{
lean_dec_ref(v_item_1347_);
v_item_1356_ = v___x_1369_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
else
{
lean_object* v_value_1885_; lean_object* v___x_1886_; 
lean_dec_ref(v___x_1369_);
v_value_1885_ = lean_ctor_get(v_item_1347_, 2);
lean_inc(v_value_1885_);
lean_dec_ref(v_item_1347_);
v___x_1886_ = lp_plausible_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2(v_value_1885_, v___y_1348_, v___y_1349_, v___y_1350_, v___y_1351_, v___y_1352_, v___y_1353_);
return v___x_1886_;
}
}
}
}
else
{
lean_dec_ref(v_config_1346_);
v_item_1356_ = v_item_1347_;
v___y_1357_ = v___y_1348_;
v___y_1358_ = v___y_1349_;
v___y_1359_ = v___y_1350_;
v___y_1360_ = v___y_1351_;
v___y_1361_ = v___y_1352_;
v___y_1362_ = v___y_1353_;
goto v___jp_1355_;
}
}
else
{
lean_object* v_a_1887_; lean_object* v___x_1889_; uint8_t v_isShared_1890_; uint8_t v_isSharedCheck_1894_; 
lean_dec_ref(v_item_1347_);
lean_dec_ref(v_config_1346_);
v_a_1887_ = lean_ctor_get(v___x_1366_, 0);
v_isSharedCheck_1894_ = !lean_is_exclusive(v___x_1366_);
if (v_isSharedCheck_1894_ == 0)
{
v___x_1889_ = v___x_1366_;
v_isShared_1890_ = v_isSharedCheck_1894_;
goto v_resetjp_1888_;
}
else
{
lean_inc(v_a_1887_);
lean_dec(v___x_1366_);
v___x_1889_ = lean_box(0);
v_isShared_1890_ = v_isSharedCheck_1894_;
goto v_resetjp_1888_;
}
v_resetjp_1888_:
{
lean_object* v___x_1892_; 
if (v_isShared_1890_ == 0)
{
v___x_1892_ = v___x_1889_;
goto v_reusejp_1891_;
}
else
{
lean_object* v_reuseFailAlloc_1893_; 
v_reuseFailAlloc_1893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1893_, 0, v_a_1887_);
v___x_1892_ = v_reuseFailAlloc_1893_;
goto v_reusejp_1891_;
}
v_reusejp_1891_:
{
return v___x_1892_;
}
}
}
v___jp_1355_:
{
lean_object* v___x_1363_; lean_object* v___x_1364_; 
v___x_1363_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___closed__0));
v___x_1364_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_1356_, v___x_1363_, v___y_1357_, v___y_1358_, v___y_1359_, v___y_1360_, v___y_1361_, v___y_1362_);
return v___x_1364_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_1895_, lean_object* v_item_1896_, lean_object* v___y_1897_, lean_object* v___y_1898_, lean_object* v___y_1899_, lean_object* v___y_1900_, lean_object* v___y_1901_, lean_object* v___y_1902_, lean_object* v___y_1903_){
_start:
{
lean_object* v_res_1904_; 
v_res_1904_ = lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___lam__0(v_config_1895_, v_item_1896_, v___y_1897_, v___y_1898_, v___y_1899_, v___y_1900_, v___y_1901_, v___y_1902_);
lean_dec(v___y_1902_);
lean_dec_ref(v___y_1901_);
lean_dec(v___y_1900_);
lean_dec_ref(v___y_1899_);
lean_dec(v___y_1898_);
lean_dec_ref(v___y_1897_);
return v_res_1904_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4(lean_object* v_e_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_, lean_object* v___y_1913_){
_start:
{
lean_object* v___x_1915_; 
v___x_1915_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___redArg(v_e_1907_, v___y_1911_);
return v___x_1915_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4___boxed(lean_object* v_e_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_){
_start:
{
lean_object* v_res_1924_; 
v_res_1924_ = lp_plausible_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__4(v_e_1916_, v___y_1917_, v___y_1918_, v___y_1919_, v___y_1920_, v___y_1921_, v___y_1922_);
lean_dec(v___y_1922_);
lean_dec_ref(v___y_1921_);
lean_dec(v___y_1920_);
lean_dec_ref(v___y_1919_);
lean_dec(v___y_1918_);
lean_dec_ref(v___y_1917_);
return v_res_1924_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6(lean_object* v_00_u03b1_1925_, lean_object* v___y_1926_, lean_object* v___y_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_){
_start:
{
lean_object* v___x_1933_; 
v___x_1933_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___redArg();
return v___x_1933_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6___boxed(lean_object* v_00_u03b1_1934_, lean_object* v___y_1935_, lean_object* v___y_1936_, lean_object* v___y_1937_, lean_object* v___y_1938_, lean_object* v___y_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_){
_start:
{
lean_object* v_res_1942_; 
v_res_1942_ = lp_plausible_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__6(v_00_u03b1_1934_, v___y_1935_, v___y_1936_, v___y_1937_, v___y_1938_, v___y_1939_, v___y_1940_);
lean_dec(v___y_1940_);
lean_dec_ref(v___y_1939_);
lean_dec(v___y_1938_);
lean_dec_ref(v___y_1937_);
lean_dec(v___y_1936_);
lean_dec_ref(v___y_1935_);
return v_res_1942_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5(lean_object* v_00_u03b1_1943_, lean_object* v_msg_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_, lean_object* v___y_1949_, lean_object* v___y_1950_){
_start:
{
lean_object* v___x_1952_; 
v___x_1952_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___redArg(v_msg_1944_, v___y_1945_, v___y_1946_, v___y_1947_, v___y_1948_, v___y_1949_, v___y_1950_);
return v___x_1952_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5___boxed(lean_object* v_00_u03b1_1953_, lean_object* v_msg_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_){
_start:
{
lean_object* v_res_1962_; 
v_res_1962_ = lp_plausible_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5(v_00_u03b1_1953_, v_msg_1954_, v___y_1955_, v___y_1956_, v___y_1957_, v___y_1958_, v___y_1959_, v___y_1960_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1959_);
lean_dec(v___y_1958_);
lean_dec_ref(v___y_1957_);
lean_dec(v___y_1956_);
lean_dec_ref(v___y_1955_);
return v_res_1962_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6(lean_object* v_msgData_1963_, lean_object* v_macroStack_1964_, lean_object* v___y_1965_, lean_object* v___y_1966_, lean_object* v___y_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_){
_start:
{
lean_object* v___x_1972_; 
v___x_1972_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___redArg(v_msgData_1963_, v_macroStack_1964_, v___y_1969_);
return v___x_1972_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6___boxed(lean_object* v_msgData_1973_, lean_object* v_macroStack_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_){
_start:
{
lean_object* v_res_1982_; 
v_res_1982_ = lp_plausible_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem_spec__2_spec__5_spec__6(v_msgData_1973_, v_macroStack_1974_, v___y_1975_, v___y_1976_, v___y_1977_, v___y_1978_, v___y_1979_, v___y_1980_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec(v___y_1978_);
lean_dec_ref(v___y_1977_);
lean_dec(v___y_1976_);
lean_dec_ref(v___y_1975_);
return v_res_1982_;
}
}
static lean_object* _init_lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1983_; lean_object* v___x_1984_; lean_object* v___x_1985_; 
v___x_1983_ = lean_box(0);
v___x_1984_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration_evalExpr___closed__1));
v___x_1985_ = l_Lean_mkConst(v___x_1984_, v___x_1983_);
return v___x_1985_;
}
}
static lean_object* _init_lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1986_; lean_object* v___x_1987_; 
v___x_1986_ = lean_obj_once(&lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0, &lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0_once, _init_lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__0);
v___x_1987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1987_, 0, v___x_1986_);
return v___x_1987_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0(lean_object* v_cfg_1988_, lean_object* v_cfgItem_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_, lean_object* v___y_1992_, lean_object* v___y_1993_, lean_object* v___y_1994_, lean_object* v___y_1995_){
_start:
{
lean_object* v___x_1997_; lean_object* v___x_1998_; 
v___x_1997_ = lean_obj_once(&lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1, &lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1_once, _init_lp_plausible_Plausible_elabConfig___redArg___lam__0___closed__1);
v___x_1998_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_1988_, v_cfgItem_1989_, v___x_1997_, v___y_1990_, v___y_1991_, v___y_1992_, v___y_1993_, v___y_1994_, v___y_1995_);
return v___x_1998_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___lam__0___boxed(lean_object* v_cfg_1999_, lean_object* v_cfgItem_2000_, lean_object* v___y_2001_, lean_object* v___y_2002_, lean_object* v___y_2003_, lean_object* v___y_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_){
_start:
{
lean_object* v_res_2008_; 
v_res_2008_ = lp_plausible_Plausible_elabConfig___redArg___lam__0(v_cfg_1999_, v_cfgItem_2000_, v___y_2001_, v___y_2002_, v___y_2003_, v___y_2004_, v___y_2005_, v___y_2006_);
lean_dec(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___y_2003_);
lean_dec(v___y_2002_);
lean_dec_ref(v___y_2001_);
lean_dec(v_cfgItem_2000_);
return v_res_2008_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg(lean_object* v_cfg_2010_, lean_object* v_init_2011_, uint8_t v_logExceptions_2012_, lean_object* v_a_2013_, lean_object* v_a_2014_, lean_object* v_a_2015_){
_start:
{
lean_object* v_onErr_2017_; lean_object* v_eval_2018_; 
v_onErr_2017_ = ((lean_object*)(lp_plausible_Plausible_elabConfig___redArg___closed__0));
v_eval_2018_ = ((lean_object*)(lp_plausible___private_Plausible_Testable_0__Plausible_elabConfig_evalConfigItem___closed__0));
if (v_logExceptions_2012_ == 0)
{
lean_object* v___x_2019_; 
v___x_2019_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2018_, v_init_2011_, v_cfg_2010_, v_onErr_2017_, v_logExceptions_2012_, v_a_2014_, v_a_2015_);
return v___x_2019_;
}
else
{
uint8_t v_recover_2020_; lean_object* v___x_2021_; 
v_recover_2020_ = lean_ctor_get_uint8(v_a_2013_, sizeof(void*)*1);
v___x_2021_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_2018_, v_init_2011_, v_cfg_2010_, v_onErr_2017_, v_recover_2020_, v_a_2014_, v_a_2015_);
return v___x_2021_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___redArg___boxed(lean_object* v_cfg_2022_, lean_object* v_init_2023_, lean_object* v_logExceptions_2024_, lean_object* v_a_2025_, lean_object* v_a_2026_, lean_object* v_a_2027_, lean_object* v_a_2028_){
_start:
{
uint8_t v_logExceptions_boxed_2029_; lean_object* v_res_2030_; 
v_logExceptions_boxed_2029_ = lean_unbox(v_logExceptions_2024_);
v_res_2030_ = lp_plausible_Plausible_elabConfig___redArg(v_cfg_2022_, v_init_2023_, v_logExceptions_boxed_2029_, v_a_2025_, v_a_2026_, v_a_2027_);
lean_dec(v_a_2027_);
lean_dec_ref(v_a_2026_);
lean_dec_ref(v_a_2025_);
return v_res_2030_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig(lean_object* v_cfg_2031_, lean_object* v_init_2032_, uint8_t v_logExceptions_2033_, lean_object* v_a_2034_, lean_object* v_a_2035_, lean_object* v_a_2036_, lean_object* v_a_2037_, lean_object* v_a_2038_, lean_object* v_a_2039_, lean_object* v_a_2040_, lean_object* v_a_2041_){
_start:
{
lean_object* v___x_2043_; 
v___x_2043_ = lp_plausible_Plausible_elabConfig___redArg(v_cfg_2031_, v_init_2032_, v_logExceptions_2033_, v_a_2034_, v_a_2040_, v_a_2041_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_elabConfig___boxed(lean_object* v_cfg_2044_, lean_object* v_init_2045_, lean_object* v_logExceptions_2046_, lean_object* v_a_2047_, lean_object* v_a_2048_, lean_object* v_a_2049_, lean_object* v_a_2050_, lean_object* v_a_2051_, lean_object* v_a_2052_, lean_object* v_a_2053_, lean_object* v_a_2054_, lean_object* v_a_2055_){
_start:
{
uint8_t v_logExceptions_boxed_2056_; lean_object* v_res_2057_; 
v_logExceptions_boxed_2056_ = lean_unbox(v_logExceptions_2046_);
v_res_2057_ = lp_plausible_Plausible_elabConfig(v_cfg_2044_, v_init_2045_, v_logExceptions_boxed_2056_, v_a_2047_, v_a_2048_, v_a_2049_, v_a_2050_, v_a_2051_, v_a_2052_, v_a_2053_, v_a_2054_);
lean_dec(v_a_2054_);
lean_dec_ref(v_a_2053_);
lean_dec(v_a_2052_);
lean_dec_ref(v_a_2051_);
lean_dec(v_a_2050_);
lean_dec_ref(v_a_2049_);
lean_dec(v_a_2048_);
lean_dec_ref(v_a_2047_);
return v_res_2057_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_instPrintableProp(lean_object* v_p_2059_){
_start:
{
lean_object* v___x_2060_; 
v___x_2060_ = ((lean_object*)(lp_plausible_Plausible_instPrintableProp___closed__0));
return v___x_2060_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0(lean_object* v_x_2062_, lean_object* v_x_2063_){
_start:
{
if (lean_obj_tag(v_x_2063_) == 0)
{
return v_x_2062_;
}
else
{
lean_object* v_head_2064_; lean_object* v_tail_2065_; lean_object* v___x_2066_; lean_object* v___x_2067_; lean_object* v___x_2068_; 
v_head_2064_ = lean_ctor_get(v_x_2063_, 0);
v_tail_2065_ = lean_ctor_get(v_x_2063_, 1);
v___x_2066_ = ((lean_object*)(lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___closed__0));
v___x_2067_ = lean_string_append(v_x_2062_, v___x_2066_);
v___x_2068_ = lean_string_append(v___x_2067_, v_head_2064_);
v_x_2062_ = v___x_2068_;
v_x_2063_ = v_tail_2065_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0___boxed(lean_object* v_x_2070_, lean_object* v_x_2071_){
_start:
{
lean_object* v_res_2072_; 
v_res_2072_ = lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0(v_x_2070_, v_x_2071_);
lean_dec(v_x_2071_);
return v_res_2072_;
}
}
LEAN_EXPORT lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0(lean_object* v_x_2076_){
_start:
{
if (lean_obj_tag(v_x_2076_) == 0)
{
lean_object* v___x_2077_; 
v___x_2077_ = ((lean_object*)(lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__0));
return v___x_2077_;
}
else
{
lean_object* v_tail_2078_; 
v_tail_2078_ = lean_ctor_get(v_x_2076_, 1);
if (lean_obj_tag(v_tail_2078_) == 0)
{
lean_object* v_head_2079_; lean_object* v___x_2080_; lean_object* v___x_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; 
v_head_2079_ = lean_ctor_get(v_x_2076_, 0);
v___x_2080_ = ((lean_object*)(lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__1));
v___x_2081_ = lean_string_append(v___x_2080_, v_head_2079_);
v___x_2082_ = ((lean_object*)(lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__2));
v___x_2083_ = lean_string_append(v___x_2081_, v___x_2082_);
return v___x_2083_;
}
else
{
lean_object* v_head_2084_; lean_object* v___x_2085_; lean_object* v___x_2086_; lean_object* v___x_2087_; uint32_t v___x_2088_; lean_object* v___x_2089_; 
v_head_2084_ = lean_ctor_get(v_x_2076_, 0);
v___x_2085_ = ((lean_object*)(lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__1));
v___x_2086_ = lean_string_append(v___x_2085_, v_head_2084_);
v___x_2087_ = lp_plausible_List_foldl___at___00List_toString___at___00Plausible_TestResult_toString_spec__0_spec__0(v___x_2086_, v_tail_2078_);
v___x_2088_ = 93;
v___x_2089_ = lean_string_push(v___x_2087_, v___x_2088_);
return v___x_2089_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___boxed(lean_object* v_x_2090_){
_start:
{
lean_object* v_res_2091_; 
v_res_2091_ = lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0(v_x_2090_);
lean_dec(v_x_2090_);
return v_res_2091_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_toString___redArg(lean_object* v_x_2097_){
_start:
{
switch(lean_obj_tag(v_x_2097_))
{
case 0:
{
lean_object* v_a_2098_; 
v_a_2098_ = lean_ctor_get(v_x_2097_, 0);
lean_inc_ref(v_a_2098_);
lean_dec_ref_known(v_x_2097_, 1);
if (lean_obj_tag(v_a_2098_) == 0)
{
lean_object* v___x_2099_; 
lean_dec_ref_known(v_a_2098_, 1);
v___x_2099_ = ((lean_object*)(lp_plausible_Plausible_TestResult_toString___redArg___closed__0));
return v___x_2099_;
}
else
{
lean_object* v___x_2100_; 
lean_dec_ref_known(v_a_2098_, 1);
v___x_2100_ = ((lean_object*)(lp_plausible_Plausible_TestResult_toString___redArg___closed__1));
return v___x_2100_;
}
}
case 1:
{
lean_object* v_a_2101_; lean_object* v___x_2102_; lean_object* v___x_2103_; lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; 
v_a_2101_ = lean_ctor_get(v_x_2097_, 0);
lean_inc(v_a_2101_);
lean_dec_ref_known(v_x_2097_, 1);
v___x_2102_ = ((lean_object*)(lp_plausible_Plausible_TestResult_toString___redArg___closed__2));
v___x_2103_ = l_Nat_reprFast(v_a_2101_);
v___x_2104_ = lean_string_append(v___x_2102_, v___x_2103_);
lean_dec_ref(v___x_2103_);
v___x_2105_ = ((lean_object*)(lp_plausible_Plausible_TestResult_toString___redArg___closed__3));
v___x_2106_ = lean_string_append(v___x_2104_, v___x_2105_);
return v___x_2106_;
}
default: 
{
lean_object* v_a_2107_; lean_object* v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; 
v_a_2107_ = lean_ctor_get(v_x_2097_, 0);
lean_inc(v_a_2107_);
lean_dec_ref_known(v_x_2097_, 2);
v___x_2108_ = ((lean_object*)(lp_plausible_Plausible_TestResult_toString___redArg___closed__4));
v___x_2109_ = lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0(v_a_2107_);
lean_dec(v_a_2107_);
v___x_2110_ = lean_string_append(v___x_2108_, v___x_2109_);
lean_dec_ref(v___x_2109_);
return v___x_2110_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_toString(lean_object* v_p_2111_, lean_object* v_x_2112_){
_start:
{
lean_object* v___x_2113_; 
v___x_2113_ = lp_plausible_Plausible_TestResult_toString___redArg(v_x_2112_);
return v___x_2113_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_instToString(lean_object* v_p_2115_){
_start:
{
lean_object* v___x_2116_; 
v___x_2116_ = ((lean_object*)(lp_plausible_Plausible_TestResult_instToString___closed__0));
return v___x_2116_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___redArg(lean_object* v_x_2119_, lean_object* v_x_2120_){
_start:
{
if (lean_obj_tag(v_x_2119_) == 1)
{
if (lean_obj_tag(v_x_2120_) == 1)
{
lean_object* v___x_2124_; uint8_t v_isShared_2125_; uint8_t v_isSharedCheck_2129_; 
v_isSharedCheck_2129_ = !lean_is_exclusive(v_x_2120_);
if (v_isSharedCheck_2129_ == 0)
{
lean_object* v_unused_2130_; 
v_unused_2130_ = lean_ctor_get(v_x_2120_, 0);
lean_dec(v_unused_2130_);
v___x_2124_ = v_x_2120_;
v_isShared_2125_ = v_isSharedCheck_2129_;
goto v_resetjp_2123_;
}
else
{
lean_dec(v_x_2120_);
v___x_2124_ = lean_box(0);
v_isShared_2125_ = v_isSharedCheck_2129_;
goto v_resetjp_2123_;
}
v_resetjp_2123_:
{
lean_object* v___x_2127_; 
if (v_isShared_2125_ == 0)
{
lean_ctor_set(v___x_2124_, 0, lean_box(0));
v___x_2127_ = v___x_2124_;
goto v_reusejp_2126_;
}
else
{
lean_object* v_reuseFailAlloc_2128_; 
v_reuseFailAlloc_2128_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2128_, 0, lean_box(0));
v___x_2127_ = v_reuseFailAlloc_2128_;
goto v_reusejp_2126_;
}
v_reusejp_2126_:
{
return v___x_2127_;
}
}
}
else
{
lean_dec_ref(v_x_2120_);
goto v___jp_2121_;
}
}
else
{
lean_dec_ref(v_x_2120_);
goto v___jp_2121_;
}
v___jp_2121_:
{
lean_object* v___x_2122_; 
v___x_2122_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
return v___x_2122_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___redArg___boxed(lean_object* v_x_2131_, lean_object* v_x_2132_){
_start:
{
lean_object* v_res_2133_; 
v_res_2133_ = lp_plausible_Plausible_TestResult_combine___redArg(v_x_2131_, v_x_2132_);
lean_dec_ref(v_x_2131_);
return v_res_2133_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine(lean_object* v_p_2134_, lean_object* v_q_2135_, lean_object* v_x_2136_, lean_object* v_x_2137_){
_start:
{
lean_object* v___x_2138_; 
v___x_2138_ = lp_plausible_Plausible_TestResult_combine___redArg(v_x_2136_, v_x_2137_);
return v___x_2138_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_combine___boxed(lean_object* v_p_2139_, lean_object* v_q_2140_, lean_object* v_x_2141_, lean_object* v_x_2142_){
_start:
{
lean_object* v_res_2143_; 
v_res_2143_ = lp_plausible_Plausible_TestResult_combine(v_p_2139_, v_q_2140_, v_x_2141_, v_x_2142_);
lean_dec_ref(v_x_2141_);
return v_res_2143_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_and___redArg(lean_object* v_x_2144_, lean_object* v_x_2145_){
_start:
{
switch(lean_obj_tag(v_x_2144_))
{
case 0:
{
switch(lean_obj_tag(v_x_2145_))
{
case 0:
{
lean_object* v_a_2146_; lean_object* v_a_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2157_; 
v_a_2146_ = lean_ctor_get(v_x_2144_, 0);
lean_inc_ref(v_a_2146_);
lean_dec_ref_known(v_x_2144_, 1);
v_a_2147_ = lean_ctor_get(v_x_2145_, 0);
v_isSharedCheck_2157_ = !lean_is_exclusive(v_x_2145_);
if (v_isSharedCheck_2157_ == 0)
{
v___x_2149_ = v_x_2145_;
v_isShared_2150_ = v_isSharedCheck_2157_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_a_2147_);
lean_dec(v_x_2145_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2157_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
lean_object* v___x_2151_; lean_object* v___x_2152_; lean_object* v___x_2153_; lean_object* v___x_2155_; 
v___x_2151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2151_, 0, lean_box(0));
v___x_2152_ = lp_plausible_Plausible_TestResult_combine___redArg(v___x_2151_, v_a_2146_);
lean_dec_ref_known(v___x_2151_, 1);
v___x_2153_ = lp_plausible_Plausible_TestResult_combine___redArg(v___x_2152_, v_a_2147_);
lean_dec_ref(v___x_2152_);
if (v_isShared_2150_ == 0)
{
lean_ctor_set(v___x_2149_, 0, v___x_2153_);
v___x_2155_ = v___x_2149_;
goto v_reusejp_2154_;
}
else
{
lean_object* v_reuseFailAlloc_2156_; 
v_reuseFailAlloc_2156_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2156_, 0, v___x_2153_);
v___x_2155_ = v_reuseFailAlloc_2156_;
goto v_reusejp_2154_;
}
v_reusejp_2154_:
{
return v___x_2155_;
}
}
}
case 1:
{
lean_dec_ref_known(v_x_2144_, 1);
return v_x_2145_;
}
default: 
{
lean_object* v_a_2158_; lean_object* v_a_2159_; lean_object* v___x_2161_; uint8_t v_isShared_2162_; uint8_t v_isSharedCheck_2166_; 
lean_dec_ref_known(v_x_2144_, 1);
v_a_2158_ = lean_ctor_get(v_x_2145_, 0);
v_a_2159_ = lean_ctor_get(v_x_2145_, 1);
v_isSharedCheck_2166_ = !lean_is_exclusive(v_x_2145_);
if (v_isSharedCheck_2166_ == 0)
{
v___x_2161_ = v_x_2145_;
v_isShared_2162_ = v_isSharedCheck_2166_;
goto v_resetjp_2160_;
}
else
{
lean_inc(v_a_2159_);
lean_inc(v_a_2158_);
lean_dec(v_x_2145_);
v___x_2161_ = lean_box(0);
v_isShared_2162_ = v_isSharedCheck_2166_;
goto v_resetjp_2160_;
}
v_resetjp_2160_:
{
lean_object* v___x_2164_; 
if (v_isShared_2162_ == 0)
{
v___x_2164_ = v___x_2161_;
goto v_reusejp_2163_;
}
else
{
lean_object* v_reuseFailAlloc_2165_; 
v_reuseFailAlloc_2165_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2165_, 0, v_a_2158_);
lean_ctor_set(v_reuseFailAlloc_2165_, 1, v_a_2159_);
v___x_2164_ = v_reuseFailAlloc_2165_;
goto v_reusejp_2163_;
}
v_reusejp_2163_:
{
return v___x_2164_;
}
}
}
}
}
case 1:
{
switch(lean_obj_tag(v_x_2145_))
{
case 2:
{
lean_object* v_a_2167_; lean_object* v_a_2168_; lean_object* v___x_2170_; uint8_t v_isShared_2171_; uint8_t v_isSharedCheck_2175_; 
lean_dec_ref_known(v_x_2144_, 1);
v_a_2167_ = lean_ctor_get(v_x_2145_, 0);
v_a_2168_ = lean_ctor_get(v_x_2145_, 1);
v_isSharedCheck_2175_ = !lean_is_exclusive(v_x_2145_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2170_ = v_x_2145_;
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
else
{
lean_inc(v_a_2168_);
lean_inc(v_a_2167_);
lean_dec(v_x_2145_);
v___x_2170_ = lean_box(0);
v_isShared_2171_ = v_isSharedCheck_2175_;
goto v_resetjp_2169_;
}
v_resetjp_2169_:
{
lean_object* v___x_2173_; 
if (v_isShared_2171_ == 0)
{
v___x_2173_ = v___x_2170_;
goto v_reusejp_2172_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v_a_2167_);
lean_ctor_set(v_reuseFailAlloc_2174_, 1, v_a_2168_);
v___x_2173_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2172_;
}
v_reusejp_2172_:
{
return v___x_2173_;
}
}
}
case 1:
{
lean_object* v_a_2176_; lean_object* v_a_2177_; lean_object* v___x_2179_; uint8_t v_isShared_2180_; uint8_t v_isSharedCheck_2185_; 
v_a_2176_ = lean_ctor_get(v_x_2144_, 0);
lean_inc(v_a_2176_);
lean_dec_ref_known(v_x_2144_, 1);
v_a_2177_ = lean_ctor_get(v_x_2145_, 0);
v_isSharedCheck_2185_ = !lean_is_exclusive(v_x_2145_);
if (v_isSharedCheck_2185_ == 0)
{
v___x_2179_ = v_x_2145_;
v_isShared_2180_ = v_isSharedCheck_2185_;
goto v_resetjp_2178_;
}
else
{
lean_inc(v_a_2177_);
lean_dec(v_x_2145_);
v___x_2179_ = lean_box(0);
v_isShared_2180_ = v_isSharedCheck_2185_;
goto v_resetjp_2178_;
}
v_resetjp_2178_:
{
lean_object* v___x_2181_; lean_object* v___x_2183_; 
v___x_2181_ = lean_nat_add(v_a_2176_, v_a_2177_);
lean_dec(v_a_2177_);
lean_dec(v_a_2176_);
if (v_isShared_2180_ == 0)
{
lean_ctor_set(v___x_2179_, 0, v___x_2181_);
v___x_2183_ = v___x_2179_;
goto v_reusejp_2182_;
}
else
{
lean_object* v_reuseFailAlloc_2184_; 
v_reuseFailAlloc_2184_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2184_, 0, v___x_2181_);
v___x_2183_ = v_reuseFailAlloc_2184_;
goto v_reusejp_2182_;
}
v_reusejp_2182_:
{
return v___x_2183_;
}
}
}
default: 
{
lean_dec_ref(v_x_2145_);
return v_x_2144_;
}
}
}
default: 
{
lean_object* v_a_2186_; lean_object* v_a_2187_; lean_object* v___x_2189_; uint8_t v_isShared_2190_; uint8_t v_isSharedCheck_2194_; 
lean_dec_ref(v_x_2145_);
v_a_2186_ = lean_ctor_get(v_x_2144_, 0);
v_a_2187_ = lean_ctor_get(v_x_2144_, 1);
v_isSharedCheck_2194_ = !lean_is_exclusive(v_x_2144_);
if (v_isSharedCheck_2194_ == 0)
{
v___x_2189_ = v_x_2144_;
v_isShared_2190_ = v_isSharedCheck_2194_;
goto v_resetjp_2188_;
}
else
{
lean_inc(v_a_2187_);
lean_inc(v_a_2186_);
lean_dec(v_x_2144_);
v___x_2189_ = lean_box(0);
v_isShared_2190_ = v_isSharedCheck_2194_;
goto v_resetjp_2188_;
}
v_resetjp_2188_:
{
lean_object* v___x_2192_; 
if (v_isShared_2190_ == 0)
{
v___x_2192_ = v___x_2189_;
goto v_reusejp_2191_;
}
else
{
lean_object* v_reuseFailAlloc_2193_; 
v_reuseFailAlloc_2193_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2193_, 0, v_a_2186_);
lean_ctor_set(v_reuseFailAlloc_2193_, 1, v_a_2187_);
v___x_2192_ = v_reuseFailAlloc_2193_;
goto v_reusejp_2191_;
}
v_reusejp_2191_:
{
return v___x_2192_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_and(lean_object* v_p_2195_, lean_object* v_q_2196_, lean_object* v_x_2197_, lean_object* v_x_2198_){
_start:
{
lean_object* v___x_2199_; 
v___x_2199_ = lp_plausible_Plausible_TestResult_and___redArg(v_x_2197_, v_x_2198_);
return v___x_2199_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_or___redArg(lean_object* v_x_2200_, lean_object* v_x_2201_){
_start:
{
lean_object* v_h_2203_; 
switch(lean_obj_tag(v_x_2200_))
{
case 0:
{
lean_object* v_a_2207_; lean_object* v___x_2209_; uint8_t v_isShared_2210_; uint8_t v_isSharedCheck_2216_; 
lean_dec_ref(v_x_2201_);
v_a_2207_ = lean_ctor_get(v_x_2200_, 0);
v_isSharedCheck_2216_ = !lean_is_exclusive(v_x_2200_);
if (v_isSharedCheck_2216_ == 0)
{
v___x_2209_ = v_x_2200_;
v_isShared_2210_ = v_isSharedCheck_2216_;
goto v_resetjp_2208_;
}
else
{
lean_inc(v_a_2207_);
lean_dec(v_x_2200_);
v___x_2209_ = lean_box(0);
v_isShared_2210_ = v_isSharedCheck_2216_;
goto v_resetjp_2208_;
}
v_resetjp_2208_:
{
lean_object* v___x_2211_; lean_object* v___x_2212_; lean_object* v___x_2214_; 
v___x_2211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2211_, 0, lean_box(0));
v___x_2212_ = lp_plausible_Plausible_TestResult_combine___redArg(v___x_2211_, v_a_2207_);
lean_dec_ref_known(v___x_2211_, 1);
if (v_isShared_2210_ == 0)
{
lean_ctor_set(v___x_2209_, 0, v___x_2212_);
v___x_2214_ = v___x_2209_;
goto v_reusejp_2213_;
}
else
{
lean_object* v_reuseFailAlloc_2215_; 
v_reuseFailAlloc_2215_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2215_, 0, v___x_2212_);
v___x_2214_ = v_reuseFailAlloc_2215_;
goto v_reusejp_2213_;
}
v_reusejp_2213_:
{
return v___x_2214_;
}
}
}
case 1:
{
switch(lean_obj_tag(v_x_2201_))
{
case 0:
{
lean_object* v_a_2217_; 
lean_dec_ref_known(v_x_2200_, 1);
v_a_2217_ = lean_ctor_get(v_x_2201_, 0);
lean_inc_ref(v_a_2217_);
lean_dec_ref_known(v_x_2201_, 1);
v_h_2203_ = v_a_2217_;
goto v___jp_2202_;
}
case 1:
{
lean_object* v_a_2218_; lean_object* v_a_2219_; lean_object* v___x_2221_; uint8_t v_isShared_2222_; uint8_t v_isSharedCheck_2227_; 
v_a_2218_ = lean_ctor_get(v_x_2200_, 0);
lean_inc(v_a_2218_);
lean_dec_ref_known(v_x_2200_, 1);
v_a_2219_ = lean_ctor_get(v_x_2201_, 0);
v_isSharedCheck_2227_ = !lean_is_exclusive(v_x_2201_);
if (v_isSharedCheck_2227_ == 0)
{
v___x_2221_ = v_x_2201_;
v_isShared_2222_ = v_isSharedCheck_2227_;
goto v_resetjp_2220_;
}
else
{
lean_inc(v_a_2219_);
lean_dec(v_x_2201_);
v___x_2221_ = lean_box(0);
v_isShared_2222_ = v_isSharedCheck_2227_;
goto v_resetjp_2220_;
}
v_resetjp_2220_:
{
lean_object* v___x_2223_; lean_object* v___x_2225_; 
v___x_2223_ = lean_nat_add(v_a_2218_, v_a_2219_);
lean_dec(v_a_2219_);
lean_dec(v_a_2218_);
if (v_isShared_2222_ == 0)
{
lean_ctor_set(v___x_2221_, 0, v___x_2223_);
v___x_2225_ = v___x_2221_;
goto v_reusejp_2224_;
}
else
{
lean_object* v_reuseFailAlloc_2226_; 
v_reuseFailAlloc_2226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2226_, 0, v___x_2223_);
v___x_2225_ = v_reuseFailAlloc_2226_;
goto v_reusejp_2224_;
}
v_reusejp_2224_:
{
return v___x_2225_;
}
}
}
default: 
{
lean_dec_ref(v_x_2201_);
return v_x_2200_;
}
}
}
default: 
{
switch(lean_obj_tag(v_x_2201_))
{
case 0:
{
lean_object* v_a_2228_; 
lean_dec_ref_known(v_x_2200_, 2);
v_a_2228_ = lean_ctor_get(v_x_2201_, 0);
lean_inc_ref(v_a_2228_);
lean_dec_ref_known(v_x_2201_, 1);
v_h_2203_ = v_a_2228_;
goto v___jp_2202_;
}
case 1:
{
lean_dec_ref_known(v_x_2200_, 2);
return v_x_2201_;
}
default: 
{
lean_object* v_a_2229_; lean_object* v_a_2230_; lean_object* v_a_2231_; lean_object* v_a_2232_; lean_object* v___x_2234_; uint8_t v_isShared_2235_; uint8_t v_isSharedCheck_2241_; 
v_a_2229_ = lean_ctor_get(v_x_2200_, 0);
lean_inc(v_a_2229_);
v_a_2230_ = lean_ctor_get(v_x_2200_, 1);
lean_inc(v_a_2230_);
lean_dec_ref_known(v_x_2200_, 2);
v_a_2231_ = lean_ctor_get(v_x_2201_, 0);
v_a_2232_ = lean_ctor_get(v_x_2201_, 1);
v_isSharedCheck_2241_ = !lean_is_exclusive(v_x_2201_);
if (v_isSharedCheck_2241_ == 0)
{
v___x_2234_ = v_x_2201_;
v_isShared_2235_ = v_isSharedCheck_2241_;
goto v_resetjp_2233_;
}
else
{
lean_inc(v_a_2232_);
lean_inc(v_a_2231_);
lean_dec(v_x_2201_);
v___x_2234_ = lean_box(0);
v_isShared_2235_ = v_isSharedCheck_2241_;
goto v_resetjp_2233_;
}
v_resetjp_2233_:
{
lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2239_; 
v___x_2236_ = l_List_appendTR___redArg(v_a_2229_, v_a_2231_);
v___x_2237_ = lean_nat_add(v_a_2230_, v_a_2232_);
lean_dec(v_a_2232_);
lean_dec(v_a_2230_);
if (v_isShared_2235_ == 0)
{
lean_ctor_set(v___x_2234_, 1, v___x_2237_);
lean_ctor_set(v___x_2234_, 0, v___x_2236_);
v___x_2239_ = v___x_2234_;
goto v_reusejp_2238_;
}
else
{
lean_object* v_reuseFailAlloc_2240_; 
v_reuseFailAlloc_2240_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2240_, 0, v___x_2236_);
lean_ctor_set(v_reuseFailAlloc_2240_, 1, v___x_2237_);
v___x_2239_ = v_reuseFailAlloc_2240_;
goto v_reusejp_2238_;
}
v_reusejp_2238_:
{
return v___x_2239_;
}
}
}
}
}
}
v___jp_2202_:
{
lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; 
v___x_2204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2204_, 0, lean_box(0));
v___x_2205_ = lp_plausible_Plausible_TestResult_combine___redArg(v___x_2204_, v_h_2203_);
lean_dec_ref_known(v___x_2204_, 1);
v___x_2206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2206_, 0, v___x_2205_);
return v___x_2206_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_or(lean_object* v_p_2242_, lean_object* v_q_2243_, lean_object* v_x_2244_, lean_object* v_x_2245_){
_start:
{
lean_object* v___x_2246_; 
v___x_2246_ = lp_plausible_Plausible_TestResult_or___redArg(v_x_2244_, v_x_2245_);
return v___x_2246_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___redArg(lean_object* v_r_2247_, lean_object* v_p_2248_){
_start:
{
switch(lean_obj_tag(v_r_2247_))
{
case 0:
{
lean_object* v_a_2249_; lean_object* v___x_2251_; uint8_t v_isShared_2252_; uint8_t v_isSharedCheck_2257_; 
v_a_2249_ = lean_ctor_get(v_r_2247_, 0);
v_isSharedCheck_2257_ = !lean_is_exclusive(v_r_2247_);
if (v_isSharedCheck_2257_ == 0)
{
v___x_2251_ = v_r_2247_;
v_isShared_2252_ = v_isSharedCheck_2257_;
goto v_resetjp_2250_;
}
else
{
lean_inc(v_a_2249_);
lean_dec(v_r_2247_);
v___x_2251_ = lean_box(0);
v_isShared_2252_ = v_isSharedCheck_2257_;
goto v_resetjp_2250_;
}
v_resetjp_2250_:
{
lean_object* v___x_2253_; lean_object* v___x_2255_; 
v___x_2253_ = lp_plausible_Plausible_TestResult_combine___redArg(v_p_2248_, v_a_2249_);
if (v_isShared_2252_ == 0)
{
lean_ctor_set(v___x_2251_, 0, v___x_2253_);
v___x_2255_ = v___x_2251_;
goto v_reusejp_2254_;
}
else
{
lean_object* v_reuseFailAlloc_2256_; 
v_reuseFailAlloc_2256_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2256_, 0, v___x_2253_);
v___x_2255_ = v_reuseFailAlloc_2256_;
goto v_reusejp_2254_;
}
v_reusejp_2254_:
{
return v___x_2255_;
}
}
}
case 1:
{
return v_r_2247_;
}
default: 
{
lean_object* v_a_2258_; lean_object* v_a_2259_; lean_object* v___x_2261_; uint8_t v_isShared_2262_; uint8_t v_isSharedCheck_2266_; 
v_a_2258_ = lean_ctor_get(v_r_2247_, 0);
v_a_2259_ = lean_ctor_get(v_r_2247_, 1);
v_isSharedCheck_2266_ = !lean_is_exclusive(v_r_2247_);
if (v_isSharedCheck_2266_ == 0)
{
v___x_2261_ = v_r_2247_;
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
else
{
lean_inc(v_a_2259_);
lean_inc(v_a_2258_);
lean_dec(v_r_2247_);
v___x_2261_ = lean_box(0);
v_isShared_2262_ = v_isSharedCheck_2266_;
goto v_resetjp_2260_;
}
v_resetjp_2260_:
{
lean_object* v___x_2264_; 
if (v_isShared_2262_ == 0)
{
v___x_2264_ = v___x_2261_;
goto v_reusejp_2263_;
}
else
{
lean_object* v_reuseFailAlloc_2265_; 
v_reuseFailAlloc_2265_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2265_, 0, v_a_2258_);
lean_ctor_set(v_reuseFailAlloc_2265_, 1, v_a_2259_);
v___x_2264_ = v_reuseFailAlloc_2265_;
goto v_reusejp_2263_;
}
v_reusejp_2263_:
{
return v___x_2264_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___redArg___boxed(lean_object* v_r_2267_, lean_object* v_p_2268_){
_start:
{
lean_object* v_res_2269_; 
v_res_2269_ = lp_plausible_Plausible_TestResult_imp___redArg(v_r_2267_, v_p_2268_);
lean_dec_ref(v_p_2268_);
return v_res_2269_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp(lean_object* v_p_2270_, lean_object* v_q_2271_, lean_object* v_h_2272_, lean_object* v_r_2273_, lean_object* v_p_2274_){
_start:
{
lean_object* v___x_2275_; 
v___x_2275_ = lp_plausible_Plausible_TestResult_imp___redArg(v_r_2273_, v_p_2274_);
return v___x_2275_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_imp___boxed(lean_object* v_p_2276_, lean_object* v_q_2277_, lean_object* v_h_2278_, lean_object* v_r_2279_, lean_object* v_p_2280_){
_start:
{
lean_object* v_res_2281_; 
v_res_2281_ = lp_plausible_Plausible_TestResult_imp(v_p_2276_, v_q_2277_, v_h_2278_, v_r_2279_, v_p_2280_);
lean_dec_ref(v_p_2280_);
return v_res_2281_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_iff___redArg(lean_object* v_r_2282_){
_start:
{
lean_object* v___x_2283_; lean_object* v___x_2284_; 
v___x_2283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2283_, 0, lean_box(0));
v___x_2284_ = lp_plausible_Plausible_TestResult_imp___redArg(v_r_2282_, v___x_2283_);
lean_dec_ref_known(v___x_2283_, 1);
return v___x_2284_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_iff(lean_object* v_p_2285_, lean_object* v_q_2286_, lean_object* v_h_2287_, lean_object* v_r_2288_){
_start:
{
lean_object* v___x_2289_; 
v___x_2289_ = lp_plausible_Plausible_TestResult_iff___redArg(v_r_2288_);
return v___x_2289_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___redArg(lean_object* v_x_2290_, lean_object* v_r_2291_, lean_object* v_p_2292_){
_start:
{
if (lean_obj_tag(v_r_2291_) == 2)
{
lean_object* v_a_2293_; lean_object* v_a_2294_; lean_object* v___x_2296_; uint8_t v_isShared_2297_; uint8_t v_isSharedCheck_2302_; 
v_a_2293_ = lean_ctor_get(v_r_2291_, 0);
v_a_2294_ = lean_ctor_get(v_r_2291_, 1);
v_isSharedCheck_2302_ = !lean_is_exclusive(v_r_2291_);
if (v_isSharedCheck_2302_ == 0)
{
v___x_2296_ = v_r_2291_;
v_isShared_2297_ = v_isSharedCheck_2302_;
goto v_resetjp_2295_;
}
else
{
lean_inc(v_a_2294_);
lean_inc(v_a_2293_);
lean_dec(v_r_2291_);
v___x_2296_ = lean_box(0);
v_isShared_2297_ = v_isSharedCheck_2302_;
goto v_resetjp_2295_;
}
v_resetjp_2295_:
{
lean_object* v___x_2298_; lean_object* v___x_2300_; 
v___x_2298_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2298_, 0, v_x_2290_);
lean_ctor_set(v___x_2298_, 1, v_a_2293_);
if (v_isShared_2297_ == 0)
{
lean_ctor_set(v___x_2296_, 0, v___x_2298_);
v___x_2300_ = v___x_2296_;
goto v_reusejp_2299_;
}
else
{
lean_object* v_reuseFailAlloc_2301_; 
v_reuseFailAlloc_2301_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2301_, 0, v___x_2298_);
lean_ctor_set(v_reuseFailAlloc_2301_, 1, v_a_2294_);
v___x_2300_ = v_reuseFailAlloc_2301_;
goto v_reusejp_2299_;
}
v_reusejp_2299_:
{
return v___x_2300_;
}
}
}
else
{
lean_object* v___x_2303_; 
lean_dec_ref(v_x_2290_);
v___x_2303_ = lp_plausible_Plausible_TestResult_imp___redArg(v_r_2291_, v_p_2292_);
return v___x_2303_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___redArg___boxed(lean_object* v_x_2304_, lean_object* v_r_2305_, lean_object* v_p_2306_){
_start:
{
lean_object* v_res_2307_; 
v_res_2307_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v_x_2304_, v_r_2305_, v_p_2306_);
lean_dec_ref(v_p_2306_);
return v_res_2307_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo(lean_object* v_p_2308_, lean_object* v_q_2309_, lean_object* v_x_2310_, lean_object* v_h_2311_, lean_object* v_r_2312_, lean_object* v_p_2313_){
_start:
{
lean_object* v___x_2314_; 
v___x_2314_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v_x_2310_, v_r_2312_, v_p_2313_);
return v___x_2314_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addInfo___boxed(lean_object* v_p_2315_, lean_object* v_q_2316_, lean_object* v_x_2317_, lean_object* v_h_2318_, lean_object* v_r_2319_, lean_object* v_p_2320_){
_start:
{
lean_object* v_res_2321_; 
v_res_2321_ = lp_plausible_Plausible_TestResult_addInfo(v_p_2315_, v_q_2316_, v_x_2317_, v_h_2318_, v_r_2319_, v_p_2320_);
lean_dec_ref(v_p_2320_);
return v_res_2321_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___redArg(lean_object* v_inst_2323_, lean_object* v_var_2324_, lean_object* v_x_2325_, lean_object* v_r_2326_, lean_object* v_p_2327_){
_start:
{
lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; lean_object* v___x_2333_; lean_object* v___x_2334_; lean_object* v___x_2335_; 
v___x_2328_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
v___x_2329_ = lean_string_append(v_var_2324_, v___x_2328_);
v___x_2330_ = lean_unsigned_to_nat(0u);
v___x_2331_ = lean_apply_2(v_inst_2323_, v_x_2325_, v___x_2330_);
v___x_2332_ = l_Std_Format_defWidth;
v___x_2333_ = l_Std_Format_pretty(v___x_2331_, v___x_2332_, v___x_2330_, v___x_2330_);
v___x_2334_ = lean_string_append(v___x_2329_, v___x_2333_);
lean_dec_ref(v___x_2333_);
v___x_2335_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v___x_2334_, v_r_2326_, v_p_2327_);
return v___x_2335_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___redArg___boxed(lean_object* v_inst_2336_, lean_object* v_var_2337_, lean_object* v_x_2338_, lean_object* v_r_2339_, lean_object* v_p_2340_){
_start:
{
lean_object* v_res_2341_; 
v_res_2341_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v_inst_2336_, v_var_2337_, v_x_2338_, v_r_2339_, v_p_2340_);
lean_dec_ref(v_p_2340_);
return v_res_2341_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo(lean_object* v_p_2342_, lean_object* v_q_2343_, lean_object* v_00_u03b3_2344_, lean_object* v_inst_2345_, lean_object* v_var_2346_, lean_object* v_x_2347_, lean_object* v_h_2348_, lean_object* v_r_2349_, lean_object* v_p_2350_){
_start:
{
lean_object* v___x_2351_; 
v___x_2351_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v_inst_2345_, v_var_2346_, v_x_2347_, v_r_2349_, v_p_2350_);
return v___x_2351_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_addVarInfo___boxed(lean_object* v_p_2352_, lean_object* v_q_2353_, lean_object* v_00_u03b3_2354_, lean_object* v_inst_2355_, lean_object* v_var_2356_, lean_object* v_x_2357_, lean_object* v_h_2358_, lean_object* v_r_2359_, lean_object* v_p_2360_){
_start:
{
lean_object* v_res_2361_; 
v_res_2361_ = lp_plausible_Plausible_TestResult_addVarInfo(v_p_2352_, v_q_2353_, v_00_u03b3_2354_, v_inst_2355_, v_var_2356_, v_x_2357_, v_h_2358_, v_r_2359_, v_p_2360_);
lean_dec_ref(v_p_2360_);
return v_res_2361_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Plausible_TestResult_isFailure___redArg(lean_object* v_x_2362_){
_start:
{
if (lean_obj_tag(v_x_2362_) == 2)
{
uint8_t v___x_2363_; 
v___x_2363_ = 1;
return v___x_2363_;
}
else
{
uint8_t v___x_2364_; 
v___x_2364_ = 0;
return v___x_2364_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_isFailure___redArg___boxed(lean_object* v_x_2365_){
_start:
{
uint8_t v_res_2366_; lean_object* v_r_2367_; 
v_res_2366_ = lp_plausible_Plausible_TestResult_isFailure___redArg(v_x_2365_);
lean_dec_ref(v_x_2365_);
v_r_2367_ = lean_box(v_res_2366_);
return v_r_2367_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Plausible_TestResult_isFailure(lean_object* v_p_2368_, lean_object* v_x_2369_){
_start:
{
uint8_t v___x_2370_; 
v___x_2370_ = lp_plausible_Plausible_TestResult_isFailure___redArg(v_x_2369_);
return v___x_2370_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_TestResult_isFailure___boxed(lean_object* v_p_2371_, lean_object* v_x_2372_){
_start:
{
uint8_t v_res_2373_; lean_object* v_r_2374_; 
v_res_2373_ = lp_plausible_Plausible_TestResult_isFailure(v_p_2371_, v_x_2372_);
lean_dec_ref(v_x_2372_);
v_r_2374_ = lean_box(v_res_2373_);
return v_r_2374_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___redArg(lean_object* v_inst_2382_, lean_object* v_cfg_2383_, uint8_t v_minimize_2384_, lean_object* v_a_2385_, lean_object* v_a_2386_){
_start:
{
lean_object* v___x_2387_; lean_object* v___x_2388_; 
v___x_2387_ = lean_box(v_minimize_2384_);
lean_inc(v_a_2386_);
v___x_2388_ = lean_apply_4(v_inst_2382_, v_cfg_2383_, v___x_2387_, v_a_2385_, v_a_2386_);
return v___x_2388_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___redArg___boxed(lean_object* v_inst_2389_, lean_object* v_cfg_2390_, lean_object* v_minimize_2391_, lean_object* v_a_2392_, lean_object* v_a_2393_){
_start:
{
uint8_t v_minimize_boxed_2394_; lean_object* v_res_2395_; 
v_minimize_boxed_2394_ = lean_unbox(v_minimize_2391_);
v_res_2395_ = lp_plausible_Plausible_Testable_runProp___redArg(v_inst_2389_, v_cfg_2390_, v_minimize_boxed_2394_, v_a_2392_, v_a_2393_);
lean_dec(v_a_2393_);
return v_res_2395_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp(lean_object* v_p_2396_, lean_object* v_inst_2397_, lean_object* v_cfg_2398_, uint8_t v_minimize_2399_, lean_object* v_a_2400_, lean_object* v_a_2401_){
_start:
{
lean_object* v___x_2402_; lean_object* v___x_2403_; 
v___x_2402_ = lean_box(v_minimize_2399_);
lean_inc(v_a_2401_);
v___x_2403_ = lean_apply_4(v_inst_2397_, v_cfg_2398_, v___x_2402_, v_a_2400_, v_a_2401_);
return v___x_2403_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runProp___boxed(lean_object* v_p_2404_, lean_object* v_inst_2405_, lean_object* v_cfg_2406_, lean_object* v_minimize_2407_, lean_object* v_a_2408_, lean_object* v_a_2409_){
_start:
{
uint8_t v_minimize_boxed_2410_; lean_object* v_res_2411_; 
v_minimize_boxed_2410_ = lean_unbox(v_minimize_2407_);
v_res_2411_ = lp_plausible_Plausible_Testable_runProp(v_p_2404_, v_inst_2405_, v_cfg_2406_, v_minimize_boxed_2410_, v_a_2408_, v_a_2409_);
lean_dec(v_a_2409_);
return v_res_2411_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___redArg(lean_object* v_inst_2414_, lean_object* v_cfg_2415_, uint8_t v_min_2416_, lean_object* v_a_2417_, lean_object* v_a_2418_){
_start:
{
lean_object* v_r_2420_; lean_object* v___y_2421_; lean_object* v___x_2424_; lean_object* v___x_2425_; 
v___x_2424_ = lean_box(v_min_2416_);
lean_inc(v_a_2418_);
lean_inc_ref(v_a_2417_);
v___x_2425_ = lean_apply_4(v_inst_2414_, v_cfg_2415_, v___x_2424_, v_a_2417_, v_a_2418_);
if (lean_obj_tag(v___x_2425_) == 0)
{
lean_object* v___x_2426_; 
lean_dec_ref_known(v___x_2425_, 1);
v___x_2426_ = ((lean_object*)(lp_plausible_Plausible_Testable_runPropE___redArg___closed__0));
v_r_2420_ = v___x_2426_;
v___y_2421_ = v_a_2417_;
goto v___jp_2419_;
}
else
{
lean_object* v_a_2427_; lean_object* v_fst_2428_; lean_object* v_snd_2429_; 
lean_dec_ref(v_a_2417_);
v_a_2427_ = lean_ctor_get(v___x_2425_, 0);
lean_inc(v_a_2427_);
lean_dec_ref_known(v___x_2425_, 1);
v_fst_2428_ = lean_ctor_get(v_a_2427_, 0);
lean_inc(v_fst_2428_);
v_snd_2429_ = lean_ctor_get(v_a_2427_, 1);
lean_inc(v_snd_2429_);
lean_dec(v_a_2427_);
v_r_2420_ = v_fst_2428_;
v___y_2421_ = v_snd_2429_;
goto v___jp_2419_;
}
v___jp_2419_:
{
lean_object* v___x_2422_; lean_object* v___x_2423_; 
v___x_2422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2422_, 0, v_r_2420_);
lean_ctor_set(v___x_2422_, 1, v___y_2421_);
v___x_2423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2423_, 0, v___x_2422_);
return v___x_2423_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___redArg___boxed(lean_object* v_inst_2430_, lean_object* v_cfg_2431_, lean_object* v_min_2432_, lean_object* v_a_2433_, lean_object* v_a_2434_){
_start:
{
uint8_t v_min_boxed_2435_; lean_object* v_res_2436_; 
v_min_boxed_2435_ = lean_unbox(v_min_2432_);
v_res_2436_ = lp_plausible_Plausible_Testable_runPropE___redArg(v_inst_2430_, v_cfg_2431_, v_min_boxed_2435_, v_a_2433_, v_a_2434_);
lean_dec(v_a_2434_);
return v_res_2436_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE(lean_object* v_p_2437_, lean_object* v_inst_2438_, lean_object* v_cfg_2439_, uint8_t v_min_2440_, lean_object* v_a_2441_, lean_object* v_a_2442_){
_start:
{
lean_object* v___x_2443_; 
v___x_2443_ = lp_plausible_Plausible_Testable_runPropE___redArg(v_inst_2438_, v_cfg_2439_, v_min_2440_, v_a_2441_, v_a_2442_);
return v___x_2443_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runPropE___boxed(lean_object* v_p_2444_, lean_object* v_inst_2445_, lean_object* v_cfg_2446_, lean_object* v_min_2447_, lean_object* v_a_2448_, lean_object* v_a_2449_){
_start:
{
uint8_t v_min_boxed_2450_; lean_object* v_res_2451_; 
v_min_boxed_2450_ = lean_unbox(v_min_2447_);
v_res_2451_ = lp_plausible_Plausible_Testable_runPropE(v_p_2444_, v_inst_2445_, v_cfg_2446_, v_min_boxed_2450_, v_a_2448_, v_a_2449_);
lean_dec(v_a_2449_);
return v_res_2451_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg___lam__0(lean_object* v_inst_2452_, lean_object* v_x_2453_){
_start:
{
lean_object* v___x_2454_; lean_object* v___x_2455_; 
v___x_2454_ = lean_box(0);
v___x_2455_ = lean_apply_2(v_inst_2452_, lean_box(0), v___x_2454_);
return v___x_2455_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg(lean_object* v_inst_2457_, lean_object* v_s_2458_){
_start:
{
lean_object* v___f_2459_; lean_object* v___x_2460_; lean_object* v___x_2461_; lean_object* v___x_2462_; lean_object* v___x_2463_; lean_object* v___x_2464_; 
v___f_2459_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_slimTrace___redArg___lam__0), 2, 1);
lean_closure_set(v___f_2459_, 0, v_inst_2457_);
v___x_2460_ = ((lean_object*)(lp_plausible_Plausible_Testable_slimTrace___redArg___closed__0));
v___x_2461_ = lean_string_append(v___x_2460_, v_s_2458_);
v___x_2462_ = ((lean_object*)(lp_plausible_List_toString___at___00Plausible_TestResult_toString_spec__0___closed__2));
v___x_2463_ = lean_string_append(v___x_2461_, v___x_2462_);
v___x_2464_ = lean_dbg_trace(v___x_2463_, v___f_2459_);
return v___x_2464_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___redArg___boxed(lean_object* v_inst_2465_, lean_object* v_s_2466_){
_start:
{
lean_object* v_res_2467_; 
v_res_2467_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_inst_2465_, v_s_2466_);
lean_dec_ref(v_s_2466_);
return v_res_2467_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace(lean_object* v_m_2468_, lean_object* v_inst_2469_, lean_object* v_s_2470_){
_start:
{
lean_object* v___x_2471_; 
v___x_2471_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_inst_2469_, v_s_2470_);
return v___x_2471_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_slimTrace___boxed(lean_object* v_m_2472_, lean_object* v_inst_2473_, lean_object* v_s_2474_){
_start:
{
lean_object* v_res_2475_; 
v_res_2475_ = lp_plausible_Plausible_Testable_slimTrace(v_m_2472_, v_inst_2473_, v_s_2474_);
lean_dec_ref(v_s_2474_);
return v_res_2475_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg___lam__0(lean_object* v_inst_2476_, lean_object* v_inst_2477_, lean_object* v_cfg_2478_, uint8_t v_min_2479_, lean_object* v___y_2480_, lean_object* v___y_2481_){
_start:
{
lean_object* v___x_2482_; lean_object* v___x_2483_; 
v___x_2482_ = lean_box(v_min_2479_);
lean_inc(v___y_2481_);
lean_inc_ref(v_cfg_2478_);
v___x_2483_ = lean_apply_4(v_inst_2476_, v_cfg_2478_, v___x_2482_, v___y_2480_, v___y_2481_);
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_dec_ref(v_cfg_2478_);
lean_dec_ref(v_inst_2477_);
return v___x_2483_;
}
else
{
lean_object* v_a_2484_; lean_object* v_fst_2485_; lean_object* v_snd_2486_; lean_object* v___x_2487_; lean_object* v___x_2488_; 
v_a_2484_ = lean_ctor_get(v___x_2483_, 0);
lean_inc(v_a_2484_);
lean_dec_ref_known(v___x_2483_, 1);
v_fst_2485_ = lean_ctor_get(v_a_2484_, 0);
lean_inc(v_fst_2485_);
v_snd_2486_ = lean_ctor_get(v_a_2484_, 1);
lean_inc(v_snd_2486_);
lean_dec(v_a_2484_);
v___x_2487_ = lean_box(v_min_2479_);
lean_inc(v___y_2481_);
v___x_2488_ = lean_apply_4(v_inst_2477_, v_cfg_2478_, v___x_2487_, v_snd_2486_, v___y_2481_);
if (lean_obj_tag(v___x_2488_) == 0)
{
lean_dec(v_fst_2485_);
return v___x_2488_;
}
else
{
lean_object* v_a_2489_; lean_object* v___x_2491_; uint8_t v_isShared_2492_; uint8_t v_isSharedCheck_2506_; 
v_a_2489_ = lean_ctor_get(v___x_2488_, 0);
v_isSharedCheck_2506_ = !lean_is_exclusive(v___x_2488_);
if (v_isSharedCheck_2506_ == 0)
{
v___x_2491_ = v___x_2488_;
v_isShared_2492_ = v_isSharedCheck_2506_;
goto v_resetjp_2490_;
}
else
{
lean_inc(v_a_2489_);
lean_dec(v___x_2488_);
v___x_2491_ = lean_box(0);
v_isShared_2492_ = v_isSharedCheck_2506_;
goto v_resetjp_2490_;
}
v_resetjp_2490_:
{
lean_object* v_fst_2493_; lean_object* v_snd_2494_; lean_object* v___x_2496_; uint8_t v_isShared_2497_; uint8_t v_isSharedCheck_2505_; 
v_fst_2493_ = lean_ctor_get(v_a_2489_, 0);
v_snd_2494_ = lean_ctor_get(v_a_2489_, 1);
v_isSharedCheck_2505_ = !lean_is_exclusive(v_a_2489_);
if (v_isSharedCheck_2505_ == 0)
{
v___x_2496_ = v_a_2489_;
v_isShared_2497_ = v_isSharedCheck_2505_;
goto v_resetjp_2495_;
}
else
{
lean_inc(v_snd_2494_);
lean_inc(v_fst_2493_);
lean_dec(v_a_2489_);
v___x_2496_ = lean_box(0);
v_isShared_2497_ = v_isSharedCheck_2505_;
goto v_resetjp_2495_;
}
v_resetjp_2495_:
{
lean_object* v___x_2498_; lean_object* v___x_2500_; 
v___x_2498_ = lp_plausible_Plausible_TestResult_and___redArg(v_fst_2485_, v_fst_2493_);
if (v_isShared_2497_ == 0)
{
lean_ctor_set(v___x_2496_, 0, v___x_2498_);
v___x_2500_ = v___x_2496_;
goto v_reusejp_2499_;
}
else
{
lean_object* v_reuseFailAlloc_2504_; 
v_reuseFailAlloc_2504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2504_, 0, v___x_2498_);
lean_ctor_set(v_reuseFailAlloc_2504_, 1, v_snd_2494_);
v___x_2500_ = v_reuseFailAlloc_2504_;
goto v_reusejp_2499_;
}
v_reusejp_2499_:
{
lean_object* v___x_2502_; 
if (v_isShared_2492_ == 0)
{
lean_ctor_set(v___x_2491_, 0, v___x_2500_);
v___x_2502_ = v___x_2491_;
goto v_reusejp_2501_;
}
else
{
lean_object* v_reuseFailAlloc_2503_; 
v_reuseFailAlloc_2503_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2503_, 0, v___x_2500_);
v___x_2502_ = v_reuseFailAlloc_2503_;
goto v_reusejp_2501_;
}
v_reusejp_2501_:
{
return v___x_2502_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg___lam__0___boxed(lean_object* v_inst_2507_, lean_object* v_inst_2508_, lean_object* v_cfg_2509_, lean_object* v_min_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_){
_start:
{
uint8_t v_min_boxed_2513_; lean_object* v_res_2514_; 
v_min_boxed_2513_ = lean_unbox(v_min_2510_);
v_res_2514_ = lp_plausible_Plausible_Testable_andTestable___redArg___lam__0(v_inst_2507_, v_inst_2508_, v_cfg_2509_, v_min_boxed_2513_, v___y_2511_, v___y_2512_);
lean_dec(v___y_2512_);
return v_res_2514_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable___redArg(lean_object* v_inst_2515_, lean_object* v_inst_2516_){
_start:
{
lean_object* v___f_2517_; 
v___f_2517_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_andTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_2517_, 0, v_inst_2515_);
lean_closure_set(v___f_2517_, 1, v_inst_2516_);
return v___f_2517_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_andTestable(lean_object* v_p_2518_, lean_object* v_q_2519_, lean_object* v_inst_2520_, lean_object* v_inst_2521_){
_start:
{
lean_object* v___f_2522_; 
v___f_2522_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_andTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_2522_, 0, v_inst_2520_);
lean_closure_set(v___f_2522_, 1, v_inst_2521_);
return v___f_2522_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_2523_; lean_object* v___x_2524_; 
v___x_2523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2523_, 0, lean_box(0));
v___x_2524_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2524_, 0, v___x_2523_);
return v___x_2524_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg___lam__0(lean_object* v_inst_2525_, lean_object* v_inst_2526_, lean_object* v_cfg_2527_, uint8_t v_min_2528_, lean_object* v___y_2529_, lean_object* v___y_2530_){
_start:
{
lean_object* v___x_2531_; lean_object* v___x_2532_; 
v___x_2531_ = lean_box(v_min_2528_);
lean_inc(v___y_2530_);
lean_inc_ref(v_cfg_2527_);
v___x_2532_ = lean_apply_4(v_inst_2525_, v_cfg_2527_, v___x_2531_, v___y_2529_, v___y_2530_);
if (lean_obj_tag(v___x_2532_) == 0)
{
lean_dec_ref(v_cfg_2527_);
lean_dec_ref(v_inst_2526_);
return v___x_2532_;
}
else
{
lean_object* v_a_2533_; lean_object* v_fst_2534_; 
v_a_2533_ = lean_ctor_get(v___x_2532_, 0);
lean_inc(v_a_2533_);
v_fst_2534_ = lean_ctor_get(v_a_2533_, 0);
if (lean_obj_tag(v_fst_2534_) == 0)
{
lean_object* v_a_2535_; 
lean_dec_ref(v_cfg_2527_);
lean_dec_ref(v_inst_2526_);
v_a_2535_ = lean_ctor_get(v_fst_2534_, 0);
if (lean_obj_tag(v_a_2535_) == 0)
{
lean_dec(v_a_2533_);
return v___x_2532_;
}
else
{
lean_object* v___x_2537_; uint8_t v_isShared_2538_; uint8_t v_isSharedCheck_2552_; 
v_isSharedCheck_2552_ = !lean_is_exclusive(v___x_2532_);
if (v_isSharedCheck_2552_ == 0)
{
lean_object* v_unused_2553_; 
v_unused_2553_ = lean_ctor_get(v___x_2532_, 0);
lean_dec(v_unused_2553_);
v___x_2537_ = v___x_2532_;
v_isShared_2538_ = v_isSharedCheck_2552_;
goto v_resetjp_2536_;
}
else
{
lean_dec(v___x_2532_);
v___x_2537_ = lean_box(0);
v_isShared_2538_ = v_isSharedCheck_2552_;
goto v_resetjp_2536_;
}
v_resetjp_2536_:
{
lean_object* v_snd_2539_; lean_object* v___x_2541_; uint8_t v_isShared_2542_; uint8_t v_isSharedCheck_2550_; 
v_snd_2539_ = lean_ctor_get(v_a_2533_, 1);
v_isSharedCheck_2550_ = !lean_is_exclusive(v_a_2533_);
if (v_isSharedCheck_2550_ == 0)
{
lean_object* v_unused_2551_; 
v_unused_2551_ = lean_ctor_get(v_a_2533_, 0);
lean_dec(v_unused_2551_);
v___x_2541_ = v_a_2533_;
v_isShared_2542_ = v_isSharedCheck_2550_;
goto v_resetjp_2540_;
}
else
{
lean_inc(v_snd_2539_);
lean_dec(v_a_2533_);
v___x_2541_ = lean_box(0);
v_isShared_2542_ = v_isSharedCheck_2550_;
goto v_resetjp_2540_;
}
v_resetjp_2540_:
{
lean_object* v___x_2543_; lean_object* v___x_2545_; 
v___x_2543_ = lean_obj_once(&lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0, &lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0_once, _init_lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0);
if (v_isShared_2542_ == 0)
{
lean_ctor_set(v___x_2541_, 0, v___x_2543_);
v___x_2545_ = v___x_2541_;
goto v_reusejp_2544_;
}
else
{
lean_object* v_reuseFailAlloc_2549_; 
v_reuseFailAlloc_2549_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2549_, 0, v___x_2543_);
lean_ctor_set(v_reuseFailAlloc_2549_, 1, v_snd_2539_);
v___x_2545_ = v_reuseFailAlloc_2549_;
goto v_reusejp_2544_;
}
v_reusejp_2544_:
{
lean_object* v___x_2547_; 
if (v_isShared_2538_ == 0)
{
lean_ctor_set(v___x_2537_, 0, v___x_2545_);
v___x_2547_ = v___x_2537_;
goto v_reusejp_2546_;
}
else
{
lean_object* v_reuseFailAlloc_2548_; 
v_reuseFailAlloc_2548_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2548_, 0, v___x_2545_);
v___x_2547_ = v_reuseFailAlloc_2548_;
goto v_reusejp_2546_;
}
v_reusejp_2546_:
{
return v___x_2547_;
}
}
}
}
}
}
else
{
lean_object* v_snd_2554_; lean_object* v___x_2555_; lean_object* v___x_2556_; 
lean_inc(v_fst_2534_);
lean_dec_ref_known(v___x_2532_, 1);
v_snd_2554_ = lean_ctor_get(v_a_2533_, 1);
lean_inc(v_snd_2554_);
lean_dec(v_a_2533_);
v___x_2555_ = lean_box(v_min_2528_);
lean_inc(v___y_2530_);
v___x_2556_ = lean_apply_4(v_inst_2526_, v_cfg_2527_, v___x_2555_, v_snd_2554_, v___y_2530_);
if (lean_obj_tag(v___x_2556_) == 0)
{
lean_dec(v_fst_2534_);
return v___x_2556_;
}
else
{
lean_object* v_a_2557_; lean_object* v___x_2559_; uint8_t v_isShared_2560_; uint8_t v_isSharedCheck_2574_; 
v_a_2557_ = lean_ctor_get(v___x_2556_, 0);
v_isSharedCheck_2574_ = !lean_is_exclusive(v___x_2556_);
if (v_isSharedCheck_2574_ == 0)
{
v___x_2559_ = v___x_2556_;
v_isShared_2560_ = v_isSharedCheck_2574_;
goto v_resetjp_2558_;
}
else
{
lean_inc(v_a_2557_);
lean_dec(v___x_2556_);
v___x_2559_ = lean_box(0);
v_isShared_2560_ = v_isSharedCheck_2574_;
goto v_resetjp_2558_;
}
v_resetjp_2558_:
{
lean_object* v_fst_2561_; lean_object* v_snd_2562_; lean_object* v___x_2564_; uint8_t v_isShared_2565_; uint8_t v_isSharedCheck_2573_; 
v_fst_2561_ = lean_ctor_get(v_a_2557_, 0);
v_snd_2562_ = lean_ctor_get(v_a_2557_, 1);
v_isSharedCheck_2573_ = !lean_is_exclusive(v_a_2557_);
if (v_isSharedCheck_2573_ == 0)
{
v___x_2564_ = v_a_2557_;
v_isShared_2565_ = v_isSharedCheck_2573_;
goto v_resetjp_2563_;
}
else
{
lean_inc(v_snd_2562_);
lean_inc(v_fst_2561_);
lean_dec(v_a_2557_);
v___x_2564_ = lean_box(0);
v_isShared_2565_ = v_isSharedCheck_2573_;
goto v_resetjp_2563_;
}
v_resetjp_2563_:
{
lean_object* v___x_2566_; lean_object* v___x_2568_; 
v___x_2566_ = lp_plausible_Plausible_TestResult_or___redArg(v_fst_2534_, v_fst_2561_);
if (v_isShared_2565_ == 0)
{
lean_ctor_set(v___x_2564_, 0, v___x_2566_);
v___x_2568_ = v___x_2564_;
goto v_reusejp_2567_;
}
else
{
lean_object* v_reuseFailAlloc_2572_; 
v_reuseFailAlloc_2572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2572_, 0, v___x_2566_);
lean_ctor_set(v_reuseFailAlloc_2572_, 1, v_snd_2562_);
v___x_2568_ = v_reuseFailAlloc_2572_;
goto v_reusejp_2567_;
}
v_reusejp_2567_:
{
lean_object* v___x_2570_; 
if (v_isShared_2560_ == 0)
{
lean_ctor_set(v___x_2559_, 0, v___x_2568_);
v___x_2570_ = v___x_2559_;
goto v_reusejp_2569_;
}
else
{
lean_object* v_reuseFailAlloc_2571_; 
v_reuseFailAlloc_2571_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2571_, 0, v___x_2568_);
v___x_2570_ = v_reuseFailAlloc_2571_;
goto v_reusejp_2569_;
}
v_reusejp_2569_:
{
return v___x_2570_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___boxed(lean_object* v_inst_2575_, lean_object* v_inst_2576_, lean_object* v_cfg_2577_, lean_object* v_min_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_){
_start:
{
uint8_t v_min_boxed_2581_; lean_object* v_res_2582_; 
v_min_boxed_2581_ = lean_unbox(v_min_2578_);
v_res_2582_ = lp_plausible_Plausible_Testable_orTestable___redArg___lam__0(v_inst_2575_, v_inst_2576_, v_cfg_2577_, v_min_boxed_2581_, v___y_2579_, v___y_2580_);
lean_dec(v___y_2580_);
return v_res_2582_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable___redArg(lean_object* v_inst_2583_, lean_object* v_inst_2584_){
_start:
{
lean_object* v___f_2585_; 
v___f_2585_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_2585_, 0, v_inst_2583_);
lean_closure_set(v___f_2585_, 1, v_inst_2584_);
return v___f_2585_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_orTestable(lean_object* v_p_2586_, lean_object* v_q_2587_, lean_object* v_inst_2588_, lean_object* v_inst_2589_){
_start:
{
lean_object* v___f_2590_; 
v___f_2590_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_2590_, 0, v_inst_2588_);
lean_closure_set(v___f_2590_, 1, v_inst_2589_);
return v___f_2590_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0(lean_object* v_inst_2591_, lean_object* v_cfg_2592_, uint8_t v_min_2593_, lean_object* v___y_2594_, lean_object* v___y_2595_){
_start:
{
lean_object* v___x_2596_; lean_object* v___x_2597_; 
v___x_2596_ = lean_box(v_min_2593_);
lean_inc(v___y_2595_);
v___x_2597_ = lean_apply_4(v_inst_2591_, v_cfg_2592_, v___x_2596_, v___y_2594_, v___y_2595_);
if (lean_obj_tag(v___x_2597_) == 0)
{
return v___x_2597_;
}
else
{
lean_object* v_a_2598_; lean_object* v___x_2600_; uint8_t v_isShared_2601_; uint8_t v_isSharedCheck_2615_; 
v_a_2598_ = lean_ctor_get(v___x_2597_, 0);
v_isSharedCheck_2615_ = !lean_is_exclusive(v___x_2597_);
if (v_isSharedCheck_2615_ == 0)
{
v___x_2600_ = v___x_2597_;
v_isShared_2601_ = v_isSharedCheck_2615_;
goto v_resetjp_2599_;
}
else
{
lean_inc(v_a_2598_);
lean_dec(v___x_2597_);
v___x_2600_ = lean_box(0);
v_isShared_2601_ = v_isSharedCheck_2615_;
goto v_resetjp_2599_;
}
v_resetjp_2599_:
{
lean_object* v_fst_2602_; lean_object* v_snd_2603_; lean_object* v___x_2605_; uint8_t v_isShared_2606_; uint8_t v_isSharedCheck_2614_; 
v_fst_2602_ = lean_ctor_get(v_a_2598_, 0);
v_snd_2603_ = lean_ctor_get(v_a_2598_, 1);
v_isSharedCheck_2614_ = !lean_is_exclusive(v_a_2598_);
if (v_isSharedCheck_2614_ == 0)
{
v___x_2605_ = v_a_2598_;
v_isShared_2606_ = v_isSharedCheck_2614_;
goto v_resetjp_2604_;
}
else
{
lean_inc(v_snd_2603_);
lean_inc(v_fst_2602_);
lean_dec(v_a_2598_);
v___x_2605_ = lean_box(0);
v_isShared_2606_ = v_isSharedCheck_2614_;
goto v_resetjp_2604_;
}
v_resetjp_2604_:
{
lean_object* v___x_2607_; lean_object* v___x_2609_; 
v___x_2607_ = lp_plausible_Plausible_TestResult_iff___redArg(v_fst_2602_);
if (v_isShared_2606_ == 0)
{
lean_ctor_set(v___x_2605_, 0, v___x_2607_);
v___x_2609_ = v___x_2605_;
goto v_reusejp_2608_;
}
else
{
lean_object* v_reuseFailAlloc_2613_; 
v_reuseFailAlloc_2613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2613_, 0, v___x_2607_);
lean_ctor_set(v_reuseFailAlloc_2613_, 1, v_snd_2603_);
v___x_2609_ = v_reuseFailAlloc_2613_;
goto v_reusejp_2608_;
}
v_reusejp_2608_:
{
lean_object* v___x_2611_; 
if (v_isShared_2601_ == 0)
{
lean_ctor_set(v___x_2600_, 0, v___x_2609_);
v___x_2611_ = v___x_2600_;
goto v_reusejp_2610_;
}
else
{
lean_object* v_reuseFailAlloc_2612_; 
v_reuseFailAlloc_2612_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2612_, 0, v___x_2609_);
v___x_2611_ = v_reuseFailAlloc_2612_;
goto v_reusejp_2610_;
}
v_reusejp_2610_:
{
return v___x_2611_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0___boxed(lean_object* v_inst_2616_, lean_object* v_cfg_2617_, lean_object* v_min_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_){
_start:
{
uint8_t v_min_boxed_2621_; lean_object* v_res_2622_; 
v_min_boxed_2621_ = lean_unbox(v_min_2618_);
v_res_2622_ = lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0(v_inst_2616_, v_cfg_2617_, v_min_boxed_2621_, v___y_2619_, v___y_2620_);
lean_dec(v___y_2620_);
return v_res_2622_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable___redArg(lean_object* v_inst_2623_){
_start:
{
lean_object* v___f_2624_; 
v___f_2624_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2624_, 0, v_inst_2623_);
return v___f_2624_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_iffTestable(lean_object* v_p_2625_, lean_object* v_q_2626_, lean_object* v_inst_2627_){
_start:
{
lean_object* v___f_2628_; 
v___f_2628_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_iffTestable___redArg___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2628_, 0, v_inst_2627_);
return v___f_2628_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10(void){
_start:
{
lean_object* v___x_2648_; lean_object* v___x_2649_; 
v___x_2648_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__9));
v___x_2649_ = l_ReaderT_instMonad___redArg(v___x_2648_);
return v___x_2649_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11(void){
_start:
{
lean_object* v___x_2650_; lean_object* v___x_2651_; 
v___x_2650_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___x_2651_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_2651_, 0, lean_box(0));
lean_closure_set(v___x_2651_, 1, lean_box(0));
lean_closure_set(v___x_2651_, 2, v___x_2650_);
return v___x_2651_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0(uint8_t v_inst_2655_, lean_object* v_inst_2656_, lean_object* v_inst_2657_, lean_object* v_cfg_2658_, uint8_t v_min_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_){
_start:
{
if (v_inst_2655_ == 0)
{
uint8_t v_traceDiscarded_2696_; 
lean_dec_ref(v_inst_2657_);
v_traceDiscarded_2696_ = lean_ctor_get_uint8(v_cfg_2658_, sizeof(void*)*4);
if (v_traceDiscarded_2696_ == 0)
{
uint8_t v_traceSuccesses_2697_; 
v_traceSuccesses_2697_ = lean_ctor_get_uint8(v_cfg_2658_, sizeof(void*)*4 + 1);
lean_dec_ref(v_cfg_2658_);
if (v_traceSuccesses_2697_ == 0)
{
lean_object* v___x_2698_; lean_object* v___x_2699_; lean_object* v___x_2700_; 
v___x_2698_ = ((lean_object*)(lp_plausible_Plausible_Testable_runPropE___redArg___closed__0));
v___x_2699_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2699_, 0, v___x_2698_);
lean_ctor_set(v___x_2699_, 1, v___y_2660_);
v___x_2700_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2700_, 0, v___x_2699_);
return v___x_2700_;
}
else
{
goto v___jp_2662_;
}
}
else
{
lean_dec_ref(v_cfg_2658_);
goto v___jp_2662_;
}
}
else
{
lean_object* v___x_2701_; lean_object* v___x_2702_; 
v___x_2701_ = lean_box(v_min_2659_);
lean_inc(v___y_2661_);
v___x_2702_ = lean_apply_5(v_inst_2657_, lean_box(0), v_cfg_2658_, v___x_2701_, v___y_2660_, v___y_2661_);
if (lean_obj_tag(v___x_2702_) == 0)
{
return v___x_2702_;
}
else
{
lean_object* v_a_2703_; lean_object* v___x_2705_; uint8_t v_isShared_2706_; uint8_t v_isSharedCheck_2723_; 
v_a_2703_ = lean_ctor_get(v___x_2702_, 0);
v_isSharedCheck_2723_ = !lean_is_exclusive(v___x_2702_);
if (v_isSharedCheck_2723_ == 0)
{
v___x_2705_ = v___x_2702_;
v_isShared_2706_ = v_isSharedCheck_2723_;
goto v_resetjp_2704_;
}
else
{
lean_inc(v_a_2703_);
lean_dec(v___x_2702_);
v___x_2705_ = lean_box(0);
v_isShared_2706_ = v_isSharedCheck_2723_;
goto v_resetjp_2704_;
}
v_resetjp_2704_:
{
lean_object* v_fst_2707_; lean_object* v_snd_2708_; lean_object* v___x_2710_; uint8_t v_isShared_2711_; uint8_t v_isSharedCheck_2722_; 
v_fst_2707_ = lean_ctor_get(v_a_2703_, 0);
v_snd_2708_ = lean_ctor_get(v_a_2703_, 1);
v_isSharedCheck_2722_ = !lean_is_exclusive(v_a_2703_);
if (v_isSharedCheck_2722_ == 0)
{
v___x_2710_ = v_a_2703_;
v_isShared_2711_ = v_isSharedCheck_2722_;
goto v_resetjp_2709_;
}
else
{
lean_inc(v_snd_2708_);
lean_inc(v_fst_2707_);
lean_dec(v_a_2703_);
v___x_2710_ = lean_box(0);
v_isShared_2711_ = v_isSharedCheck_2722_;
goto v_resetjp_2709_;
}
v_resetjp_2709_:
{
lean_object* v___x_2712_; lean_object* v___x_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; lean_object* v___x_2717_; 
v___x_2712_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__14));
v___x_2713_ = lean_string_append(v___x_2712_, v_inst_2656_);
v___x_2714_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2714_, 0, lean_box(0));
v___x_2715_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v___x_2713_, v_fst_2707_, v___x_2714_);
lean_dec_ref_known(v___x_2714_, 1);
if (v_isShared_2711_ == 0)
{
lean_ctor_set(v___x_2710_, 0, v___x_2715_);
v___x_2717_ = v___x_2710_;
goto v_reusejp_2716_;
}
else
{
lean_object* v_reuseFailAlloc_2721_; 
v_reuseFailAlloc_2721_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2721_, 0, v___x_2715_);
lean_ctor_set(v_reuseFailAlloc_2721_, 1, v_snd_2708_);
v___x_2717_ = v_reuseFailAlloc_2721_;
goto v_reusejp_2716_;
}
v_reusejp_2716_:
{
lean_object* v___x_2719_; 
if (v_isShared_2706_ == 0)
{
lean_ctor_set(v___x_2705_, 0, v___x_2717_);
v___x_2719_ = v___x_2705_;
goto v_reusejp_2718_;
}
else
{
lean_object* v_reuseFailAlloc_2720_; 
v_reuseFailAlloc_2720_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2720_, 0, v___x_2717_);
v___x_2719_ = v_reuseFailAlloc_2720_;
goto v_reusejp_2718_;
}
v_reusejp_2718_:
{
return v___x_2719_;
}
}
}
}
}
}
v___jp_2662_:
{
lean_object* v___x_2663_; lean_object* v___x_2664_; lean_object* v___x_2665_; lean_object* v___x_2666_; lean_object* v___x_2667_; lean_object* v___x_1190__overap_2668_; lean_object* v___x_2669_; 
v___x_2663_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___x_2664_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__12));
v___x_2665_ = lean_string_append(v___x_2664_, v_inst_2656_);
v___x_2666_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__13));
v___x_2667_ = lean_string_append(v___x_2665_, v___x_2666_);
v___x_1190__overap_2668_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_2663_, v___x_2667_);
lean_dec_ref(v___x_2667_);
lean_inc(v___y_2661_);
v___x_2669_ = lean_apply_2(v___x_1190__overap_2668_, v___y_2660_, v___y_2661_);
if (lean_obj_tag(v___x_2669_) == 0)
{
lean_object* v_a_2670_; lean_object* v___x_2672_; uint8_t v_isShared_2673_; uint8_t v_isSharedCheck_2677_; 
v_a_2670_ = lean_ctor_get(v___x_2669_, 0);
v_isSharedCheck_2677_ = !lean_is_exclusive(v___x_2669_);
if (v_isSharedCheck_2677_ == 0)
{
v___x_2672_ = v___x_2669_;
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
else
{
lean_inc(v_a_2670_);
lean_dec(v___x_2669_);
v___x_2672_ = lean_box(0);
v_isShared_2673_ = v_isSharedCheck_2677_;
goto v_resetjp_2671_;
}
v_resetjp_2671_:
{
lean_object* v___x_2675_; 
if (v_isShared_2673_ == 0)
{
v___x_2675_ = v___x_2672_;
goto v_reusejp_2674_;
}
else
{
lean_object* v_reuseFailAlloc_2676_; 
v_reuseFailAlloc_2676_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2676_, 0, v_a_2670_);
v___x_2675_ = v_reuseFailAlloc_2676_;
goto v_reusejp_2674_;
}
v_reusejp_2674_:
{
return v___x_2675_;
}
}
}
else
{
lean_object* v_a_2678_; lean_object* v___x_2680_; uint8_t v_isShared_2681_; uint8_t v_isSharedCheck_2695_; 
v_a_2678_ = lean_ctor_get(v___x_2669_, 0);
v_isSharedCheck_2695_ = !lean_is_exclusive(v___x_2669_);
if (v_isSharedCheck_2695_ == 0)
{
v___x_2680_ = v___x_2669_;
v_isShared_2681_ = v_isSharedCheck_2695_;
goto v_resetjp_2679_;
}
else
{
lean_inc(v_a_2678_);
lean_dec(v___x_2669_);
v___x_2680_ = lean_box(0);
v_isShared_2681_ = v_isSharedCheck_2695_;
goto v_resetjp_2679_;
}
v_resetjp_2679_:
{
lean_object* v_snd_2682_; lean_object* v___x_2684_; uint8_t v_isShared_2685_; uint8_t v_isSharedCheck_2693_; 
v_snd_2682_ = lean_ctor_get(v_a_2678_, 1);
v_isSharedCheck_2693_ = !lean_is_exclusive(v_a_2678_);
if (v_isSharedCheck_2693_ == 0)
{
lean_object* v_unused_2694_; 
v_unused_2694_ = lean_ctor_get(v_a_2678_, 0);
lean_dec(v_unused_2694_);
v___x_2684_ = v_a_2678_;
v_isShared_2685_ = v_isSharedCheck_2693_;
goto v_resetjp_2683_;
}
else
{
lean_inc(v_snd_2682_);
lean_dec(v_a_2678_);
v___x_2684_ = lean_box(0);
v_isShared_2685_ = v_isSharedCheck_2693_;
goto v_resetjp_2683_;
}
v_resetjp_2683_:
{
lean_object* v___x_2686_; lean_object* v___x_2688_; 
v___x_2686_ = ((lean_object*)(lp_plausible_Plausible_Testable_runPropE___redArg___closed__0));
if (v_isShared_2685_ == 0)
{
lean_ctor_set(v___x_2684_, 0, v___x_2686_);
v___x_2688_ = v___x_2684_;
goto v_reusejp_2687_;
}
else
{
lean_object* v_reuseFailAlloc_2692_; 
v_reuseFailAlloc_2692_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2692_, 0, v___x_2686_);
lean_ctor_set(v_reuseFailAlloc_2692_, 1, v_snd_2682_);
v___x_2688_ = v_reuseFailAlloc_2692_;
goto v_reusejp_2687_;
}
v_reusejp_2687_:
{
lean_object* v___x_2690_; 
if (v_isShared_2681_ == 0)
{
lean_ctor_set(v___x_2680_, 0, v___x_2688_);
v___x_2690_ = v___x_2680_;
goto v_reusejp_2689_;
}
else
{
lean_object* v_reuseFailAlloc_2691_; 
v_reuseFailAlloc_2691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2691_, 0, v___x_2688_);
v___x_2690_ = v_reuseFailAlloc_2691_;
goto v_reusejp_2689_;
}
v_reusejp_2689_:
{
return v___x_2690_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___boxed(lean_object* v_inst_2724_, lean_object* v_inst_2725_, lean_object* v_inst_2726_, lean_object* v_cfg_2727_, lean_object* v_min_2728_, lean_object* v___y_2729_, lean_object* v___y_2730_){
_start:
{
uint8_t v_inst_1271__boxed_2731_; uint8_t v_min_boxed_2732_; lean_object* v_res_2733_; 
v_inst_1271__boxed_2731_ = lean_unbox(v_inst_2724_);
v_min_boxed_2732_ = lean_unbox(v_min_2728_);
v_res_2733_ = lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0(v_inst_1271__boxed_2731_, v_inst_2725_, v_inst_2726_, v_cfg_2727_, v_min_boxed_2732_, v___y_2729_, v___y_2730_);
lean_dec(v___y_2730_);
lean_dec_ref(v_inst_2725_);
return v_res_2733_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg(lean_object* v_inst_2734_, uint8_t v_inst_2735_, lean_object* v_inst_2736_){
_start:
{
lean_object* v___x_2737_; lean_object* v___f_2738_; 
v___x_2737_ = lean_box(v_inst_2735_);
v___f_2738_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_2738_, 0, v___x_2737_);
lean_closure_set(v___f_2738_, 1, v_inst_2734_);
lean_closure_set(v___f_2738_, 2, v_inst_2736_);
return v___f_2738_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___redArg___boxed(lean_object* v_inst_2739_, lean_object* v_inst_2740_, lean_object* v_inst_2741_){
_start:
{
uint8_t v_inst_1433__boxed_2742_; lean_object* v_res_2743_; 
v_inst_1433__boxed_2742_ = lean_unbox(v_inst_2740_);
v_res_2743_ = lp_plausible_Plausible_Testable_decGuardTestable___redArg(v_inst_2739_, v_inst_1433__boxed_2742_, v_inst_2741_);
return v_res_2743_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable(lean_object* v_p_2744_, lean_object* v_var_2745_, lean_object* v_inst_2746_, uint8_t v_inst_2747_, lean_object* v_00_u03b2_2748_, lean_object* v_inst_2749_){
_start:
{
lean_object* v___x_2750_; lean_object* v___f_2751_; 
v___x_2750_ = lean_box(v_inst_2747_);
v___f_2751_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_2751_, 0, v___x_2750_);
lean_closure_set(v___f_2751_, 1, v_inst_2746_);
lean_closure_set(v___f_2751_, 2, v_inst_2749_);
return v___f_2751_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decGuardTestable___boxed(lean_object* v_p_2752_, lean_object* v_var_2753_, lean_object* v_inst_2754_, lean_object* v_inst_2755_, lean_object* v_00_u03b2_2756_, lean_object* v_inst_2757_){
_start:
{
uint8_t v_inst_1447__boxed_2758_; lean_object* v_res_2759_; 
v_inst_1447__boxed_2758_ = lean_unbox(v_inst_2755_);
v_res_2759_ = lp_plausible_Plausible_Testable_decGuardTestable(v_p_2752_, v_var_2753_, v_inst_2754_, v_inst_1447__boxed_2758_, v_00_u03b2_2756_, v_inst_2757_);
lean_dec_ref(v_var_2753_);
return v_res_2759_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0(lean_object* v_inst_2761_, lean_object* v___f_2762_, lean_object* v_var_2763_, lean_object* v_cfg_2764_, uint8_t v_min_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_){
_start:
{
lean_object* v___x_2768_; lean_object* v___x_2769_; 
v___x_2768_ = lean_box(v_min_2765_);
lean_inc(v___y_2767_);
v___x_2769_ = lean_apply_4(v_inst_2761_, v_cfg_2764_, v___x_2768_, v___y_2766_, v___y_2767_);
if (lean_obj_tag(v___x_2769_) == 0)
{
lean_dec_ref(v_var_2763_);
lean_dec_ref(v___f_2762_);
return v___x_2769_;
}
else
{
lean_object* v_a_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2789_; 
v_a_2770_ = lean_ctor_get(v___x_2769_, 0);
v_isSharedCheck_2789_ = !lean_is_exclusive(v___x_2769_);
if (v_isSharedCheck_2789_ == 0)
{
v___x_2772_ = v___x_2769_;
v_isShared_2773_ = v_isSharedCheck_2789_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_a_2770_);
lean_dec(v___x_2769_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2789_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v_fst_2774_; lean_object* v_snd_2775_; lean_object* v___x_2777_; uint8_t v_isShared_2778_; uint8_t v_isSharedCheck_2788_; 
v_fst_2774_ = lean_ctor_get(v_a_2770_, 0);
v_snd_2775_ = lean_ctor_get(v_a_2770_, 1);
v_isSharedCheck_2788_ = !lean_is_exclusive(v_a_2770_);
if (v_isSharedCheck_2788_ == 0)
{
v___x_2777_ = v_a_2770_;
v_isShared_2778_ = v_isSharedCheck_2788_;
goto v_resetjp_2776_;
}
else
{
lean_inc(v_snd_2775_);
lean_inc(v_fst_2774_);
lean_dec(v_a_2770_);
v___x_2777_ = lean_box(0);
v_isShared_2778_ = v_isSharedCheck_2788_;
goto v_resetjp_2776_;
}
v_resetjp_2776_:
{
lean_object* v___x_2779_; lean_object* v___x_2780_; lean_object* v___x_2781_; lean_object* v___x_2783_; 
v___x_2779_ = ((lean_object*)(lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___closed__0));
v___x_2780_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_2781_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v___f_2762_, v_var_2763_, v___x_2779_, v_fst_2774_, v___x_2780_);
if (v_isShared_2778_ == 0)
{
lean_ctor_set(v___x_2777_, 0, v___x_2781_);
v___x_2783_ = v___x_2777_;
goto v_reusejp_2782_;
}
else
{
lean_object* v_reuseFailAlloc_2787_; 
v_reuseFailAlloc_2787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2787_, 0, v___x_2781_);
lean_ctor_set(v_reuseFailAlloc_2787_, 1, v_snd_2775_);
v___x_2783_ = v_reuseFailAlloc_2787_;
goto v_reusejp_2782_;
}
v_reusejp_2782_:
{
lean_object* v___x_2785_; 
if (v_isShared_2773_ == 0)
{
lean_ctor_set(v___x_2772_, 0, v___x_2783_);
v___x_2785_ = v___x_2772_;
goto v_reusejp_2784_;
}
else
{
lean_object* v_reuseFailAlloc_2786_; 
v_reuseFailAlloc_2786_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2786_, 0, v___x_2783_);
v___x_2785_ = v_reuseFailAlloc_2786_;
goto v_reusejp_2784_;
}
v_reusejp_2784_:
{
return v___x_2785_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___boxed(lean_object* v_inst_2790_, lean_object* v___f_2791_, lean_object* v_var_2792_, lean_object* v_cfg_2793_, lean_object* v_min_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_){
_start:
{
uint8_t v_min_boxed_2797_; lean_object* v_res_2798_; 
v_min_boxed_2797_ = lean_unbox(v_min_2794_);
v_res_2798_ = lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0(v_inst_2790_, v___f_2791_, v_var_2792_, v_cfg_2793_, v_min_boxed_2797_, v___y_2795_, v___y_2796_);
lean_dec(v___y_2796_);
return v_res_2798_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable___redArg(lean_object* v_var_2800_, lean_object* v_inst_2801_){
_start:
{
lean_object* v___f_2802_; lean_object* v___f_2803_; 
v___f_2802_ = ((lean_object*)(lp_plausible_Plausible_Testable_forallTypesTestable___redArg___closed__0));
v___f_2803_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_forallTypesTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_2803_, 0, v_inst_2801_);
lean_closure_set(v___f_2803_, 1, v___f_2802_);
lean_closure_set(v___f_2803_, 2, v_var_2800_);
return v___f_2803_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesTestable(lean_object* v_var_2804_, lean_object* v_f_2805_, lean_object* v_inst_2806_){
_start:
{
lean_object* v___x_2807_; 
v___x_2807_ = lp_plausible_Plausible_Testable_forallTypesTestable___redArg(v_var_2804_, v_inst_2806_);
return v___x_2807_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0(lean_object* v_inst_2809_, lean_object* v___f_2810_, lean_object* v_var_2811_, lean_object* v_cfg_2812_, uint8_t v_min_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_){
_start:
{
lean_object* v___x_2816_; lean_object* v___x_2817_; 
v___x_2816_ = lean_box(v_min_2813_);
lean_inc(v___y_2815_);
v___x_2817_ = lean_apply_4(v_inst_2809_, v_cfg_2812_, v___x_2816_, v___y_2814_, v___y_2815_);
if (lean_obj_tag(v___x_2817_) == 0)
{
lean_dec_ref(v_var_2811_);
lean_dec_ref(v___f_2810_);
return v___x_2817_;
}
else
{
lean_object* v_a_2818_; lean_object* v___x_2820_; uint8_t v_isShared_2821_; uint8_t v_isSharedCheck_2837_; 
v_a_2818_ = lean_ctor_get(v___x_2817_, 0);
v_isSharedCheck_2837_ = !lean_is_exclusive(v___x_2817_);
if (v_isSharedCheck_2837_ == 0)
{
v___x_2820_ = v___x_2817_;
v_isShared_2821_ = v_isSharedCheck_2837_;
goto v_resetjp_2819_;
}
else
{
lean_inc(v_a_2818_);
lean_dec(v___x_2817_);
v___x_2820_ = lean_box(0);
v_isShared_2821_ = v_isSharedCheck_2837_;
goto v_resetjp_2819_;
}
v_resetjp_2819_:
{
lean_object* v_fst_2822_; lean_object* v_snd_2823_; lean_object* v___x_2825_; uint8_t v_isShared_2826_; uint8_t v_isSharedCheck_2836_; 
v_fst_2822_ = lean_ctor_get(v_a_2818_, 0);
v_snd_2823_ = lean_ctor_get(v_a_2818_, 1);
v_isSharedCheck_2836_ = !lean_is_exclusive(v_a_2818_);
if (v_isSharedCheck_2836_ == 0)
{
v___x_2825_ = v_a_2818_;
v_isShared_2826_ = v_isSharedCheck_2836_;
goto v_resetjp_2824_;
}
else
{
lean_inc(v_snd_2823_);
lean_inc(v_fst_2822_);
lean_dec(v_a_2818_);
v___x_2825_ = lean_box(0);
v_isShared_2826_ = v_isSharedCheck_2836_;
goto v_resetjp_2824_;
}
v_resetjp_2824_:
{
lean_object* v___x_2827_; lean_object* v___x_2828_; lean_object* v___x_2829_; lean_object* v___x_2831_; 
v___x_2827_ = ((lean_object*)(lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___closed__0));
v___x_2828_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_2829_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v___f_2810_, v_var_2811_, v___x_2827_, v_fst_2822_, v___x_2828_);
if (v_isShared_2826_ == 0)
{
lean_ctor_set(v___x_2825_, 0, v___x_2829_);
v___x_2831_ = v___x_2825_;
goto v_reusejp_2830_;
}
else
{
lean_object* v_reuseFailAlloc_2835_; 
v_reuseFailAlloc_2835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2835_, 0, v___x_2829_);
lean_ctor_set(v_reuseFailAlloc_2835_, 1, v_snd_2823_);
v___x_2831_ = v_reuseFailAlloc_2835_;
goto v_reusejp_2830_;
}
v_reusejp_2830_:
{
lean_object* v___x_2833_; 
if (v_isShared_2821_ == 0)
{
lean_ctor_set(v___x_2820_, 0, v___x_2831_);
v___x_2833_ = v___x_2820_;
goto v_reusejp_2832_;
}
else
{
lean_object* v_reuseFailAlloc_2834_; 
v_reuseFailAlloc_2834_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2834_, 0, v___x_2831_);
v___x_2833_ = v_reuseFailAlloc_2834_;
goto v_reusejp_2832_;
}
v_reusejp_2832_:
{
return v___x_2833_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___boxed(lean_object* v_inst_2838_, lean_object* v___f_2839_, lean_object* v_var_2840_, lean_object* v_cfg_2841_, lean_object* v_min_2842_, lean_object* v___y_2843_, lean_object* v___y_2844_){
_start:
{
uint8_t v_min_boxed_2845_; lean_object* v_res_2846_; 
v_min_boxed_2845_ = lean_unbox(v_min_2842_);
v_res_2846_ = lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0(v_inst_2838_, v___f_2839_, v_var_2840_, v_cfg_2841_, v_min_boxed_2845_, v___y_2843_, v___y_2844_);
lean_dec(v___y_2844_);
return v_res_2846_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg(lean_object* v_var_2847_, lean_object* v_inst_2848_){
_start:
{
lean_object* v___f_2849_; lean_object* v___f_2850_; 
v___f_2849_ = ((lean_object*)(lp_plausible_Plausible_Testable_forallTypesTestable___redArg___closed__0));
v___f_2850_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_2850_, 0, v_inst_2848_);
lean_closure_set(v___f_2850_, 1, v___f_2849_);
lean_closure_set(v___f_2850_, 2, v_var_2847_);
return v___f_2850_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_forallTypesULiftTestable(lean_object* v_var_2851_, lean_object* v_f_2852_, lean_object* v_inst_2853_){
_start:
{
lean_object* v___x_2854_; 
v___x_2854_ = lp_plausible_Plausible_Testable_forallTypesULiftTestable___redArg(v_var_2851_, v_inst_2853_);
return v___x_2854_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_formatFailure(lean_object* v_s_2863_, lean_object* v_xs_2864_, lean_object* v_n_2865_){
_start:
{
lean_object* v___x_2866_; lean_object* v_counter_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; lean_object* v___x_2870_; lean_object* v___x_2871_; lean_object* v___x_2872_; lean_object* v___x_2873_; lean_object* v___x_2874_; lean_object* v___x_2875_; lean_object* v___x_2876_; lean_object* v___x_2877_; lean_object* v_parts_2878_; lean_object* v___x_2879_; 
v___x_2866_ = ((lean_object*)(lp_plausible_Plausible_Testable_formatFailure___closed__0));
v_counter_2867_ = l_String_intercalate(v___x_2866_, v_xs_2864_);
v___x_2868_ = ((lean_object*)(lp_plausible_Plausible_Testable_formatFailure___closed__1));
v___x_2869_ = ((lean_object*)(lp_plausible_Plausible_Testable_formatFailure___closed__2));
v___x_2870_ = l_Nat_reprFast(v_n_2865_);
v___x_2871_ = lean_string_append(v___x_2869_, v___x_2870_);
lean_dec_ref(v___x_2870_);
v___x_2872_ = ((lean_object*)(lp_plausible_Plausible_Testable_formatFailure___closed__3));
v___x_2873_ = lean_string_append(v___x_2871_, v___x_2872_);
v___x_2874_ = ((lean_object*)(lp_plausible_Plausible_Testable_formatFailure___closed__5));
v___x_2875_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2875_, 0, v___x_2873_);
lean_ctor_set(v___x_2875_, 1, v___x_2874_);
v___x_2876_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2876_, 0, v_counter_2867_);
lean_ctor_set(v___x_2876_, 1, v___x_2875_);
v___x_2877_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2877_, 0, v_s_2863_);
lean_ctor_set(v___x_2877_, 1, v___x_2876_);
v_parts_2878_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_parts_2878_, 0, v___x_2868_);
lean_ctor_set(v_parts_2878_, 1, v___x_2877_);
v___x_2879_ = l_String_intercalate(v___x_2866_, v_parts_2878_);
return v___x_2879_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___redArg(lean_object* v_n_2880_, lean_object* v_x_2881_){
_start:
{
if (lean_obj_tag(v_x_2881_) == 2)
{
lean_object* v_a_2882_; lean_object* v_a_2883_; lean_object* v___x_2885_; uint8_t v_isShared_2886_; uint8_t v_isSharedCheck_2891_; 
v_a_2882_ = lean_ctor_get(v_x_2881_, 0);
v_a_2883_ = lean_ctor_get(v_x_2881_, 1);
v_isSharedCheck_2891_ = !lean_is_exclusive(v_x_2881_);
if (v_isSharedCheck_2891_ == 0)
{
v___x_2885_ = v_x_2881_;
v_isShared_2886_ = v_isSharedCheck_2891_;
goto v_resetjp_2884_;
}
else
{
lean_inc(v_a_2883_);
lean_inc(v_a_2882_);
lean_dec(v_x_2881_);
v___x_2885_ = lean_box(0);
v_isShared_2886_ = v_isSharedCheck_2891_;
goto v_resetjp_2884_;
}
v_resetjp_2884_:
{
lean_object* v___x_2887_; lean_object* v___x_2889_; 
v___x_2887_ = lean_nat_add(v_a_2883_, v_n_2880_);
lean_dec(v_a_2883_);
if (v_isShared_2886_ == 0)
{
lean_ctor_set(v___x_2885_, 1, v___x_2887_);
v___x_2889_ = v___x_2885_;
goto v_reusejp_2888_;
}
else
{
lean_object* v_reuseFailAlloc_2890_; 
v_reuseFailAlloc_2890_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2890_, 0, v_a_2882_);
lean_ctor_set(v_reuseFailAlloc_2890_, 1, v___x_2887_);
v___x_2889_ = v_reuseFailAlloc_2890_;
goto v_reusejp_2888_;
}
v_reusejp_2888_:
{
return v___x_2889_;
}
}
}
else
{
return v_x_2881_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___redArg___boxed(lean_object* v_n_2892_, lean_object* v_x_2893_){
_start:
{
lean_object* v_res_2894_; 
v_res_2894_ = lp_plausible_Plausible_Testable_addShrinks___redArg(v_n_2892_, v_x_2893_);
lean_dec(v_n_2892_);
return v_res_2894_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks(lean_object* v_p_2895_, lean_object* v_n_2896_, lean_object* v_x_2897_){
_start:
{
lean_object* v___x_2898_; 
v___x_2898_ = lp_plausible_Plausible_Testable_addShrinks___redArg(v_n_2896_, v_x_2897_);
return v___x_2898_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_addShrinks___boxed(lean_object* v_p_2899_, lean_object* v_n_2900_, lean_object* v_x_2901_){
_start:
{
lean_object* v_res_2902_; 
v_res_2902_ = lp_plausible_Plausible_Testable_addShrinks(v_p_2899_, v_n_2900_, v_x_2901_);
lean_dec(v_n_2900_);
return v_res_2902_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_instInhabitedOptionTOfPure___redArg(lean_object* v_inst_2903_){
_start:
{
lean_object* v___x_2904_; lean_object* v___x_2905_; 
v___x_2904_ = lean_box(0);
v___x_2905_ = lean_apply_2(v_inst_2903_, lean_box(0), v___x_2904_);
return v___x_2905_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_instInhabitedOptionTOfPure(lean_object* v_00_u03b1_2906_, lean_object* v_m_2907_, lean_object* v_inst_2908_){
_start:
{
lean_object* v___x_2909_; 
v___x_2909_ = lp_plausible_Plausible_Testable_instInhabitedOptionTOfPure___redArg(v_inst_2908_);
return v___x_2909_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7(void){
_start:
{
lean_object* v___x_2913_; lean_object* v___x_2914_; 
v___x_2913_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___x_2914_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_2914_, 0, lean_box(0));
lean_closure_set(v___x_2914_, 1, lean_box(0));
lean_closure_set(v___x_2914_, 2, v___x_2913_);
return v___x_2914_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3(void){
_start:
{
lean_object* v___x_2915_; lean_object* v___f_2916_; 
v___x_2915_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___f_2916_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_2916_, 0, v___x_2915_);
return v___f_2916_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2(void){
_start:
{
lean_object* v___x_2917_; lean_object* v___f_2918_; 
v___x_2917_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___f_2918_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_2918_, 0, v___x_2917_);
return v___f_2918_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1(void){
_start:
{
lean_object* v___x_2919_; lean_object* v___f_2920_; 
v___x_2919_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___f_2920_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_2920_, 0, v___x_2919_);
return v___f_2920_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0(void){
_start:
{
lean_object* v___x_2921_; lean_object* v___f_2922_; 
v___x_2921_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___f_2922_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_2922_, 0, v___x_2921_);
return v___f_2922_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4(void){
_start:
{
lean_object* v___x_2923_; lean_object* v___x_2924_; 
v___x_2923_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__10);
v___x_2924_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_2924_, 0, lean_box(0));
lean_closure_set(v___x_2924_, 1, lean_box(0));
lean_closure_set(v___x_2924_, 2, v___x_2923_);
return v___x_2924_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5(void){
_start:
{
lean_object* v___f_2925_; lean_object* v___x_2926_; lean_object* v___x_2927_; 
v___f_2925_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__0);
v___x_2926_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__4);
v___x_2927_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2927_, 0, v___x_2926_);
lean_ctor_set(v___x_2927_, 1, v___f_2925_);
return v___x_2927_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6(void){
_start:
{
lean_object* v___f_2928_; lean_object* v___f_2929_; lean_object* v___f_2930_; lean_object* v___x_2931_; lean_object* v___x_2932_; lean_object* v___x_2933_; 
v___f_2928_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__3);
v___f_2929_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__2);
v___f_2930_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__1);
v___x_2931_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___x_2932_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__5);
v___x_2933_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2933_, 0, v___x_2932_);
lean_ctor_set(v___x_2933_, 1, v___x_2931_);
lean_ctor_set(v___x_2933_, 2, v___f_2930_);
lean_ctor_set(v___x_2933_, 3, v___f_2929_);
lean_ctor_set(v___x_2933_, 4, v___f_2928_);
return v___x_2933_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8(void){
_start:
{
lean_object* v___x_2934_; lean_object* v___x_2935_; lean_object* v___x_2936_; 
v___x_2934_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__7);
v___x_2935_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__6);
v___x_2936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2936_, 0, v___x_2935_);
lean_ctor_set(v___x_2936_, 1, v___x_2934_);
return v___x_2936_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17(void){
_start:
{
lean_object* v___x_2937_; lean_object* v___x_2938_; 
v___x_2937_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___x_2938_ = lean_alloc_closure((void*)(l_OptionT_bind), 6, 2);
lean_closure_set(v___x_2938_, 0, lean_box(0));
lean_closure_set(v___x_2938_, 1, v___x_2937_);
return v___x_2938_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13(void){
_start:
{
lean_object* v___x_2939_; lean_object* v___f_2940_; 
v___x_2939_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___f_2940_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__11), 5, 1);
lean_closure_set(v___f_2940_, 0, v___x_2939_);
return v___f_2940_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12(void){
_start:
{
lean_object* v___x_2941_; lean_object* v___f_2942_; 
v___x_2941_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___f_2942_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_2942_, 0, v___x_2941_);
return v___f_2942_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11(void){
_start:
{
lean_object* v___x_2943_; lean_object* v___f_2944_; 
v___x_2943_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___f_2944_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__6), 5, 1);
lean_closure_set(v___f_2944_, 0, v___x_2943_);
return v___f_2944_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15(void){
_start:
{
lean_object* v___x_2945_; lean_object* v___x_2946_; 
v___x_2945_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___x_2946_ = lean_alloc_closure((void*)(l_OptionT_pure), 4, 2);
lean_closure_set(v___x_2946_, 0, lean_box(0));
lean_closure_set(v___x_2946_, 1, v___x_2945_);
return v___x_2946_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10(void){
_start:
{
lean_object* v___x_2947_; lean_object* v___f_2948_; 
v___x_2947_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___f_2948_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__3), 5, 1);
lean_closure_set(v___f_2948_, 0, v___x_2947_);
return v___f_2948_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9(void){
_start:
{
lean_object* v___x_2949_; lean_object* v___f_2950_; 
v___x_2949_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___f_2950_ = lean_alloc_closure((void*)(l_OptionT_instMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_2950_, 0, v___x_2949_);
return v___f_2950_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14(void){
_start:
{
lean_object* v___f_2951_; lean_object* v___f_2952_; lean_object* v___x_2953_; 
v___f_2951_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__10);
v___f_2952_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__9);
v___x_2953_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2953_, 0, v___f_2952_);
lean_ctor_set(v___x_2953_, 1, v___f_2951_);
return v___x_2953_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16(void){
_start:
{
lean_object* v___f_2954_; lean_object* v___f_2955_; lean_object* v___f_2956_; lean_object* v___x_2957_; lean_object* v___x_2958_; lean_object* v___x_2959_; 
v___f_2954_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__13);
v___f_2955_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__12);
v___f_2956_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__11);
v___x_2957_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__15);
v___x_2958_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__14);
v___x_2959_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2959_, 0, v___x_2958_);
lean_ctor_set(v___x_2959_, 1, v___x_2957_);
lean_ctor_set(v___x_2959_, 2, v___f_2956_);
lean_ctor_set(v___x_2959_, 3, v___f_2955_);
lean_ctor_set(v___x_2959_, 4, v___f_2954_);
return v___x_2959_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18(void){
_start:
{
lean_object* v___x_2960_; lean_object* v___x_2961_; lean_object* v___x_2962_; 
v___x_2960_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__17);
v___x_2961_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__16);
v___x_2962_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2962_, 0, v___x_2961_);
lean_ctor_set(v___x_2962_, 1, v___x_2960_);
return v___x_2962_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19(void){
_start:
{
lean_object* v___x_2963_; lean_object* v___x_2964_; 
v___x_2963_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__8);
v___x_2964_ = l_OptionT_instAlternative___redArg(v___x_2963_);
return v___x_2964_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___boxed(lean_object** _args){
lean_object* v___x_2968_ = _args[0];
lean_object* v_n_2969_ = _args[1];
lean_object* v_inst_2970_ = _args[2];
lean_object* v_inst_2971_ = _args[3];
lean_object* v_cfg_2972_ = _args[4];
lean_object* v_var_2973_ = _args[5];
lean_object* v_interp_2974_ = _args[6];
lean_object* v___x_2975_ = _args[7];
lean_object* v_traceShrink_2976_ = _args[8];
lean_object* v_proxyRepr_2977_ = _args[9];
lean_object* v_x_2978_ = _args[10];
lean_object* v_toPure_2979_ = _args[11];
lean_object* v_traceShrinkCandidates_2980_ = _args[12];
lean_object* v_a_2981_ = _args[13];
lean_object* v_x_2982_ = _args[14];
lean_object* v___y_2983_ = _args[15];
lean_object* v___y_2984_ = _args[16];
lean_object* v___y_2985_ = _args[17];
_start:
{
uint8_t v_traceShrink_boxed_2986_; uint8_t v_traceShrinkCandidates_boxed_2987_; lean_object* v_res_2988_; 
v_traceShrink_boxed_2986_ = lean_unbox(v_traceShrink_2976_);
v_traceShrinkCandidates_boxed_2987_ = lean_unbox(v_traceShrinkCandidates_2980_);
v_res_2988_ = lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0(v___x_2968_, v_n_2969_, v_inst_2970_, v_inst_2971_, v_cfg_2972_, v_var_2973_, v_interp_2974_, v___x_2975_, v_traceShrink_boxed_2986_, v_proxyRepr_2977_, v_x_2978_, v_toPure_2979_, v_traceShrinkCandidates_boxed_2987_, v_a_2981_, v_x_2982_, v___y_2983_, v___y_2984_, v___y_2985_);
lean_dec(v___y_2985_);
lean_dec_ref(v___y_2983_);
lean_dec(v_n_2969_);
return v_res_2988_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg(lean_object* v_inst_2992_, lean_object* v_inst_2993_, lean_object* v_cfg_2994_, lean_object* v_var_2995_, lean_object* v_x_2996_, lean_object* v_n_2997_, lean_object* v_a_2998_, lean_object* v_a_2999_){
_start:
{
lean_object* v___y_3001_; lean_object* v___y_3002_; lean_object* v___x_3005_; lean_object* v___x_3006_; lean_object* v_toApplicative_3007_; lean_object* v_toPure_3008_; lean_object* v_proxyRepr_3009_; lean_object* v_shrink_3010_; lean_object* v_interp_3011_; uint8_t v_traceShrink_3012_; uint8_t v_traceShrinkCandidates_3013_; lean_object* v_candidates_3014_; lean_object* v___y_3016_; lean_object* v___y_3017_; 
v___x_3005_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__18);
v___x_3006_ = lean_obj_once(&lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19, &lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19_once, _init_lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__19);
v_toApplicative_3007_ = lean_ctor_get(v___x_3006_, 0);
v_toPure_3008_ = lean_ctor_get(v_toApplicative_3007_, 1);
v_proxyRepr_3009_ = lean_ctor_get(v_inst_2992_, 0);
lean_inc_ref(v_proxyRepr_3009_);
v_shrink_3010_ = lean_ctor_get(v_inst_2992_, 1);
v_interp_3011_ = lean_ctor_get(v_inst_2992_, 3);
lean_inc(v_interp_3011_);
v_traceShrink_3012_ = lean_ctor_get_uint8(v_cfg_2994_, sizeof(void*)*4 + 2);
v_traceShrinkCandidates_3013_ = lean_ctor_get_uint8(v_cfg_2994_, sizeof(void*)*4 + 3);
lean_inc_ref(v_shrink_3010_);
lean_inc(v_x_2996_);
v_candidates_3014_ = lean_apply_1(v_shrink_3010_, v_x_2996_);
if (v_traceShrinkCandidates_3013_ == 0)
{
v___y_3016_ = v_a_2998_;
v___y_3017_ = v_a_2999_;
goto v___jp_3015_;
}
else
{
lean_object* v___x_3106_; lean_object* v___x_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; lean_object* v___x_3110_; lean_object* v___x_3111_; lean_object* v___x_3112_; lean_object* v___x_3113_; lean_object* v___x_3114_; lean_object* v___x_3115_; lean_object* v___x_3116_; lean_object* v___x_3117_; lean_object* v___x_3118_; lean_object* v___x_3119_; lean_object* v___x_10046__overap_3120_; lean_object* v___x_3121_; 
v___x_3106_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__22));
v___x_3107_ = lean_string_append(v___x_3106_, v_var_2995_);
v___x_3108_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
v___x_3109_ = lean_string_append(v___x_3107_, v___x_3108_);
v___x_3110_ = lean_unsigned_to_nat(0u);
lean_inc_ref_n(v_proxyRepr_3009_, 2);
lean_inc(v_x_2996_);
v___x_3111_ = lean_apply_2(v_proxyRepr_3009_, v_x_2996_, v___x_3110_);
v___x_3112_ = l_Std_Format_defWidth;
v___x_3113_ = l_Std_Format_pretty(v___x_3111_, v___x_3112_, v___x_3110_, v___x_3110_);
v___x_3114_ = lean_string_append(v___x_3109_, v___x_3113_);
lean_dec_ref(v___x_3113_);
v___x_3115_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__23));
v___x_3116_ = lean_string_append(v___x_3114_, v___x_3115_);
lean_inc(v_candidates_3014_);
v___x_3117_ = l_List_repr___redArg(v_proxyRepr_3009_, v_candidates_3014_);
v___x_3118_ = l_Std_Format_pretty(v___x_3117_, v___x_3112_, v___x_3110_, v___x_3110_);
v___x_3119_ = lean_string_append(v___x_3116_, v___x_3118_);
lean_dec_ref(v___x_3118_);
lean_inc(v_toPure_3008_);
v___x_10046__overap_3120_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_toPure_3008_, v___x_3119_);
lean_dec_ref(v___x_3119_);
lean_inc(v_a_2999_);
v___x_3121_ = lean_apply_2(v___x_10046__overap_3120_, v_a_2998_, v_a_2999_);
if (lean_obj_tag(v___x_3121_) == 0)
{
lean_object* v_a_3122_; lean_object* v___x_3124_; uint8_t v_isShared_3125_; uint8_t v_isSharedCheck_3129_; 
lean_dec(v_candidates_3014_);
lean_dec(v_interp_3011_);
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_n_2997_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
lean_dec_ref(v_cfg_2994_);
lean_dec_ref(v_inst_2993_);
lean_dec_ref(v_inst_2992_);
v_a_3122_ = lean_ctor_get(v___x_3121_, 0);
v_isSharedCheck_3129_ = !lean_is_exclusive(v___x_3121_);
if (v_isSharedCheck_3129_ == 0)
{
v___x_3124_ = v___x_3121_;
v_isShared_3125_ = v_isSharedCheck_3129_;
goto v_resetjp_3123_;
}
else
{
lean_inc(v_a_3122_);
lean_dec(v___x_3121_);
v___x_3124_ = lean_box(0);
v_isShared_3125_ = v_isSharedCheck_3129_;
goto v_resetjp_3123_;
}
v_resetjp_3123_:
{
lean_object* v___x_3127_; 
if (v_isShared_3125_ == 0)
{
v___x_3127_ = v___x_3124_;
goto v_reusejp_3126_;
}
else
{
lean_object* v_reuseFailAlloc_3128_; 
v_reuseFailAlloc_3128_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3128_, 0, v_a_3122_);
v___x_3127_ = v_reuseFailAlloc_3128_;
goto v_reusejp_3126_;
}
v_reusejp_3126_:
{
return v___x_3127_;
}
}
}
else
{
lean_object* v_a_3130_; lean_object* v___x_3132_; uint8_t v_isShared_3133_; uint8_t v_isSharedCheck_3149_; 
v_a_3130_ = lean_ctor_get(v___x_3121_, 0);
v_isSharedCheck_3149_ = !lean_is_exclusive(v___x_3121_);
if (v_isSharedCheck_3149_ == 0)
{
v___x_3132_ = v___x_3121_;
v_isShared_3133_ = v_isSharedCheck_3149_;
goto v_resetjp_3131_;
}
else
{
lean_inc(v_a_3130_);
lean_dec(v___x_3121_);
v___x_3132_ = lean_box(0);
v_isShared_3133_ = v_isSharedCheck_3149_;
goto v_resetjp_3131_;
}
v_resetjp_3131_:
{
lean_object* v_fst_3134_; 
v_fst_3134_ = lean_ctor_get(v_a_3130_, 0);
if (lean_obj_tag(v_fst_3134_) == 0)
{
lean_object* v_snd_3135_; lean_object* v___x_3137_; uint8_t v_isShared_3138_; uint8_t v_isSharedCheck_3146_; 
lean_dec(v_candidates_3014_);
lean_dec(v_interp_3011_);
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_n_2997_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
lean_dec_ref(v_cfg_2994_);
lean_dec_ref(v_inst_2993_);
lean_dec_ref(v_inst_2992_);
v_snd_3135_ = lean_ctor_get(v_a_3130_, 1);
v_isSharedCheck_3146_ = !lean_is_exclusive(v_a_3130_);
if (v_isSharedCheck_3146_ == 0)
{
lean_object* v_unused_3147_; 
v_unused_3147_ = lean_ctor_get(v_a_3130_, 0);
lean_dec(v_unused_3147_);
v___x_3137_ = v_a_3130_;
v_isShared_3138_ = v_isSharedCheck_3146_;
goto v_resetjp_3136_;
}
else
{
lean_inc(v_snd_3135_);
lean_dec(v_a_3130_);
v___x_3137_ = lean_box(0);
v_isShared_3138_ = v_isSharedCheck_3146_;
goto v_resetjp_3136_;
}
v_resetjp_3136_:
{
lean_object* v___x_3139_; lean_object* v___x_3141_; 
v___x_3139_ = lean_box(0);
if (v_isShared_3138_ == 0)
{
lean_ctor_set(v___x_3137_, 0, v___x_3139_);
v___x_3141_ = v___x_3137_;
goto v_reusejp_3140_;
}
else
{
lean_object* v_reuseFailAlloc_3145_; 
v_reuseFailAlloc_3145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3145_, 0, v___x_3139_);
lean_ctor_set(v_reuseFailAlloc_3145_, 1, v_snd_3135_);
v___x_3141_ = v_reuseFailAlloc_3145_;
goto v_reusejp_3140_;
}
v_reusejp_3140_:
{
lean_object* v___x_3143_; 
if (v_isShared_3133_ == 0)
{
lean_ctor_set(v___x_3132_, 0, v___x_3141_);
v___x_3143_ = v___x_3132_;
goto v_reusejp_3142_;
}
else
{
lean_object* v_reuseFailAlloc_3144_; 
v_reuseFailAlloc_3144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3144_, 0, v___x_3141_);
v___x_3143_ = v_reuseFailAlloc_3144_;
goto v_reusejp_3142_;
}
v_reusejp_3142_:
{
return v___x_3143_;
}
}
}
}
else
{
lean_object* v_snd_3148_; 
lean_del_object(v___x_3132_);
v_snd_3148_ = lean_ctor_get(v_a_3130_, 1);
lean_inc(v_snd_3148_);
lean_dec(v_a_3130_);
v___y_3016_ = v_snd_3148_;
v___y_3017_ = v_a_2999_;
goto v___jp_3015_;
}
}
}
}
v___jp_3000_:
{
lean_object* v___x_3003_; lean_object* v___x_3004_; 
v___x_3003_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3003_, 0, v___y_3001_);
lean_ctor_set(v___x_3003_, 1, v___y_3002_);
v___x_3004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3004_, 0, v___x_3003_);
return v___x_3004_;
}
v___jp_3015_:
{
lean_object* v___x_3018_; lean_object* v___x_3019_; lean_object* v___x_3020_; lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___f_3023_; lean_object* v___x_8250__overap_3024_; lean_object* v___x_3025_; 
v___x_3018_ = lean_box(0);
v___x_3019_ = lean_box(0);
v___x_3020_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__20));
v___x_3021_ = lean_box(v_traceShrink_3012_);
v___x_3022_ = lean_box(v_traceShrinkCandidates_3013_);
lean_inc(v_toPure_3008_);
lean_inc(v_x_2996_);
lean_inc_ref(v_proxyRepr_3009_);
lean_inc_ref(v_var_2995_);
v___f_3023_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___boxed), 18, 13);
lean_closure_set(v___f_3023_, 0, v___x_3019_);
lean_closure_set(v___f_3023_, 1, v_n_2997_);
lean_closure_set(v___f_3023_, 2, v_inst_2992_);
lean_closure_set(v___f_3023_, 3, v_inst_2993_);
lean_closure_set(v___f_3023_, 4, v_cfg_2994_);
lean_closure_set(v___f_3023_, 5, v_var_2995_);
lean_closure_set(v___f_3023_, 6, v_interp_3011_);
lean_closure_set(v___f_3023_, 7, v___x_3020_);
lean_closure_set(v___f_3023_, 8, v___x_3021_);
lean_closure_set(v___f_3023_, 9, v_proxyRepr_3009_);
lean_closure_set(v___f_3023_, 10, v_x_2996_);
lean_closure_set(v___f_3023_, 11, v_toPure_3008_);
lean_closure_set(v___f_3023_, 12, v___x_3022_);
v___x_8250__overap_3024_ = l_List_forIn_x27_loop___redArg(v___x_3005_, v___f_3023_, v_candidates_3014_, v___x_3020_);
lean_dec(v_candidates_3014_);
lean_inc(v___y_3017_);
v___x_3025_ = lean_apply_2(v___x_8250__overap_3024_, v___y_3016_, v___y_3017_);
if (lean_obj_tag(v___x_3025_) == 0)
{
lean_object* v_a_3026_; lean_object* v___x_3028_; uint8_t v_isShared_3029_; uint8_t v_isSharedCheck_3033_; 
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
v_a_3026_ = lean_ctor_get(v___x_3025_, 0);
v_isSharedCheck_3033_ = !lean_is_exclusive(v___x_3025_);
if (v_isSharedCheck_3033_ == 0)
{
v___x_3028_ = v___x_3025_;
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
else
{
lean_inc(v_a_3026_);
lean_dec(v___x_3025_);
v___x_3028_ = lean_box(0);
v_isShared_3029_ = v_isSharedCheck_3033_;
goto v_resetjp_3027_;
}
v_resetjp_3027_:
{
lean_object* v___x_3031_; 
if (v_isShared_3029_ == 0)
{
v___x_3031_ = v___x_3028_;
goto v_reusejp_3030_;
}
else
{
lean_object* v_reuseFailAlloc_3032_; 
v_reuseFailAlloc_3032_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3032_, 0, v_a_3026_);
v___x_3031_ = v_reuseFailAlloc_3032_;
goto v_reusejp_3030_;
}
v_reusejp_3030_:
{
return v___x_3031_;
}
}
}
else
{
lean_object* v_a_3034_; lean_object* v___x_3036_; uint8_t v_isShared_3037_; uint8_t v_isSharedCheck_3105_; 
v_a_3034_ = lean_ctor_get(v___x_3025_, 0);
v_isSharedCheck_3105_ = !lean_is_exclusive(v___x_3025_);
if (v_isSharedCheck_3105_ == 0)
{
v___x_3036_ = v___x_3025_;
v_isShared_3037_ = v_isSharedCheck_3105_;
goto v_resetjp_3035_;
}
else
{
lean_inc(v_a_3034_);
lean_dec(v___x_3025_);
v___x_3036_ = lean_box(0);
v_isShared_3037_ = v_isSharedCheck_3105_;
goto v_resetjp_3035_;
}
v_resetjp_3035_:
{
lean_object* v_fst_3038_; 
v_fst_3038_ = lean_ctor_get(v_a_3034_, 0);
if (lean_obj_tag(v_fst_3038_) == 0)
{
lean_object* v_snd_3039_; lean_object* v___x_3041_; uint8_t v_isShared_3042_; uint8_t v_isSharedCheck_3049_; 
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
v_snd_3039_ = lean_ctor_get(v_a_3034_, 1);
v_isSharedCheck_3049_ = !lean_is_exclusive(v_a_3034_);
if (v_isSharedCheck_3049_ == 0)
{
lean_object* v_unused_3050_; 
v_unused_3050_ = lean_ctor_get(v_a_3034_, 0);
lean_dec(v_unused_3050_);
v___x_3041_ = v_a_3034_;
v_isShared_3042_ = v_isSharedCheck_3049_;
goto v_resetjp_3040_;
}
else
{
lean_inc(v_snd_3039_);
lean_dec(v_a_3034_);
v___x_3041_ = lean_box(0);
v_isShared_3042_ = v_isSharedCheck_3049_;
goto v_resetjp_3040_;
}
v_resetjp_3040_:
{
lean_object* v___x_3044_; 
if (v_isShared_3042_ == 0)
{
lean_ctor_set(v___x_3041_, 0, v___x_3018_);
v___x_3044_ = v___x_3041_;
goto v_reusejp_3043_;
}
else
{
lean_object* v_reuseFailAlloc_3048_; 
v_reuseFailAlloc_3048_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3048_, 0, v___x_3018_);
lean_ctor_set(v_reuseFailAlloc_3048_, 1, v_snd_3039_);
v___x_3044_ = v_reuseFailAlloc_3048_;
goto v_reusejp_3043_;
}
v_reusejp_3043_:
{
lean_object* v___x_3046_; 
if (v_isShared_3037_ == 0)
{
lean_ctor_set(v___x_3036_, 0, v___x_3044_);
v___x_3046_ = v___x_3036_;
goto v_reusejp_3045_;
}
else
{
lean_object* v_reuseFailAlloc_3047_; 
v_reuseFailAlloc_3047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3047_, 0, v___x_3044_);
v___x_3046_ = v_reuseFailAlloc_3047_;
goto v_reusejp_3045_;
}
v_reusejp_3045_:
{
return v___x_3046_;
}
}
}
}
else
{
lean_object* v_val_3051_; lean_object* v_fst_3052_; lean_object* v___x_3054_; uint8_t v_isShared_3055_; uint8_t v_isSharedCheck_3103_; 
v_val_3051_ = lean_ctor_get(v_fst_3038_, 0);
lean_inc(v_val_3051_);
v_fst_3052_ = lean_ctor_get(v_val_3051_, 0);
v_isSharedCheck_3103_ = !lean_is_exclusive(v_val_3051_);
if (v_isSharedCheck_3103_ == 0)
{
lean_object* v_unused_3104_; 
v_unused_3104_ = lean_ctor_get(v_val_3051_, 1);
lean_dec(v_unused_3104_);
v___x_3054_ = v_val_3051_;
v_isShared_3055_ = v_isSharedCheck_3103_;
goto v_resetjp_3053_;
}
else
{
lean_inc(v_fst_3052_);
lean_dec(v_val_3051_);
v___x_3054_ = lean_box(0);
v_isShared_3055_ = v_isSharedCheck_3103_;
goto v_resetjp_3053_;
}
v_resetjp_3053_:
{
if (lean_obj_tag(v_fst_3052_) == 0)
{
lean_del_object(v___x_3054_);
lean_del_object(v___x_3036_);
if (v_traceShrink_3012_ == 0)
{
lean_object* v_snd_3056_; 
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
v_snd_3056_ = lean_ctor_get(v_a_3034_, 1);
lean_inc(v_snd_3056_);
lean_dec(v_a_3034_);
v___y_3001_ = v_fst_3052_;
v___y_3002_ = v_snd_3056_;
goto v___jp_3000_;
}
else
{
lean_object* v_snd_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; lean_object* v___x_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; lean_object* v___x_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; lean_object* v___x_3066_; lean_object* v___x_10014__overap_3067_; lean_object* v___x_3068_; 
v_snd_3057_ = lean_ctor_get(v_a_3034_, 1);
lean_inc(v_snd_3057_);
lean_dec(v_a_3034_);
v___x_3058_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___closed__21));
v___x_3059_ = lean_string_append(v___x_3058_, v_var_2995_);
lean_dec_ref(v_var_2995_);
v___x_3060_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
v___x_3061_ = lean_string_append(v___x_3059_, v___x_3060_);
v___x_3062_ = lean_unsigned_to_nat(0u);
v___x_3063_ = lean_apply_2(v_proxyRepr_3009_, v_x_2996_, v___x_3062_);
v___x_3064_ = l_Std_Format_defWidth;
v___x_3065_ = l_Std_Format_pretty(v___x_3063_, v___x_3064_, v___x_3062_, v___x_3062_);
v___x_3066_ = lean_string_append(v___x_3061_, v___x_3065_);
lean_dec_ref(v___x_3065_);
lean_inc(v_toPure_3008_);
v___x_10014__overap_3067_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_toPure_3008_, v___x_3066_);
lean_dec_ref(v___x_3066_);
lean_inc(v___y_3017_);
v___x_3068_ = lean_apply_2(v___x_10014__overap_3067_, v_snd_3057_, v___y_3017_);
if (lean_obj_tag(v___x_3068_) == 0)
{
lean_object* v_a_3069_; lean_object* v___x_3071_; uint8_t v_isShared_3072_; uint8_t v_isSharedCheck_3076_; 
v_a_3069_ = lean_ctor_get(v___x_3068_, 0);
v_isSharedCheck_3076_ = !lean_is_exclusive(v___x_3068_);
if (v_isSharedCheck_3076_ == 0)
{
v___x_3071_ = v___x_3068_;
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
else
{
lean_inc(v_a_3069_);
lean_dec(v___x_3068_);
v___x_3071_ = lean_box(0);
v_isShared_3072_ = v_isSharedCheck_3076_;
goto v_resetjp_3070_;
}
v_resetjp_3070_:
{
lean_object* v___x_3074_; 
if (v_isShared_3072_ == 0)
{
v___x_3074_ = v___x_3071_;
goto v_reusejp_3073_;
}
else
{
lean_object* v_reuseFailAlloc_3075_; 
v_reuseFailAlloc_3075_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3075_, 0, v_a_3069_);
v___x_3074_ = v_reuseFailAlloc_3075_;
goto v_reusejp_3073_;
}
v_reusejp_3073_:
{
return v___x_3074_;
}
}
}
else
{
lean_object* v_a_3077_; lean_object* v___x_3079_; uint8_t v_isShared_3080_; uint8_t v_isSharedCheck_3095_; 
v_a_3077_ = lean_ctor_get(v___x_3068_, 0);
v_isSharedCheck_3095_ = !lean_is_exclusive(v___x_3068_);
if (v_isSharedCheck_3095_ == 0)
{
v___x_3079_ = v___x_3068_;
v_isShared_3080_ = v_isSharedCheck_3095_;
goto v_resetjp_3078_;
}
else
{
lean_inc(v_a_3077_);
lean_dec(v___x_3068_);
v___x_3079_ = lean_box(0);
v_isShared_3080_ = v_isSharedCheck_3095_;
goto v_resetjp_3078_;
}
v_resetjp_3078_:
{
lean_object* v_fst_3081_; 
v_fst_3081_ = lean_ctor_get(v_a_3077_, 0);
if (lean_obj_tag(v_fst_3081_) == 0)
{
lean_object* v_snd_3082_; lean_object* v___x_3084_; uint8_t v_isShared_3085_; uint8_t v_isSharedCheck_3092_; 
v_snd_3082_ = lean_ctor_get(v_a_3077_, 1);
v_isSharedCheck_3092_ = !lean_is_exclusive(v_a_3077_);
if (v_isSharedCheck_3092_ == 0)
{
lean_object* v_unused_3093_; 
v_unused_3093_ = lean_ctor_get(v_a_3077_, 0);
lean_dec(v_unused_3093_);
v___x_3084_ = v_a_3077_;
v_isShared_3085_ = v_isSharedCheck_3092_;
goto v_resetjp_3083_;
}
else
{
lean_inc(v_snd_3082_);
lean_dec(v_a_3077_);
v___x_3084_ = lean_box(0);
v_isShared_3085_ = v_isSharedCheck_3092_;
goto v_resetjp_3083_;
}
v_resetjp_3083_:
{
lean_object* v___x_3087_; 
if (v_isShared_3085_ == 0)
{
lean_ctor_set(v___x_3084_, 0, v_fst_3052_);
v___x_3087_ = v___x_3084_;
goto v_reusejp_3086_;
}
else
{
lean_object* v_reuseFailAlloc_3091_; 
v_reuseFailAlloc_3091_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3091_, 0, v_fst_3052_);
lean_ctor_set(v_reuseFailAlloc_3091_, 1, v_snd_3082_);
v___x_3087_ = v_reuseFailAlloc_3091_;
goto v_reusejp_3086_;
}
v_reusejp_3086_:
{
lean_object* v___x_3089_; 
if (v_isShared_3080_ == 0)
{
lean_ctor_set(v___x_3079_, 0, v___x_3087_);
v___x_3089_ = v___x_3079_;
goto v_reusejp_3088_;
}
else
{
lean_object* v_reuseFailAlloc_3090_; 
v_reuseFailAlloc_3090_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3090_, 0, v___x_3087_);
v___x_3089_ = v_reuseFailAlloc_3090_;
goto v_reusejp_3088_;
}
v_reusejp_3088_:
{
return v___x_3089_;
}
}
}
}
else
{
lean_object* v_snd_3094_; 
lean_del_object(v___x_3079_);
v_snd_3094_ = lean_ctor_get(v_a_3077_, 1);
lean_inc(v_snd_3094_);
lean_dec(v_a_3077_);
v___y_3001_ = v_fst_3052_;
v___y_3002_ = v_snd_3094_;
goto v___jp_3000_;
}
}
}
}
}
else
{
lean_object* v_snd_3096_; lean_object* v___x_3098_; 
lean_dec_ref(v_proxyRepr_3009_);
lean_dec(v_x_2996_);
lean_dec_ref(v_var_2995_);
v_snd_3096_ = lean_ctor_get(v_a_3034_, 1);
lean_inc(v_snd_3096_);
lean_dec(v_a_3034_);
if (v_isShared_3055_ == 0)
{
lean_ctor_set(v___x_3054_, 1, v_snd_3096_);
v___x_3098_ = v___x_3054_;
goto v_reusejp_3097_;
}
else
{
lean_object* v_reuseFailAlloc_3102_; 
v_reuseFailAlloc_3102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3102_, 0, v_fst_3052_);
lean_ctor_set(v_reuseFailAlloc_3102_, 1, v_snd_3096_);
v___x_3098_ = v_reuseFailAlloc_3102_;
goto v_reusejp_3097_;
}
v_reusejp_3097_:
{
lean_object* v___x_3100_; 
if (v_isShared_3037_ == 0)
{
lean_ctor_set(v___x_3036_, 0, v___x_3098_);
v___x_3100_ = v___x_3036_;
goto v_reusejp_3099_;
}
else
{
lean_object* v_reuseFailAlloc_3101_; 
v_reuseFailAlloc_3101_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3101_, 0, v___x_3098_);
v___x_3100_ = v_reuseFailAlloc_3101_;
goto v_reusejp_3099_;
}
v_reusejp_3099_:
{
return v___x_3100_;
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0(lean_object* v___x_3150_, lean_object* v_n_3151_, lean_object* v_inst_3152_, lean_object* v_inst_3153_, lean_object* v_cfg_3154_, lean_object* v_var_3155_, lean_object* v_interp_3156_, lean_object* v___x_3157_, uint8_t v_traceShrink_3158_, lean_object* v_proxyRepr_3159_, lean_object* v_x_3160_, lean_object* v_toPure_3161_, uint8_t v_traceShrinkCandidates_3162_, lean_object* v_a_3163_, lean_object* v_x_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_, lean_object* v___y_3167_){
_start:
{
lean_object* v_fst_3169_; lean_object* v_snd_3170_; lean_object* v___y_3177_; lean_object* v___y_3207_; lean_object* v___y_3208_; lean_object* v___y_3209_; lean_object* v___y_3227_; lean_object* v___y_3228_; 
if (v_traceShrinkCandidates_3162_ == 0)
{
v___y_3227_ = v___y_3166_;
v___y_3228_ = v___y_3167_;
goto v___jp_3226_;
}
else
{
lean_object* v___x_3303_; lean_object* v___x_3304_; lean_object* v___x_3305_; lean_object* v___x_3306_; lean_object* v___x_3307_; lean_object* v___x_3308_; lean_object* v___x_3309_; lean_object* v___x_3310_; lean_object* v___x_3311_; lean_object* v___x_10351__overap_3312_; lean_object* v___x_3313_; 
v___x_3303_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__2));
v___x_3304_ = lean_string_append(v___x_3303_, v_var_3155_);
v___x_3305_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
v___x_3306_ = lean_string_append(v___x_3304_, v___x_3305_);
v___x_3307_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3159_);
lean_inc(v_a_3163_);
v___x_3308_ = lean_apply_2(v_proxyRepr_3159_, v_a_3163_, v___x_3307_);
v___x_3309_ = l_Std_Format_defWidth;
v___x_3310_ = l_Std_Format_pretty(v___x_3308_, v___x_3309_, v___x_3307_, v___x_3307_);
v___x_3311_ = lean_string_append(v___x_3306_, v___x_3310_);
lean_dec_ref(v___x_3310_);
lean_inc(v_toPure_3161_);
v___x_10351__overap_3312_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_toPure_3161_, v___x_3311_);
lean_dec_ref(v___x_3311_);
lean_inc(v___y_3167_);
v___x_3313_ = lean_apply_2(v___x_10351__overap_3312_, v___y_3166_, v___y_3167_);
if (lean_obj_tag(v___x_3313_) == 0)
{
lean_object* v_a_3314_; lean_object* v___x_3316_; uint8_t v_isShared_3317_; uint8_t v_isSharedCheck_3321_; 
lean_dec(v_a_3163_);
lean_dec(v_toPure_3161_);
lean_dec(v_x_3160_);
lean_dec_ref(v_proxyRepr_3159_);
lean_dec_ref(v___x_3157_);
lean_dec(v_interp_3156_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v_a_3314_ = lean_ctor_get(v___x_3313_, 0);
v_isSharedCheck_3321_ = !lean_is_exclusive(v___x_3313_);
if (v_isSharedCheck_3321_ == 0)
{
v___x_3316_ = v___x_3313_;
v_isShared_3317_ = v_isSharedCheck_3321_;
goto v_resetjp_3315_;
}
else
{
lean_inc(v_a_3314_);
lean_dec(v___x_3313_);
v___x_3316_ = lean_box(0);
v_isShared_3317_ = v_isSharedCheck_3321_;
goto v_resetjp_3315_;
}
v_resetjp_3315_:
{
lean_object* v___x_3319_; 
if (v_isShared_3317_ == 0)
{
v___x_3319_ = v___x_3316_;
goto v_reusejp_3318_;
}
else
{
lean_object* v_reuseFailAlloc_3320_; 
v_reuseFailAlloc_3320_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3320_, 0, v_a_3314_);
v___x_3319_ = v_reuseFailAlloc_3320_;
goto v_reusejp_3318_;
}
v_reusejp_3318_:
{
return v___x_3319_;
}
}
}
else
{
lean_object* v_a_3322_; lean_object* v___x_3324_; uint8_t v_isShared_3325_; uint8_t v_isSharedCheck_3341_; 
v_a_3322_ = lean_ctor_get(v___x_3313_, 0);
v_isSharedCheck_3341_ = !lean_is_exclusive(v___x_3313_);
if (v_isSharedCheck_3341_ == 0)
{
v___x_3324_ = v___x_3313_;
v_isShared_3325_ = v_isSharedCheck_3341_;
goto v_resetjp_3323_;
}
else
{
lean_inc(v_a_3322_);
lean_dec(v___x_3313_);
v___x_3324_ = lean_box(0);
v_isShared_3325_ = v_isSharedCheck_3341_;
goto v_resetjp_3323_;
}
v_resetjp_3323_:
{
lean_object* v_fst_3326_; 
v_fst_3326_ = lean_ctor_get(v_a_3322_, 0);
if (lean_obj_tag(v_fst_3326_) == 0)
{
lean_object* v_snd_3327_; lean_object* v___x_3329_; uint8_t v_isShared_3330_; uint8_t v_isSharedCheck_3338_; 
lean_dec(v_a_3163_);
lean_dec(v_toPure_3161_);
lean_dec(v_x_3160_);
lean_dec_ref(v_proxyRepr_3159_);
lean_dec_ref(v___x_3157_);
lean_dec(v_interp_3156_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v_snd_3327_ = lean_ctor_get(v_a_3322_, 1);
v_isSharedCheck_3338_ = !lean_is_exclusive(v_a_3322_);
if (v_isSharedCheck_3338_ == 0)
{
lean_object* v_unused_3339_; 
v_unused_3339_ = lean_ctor_get(v_a_3322_, 0);
lean_dec(v_unused_3339_);
v___x_3329_ = v_a_3322_;
v_isShared_3330_ = v_isSharedCheck_3338_;
goto v_resetjp_3328_;
}
else
{
lean_inc(v_snd_3327_);
lean_dec(v_a_3322_);
v___x_3329_ = lean_box(0);
v_isShared_3330_ = v_isSharedCheck_3338_;
goto v_resetjp_3328_;
}
v_resetjp_3328_:
{
lean_object* v___x_3331_; lean_object* v___x_3333_; 
v___x_3331_ = lean_box(0);
if (v_isShared_3330_ == 0)
{
lean_ctor_set(v___x_3329_, 0, v___x_3331_);
v___x_3333_ = v___x_3329_;
goto v_reusejp_3332_;
}
else
{
lean_object* v_reuseFailAlloc_3337_; 
v_reuseFailAlloc_3337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3337_, 0, v___x_3331_);
lean_ctor_set(v_reuseFailAlloc_3337_, 1, v_snd_3327_);
v___x_3333_ = v_reuseFailAlloc_3337_;
goto v_reusejp_3332_;
}
v_reusejp_3332_:
{
lean_object* v___x_3335_; 
if (v_isShared_3325_ == 0)
{
lean_ctor_set(v___x_3324_, 0, v___x_3333_);
v___x_3335_ = v___x_3324_;
goto v_reusejp_3334_;
}
else
{
lean_object* v_reuseFailAlloc_3336_; 
v_reuseFailAlloc_3336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3336_, 0, v___x_3333_);
v___x_3335_ = v_reuseFailAlloc_3336_;
goto v_reusejp_3334_;
}
v_reusejp_3334_:
{
return v___x_3335_;
}
}
}
}
else
{
lean_object* v_snd_3340_; 
lean_del_object(v___x_3324_);
v_snd_3340_ = lean_ctor_get(v_a_3322_, 1);
lean_inc(v_snd_3340_);
lean_dec(v_a_3322_);
v___y_3227_ = v_snd_3340_;
v___y_3228_ = v___y_3167_;
goto v___jp_3226_;
}
}
}
}
v___jp_3168_:
{
lean_object* v___x_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; lean_object* v___x_3174_; lean_object* v___x_3175_; 
v___x_3171_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3171_, 0, v_fst_3169_);
lean_ctor_set(v___x_3171_, 1, v___x_3150_);
v___x_3172_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3172_, 0, v___x_3171_);
v___x_3173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3173_, 0, v___x_3172_);
v___x_3174_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3174_, 0, v___x_3173_);
lean_ctor_set(v___x_3174_, 1, v_snd_3170_);
v___x_3175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3175_, 0, v___x_3174_);
return v___x_3175_;
}
v___jp_3176_:
{
if (lean_obj_tag(v___y_3177_) == 0)
{
lean_object* v_a_3178_; lean_object* v___x_3180_; uint8_t v_isShared_3181_; uint8_t v_isSharedCheck_3185_; 
v_a_3178_ = lean_ctor_get(v___y_3177_, 0);
v_isSharedCheck_3185_ = !lean_is_exclusive(v___y_3177_);
if (v_isSharedCheck_3185_ == 0)
{
v___x_3180_ = v___y_3177_;
v_isShared_3181_ = v_isSharedCheck_3185_;
goto v_resetjp_3179_;
}
else
{
lean_inc(v_a_3178_);
lean_dec(v___y_3177_);
v___x_3180_ = lean_box(0);
v_isShared_3181_ = v_isSharedCheck_3185_;
goto v_resetjp_3179_;
}
v_resetjp_3179_:
{
lean_object* v___x_3183_; 
if (v_isShared_3181_ == 0)
{
v___x_3183_ = v___x_3180_;
goto v_reusejp_3182_;
}
else
{
lean_object* v_reuseFailAlloc_3184_; 
v_reuseFailAlloc_3184_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3184_, 0, v_a_3178_);
v___x_3183_ = v_reuseFailAlloc_3184_;
goto v_reusejp_3182_;
}
v_reusejp_3182_:
{
return v___x_3183_;
}
}
}
else
{
lean_object* v_a_3186_; lean_object* v___x_3188_; uint8_t v_isShared_3189_; uint8_t v_isSharedCheck_3205_; 
v_a_3186_ = lean_ctor_get(v___y_3177_, 0);
v_isSharedCheck_3205_ = !lean_is_exclusive(v___y_3177_);
if (v_isSharedCheck_3205_ == 0)
{
v___x_3188_ = v___y_3177_;
v_isShared_3189_ = v_isSharedCheck_3205_;
goto v_resetjp_3187_;
}
else
{
lean_inc(v_a_3186_);
lean_dec(v___y_3177_);
v___x_3188_ = lean_box(0);
v_isShared_3189_ = v_isSharedCheck_3205_;
goto v_resetjp_3187_;
}
v_resetjp_3187_:
{
lean_object* v_fst_3190_; 
v_fst_3190_ = lean_ctor_get(v_a_3186_, 0);
if (lean_obj_tag(v_fst_3190_) == 0)
{
lean_object* v_snd_3191_; lean_object* v___x_3193_; uint8_t v_isShared_3194_; uint8_t v_isSharedCheck_3202_; 
v_snd_3191_ = lean_ctor_get(v_a_3186_, 1);
v_isSharedCheck_3202_ = !lean_is_exclusive(v_a_3186_);
if (v_isSharedCheck_3202_ == 0)
{
lean_object* v_unused_3203_; 
v_unused_3203_ = lean_ctor_get(v_a_3186_, 0);
lean_dec(v_unused_3203_);
v___x_3193_ = v_a_3186_;
v_isShared_3194_ = v_isSharedCheck_3202_;
goto v_resetjp_3192_;
}
else
{
lean_inc(v_snd_3191_);
lean_dec(v_a_3186_);
v___x_3193_ = lean_box(0);
v_isShared_3194_ = v_isSharedCheck_3202_;
goto v_resetjp_3192_;
}
v_resetjp_3192_:
{
lean_object* v___x_3195_; lean_object* v___x_3197_; 
v___x_3195_ = lean_box(0);
if (v_isShared_3194_ == 0)
{
lean_ctor_set(v___x_3193_, 0, v___x_3195_);
v___x_3197_ = v___x_3193_;
goto v_reusejp_3196_;
}
else
{
lean_object* v_reuseFailAlloc_3201_; 
v_reuseFailAlloc_3201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3201_, 0, v___x_3195_);
lean_ctor_set(v_reuseFailAlloc_3201_, 1, v_snd_3191_);
v___x_3197_ = v_reuseFailAlloc_3201_;
goto v_reusejp_3196_;
}
v_reusejp_3196_:
{
lean_object* v___x_3199_; 
if (v_isShared_3189_ == 0)
{
lean_ctor_set(v___x_3188_, 0, v___x_3197_);
v___x_3199_ = v___x_3188_;
goto v_reusejp_3198_;
}
else
{
lean_object* v_reuseFailAlloc_3200_; 
v_reuseFailAlloc_3200_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3200_, 0, v___x_3197_);
v___x_3199_ = v_reuseFailAlloc_3200_;
goto v_reusejp_3198_;
}
v_reusejp_3198_:
{
return v___x_3199_;
}
}
}
}
else
{
lean_object* v_snd_3204_; 
lean_inc_ref(v_fst_3190_);
lean_del_object(v___x_3188_);
v_snd_3204_ = lean_ctor_get(v_a_3186_, 1);
lean_inc(v_snd_3204_);
lean_dec(v_a_3186_);
v_fst_3169_ = v_fst_3190_;
v_snd_3170_ = v_snd_3204_;
goto v___jp_3168_;
}
}
}
}
v___jp_3206_:
{
lean_object* v___x_3210_; lean_object* v___x_3211_; lean_object* v___x_3212_; lean_object* v___x_3213_; lean_object* v___x_3214_; 
v___x_3210_ = lean_unsigned_to_nat(1u);
v___x_3211_ = lean_nat_add(v_n_3151_, v___x_3210_);
v___x_3212_ = lp_plausible_Plausible_Testable_addShrinks___redArg(v___x_3211_, v___y_3207_);
lean_inc(v_a_3163_);
v___x_3213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3213_, 0, v_a_3163_);
lean_ctor_set(v___x_3213_, 1, v___x_3212_);
v___x_3214_ = lp_plausible_Plausible_Testable_minimizeAux___redArg(v_inst_3152_, v_inst_3153_, v_cfg_3154_, v_var_3155_, v_a_3163_, v___x_3211_, v___y_3208_, v___y_3209_);
if (lean_obj_tag(v___x_3214_) == 0)
{
lean_dec_ref_known(v___x_3213_, 2);
v___y_3177_ = v___x_3214_;
goto v___jp_3176_;
}
else
{
lean_object* v_a_3215_; lean_object* v_fst_3216_; 
v_a_3215_ = lean_ctor_get(v___x_3214_, 0);
lean_inc(v_a_3215_);
v_fst_3216_ = lean_ctor_get(v_a_3215_, 0);
if (lean_obj_tag(v_fst_3216_) == 0)
{
lean_object* v___x_3218_; uint8_t v_isShared_3219_; uint8_t v_isSharedCheck_3224_; 
v_isSharedCheck_3224_ = !lean_is_exclusive(v___x_3214_);
if (v_isSharedCheck_3224_ == 0)
{
lean_object* v_unused_3225_; 
v_unused_3225_ = lean_ctor_get(v___x_3214_, 0);
lean_dec(v_unused_3225_);
v___x_3218_ = v___x_3214_;
v_isShared_3219_ = v_isSharedCheck_3224_;
goto v_resetjp_3217_;
}
else
{
lean_dec(v___x_3214_);
v___x_3218_ = lean_box(0);
v_isShared_3219_ = v_isSharedCheck_3224_;
goto v_resetjp_3217_;
}
v_resetjp_3217_:
{
lean_object* v_snd_3220_; lean_object* v___x_3222_; 
v_snd_3220_ = lean_ctor_get(v_a_3215_, 1);
lean_inc(v_snd_3220_);
lean_dec(v_a_3215_);
if (v_isShared_3219_ == 0)
{
lean_ctor_set(v___x_3218_, 0, v___x_3213_);
v___x_3222_ = v___x_3218_;
goto v_reusejp_3221_;
}
else
{
lean_object* v_reuseFailAlloc_3223_; 
v_reuseFailAlloc_3223_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3223_, 0, v___x_3213_);
v___x_3222_ = v_reuseFailAlloc_3223_;
goto v_reusejp_3221_;
}
v_reusejp_3221_:
{
v_fst_3169_ = v___x_3222_;
v_snd_3170_ = v_snd_3220_;
goto v___jp_3168_;
}
}
}
else
{
lean_dec(v_a_3215_);
lean_dec_ref_known(v___x_3213_, 2);
v___y_3177_ = v___x_3214_;
goto v___jp_3176_;
}
}
}
v___jp_3226_:
{
lean_object* v___x_3229_; uint8_t v___x_3230_; lean_object* v___x_3231_; lean_object* v___x_3232_; 
lean_inc(v_a_3163_);
v___x_3229_ = lean_apply_1(v_interp_3156_, v_a_3163_);
v___x_3230_ = 1;
v___x_3231_ = lean_box(v___x_3230_);
lean_inc_ref(v_inst_3153_);
lean_inc(v___y_3228_);
lean_inc_ref(v_cfg_3154_);
v___x_3232_ = lean_apply_5(v_inst_3153_, v___x_3229_, v_cfg_3154_, v___x_3231_, v___y_3227_, v___y_3228_);
if (lean_obj_tag(v___x_3232_) == 0)
{
lean_object* v_a_3233_; lean_object* v___x_3235_; uint8_t v_isShared_3236_; uint8_t v_isSharedCheck_3240_; 
lean_dec(v_a_3163_);
lean_dec(v_toPure_3161_);
lean_dec(v_x_3160_);
lean_dec_ref(v_proxyRepr_3159_);
lean_dec_ref(v___x_3157_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v_a_3233_ = lean_ctor_get(v___x_3232_, 0);
v_isSharedCheck_3240_ = !lean_is_exclusive(v___x_3232_);
if (v_isSharedCheck_3240_ == 0)
{
v___x_3235_ = v___x_3232_;
v_isShared_3236_ = v_isSharedCheck_3240_;
goto v_resetjp_3234_;
}
else
{
lean_inc(v_a_3233_);
lean_dec(v___x_3232_);
v___x_3235_ = lean_box(0);
v_isShared_3236_ = v_isSharedCheck_3240_;
goto v_resetjp_3234_;
}
v_resetjp_3234_:
{
lean_object* v___x_3238_; 
if (v_isShared_3236_ == 0)
{
v___x_3238_ = v___x_3235_;
goto v_reusejp_3237_;
}
else
{
lean_object* v_reuseFailAlloc_3239_; 
v_reuseFailAlloc_3239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3239_, 0, v_a_3233_);
v___x_3238_ = v_reuseFailAlloc_3239_;
goto v_reusejp_3237_;
}
v_reusejp_3237_:
{
return v___x_3238_;
}
}
}
else
{
lean_object* v_a_3241_; lean_object* v___x_3243_; uint8_t v_isShared_3244_; uint8_t v_isSharedCheck_3302_; 
v_a_3241_ = lean_ctor_get(v___x_3232_, 0);
v_isSharedCheck_3302_ = !lean_is_exclusive(v___x_3232_);
if (v_isSharedCheck_3302_ == 0)
{
v___x_3243_ = v___x_3232_;
v_isShared_3244_ = v_isSharedCheck_3302_;
goto v_resetjp_3242_;
}
else
{
lean_inc(v_a_3241_);
lean_dec(v___x_3232_);
v___x_3243_ = lean_box(0);
v_isShared_3244_ = v_isSharedCheck_3302_;
goto v_resetjp_3242_;
}
v_resetjp_3242_:
{
lean_object* v_fst_3245_; lean_object* v_snd_3246_; lean_object* v___x_3248_; uint8_t v_isShared_3249_; uint8_t v_isSharedCheck_3301_; 
v_fst_3245_ = lean_ctor_get(v_a_3241_, 0);
v_snd_3246_ = lean_ctor_get(v_a_3241_, 1);
v_isSharedCheck_3301_ = !lean_is_exclusive(v_a_3241_);
if (v_isSharedCheck_3301_ == 0)
{
v___x_3248_ = v_a_3241_;
v_isShared_3249_ = v_isSharedCheck_3301_;
goto v_resetjp_3247_;
}
else
{
lean_inc(v_snd_3246_);
lean_inc(v_fst_3245_);
lean_dec(v_a_3241_);
v___x_3248_ = lean_box(0);
v_isShared_3249_ = v_isSharedCheck_3301_;
goto v_resetjp_3247_;
}
v_resetjp_3247_:
{
uint8_t v___x_3250_; 
v___x_3250_ = lp_plausible_Plausible_TestResult_isFailure___redArg(v_fst_3245_);
if (v___x_3250_ == 0)
{
lean_object* v___x_3251_; lean_object* v___x_3252_; lean_object* v___x_3254_; 
lean_dec(v_fst_3245_);
lean_dec(v_a_3163_);
lean_dec(v_toPure_3161_);
lean_dec(v_x_3160_);
lean_dec_ref(v_proxyRepr_3159_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v___x_3251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3251_, 0, v___x_3157_);
v___x_3252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3252_, 0, v___x_3251_);
if (v_isShared_3249_ == 0)
{
lean_ctor_set(v___x_3248_, 0, v___x_3252_);
v___x_3254_ = v___x_3248_;
goto v_reusejp_3253_;
}
else
{
lean_object* v_reuseFailAlloc_3258_; 
v_reuseFailAlloc_3258_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3258_, 0, v___x_3252_);
lean_ctor_set(v_reuseFailAlloc_3258_, 1, v_snd_3246_);
v___x_3254_ = v_reuseFailAlloc_3258_;
goto v_reusejp_3253_;
}
v_reusejp_3253_:
{
lean_object* v___x_3256_; 
if (v_isShared_3244_ == 0)
{
lean_ctor_set(v___x_3243_, 0, v___x_3254_);
v___x_3256_ = v___x_3243_;
goto v_reusejp_3255_;
}
else
{
lean_object* v_reuseFailAlloc_3257_; 
v_reuseFailAlloc_3257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3257_, 0, v___x_3254_);
v___x_3256_ = v_reuseFailAlloc_3257_;
goto v_reusejp_3255_;
}
v_reusejp_3255_:
{
return v___x_3256_;
}
}
}
else
{
lean_del_object(v___x_3248_);
lean_del_object(v___x_3243_);
lean_dec_ref(v___x_3157_);
if (v_traceShrink_3158_ == 0)
{
lean_dec(v_toPure_3161_);
lean_dec(v_x_3160_);
lean_dec_ref(v_proxyRepr_3159_);
v___y_3207_ = v_fst_3245_;
v___y_3208_ = v_snd_3246_;
v___y_3209_ = v___y_3228_;
goto v___jp_3206_;
}
else
{
lean_object* v___x_3259_; lean_object* v___x_3260_; lean_object* v___x_3261_; lean_object* v___x_3262_; lean_object* v___x_3263_; lean_object* v___x_3264_; lean_object* v___x_3265_; lean_object* v___x_3266_; lean_object* v___x_3267_; lean_object* v___x_3268_; lean_object* v___x_3269_; lean_object* v___x_3270_; lean_object* v___x_10332__overap_3271_; lean_object* v___x_3272_; 
v___x_3259_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__0));
lean_inc_ref(v_var_3155_);
v___x_3260_ = lean_string_append(v_var_3155_, v___x_3259_);
v___x_3261_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3159_);
lean_inc(v_a_3163_);
v___x_3262_ = lean_apply_2(v_proxyRepr_3159_, v_a_3163_, v___x_3261_);
v___x_3263_ = l_Std_Format_defWidth;
v___x_3264_ = l_Std_Format_pretty(v___x_3262_, v___x_3263_, v___x_3261_, v___x_3261_);
v___x_3265_ = lean_string_append(v___x_3260_, v___x_3264_);
lean_dec_ref(v___x_3264_);
v___x_3266_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimizeAux___redArg___lam__0___closed__1));
v___x_3267_ = lean_string_append(v___x_3265_, v___x_3266_);
v___x_3268_ = lean_apply_2(v_proxyRepr_3159_, v_x_3160_, v___x_3261_);
v___x_3269_ = l_Std_Format_pretty(v___x_3268_, v___x_3263_, v___x_3261_, v___x_3261_);
v___x_3270_ = lean_string_append(v___x_3267_, v___x_3269_);
lean_dec_ref(v___x_3269_);
v___x_10332__overap_3271_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v_toPure_3161_, v___x_3270_);
lean_dec_ref(v___x_3270_);
lean_inc(v___y_3228_);
v___x_3272_ = lean_apply_2(v___x_10332__overap_3271_, v_snd_3246_, v___y_3228_);
if (lean_obj_tag(v___x_3272_) == 0)
{
lean_object* v_a_3273_; lean_object* v___x_3275_; uint8_t v_isShared_3276_; uint8_t v_isSharedCheck_3280_; 
lean_dec(v_fst_3245_);
lean_dec(v_a_3163_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v_a_3273_ = lean_ctor_get(v___x_3272_, 0);
v_isSharedCheck_3280_ = !lean_is_exclusive(v___x_3272_);
if (v_isSharedCheck_3280_ == 0)
{
v___x_3275_ = v___x_3272_;
v_isShared_3276_ = v_isSharedCheck_3280_;
goto v_resetjp_3274_;
}
else
{
lean_inc(v_a_3273_);
lean_dec(v___x_3272_);
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
v_reuseFailAlloc_3279_ = lean_alloc_ctor(0, 1, 0);
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
else
{
lean_object* v_a_3281_; lean_object* v___x_3283_; uint8_t v_isShared_3284_; uint8_t v_isSharedCheck_3300_; 
v_a_3281_ = lean_ctor_get(v___x_3272_, 0);
v_isSharedCheck_3300_ = !lean_is_exclusive(v___x_3272_);
if (v_isSharedCheck_3300_ == 0)
{
v___x_3283_ = v___x_3272_;
v_isShared_3284_ = v_isSharedCheck_3300_;
goto v_resetjp_3282_;
}
else
{
lean_inc(v_a_3281_);
lean_dec(v___x_3272_);
v___x_3283_ = lean_box(0);
v_isShared_3284_ = v_isSharedCheck_3300_;
goto v_resetjp_3282_;
}
v_resetjp_3282_:
{
lean_object* v_fst_3285_; 
v_fst_3285_ = lean_ctor_get(v_a_3281_, 0);
if (lean_obj_tag(v_fst_3285_) == 0)
{
lean_object* v_snd_3286_; lean_object* v___x_3288_; uint8_t v_isShared_3289_; uint8_t v_isSharedCheck_3297_; 
lean_dec(v_fst_3245_);
lean_dec(v_a_3163_);
lean_dec_ref(v_var_3155_);
lean_dec_ref(v_cfg_3154_);
lean_dec_ref(v_inst_3153_);
lean_dec_ref(v_inst_3152_);
v_snd_3286_ = lean_ctor_get(v_a_3281_, 1);
v_isSharedCheck_3297_ = !lean_is_exclusive(v_a_3281_);
if (v_isSharedCheck_3297_ == 0)
{
lean_object* v_unused_3298_; 
v_unused_3298_ = lean_ctor_get(v_a_3281_, 0);
lean_dec(v_unused_3298_);
v___x_3288_ = v_a_3281_;
v_isShared_3289_ = v_isSharedCheck_3297_;
goto v_resetjp_3287_;
}
else
{
lean_inc(v_snd_3286_);
lean_dec(v_a_3281_);
v___x_3288_ = lean_box(0);
v_isShared_3289_ = v_isSharedCheck_3297_;
goto v_resetjp_3287_;
}
v_resetjp_3287_:
{
lean_object* v___x_3290_; lean_object* v___x_3292_; 
v___x_3290_ = lean_box(0);
if (v_isShared_3289_ == 0)
{
lean_ctor_set(v___x_3288_, 0, v___x_3290_);
v___x_3292_ = v___x_3288_;
goto v_reusejp_3291_;
}
else
{
lean_object* v_reuseFailAlloc_3296_; 
v_reuseFailAlloc_3296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3296_, 0, v___x_3290_);
lean_ctor_set(v_reuseFailAlloc_3296_, 1, v_snd_3286_);
v___x_3292_ = v_reuseFailAlloc_3296_;
goto v_reusejp_3291_;
}
v_reusejp_3291_:
{
lean_object* v___x_3294_; 
if (v_isShared_3284_ == 0)
{
lean_ctor_set(v___x_3283_, 0, v___x_3292_);
v___x_3294_ = v___x_3283_;
goto v_reusejp_3293_;
}
else
{
lean_object* v_reuseFailAlloc_3295_; 
v_reuseFailAlloc_3295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3295_, 0, v___x_3292_);
v___x_3294_ = v_reuseFailAlloc_3295_;
goto v_reusejp_3293_;
}
v_reusejp_3293_:
{
return v___x_3294_;
}
}
}
}
else
{
lean_object* v_snd_3299_; 
lean_del_object(v___x_3283_);
v_snd_3299_ = lean_ctor_get(v_a_3281_, 1);
lean_inc(v_snd_3299_);
lean_dec(v_a_3281_);
v___y_3207_ = v_fst_3245_;
v___y_3208_ = v_snd_3299_;
v___y_3209_ = v___y_3228_;
goto v___jp_3206_;
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___redArg___boxed(lean_object* v_inst_3342_, lean_object* v_inst_3343_, lean_object* v_cfg_3344_, lean_object* v_var_3345_, lean_object* v_x_3346_, lean_object* v_n_3347_, lean_object* v_a_3348_, lean_object* v_a_3349_){
_start:
{
lean_object* v_res_3350_; 
v_res_3350_ = lp_plausible_Plausible_Testable_minimizeAux___redArg(v_inst_3342_, v_inst_3343_, v_cfg_3344_, v_var_3345_, v_x_3346_, v_n_3347_, v_a_3348_, v_a_3349_);
lean_dec(v_a_3349_);
return v_res_3350_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux(lean_object* v_00_u03b1_3351_, lean_object* v_inst_3352_, lean_object* v_00_u03b2_3353_, lean_object* v_inst_3354_, lean_object* v_cfg_3355_, lean_object* v_var_3356_, lean_object* v_x_3357_, lean_object* v_n_3358_, lean_object* v_a_3359_, lean_object* v_a_3360_){
_start:
{
lean_object* v___x_3361_; 
v___x_3361_ = lp_plausible_Plausible_Testable_minimizeAux___redArg(v_inst_3352_, v_inst_3354_, v_cfg_3355_, v_var_3356_, v_x_3357_, v_n_3358_, v_a_3359_, v_a_3360_);
return v___x_3361_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimizeAux___boxed(lean_object* v_00_u03b1_3362_, lean_object* v_inst_3363_, lean_object* v_00_u03b2_3364_, lean_object* v_inst_3365_, lean_object* v_cfg_3366_, lean_object* v_var_3367_, lean_object* v_x_3368_, lean_object* v_n_3369_, lean_object* v_a_3370_, lean_object* v_a_3371_){
_start:
{
lean_object* v_res_3372_; 
v_res_3372_ = lp_plausible_Plausible_Testable_minimizeAux(v_00_u03b1_3362_, v_inst_3363_, v_00_u03b2_3364_, v_inst_3365_, v_cfg_3366_, v_var_3367_, v_x_3368_, v_n_3369_, v_a_3370_, v_a_3371_);
lean_dec(v_a_3371_);
return v_res_3372_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___redArg(lean_object* v_inst_3375_, lean_object* v_inst_3376_, lean_object* v_cfg_3377_, lean_object* v_var_3378_, lean_object* v_x_3379_, lean_object* v_r_3380_, lean_object* v_a_3381_, lean_object* v_a_3382_){
_start:
{
lean_object* v___y_3384_; lean_object* v___y_3385_; lean_object* v___y_3389_; lean_object* v___y_3390_; uint8_t v_traceShrink_3414_; 
v_traceShrink_3414_ = lean_ctor_get_uint8(v_cfg_3377_, sizeof(void*)*4 + 2);
if (v_traceShrink_3414_ == 0)
{
v___y_3389_ = v_a_3381_;
v___y_3390_ = v_a_3382_;
goto v___jp_3388_;
}
else
{
lean_object* v___x_3415_; lean_object* v___x_3416_; lean_object* v___x_1048__overap_3417_; lean_object* v___x_3418_; 
v___x_3415_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___x_3416_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimize___redArg___closed__0));
v___x_1048__overap_3417_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3415_, v___x_3416_);
lean_inc(v_a_3382_);
v___x_3418_ = lean_apply_2(v___x_1048__overap_3417_, v_a_3381_, v_a_3382_);
if (lean_obj_tag(v___x_3418_) == 0)
{
lean_object* v_a_3419_; lean_object* v___x_3421_; uint8_t v_isShared_3422_; uint8_t v_isSharedCheck_3426_; 
lean_dec_ref(v_r_3380_);
lean_dec(v_x_3379_);
lean_dec_ref(v_var_3378_);
lean_dec_ref(v_cfg_3377_);
lean_dec_ref(v_inst_3376_);
lean_dec_ref(v_inst_3375_);
v_a_3419_ = lean_ctor_get(v___x_3418_, 0);
v_isSharedCheck_3426_ = !lean_is_exclusive(v___x_3418_);
if (v_isSharedCheck_3426_ == 0)
{
v___x_3421_ = v___x_3418_;
v_isShared_3422_ = v_isSharedCheck_3426_;
goto v_resetjp_3420_;
}
else
{
lean_inc(v_a_3419_);
lean_dec(v___x_3418_);
v___x_3421_ = lean_box(0);
v_isShared_3422_ = v_isSharedCheck_3426_;
goto v_resetjp_3420_;
}
v_resetjp_3420_:
{
lean_object* v___x_3424_; 
if (v_isShared_3422_ == 0)
{
v___x_3424_ = v___x_3421_;
goto v_reusejp_3423_;
}
else
{
lean_object* v_reuseFailAlloc_3425_; 
v_reuseFailAlloc_3425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3425_, 0, v_a_3419_);
v___x_3424_ = v_reuseFailAlloc_3425_;
goto v_reusejp_3423_;
}
v_reusejp_3423_:
{
return v___x_3424_;
}
}
}
else
{
lean_object* v_a_3427_; lean_object* v_snd_3428_; lean_object* v_proxyRepr_3429_; lean_object* v___x_3430_; lean_object* v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3433_; lean_object* v___x_3434_; lean_object* v___x_3435_; lean_object* v___x_3436_; lean_object* v___x_3437_; lean_object* v___x_3438_; lean_object* v___x_1066__overap_3439_; lean_object* v___x_3440_; 
v_a_3427_ = lean_ctor_get(v___x_3418_, 0);
lean_inc(v_a_3427_);
lean_dec_ref_known(v___x_3418_, 1);
v_snd_3428_ = lean_ctor_get(v_a_3427_, 1);
lean_inc(v_snd_3428_);
lean_dec(v_a_3427_);
v_proxyRepr_3429_ = lean_ctor_get(v_inst_3375_, 0);
v___x_3430_ = ((lean_object*)(lp_plausible_Plausible_Testable_minimize___redArg___closed__1));
v___x_3431_ = lean_string_append(v___x_3430_, v_var_3378_);
v___x_3432_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
v___x_3433_ = lean_string_append(v___x_3431_, v___x_3432_);
v___x_3434_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3429_);
lean_inc(v_x_3379_);
v___x_3435_ = lean_apply_2(v_proxyRepr_3429_, v_x_3379_, v___x_3434_);
v___x_3436_ = l_Std_Format_defWidth;
v___x_3437_ = l_Std_Format_pretty(v___x_3435_, v___x_3436_, v___x_3434_, v___x_3434_);
v___x_3438_ = lean_string_append(v___x_3433_, v___x_3437_);
lean_dec_ref(v___x_3437_);
v___x_1066__overap_3439_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3415_, v___x_3438_);
lean_dec_ref(v___x_3438_);
lean_inc(v_a_3382_);
v___x_3440_ = lean_apply_2(v___x_1066__overap_3439_, v_snd_3428_, v_a_3382_);
if (lean_obj_tag(v___x_3440_) == 0)
{
lean_object* v_a_3441_; lean_object* v___x_3443_; uint8_t v_isShared_3444_; uint8_t v_isSharedCheck_3448_; 
lean_dec_ref(v_r_3380_);
lean_dec(v_x_3379_);
lean_dec_ref(v_var_3378_);
lean_dec_ref(v_cfg_3377_);
lean_dec_ref(v_inst_3376_);
lean_dec_ref(v_inst_3375_);
v_a_3441_ = lean_ctor_get(v___x_3440_, 0);
v_isSharedCheck_3448_ = !lean_is_exclusive(v___x_3440_);
if (v_isSharedCheck_3448_ == 0)
{
v___x_3443_ = v___x_3440_;
v_isShared_3444_ = v_isSharedCheck_3448_;
goto v_resetjp_3442_;
}
else
{
lean_inc(v_a_3441_);
lean_dec(v___x_3440_);
v___x_3443_ = lean_box(0);
v_isShared_3444_ = v_isSharedCheck_3448_;
goto v_resetjp_3442_;
}
v_resetjp_3442_:
{
lean_object* v___x_3446_; 
if (v_isShared_3444_ == 0)
{
v___x_3446_ = v___x_3443_;
goto v_reusejp_3445_;
}
else
{
lean_object* v_reuseFailAlloc_3447_; 
v_reuseFailAlloc_3447_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3447_, 0, v_a_3441_);
v___x_3446_ = v_reuseFailAlloc_3447_;
goto v_reusejp_3445_;
}
v_reusejp_3445_:
{
return v___x_3446_;
}
}
}
else
{
lean_object* v_a_3449_; lean_object* v_snd_3450_; 
v_a_3449_ = lean_ctor_get(v___x_3440_, 0);
lean_inc(v_a_3449_);
lean_dec_ref_known(v___x_3440_, 1);
v_snd_3450_ = lean_ctor_get(v_a_3449_, 1);
lean_inc(v_snd_3450_);
lean_dec(v_a_3449_);
v___y_3389_ = v_snd_3450_;
v___y_3390_ = v_a_3382_;
goto v___jp_3388_;
}
}
}
v___jp_3383_:
{
lean_object* v___x_3386_; lean_object* v___x_3387_; 
v___x_3386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3386_, 0, v___y_3385_);
lean_ctor_set(v___x_3386_, 1, v___y_3384_);
v___x_3387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3387_, 0, v___x_3386_);
return v___x_3387_;
}
v___jp_3388_:
{
lean_object* v___x_3391_; lean_object* v___x_3392_; 
v___x_3391_ = lean_unsigned_to_nat(0u);
lean_inc(v_x_3379_);
v___x_3392_ = lp_plausible_Plausible_Testable_minimizeAux___redArg(v_inst_3375_, v_inst_3376_, v_cfg_3377_, v_var_3378_, v_x_3379_, v___x_3391_, v___y_3389_, v___y_3390_);
if (lean_obj_tag(v___x_3392_) == 0)
{
lean_object* v_a_3393_; lean_object* v___x_3395_; uint8_t v_isShared_3396_; uint8_t v_isSharedCheck_3400_; 
lean_dec_ref(v_r_3380_);
lean_dec(v_x_3379_);
v_a_3393_ = lean_ctor_get(v___x_3392_, 0);
v_isSharedCheck_3400_ = !lean_is_exclusive(v___x_3392_);
if (v_isSharedCheck_3400_ == 0)
{
v___x_3395_ = v___x_3392_;
v_isShared_3396_ = v_isSharedCheck_3400_;
goto v_resetjp_3394_;
}
else
{
lean_inc(v_a_3393_);
lean_dec(v___x_3392_);
v___x_3395_ = lean_box(0);
v_isShared_3396_ = v_isSharedCheck_3400_;
goto v_resetjp_3394_;
}
v_resetjp_3394_:
{
lean_object* v___x_3398_; 
if (v_isShared_3396_ == 0)
{
v___x_3398_ = v___x_3395_;
goto v_reusejp_3397_;
}
else
{
lean_object* v_reuseFailAlloc_3399_; 
v_reuseFailAlloc_3399_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3399_, 0, v_a_3393_);
v___x_3398_ = v_reuseFailAlloc_3399_;
goto v_reusejp_3397_;
}
v_reusejp_3397_:
{
return v___x_3398_;
}
}
}
else
{
lean_object* v_a_3401_; lean_object* v_fst_3402_; 
v_a_3401_ = lean_ctor_get(v___x_3392_, 0);
lean_inc(v_a_3401_);
lean_dec_ref_known(v___x_3392_, 1);
v_fst_3402_ = lean_ctor_get(v_a_3401_, 0);
if (lean_obj_tag(v_fst_3402_) == 0)
{
lean_object* v_snd_3403_; lean_object* v___x_3405_; uint8_t v_isShared_3406_; uint8_t v_isSharedCheck_3410_; 
v_snd_3403_ = lean_ctor_get(v_a_3401_, 1);
v_isSharedCheck_3410_ = !lean_is_exclusive(v_a_3401_);
if (v_isSharedCheck_3410_ == 0)
{
lean_object* v_unused_3411_; 
v_unused_3411_ = lean_ctor_get(v_a_3401_, 0);
lean_dec(v_unused_3411_);
v___x_3405_ = v_a_3401_;
v_isShared_3406_ = v_isSharedCheck_3410_;
goto v_resetjp_3404_;
}
else
{
lean_inc(v_snd_3403_);
lean_dec(v_a_3401_);
v___x_3405_ = lean_box(0);
v_isShared_3406_ = v_isSharedCheck_3410_;
goto v_resetjp_3404_;
}
v_resetjp_3404_:
{
lean_object* v___x_3408_; 
if (v_isShared_3406_ == 0)
{
lean_ctor_set(v___x_3405_, 1, v_r_3380_);
lean_ctor_set(v___x_3405_, 0, v_x_3379_);
v___x_3408_ = v___x_3405_;
goto v_reusejp_3407_;
}
else
{
lean_object* v_reuseFailAlloc_3409_; 
v_reuseFailAlloc_3409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3409_, 0, v_x_3379_);
lean_ctor_set(v_reuseFailAlloc_3409_, 1, v_r_3380_);
v___x_3408_ = v_reuseFailAlloc_3409_;
goto v_reusejp_3407_;
}
v_reusejp_3407_:
{
v___y_3384_ = v_snd_3403_;
v___y_3385_ = v___x_3408_;
goto v___jp_3383_;
}
}
}
else
{
lean_object* v_snd_3412_; lean_object* v_val_3413_; 
lean_inc_ref(v_fst_3402_);
lean_dec_ref(v_r_3380_);
lean_dec(v_x_3379_);
v_snd_3412_ = lean_ctor_get(v_a_3401_, 1);
lean_inc(v_snd_3412_);
lean_dec(v_a_3401_);
v_val_3413_ = lean_ctor_get(v_fst_3402_, 0);
lean_inc(v_val_3413_);
lean_dec_ref_known(v_fst_3402_, 1);
v___y_3384_ = v_snd_3412_;
v___y_3385_ = v_val_3413_;
goto v___jp_3383_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___redArg___boxed(lean_object* v_inst_3451_, lean_object* v_inst_3452_, lean_object* v_cfg_3453_, lean_object* v_var_3454_, lean_object* v_x_3455_, lean_object* v_r_3456_, lean_object* v_a_3457_, lean_object* v_a_3458_){
_start:
{
lean_object* v_res_3459_; 
v_res_3459_ = lp_plausible_Plausible_Testable_minimize___redArg(v_inst_3451_, v_inst_3452_, v_cfg_3453_, v_var_3454_, v_x_3455_, v_r_3456_, v_a_3457_, v_a_3458_);
lean_dec(v_a_3458_);
return v_res_3459_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize(lean_object* v_00_u03b1_3460_, lean_object* v_inst_3461_, lean_object* v_00_u03b2_3462_, lean_object* v_inst_3463_, lean_object* v_cfg_3464_, lean_object* v_var_3465_, lean_object* v_x_3466_, lean_object* v_r_3467_, lean_object* v_a_3468_, lean_object* v_a_3469_){
_start:
{
lean_object* v___x_3470_; 
v___x_3470_ = lp_plausible_Plausible_Testable_minimize___redArg(v_inst_3461_, v_inst_3463_, v_cfg_3464_, v_var_3465_, v_x_3466_, v_r_3467_, v_a_3468_, v_a_3469_);
return v___x_3470_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_minimize___boxed(lean_object* v_00_u03b1_3471_, lean_object* v_inst_3472_, lean_object* v_00_u03b2_3473_, lean_object* v_inst_3474_, lean_object* v_cfg_3475_, lean_object* v_var_3476_, lean_object* v_x_3477_, lean_object* v_r_3478_, lean_object* v_a_3479_, lean_object* v_a_3480_){
_start:
{
lean_object* v_res_3481_; 
v_res_3481_ = lp_plausible_Plausible_Testable_minimize(v_00_u03b1_3471_, v_inst_3472_, v_00_u03b2_3473_, v_inst_3474_, v_cfg_3475_, v_var_3476_, v_x_3477_, v_r_3478_, v_a_3479_, v_a_3480_);
lean_dec(v_a_3480_);
return v_res_3481_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg___lam__0(lean_object* v_sample_3483_, lean_object* v_proxyRepr_3484_, lean_object* v_var_3485_, lean_object* v_inst_3486_, lean_object* v_inst_3487_, lean_object* v_interp_3488_, lean_object* v___x_3489_, lean_object* v_cfg_3490_, uint8_t v_min_3491_, lean_object* v___y_3492_, lean_object* v___y_3493_){
_start:
{
lean_object* v_fst_3495_; lean_object* v_snd_3496_; lean_object* v___y_3497_; lean_object* v___x_3502_; 
lean_inc(v___y_3493_);
v___x_3502_ = lean_apply_2(v_sample_3483_, v___y_3492_, v___y_3493_);
if (lean_obj_tag(v___x_3502_) == 0)
{
lean_object* v_a_3503_; lean_object* v___x_3505_; uint8_t v_isShared_3506_; uint8_t v_isSharedCheck_3510_; 
lean_dec_ref(v_cfg_3490_);
lean_dec(v___x_3489_);
lean_dec(v_interp_3488_);
lean_dec_ref(v_inst_3487_);
lean_dec_ref(v_inst_3486_);
lean_dec_ref(v_var_3485_);
lean_dec_ref(v_proxyRepr_3484_);
v_a_3503_ = lean_ctor_get(v___x_3502_, 0);
v_isSharedCheck_3510_ = !lean_is_exclusive(v___x_3502_);
if (v_isSharedCheck_3510_ == 0)
{
v___x_3505_ = v___x_3502_;
v_isShared_3506_ = v_isSharedCheck_3510_;
goto v_resetjp_3504_;
}
else
{
lean_inc(v_a_3503_);
lean_dec(v___x_3502_);
v___x_3505_ = lean_box(0);
v_isShared_3506_ = v_isSharedCheck_3510_;
goto v_resetjp_3504_;
}
v_resetjp_3504_:
{
lean_object* v___x_3508_; 
if (v_isShared_3506_ == 0)
{
v___x_3508_ = v___x_3505_;
goto v_reusejp_3507_;
}
else
{
lean_object* v_reuseFailAlloc_3509_; 
v_reuseFailAlloc_3509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3509_, 0, v_a_3503_);
v___x_3508_ = v_reuseFailAlloc_3509_;
goto v_reusejp_3507_;
}
v_reusejp_3507_:
{
return v___x_3508_;
}
}
}
else
{
lean_object* v_a_3511_; lean_object* v_fst_3512_; lean_object* v_snd_3513_; lean_object* v___y_3515_; lean_object* v___y_3516_; lean_object* v___y_3517_; uint8_t v_traceDiscarded_3532_; uint8_t v_traceSuccesses_3533_; lean_object* v___y_3535_; lean_object* v___y_3536_; 
v_a_3511_ = lean_ctor_get(v___x_3502_, 0);
lean_inc(v_a_3511_);
lean_dec_ref_known(v___x_3502_, 1);
v_fst_3512_ = lean_ctor_get(v_a_3511_, 0);
lean_inc(v_fst_3512_);
v_snd_3513_ = lean_ctor_get(v_a_3511_, 1);
lean_inc(v_snd_3513_);
lean_dec(v_a_3511_);
v_traceDiscarded_3532_ = lean_ctor_get_uint8(v_cfg_3490_, sizeof(void*)*4);
v_traceSuccesses_3533_ = lean_ctor_get_uint8(v_cfg_3490_, sizeof(void*)*4 + 1);
if (v_traceSuccesses_3533_ == 0)
{
if (v_traceDiscarded_3532_ == 0)
{
v___y_3535_ = v_snd_3513_;
v___y_3536_ = v___y_3493_;
goto v___jp_3534_;
}
else
{
goto v___jp_3566_;
}
}
else
{
goto v___jp_3566_;
}
v___jp_3514_:
{
if (v_min_3491_ == 0)
{
lean_dec_ref(v_cfg_3490_);
lean_dec_ref(v_inst_3487_);
lean_dec_ref(v_inst_3486_);
v_fst_3495_ = v_fst_3512_;
v_snd_3496_ = v___y_3515_;
v___y_3497_ = v___y_3516_;
goto v___jp_3494_;
}
else
{
lean_object* v___x_3518_; 
lean_inc_ref(v_var_3485_);
v___x_3518_ = lp_plausible_Plausible_Testable_minimize___redArg(v_inst_3486_, v_inst_3487_, v_cfg_3490_, v_var_3485_, v_fst_3512_, v___y_3515_, v___y_3516_, v___y_3517_);
if (lean_obj_tag(v___x_3518_) == 0)
{
lean_object* v_a_3519_; lean_object* v___x_3521_; uint8_t v_isShared_3522_; uint8_t v_isSharedCheck_3526_; 
lean_dec_ref(v_var_3485_);
lean_dec_ref(v_proxyRepr_3484_);
v_a_3519_ = lean_ctor_get(v___x_3518_, 0);
v_isSharedCheck_3526_ = !lean_is_exclusive(v___x_3518_);
if (v_isSharedCheck_3526_ == 0)
{
v___x_3521_ = v___x_3518_;
v_isShared_3522_ = v_isSharedCheck_3526_;
goto v_resetjp_3520_;
}
else
{
lean_inc(v_a_3519_);
lean_dec(v___x_3518_);
v___x_3521_ = lean_box(0);
v_isShared_3522_ = v_isSharedCheck_3526_;
goto v_resetjp_3520_;
}
v_resetjp_3520_:
{
lean_object* v___x_3524_; 
if (v_isShared_3522_ == 0)
{
v___x_3524_ = v___x_3521_;
goto v_reusejp_3523_;
}
else
{
lean_object* v_reuseFailAlloc_3525_; 
v_reuseFailAlloc_3525_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3525_, 0, v_a_3519_);
v___x_3524_ = v_reuseFailAlloc_3525_;
goto v_reusejp_3523_;
}
v_reusejp_3523_:
{
return v___x_3524_;
}
}
}
else
{
lean_object* v_a_3527_; lean_object* v_fst_3528_; lean_object* v_snd_3529_; lean_object* v_fst_3530_; lean_object* v_snd_3531_; 
v_a_3527_ = lean_ctor_get(v___x_3518_, 0);
lean_inc(v_a_3527_);
lean_dec_ref_known(v___x_3518_, 1);
v_fst_3528_ = lean_ctor_get(v_a_3527_, 0);
lean_inc(v_fst_3528_);
v_snd_3529_ = lean_ctor_get(v_a_3527_, 1);
lean_inc(v_snd_3529_);
lean_dec(v_a_3527_);
v_fst_3530_ = lean_ctor_get(v_fst_3528_, 0);
lean_inc(v_fst_3530_);
v_snd_3531_ = lean_ctor_get(v_fst_3528_, 1);
lean_inc(v_snd_3531_);
lean_dec(v_fst_3528_);
v_fst_3495_ = v_fst_3530_;
v_snd_3496_ = v_snd_3531_;
v___y_3497_ = v_snd_3529_;
goto v___jp_3494_;
}
}
}
v___jp_3534_:
{
lean_object* v___x_3537_; lean_object* v___x_3538_; uint8_t v___x_3539_; lean_object* v___x_3540_; lean_object* v_a_3541_; lean_object* v_fst_3542_; lean_object* v_snd_3543_; uint8_t v___x_3544_; 
lean_inc(v_fst_3512_);
v___x_3537_ = lean_apply_1(v_interp_3488_, v_fst_3512_);
lean_inc_ref(v_inst_3487_);
v___x_3538_ = lean_apply_1(v_inst_3487_, v___x_3537_);
v___x_3539_ = 0;
lean_inc_ref(v_cfg_3490_);
v___x_3540_ = lp_plausible_Plausible_Testable_runPropE___redArg(v___x_3538_, v_cfg_3490_, v___x_3539_, v___y_3535_, v___y_3536_);
v_a_3541_ = lean_ctor_get(v___x_3540_, 0);
lean_inc(v_a_3541_);
lean_dec_ref(v___x_3540_);
v_fst_3542_ = lean_ctor_get(v_a_3541_, 0);
lean_inc(v_fst_3542_);
v_snd_3543_ = lean_ctor_get(v_a_3541_, 1);
lean_inc(v_snd_3543_);
lean_dec(v_a_3541_);
v___x_3544_ = lp_plausible_Plausible_TestResult_isFailure___redArg(v_fst_3542_);
if (v___x_3544_ == 0)
{
lean_dec_ref(v_cfg_3490_);
lean_dec(v___x_3489_);
lean_dec_ref(v_inst_3487_);
lean_dec_ref(v_inst_3486_);
v_fst_3495_ = v_fst_3512_;
v_snd_3496_ = v_fst_3542_;
v___y_3497_ = v_snd_3543_;
goto v___jp_3494_;
}
else
{
if (v_traceSuccesses_3533_ == 0)
{
lean_dec(v___x_3489_);
v___y_3515_ = v_fst_3542_;
v___y_3516_ = v_snd_3543_;
v___y_3517_ = v___y_3536_;
goto v___jp_3514_;
}
else
{
lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3547_; lean_object* v___x_3548_; lean_object* v___x_3549_; lean_object* v___x_3550_; lean_object* v___x_3551_; lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3077__overap_3554_; lean_object* v___x_3555_; 
v___x_3545_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
lean_inc_ref(v_var_3485_);
v___x_3546_ = lean_string_append(v_var_3485_, v___x_3545_);
v___x_3547_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3484_);
lean_inc(v_fst_3512_);
v___x_3548_ = lean_apply_2(v_proxyRepr_3484_, v_fst_3512_, v___x_3547_);
v___x_3549_ = l_Std_Format_defWidth;
v___x_3550_ = l_Std_Format_pretty(v___x_3548_, v___x_3549_, v___x_3547_, v___x_3547_);
v___x_3551_ = lean_string_append(v___x_3546_, v___x_3550_);
lean_dec_ref(v___x_3550_);
v___x_3552_ = ((lean_object*)(lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___closed__0));
v___x_3553_ = lean_string_append(v___x_3551_, v___x_3552_);
v___x_3077__overap_3554_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3489_, v___x_3553_);
lean_dec_ref(v___x_3553_);
lean_inc(v___y_3536_);
v___x_3555_ = lean_apply_2(v___x_3077__overap_3554_, v_snd_3543_, v___y_3536_);
if (lean_obj_tag(v___x_3555_) == 0)
{
lean_object* v_a_3556_; lean_object* v___x_3558_; uint8_t v_isShared_3559_; uint8_t v_isSharedCheck_3563_; 
lean_dec(v_fst_3542_);
lean_dec(v_fst_3512_);
lean_dec_ref(v_cfg_3490_);
lean_dec_ref(v_inst_3487_);
lean_dec_ref(v_inst_3486_);
lean_dec_ref(v_var_3485_);
lean_dec_ref(v_proxyRepr_3484_);
v_a_3556_ = lean_ctor_get(v___x_3555_, 0);
v_isSharedCheck_3563_ = !lean_is_exclusive(v___x_3555_);
if (v_isSharedCheck_3563_ == 0)
{
v___x_3558_ = v___x_3555_;
v_isShared_3559_ = v_isSharedCheck_3563_;
goto v_resetjp_3557_;
}
else
{
lean_inc(v_a_3556_);
lean_dec(v___x_3555_);
v___x_3558_ = lean_box(0);
v_isShared_3559_ = v_isSharedCheck_3563_;
goto v_resetjp_3557_;
}
v_resetjp_3557_:
{
lean_object* v___x_3561_; 
if (v_isShared_3559_ == 0)
{
v___x_3561_ = v___x_3558_;
goto v_reusejp_3560_;
}
else
{
lean_object* v_reuseFailAlloc_3562_; 
v_reuseFailAlloc_3562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3562_, 0, v_a_3556_);
v___x_3561_ = v_reuseFailAlloc_3562_;
goto v_reusejp_3560_;
}
v_reusejp_3560_:
{
return v___x_3561_;
}
}
}
else
{
lean_object* v_a_3564_; lean_object* v_snd_3565_; 
v_a_3564_ = lean_ctor_get(v___x_3555_, 0);
lean_inc(v_a_3564_);
lean_dec_ref_known(v___x_3555_, 1);
v_snd_3565_ = lean_ctor_get(v_a_3564_, 1);
lean_inc(v_snd_3565_);
lean_dec(v_a_3564_);
v___y_3515_ = v_fst_3542_;
v___y_3516_ = v_snd_3565_;
v___y_3517_ = v___y_3536_;
goto v___jp_3514_;
}
}
}
}
v___jp_3566_:
{
lean_object* v___x_3567_; lean_object* v___x_3568_; lean_object* v___x_3569_; lean_object* v___x_3570_; lean_object* v___x_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; lean_object* v___x_3091__overap_3574_; lean_object* v___x_3575_; 
v___x_3567_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
lean_inc_ref(v_var_3485_);
v___x_3568_ = lean_string_append(v_var_3485_, v___x_3567_);
v___x_3569_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3484_);
lean_inc(v_fst_3512_);
v___x_3570_ = lean_apply_2(v_proxyRepr_3484_, v_fst_3512_, v___x_3569_);
v___x_3571_ = l_Std_Format_defWidth;
v___x_3572_ = l_Std_Format_pretty(v___x_3570_, v___x_3571_, v___x_3569_, v___x_3569_);
v___x_3573_ = lean_string_append(v___x_3568_, v___x_3572_);
lean_dec_ref(v___x_3572_);
lean_inc(v___x_3489_);
v___x_3091__overap_3574_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3489_, v___x_3573_);
lean_dec_ref(v___x_3573_);
lean_inc(v___y_3493_);
v___x_3575_ = lean_apply_2(v___x_3091__overap_3574_, v_snd_3513_, v___y_3493_);
if (lean_obj_tag(v___x_3575_) == 0)
{
lean_object* v_a_3576_; lean_object* v___x_3578_; uint8_t v_isShared_3579_; uint8_t v_isSharedCheck_3583_; 
lean_dec(v_fst_3512_);
lean_dec_ref(v_cfg_3490_);
lean_dec(v___x_3489_);
lean_dec(v_interp_3488_);
lean_dec_ref(v_inst_3487_);
lean_dec_ref(v_inst_3486_);
lean_dec_ref(v_var_3485_);
lean_dec_ref(v_proxyRepr_3484_);
v_a_3576_ = lean_ctor_get(v___x_3575_, 0);
v_isSharedCheck_3583_ = !lean_is_exclusive(v___x_3575_);
if (v_isSharedCheck_3583_ == 0)
{
v___x_3578_ = v___x_3575_;
v_isShared_3579_ = v_isSharedCheck_3583_;
goto v_resetjp_3577_;
}
else
{
lean_inc(v_a_3576_);
lean_dec(v___x_3575_);
v___x_3578_ = lean_box(0);
v_isShared_3579_ = v_isSharedCheck_3583_;
goto v_resetjp_3577_;
}
v_resetjp_3577_:
{
lean_object* v___x_3581_; 
if (v_isShared_3579_ == 0)
{
v___x_3581_ = v___x_3578_;
goto v_reusejp_3580_;
}
else
{
lean_object* v_reuseFailAlloc_3582_; 
v_reuseFailAlloc_3582_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3582_, 0, v_a_3576_);
v___x_3581_ = v_reuseFailAlloc_3582_;
goto v_reusejp_3580_;
}
v_reusejp_3580_:
{
return v___x_3581_;
}
}
}
else
{
lean_object* v_a_3584_; lean_object* v_snd_3585_; 
v_a_3584_ = lean_ctor_get(v___x_3575_, 0);
lean_inc(v_a_3584_);
lean_dec_ref_known(v___x_3575_, 1);
v_snd_3585_ = lean_ctor_get(v_a_3584_, 1);
lean_inc(v_snd_3585_);
lean_dec(v_a_3584_);
v___y_3535_ = v_snd_3585_;
v___y_3536_ = v___y_3493_;
goto v___jp_3534_;
}
}
}
v___jp_3494_:
{
lean_object* v___x_3498_; lean_object* v___x_3499_; lean_object* v___x_3500_; lean_object* v___x_3501_; 
v___x_3498_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_3499_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v_proxyRepr_3484_, v_var_3485_, v_fst_3495_, v_snd_3496_, v___x_3498_);
v___x_3500_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3500_, 0, v___x_3499_);
lean_ctor_set(v___x_3500_, 1, v___y_3497_);
v___x_3501_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3501_, 0, v___x_3500_);
return v___x_3501_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___boxed(lean_object* v_sample_3586_, lean_object* v_proxyRepr_3587_, lean_object* v_var_3588_, lean_object* v_inst_3589_, lean_object* v_inst_3590_, lean_object* v_interp_3591_, lean_object* v___x_3592_, lean_object* v_cfg_3593_, lean_object* v_min_3594_, lean_object* v___y_3595_, lean_object* v___y_3596_){
_start:
{
uint8_t v_min_boxed_3597_; lean_object* v_res_3598_; 
v_min_boxed_3597_ = lean_unbox(v_min_3594_);
v_res_3598_ = lp_plausible_Plausible_Testable_varTestable___redArg___lam__0(v_sample_3586_, v_proxyRepr_3587_, v_var_3588_, v_inst_3589_, v_inst_3590_, v_interp_3591_, v___x_3592_, v_cfg_3593_, v_min_boxed_3597_, v___y_3595_, v___y_3596_);
lean_dec(v___y_3596_);
return v_res_3598_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable___redArg(lean_object* v_var_3599_, lean_object* v_inst_3600_, lean_object* v_inst_3601_){
_start:
{
lean_object* v_proxyRepr_3602_; lean_object* v_sample_3603_; lean_object* v_interp_3604_; lean_object* v___x_3605_; lean_object* v___f_3606_; 
v_proxyRepr_3602_ = lean_ctor_get(v_inst_3600_, 0);
lean_inc_ref(v_proxyRepr_3602_);
v_sample_3603_ = lean_ctor_get(v_inst_3600_, 2);
lean_inc_ref(v_sample_3603_);
v_interp_3604_ = lean_ctor_get(v_inst_3600_, 3);
lean_inc(v_interp_3604_);
v___x_3605_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___f_3606_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___boxed), 11, 7);
lean_closure_set(v___f_3606_, 0, v_sample_3603_);
lean_closure_set(v___f_3606_, 1, v_proxyRepr_3602_);
lean_closure_set(v___f_3606_, 2, v_var_3599_);
lean_closure_set(v___f_3606_, 3, v_inst_3600_);
lean_closure_set(v___f_3606_, 4, v_inst_3601_);
lean_closure_set(v___f_3606_, 5, v_interp_3604_);
lean_closure_set(v___f_3606_, 6, v___x_3605_);
return v___f_3606_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_varTestable(lean_object* v_var_3607_, lean_object* v_00_u03b1_3608_, lean_object* v_inst_3609_, lean_object* v_00_u03b2_3610_, lean_object* v_inst_3611_){
_start:
{
lean_object* v___x_3612_; 
v___x_3612_ = lp_plausible_Plausible_Testable_varTestable___redArg(v_var_3607_, v_inst_3609_, v_inst_3611_);
return v___x_3612_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0(lean_object* v_var_3613_, lean_object* v___x_3614_, lean_object* v_inst_3615_, lean_object* v_cfg_3616_, uint8_t v_min_3617_, lean_object* v___y_3618_, lean_object* v___y_3619_){
_start:
{
lean_object* v___x_334__overap_3620_; lean_object* v___x_3621_; lean_object* v___x_3622_; 
v___x_334__overap_3620_ = lp_plausible_Plausible_Testable_varTestable___redArg(v_var_3613_, v___x_3614_, v_inst_3615_);
v___x_3621_ = lean_box(v_min_3617_);
lean_inc(v___y_3619_);
v___x_3622_ = lean_apply_4(v___x_334__overap_3620_, v_cfg_3616_, v___x_3621_, v___y_3618_, v___y_3619_);
if (lean_obj_tag(v___x_3622_) == 0)
{
return v___x_3622_;
}
else
{
lean_object* v_a_3623_; lean_object* v___x_3625_; uint8_t v_isShared_3626_; uint8_t v_isSharedCheck_3641_; 
v_a_3623_ = lean_ctor_get(v___x_3622_, 0);
v_isSharedCheck_3641_ = !lean_is_exclusive(v___x_3622_);
if (v_isSharedCheck_3641_ == 0)
{
v___x_3625_ = v___x_3622_;
v_isShared_3626_ = v_isSharedCheck_3641_;
goto v_resetjp_3624_;
}
else
{
lean_inc(v_a_3623_);
lean_dec(v___x_3622_);
v___x_3625_ = lean_box(0);
v_isShared_3626_ = v_isSharedCheck_3641_;
goto v_resetjp_3624_;
}
v_resetjp_3624_:
{
lean_object* v_fst_3627_; lean_object* v_snd_3628_; lean_object* v___x_3630_; uint8_t v_isShared_3631_; uint8_t v_isSharedCheck_3640_; 
v_fst_3627_ = lean_ctor_get(v_a_3623_, 0);
v_snd_3628_ = lean_ctor_get(v_a_3623_, 1);
v_isSharedCheck_3640_ = !lean_is_exclusive(v_a_3623_);
if (v_isSharedCheck_3640_ == 0)
{
v___x_3630_ = v_a_3623_;
v_isShared_3631_ = v_isSharedCheck_3640_;
goto v_resetjp_3629_;
}
else
{
lean_inc(v_snd_3628_);
lean_inc(v_fst_3627_);
lean_dec(v_a_3623_);
v___x_3630_ = lean_box(0);
v_isShared_3631_ = v_isSharedCheck_3640_;
goto v_resetjp_3629_;
}
v_resetjp_3629_:
{
lean_object* v___x_3632_; lean_object* v___x_3633_; lean_object* v___x_3635_; 
v___x_3632_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_3633_ = lp_plausible_Plausible_TestResult_imp___redArg(v_fst_3627_, v___x_3632_);
if (v_isShared_3631_ == 0)
{
lean_ctor_set(v___x_3630_, 0, v___x_3633_);
v___x_3635_ = v___x_3630_;
goto v_reusejp_3634_;
}
else
{
lean_object* v_reuseFailAlloc_3639_; 
v_reuseFailAlloc_3639_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3639_, 0, v___x_3633_);
lean_ctor_set(v_reuseFailAlloc_3639_, 1, v_snd_3628_);
v___x_3635_ = v_reuseFailAlloc_3639_;
goto v_reusejp_3634_;
}
v_reusejp_3634_:
{
lean_object* v___x_3637_; 
if (v_isShared_3626_ == 0)
{
lean_ctor_set(v___x_3625_, 0, v___x_3635_);
v___x_3637_ = v___x_3625_;
goto v_reusejp_3636_;
}
else
{
lean_object* v_reuseFailAlloc_3638_; 
v_reuseFailAlloc_3638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3638_, 0, v___x_3635_);
v___x_3637_ = v_reuseFailAlloc_3638_;
goto v_reusejp_3636_;
}
v_reusejp_3636_:
{
return v___x_3637_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0___boxed(lean_object* v_var_3642_, lean_object* v___x_3643_, lean_object* v_inst_3644_, lean_object* v_cfg_3645_, lean_object* v_min_3646_, lean_object* v___y_3647_, lean_object* v___y_3648_){
_start:
{
uint8_t v_min_boxed_3649_; lean_object* v_res_3650_; 
v_min_boxed_3649_ = lean_unbox(v_min_3646_);
v_res_3650_ = lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0(v_var_3642_, v___x_3643_, v_inst_3644_, v_cfg_3645_, v_min_boxed_3649_, v___y_3647_, v___y_3648_);
lean_dec(v___y_3648_);
return v_res_3650_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2(void){
_start:
{
lean_object* v___x_3653_; lean_object* v___f_3654_; lean_object* v___x_3655_; lean_object* v___x_3656_; 
v___x_3653_ = lp_plausible_Plausible_Bool_Arbitrary;
v___f_3654_ = ((lean_object*)(lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__1));
v___x_3655_ = ((lean_object*)(lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__0));
v___x_3656_ = lp_plausible_Plausible_SampleableExt_selfContained___redArg(v___x_3655_, v___f_3654_, v___x_3653_);
return v___x_3656_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable___redArg(lean_object* v_var_3657_, lean_object* v_inst_3658_){
_start:
{
lean_object* v___x_3659_; lean_object* v___f_3660_; 
v___x_3659_ = lean_obj_once(&lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2, &lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2_once, _init_lp_plausible_Plausible_Testable_propVarTestable___redArg___closed__2);
v___f_3660_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_propVarTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_3660_, 0, v_var_3657_);
lean_closure_set(v___f_3660_, 1, v___x_3659_);
lean_closure_set(v___f_3660_, 2, v_inst_3658_);
return v___f_3660_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_propVarTestable(lean_object* v_var_3661_, lean_object* v_00_u03b2_3662_, lean_object* v_inst_3663_){
_start:
{
lean_object* v___x_3664_; 
v___x_3664_ = lp_plausible_Plausible_Testable_propVarTestable___redArg(v_var_3661_, v_inst_3663_);
return v___x_3664_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0(lean_object* v_inst_3667_, lean_object* v_var_3668_, lean_object* v___x_3669_, lean_object* v_cfg_3670_, uint8_t v_min_3671_, lean_object* v___y_3672_, lean_object* v___y_3673_){
_start:
{
lean_object* v___y_3675_; lean_object* v___y_3676_; uint8_t v_traceDiscarded_3717_; 
v_traceDiscarded_3717_ = lean_ctor_get_uint8(v_cfg_3670_, sizeof(void*)*4);
if (v_traceDiscarded_3717_ == 0)
{
uint8_t v_traceSuccesses_3718_; 
v_traceSuccesses_3718_ = lean_ctor_get_uint8(v_cfg_3670_, sizeof(void*)*4 + 1);
if (v_traceSuccesses_3718_ == 0)
{
lean_dec(v___x_3669_);
v___y_3675_ = v___y_3672_;
v___y_3676_ = v___y_3673_;
goto v___jp_3674_;
}
else
{
goto v___jp_3702_;
}
}
else
{
goto v___jp_3702_;
}
v___jp_3674_:
{
lean_object* v___x_3677_; lean_object* v___x_3678_; 
v___x_3677_ = lean_box(v_min_3671_);
lean_inc(v___y_3676_);
v___x_3678_ = lean_apply_4(v_inst_3667_, v_cfg_3670_, v___x_3677_, v___y_3675_, v___y_3676_);
if (lean_obj_tag(v___x_3678_) == 0)
{
lean_dec_ref(v_var_3668_);
return v___x_3678_;
}
else
{
lean_object* v_a_3679_; lean_object* v___x_3681_; uint8_t v_isShared_3682_; uint8_t v_isSharedCheck_3701_; 
v_a_3679_ = lean_ctor_get(v___x_3678_, 0);
v_isSharedCheck_3701_ = !lean_is_exclusive(v___x_3678_);
if (v_isSharedCheck_3701_ == 0)
{
v___x_3681_ = v___x_3678_;
v_isShared_3682_ = v_isSharedCheck_3701_;
goto v_resetjp_3680_;
}
else
{
lean_inc(v_a_3679_);
lean_dec(v___x_3678_);
v___x_3681_ = lean_box(0);
v_isShared_3682_ = v_isSharedCheck_3701_;
goto v_resetjp_3680_;
}
v_resetjp_3680_:
{
lean_object* v_fst_3683_; lean_object* v_snd_3684_; lean_object* v___x_3686_; uint8_t v_isShared_3687_; uint8_t v_isSharedCheck_3700_; 
v_fst_3683_ = lean_ctor_get(v_a_3679_, 0);
v_snd_3684_ = lean_ctor_get(v_a_3679_, 1);
v_isSharedCheck_3700_ = !lean_is_exclusive(v_a_3679_);
if (v_isSharedCheck_3700_ == 0)
{
v___x_3686_ = v_a_3679_;
v_isShared_3687_ = v_isSharedCheck_3700_;
goto v_resetjp_3685_;
}
else
{
lean_inc(v_snd_3684_);
lean_inc(v_fst_3683_);
lean_dec(v_a_3679_);
v___x_3686_ = lean_box(0);
v_isShared_3687_ = v_isSharedCheck_3700_;
goto v_resetjp_3685_;
}
v_resetjp_3685_:
{
lean_object* v___x_3688_; lean_object* v___x_3689_; lean_object* v___x_3690_; lean_object* v___x_3691_; lean_object* v___x_3692_; lean_object* v___x_3693_; lean_object* v___x_3695_; 
v___x_3688_ = ((lean_object*)(lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__0));
v___x_3689_ = lean_string_append(v_var_3668_, v___x_3688_);
v___x_3690_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_3691_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v___x_3689_, v_fst_3683_, v___x_3690_);
v___x_3692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3692_, 0, lean_box(0));
v___x_3693_ = lp_plausible_Plausible_TestResult_imp___redArg(v___x_3691_, v___x_3692_);
lean_dec_ref_known(v___x_3692_, 1);
if (v_isShared_3687_ == 0)
{
lean_ctor_set(v___x_3686_, 0, v___x_3693_);
v___x_3695_ = v___x_3686_;
goto v_reusejp_3694_;
}
else
{
lean_object* v_reuseFailAlloc_3699_; 
v_reuseFailAlloc_3699_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3699_, 0, v___x_3693_);
lean_ctor_set(v_reuseFailAlloc_3699_, 1, v_snd_3684_);
v___x_3695_ = v_reuseFailAlloc_3699_;
goto v_reusejp_3694_;
}
v_reusejp_3694_:
{
lean_object* v___x_3697_; 
if (v_isShared_3682_ == 0)
{
lean_ctor_set(v___x_3681_, 0, v___x_3695_);
v___x_3697_ = v___x_3681_;
goto v_reusejp_3696_;
}
else
{
lean_object* v_reuseFailAlloc_3698_; 
v_reuseFailAlloc_3698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3698_, 0, v___x_3695_);
v___x_3697_ = v_reuseFailAlloc_3698_;
goto v_reusejp_3696_;
}
v_reusejp_3696_:
{
return v___x_3697_;
}
}
}
}
}
}
v___jp_3702_:
{
lean_object* v___x_3703_; lean_object* v___x_3704_; lean_object* v___x_851__overap_3705_; lean_object* v___x_3706_; 
v___x_3703_ = ((lean_object*)(lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___closed__1));
lean_inc_ref(v_var_3668_);
v___x_3704_ = lean_string_append(v_var_3668_, v___x_3703_);
v___x_851__overap_3705_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3669_, v___x_3704_);
lean_dec_ref(v___x_3704_);
lean_inc(v___y_3673_);
v___x_3706_ = lean_apply_2(v___x_851__overap_3705_, v___y_3672_, v___y_3673_);
if (lean_obj_tag(v___x_3706_) == 0)
{
lean_object* v_a_3707_; lean_object* v___x_3709_; uint8_t v_isShared_3710_; uint8_t v_isSharedCheck_3714_; 
lean_dec_ref(v_cfg_3670_);
lean_dec_ref(v_var_3668_);
lean_dec_ref(v_inst_3667_);
v_a_3707_ = lean_ctor_get(v___x_3706_, 0);
v_isSharedCheck_3714_ = !lean_is_exclusive(v___x_3706_);
if (v_isSharedCheck_3714_ == 0)
{
v___x_3709_ = v___x_3706_;
v_isShared_3710_ = v_isSharedCheck_3714_;
goto v_resetjp_3708_;
}
else
{
lean_inc(v_a_3707_);
lean_dec(v___x_3706_);
v___x_3709_ = lean_box(0);
v_isShared_3710_ = v_isSharedCheck_3714_;
goto v_resetjp_3708_;
}
v_resetjp_3708_:
{
lean_object* v___x_3712_; 
if (v_isShared_3710_ == 0)
{
v___x_3712_ = v___x_3709_;
goto v_reusejp_3711_;
}
else
{
lean_object* v_reuseFailAlloc_3713_; 
v_reuseFailAlloc_3713_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3713_, 0, v_a_3707_);
v___x_3712_ = v_reuseFailAlloc_3713_;
goto v_reusejp_3711_;
}
v_reusejp_3711_:
{
return v___x_3712_;
}
}
}
else
{
lean_object* v_a_3715_; lean_object* v_snd_3716_; 
v_a_3715_ = lean_ctor_get(v___x_3706_, 0);
lean_inc(v_a_3715_);
lean_dec_ref_known(v___x_3706_, 1);
v_snd_3716_ = lean_ctor_get(v_a_3715_, 1);
lean_inc(v_snd_3716_);
lean_dec(v_a_3715_);
v___y_3675_ = v_snd_3716_;
v___y_3676_ = v___y_3673_;
goto v___jp_3674_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___boxed(lean_object* v_inst_3719_, lean_object* v_var_3720_, lean_object* v___x_3721_, lean_object* v_cfg_3722_, lean_object* v_min_3723_, lean_object* v___y_3724_, lean_object* v___y_3725_){
_start:
{
uint8_t v_min_boxed_3726_; lean_object* v_res_3727_; 
v_min_boxed_3726_ = lean_unbox(v_min_3723_);
v_res_3727_ = lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0(v_inst_3719_, v_var_3720_, v___x_3721_, v_cfg_3722_, v_min_boxed_3726_, v___y_3724_, v___y_3725_);
lean_dec(v___y_3725_);
return v_res_3727_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable___redArg(lean_object* v_var_3728_, lean_object* v_inst_3729_){
_start:
{
lean_object* v___x_3730_; lean_object* v___f_3731_; 
v___x_3730_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___f_3731_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_unusedVarTestable___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_3731_, 0, v_inst_3729_);
lean_closure_set(v___f_3731_, 1, v_var_3728_);
lean_closure_set(v___f_3731_, 2, v___x_3730_);
return v___f_3731_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_unusedVarTestable(lean_object* v_var_3732_, lean_object* v_00_u03b1_3733_, lean_object* v_00_u03b2_3734_, lean_object* v_inst_3735_, lean_object* v_inst_3736_){
_start:
{
lean_object* v___x_3737_; 
v___x_3737_ = lp_plausible_Plausible_Testable_unusedVarTestable___redArg(v_var_3732_, v_inst_3736_);
return v___x_3737_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0(lean_object* v_inst_3739_, lean_object* v_inst_3740_, lean_object* v_x_3741_, lean_object* v___y_3742_, uint8_t v___y_3743_, lean_object* v___y_3744_, lean_object* v___y_3745_){
_start:
{
lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___x_3748_; 
lean_inc(v_x_3741_);
v___x_3746_ = lean_apply_1(v_inst_3739_, v_x_3741_);
v___x_3747_ = lean_box(v___y_3743_);
lean_inc(v___y_3745_);
v___x_3748_ = lean_apply_5(v_inst_3740_, v_x_3741_, v___y_3742_, v___x_3747_, v___y_3744_, v___y_3745_);
if (lean_obj_tag(v___x_3748_) == 0)
{
lean_dec_ref(v___x_3746_);
return v___x_3748_;
}
else
{
lean_object* v_a_3749_; lean_object* v___x_3751_; uint8_t v_isShared_3752_; uint8_t v_isSharedCheck_3771_; 
v_a_3749_ = lean_ctor_get(v___x_3748_, 0);
v_isSharedCheck_3771_ = !lean_is_exclusive(v___x_3748_);
if (v_isSharedCheck_3771_ == 0)
{
v___x_3751_ = v___x_3748_;
v_isShared_3752_ = v_isSharedCheck_3771_;
goto v_resetjp_3750_;
}
else
{
lean_inc(v_a_3749_);
lean_dec(v___x_3748_);
v___x_3751_ = lean_box(0);
v_isShared_3752_ = v_isSharedCheck_3771_;
goto v_resetjp_3750_;
}
v_resetjp_3750_:
{
lean_object* v_fst_3753_; lean_object* v_snd_3754_; lean_object* v___x_3756_; uint8_t v_isShared_3757_; uint8_t v_isSharedCheck_3770_; 
v_fst_3753_ = lean_ctor_get(v_a_3749_, 0);
v_snd_3754_ = lean_ctor_get(v_a_3749_, 1);
v_isSharedCheck_3770_ = !lean_is_exclusive(v_a_3749_);
if (v_isSharedCheck_3770_ == 0)
{
v___x_3756_ = v_a_3749_;
v_isShared_3757_ = v_isSharedCheck_3770_;
goto v_resetjp_3755_;
}
else
{
lean_inc(v_snd_3754_);
lean_inc(v_fst_3753_);
lean_dec(v_a_3749_);
v___x_3756_ = lean_box(0);
v_isShared_3757_ = v_isSharedCheck_3770_;
goto v_resetjp_3755_;
}
v_resetjp_3755_:
{
lean_object* v___x_3758_; lean_object* v___x_3759_; lean_object* v___x_3760_; lean_object* v___x_3761_; lean_object* v___x_3762_; lean_object* v___x_3763_; lean_object* v___x_3765_; 
v___x_3758_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__14));
v___x_3759_ = lean_string_append(v___x_3758_, v___x_3746_);
lean_dec_ref(v___x_3746_);
v___x_3760_ = ((lean_object*)(lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___closed__0));
v___x_3761_ = lean_string_append(v___x_3759_, v___x_3760_);
v___x_3762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3762_, 0, lean_box(0));
v___x_3763_ = lp_plausible_Plausible_TestResult_addInfo___redArg(v___x_3761_, v_fst_3753_, v___x_3762_);
lean_dec_ref_known(v___x_3762_, 1);
if (v_isShared_3757_ == 0)
{
lean_ctor_set(v___x_3756_, 0, v___x_3763_);
v___x_3765_ = v___x_3756_;
goto v_reusejp_3764_;
}
else
{
lean_object* v_reuseFailAlloc_3769_; 
v_reuseFailAlloc_3769_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3769_, 0, v___x_3763_);
lean_ctor_set(v_reuseFailAlloc_3769_, 1, v_snd_3754_);
v___x_3765_ = v_reuseFailAlloc_3769_;
goto v_reusejp_3764_;
}
v_reusejp_3764_:
{
lean_object* v___x_3767_; 
if (v_isShared_3752_ == 0)
{
lean_ctor_set(v___x_3751_, 0, v___x_3765_);
v___x_3767_ = v___x_3751_;
goto v_reusejp_3766_;
}
else
{
lean_object* v_reuseFailAlloc_3768_; 
v_reuseFailAlloc_3768_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3768_, 0, v___x_3765_);
v___x_3767_ = v_reuseFailAlloc_3768_;
goto v_reusejp_3766_;
}
v_reusejp_3766_:
{
return v___x_3767_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___boxed(lean_object* v_inst_3772_, lean_object* v_inst_3773_, lean_object* v_x_3774_, lean_object* v___y_3775_, lean_object* v___y_3776_, lean_object* v___y_3777_, lean_object* v___y_3778_){
_start:
{
uint8_t v___y_1332__boxed_3779_; lean_object* v_res_3780_; 
v___y_1332__boxed_3779_ = lean_unbox(v___y_3776_);
v_res_3780_ = lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0(v_inst_3772_, v_inst_3773_, v_x_3774_, v___y_3775_, v___y_1332__boxed_3779_, v___y_3777_, v___y_3778_);
lean_dec(v___y_3778_);
return v_res_3780_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1(lean_object* v_inst_3781_, lean_object* v_var_3782_, lean_object* v___f_3783_, lean_object* v_cfg_3784_, uint8_t v_min_3785_, lean_object* v___y_3786_, lean_object* v___y_3787_){
_start:
{
lean_object* v_fst_3789_; lean_object* v_snd_3790_; lean_object* v_proxyRepr_3794_; lean_object* v_sample_3795_; lean_object* v_interp_3796_; lean_object* v_fst_3798_; lean_object* v_snd_3799_; lean_object* v___y_3800_; lean_object* v___x_3803_; 
v_proxyRepr_3794_ = lean_ctor_get(v_inst_3781_, 0);
lean_inc_ref(v_proxyRepr_3794_);
v_sample_3795_ = lean_ctor_get(v_inst_3781_, 2);
v_interp_3796_ = lean_ctor_get(v_inst_3781_, 3);
lean_inc_ref(v_sample_3795_);
lean_inc(v___y_3787_);
v___x_3803_ = lean_apply_2(v_sample_3795_, v___y_3786_, v___y_3787_);
if (lean_obj_tag(v___x_3803_) == 0)
{
lean_object* v_a_3804_; lean_object* v___x_3806_; uint8_t v_isShared_3807_; uint8_t v_isSharedCheck_3811_; 
lean_dec_ref(v_proxyRepr_3794_);
lean_dec_ref(v_cfg_3784_);
lean_dec_ref(v___f_3783_);
lean_dec_ref(v_var_3782_);
lean_dec_ref(v_inst_3781_);
v_a_3804_ = lean_ctor_get(v___x_3803_, 0);
v_isSharedCheck_3811_ = !lean_is_exclusive(v___x_3803_);
if (v_isSharedCheck_3811_ == 0)
{
v___x_3806_ = v___x_3803_;
v_isShared_3807_ = v_isSharedCheck_3811_;
goto v_resetjp_3805_;
}
else
{
lean_inc(v_a_3804_);
lean_dec(v___x_3803_);
v___x_3806_ = lean_box(0);
v_isShared_3807_ = v_isSharedCheck_3811_;
goto v_resetjp_3805_;
}
v_resetjp_3805_:
{
lean_object* v___x_3809_; 
if (v_isShared_3807_ == 0)
{
v___x_3809_ = v___x_3806_;
goto v_reusejp_3808_;
}
else
{
lean_object* v_reuseFailAlloc_3810_; 
v_reuseFailAlloc_3810_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3810_, 0, v_a_3804_);
v___x_3809_ = v_reuseFailAlloc_3810_;
goto v_reusejp_3808_;
}
v_reusejp_3808_:
{
return v___x_3809_;
}
}
}
else
{
lean_object* v_a_3812_; lean_object* v_fst_3813_; lean_object* v_snd_3814_; lean_object* v___y_3816_; lean_object* v___y_3817_; lean_object* v___y_3818_; uint8_t v_traceDiscarded_3833_; uint8_t v_traceSuccesses_3834_; lean_object* v___x_3835_; lean_object* v___y_3837_; lean_object* v___y_3838_; 
v_a_3812_ = lean_ctor_get(v___x_3803_, 0);
lean_inc(v_a_3812_);
lean_dec_ref_known(v___x_3803_, 1);
v_fst_3813_ = lean_ctor_get(v_a_3812_, 0);
lean_inc(v_fst_3813_);
v_snd_3814_ = lean_ctor_get(v_a_3812_, 1);
lean_inc(v_snd_3814_);
lean_dec(v_a_3812_);
v_traceDiscarded_3833_ = lean_ctor_get_uint8(v_cfg_3784_, sizeof(void*)*4);
v_traceSuccesses_3834_ = lean_ctor_get_uint8(v_cfg_3784_, sizeof(void*)*4 + 1);
v___x_3835_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
if (v_traceSuccesses_3834_ == 0)
{
if (v_traceDiscarded_3833_ == 0)
{
v___y_3837_ = v_snd_3814_;
v___y_3838_ = v___y_3787_;
goto v___jp_3836_;
}
else
{
goto v___jp_3868_;
}
}
else
{
goto v___jp_3868_;
}
v___jp_3815_:
{
if (v_min_3785_ == 0)
{
lean_dec_ref(v_cfg_3784_);
lean_dec_ref(v___f_3783_);
lean_dec_ref(v_inst_3781_);
v_fst_3798_ = v_fst_3813_;
v_snd_3799_ = v___y_3816_;
v___y_3800_ = v___y_3817_;
goto v___jp_3797_;
}
else
{
lean_object* v___x_3819_; 
lean_inc_ref(v_var_3782_);
v___x_3819_ = lp_plausible_Plausible_Testable_minimize___redArg(v_inst_3781_, v___f_3783_, v_cfg_3784_, v_var_3782_, v_fst_3813_, v___y_3816_, v___y_3817_, v___y_3818_);
if (lean_obj_tag(v___x_3819_) == 0)
{
lean_object* v_a_3820_; lean_object* v___x_3822_; uint8_t v_isShared_3823_; uint8_t v_isSharedCheck_3827_; 
lean_dec_ref(v_proxyRepr_3794_);
lean_dec_ref(v_var_3782_);
v_a_3820_ = lean_ctor_get(v___x_3819_, 0);
v_isSharedCheck_3827_ = !lean_is_exclusive(v___x_3819_);
if (v_isSharedCheck_3827_ == 0)
{
v___x_3822_ = v___x_3819_;
v_isShared_3823_ = v_isSharedCheck_3827_;
goto v_resetjp_3821_;
}
else
{
lean_inc(v_a_3820_);
lean_dec(v___x_3819_);
v___x_3822_ = lean_box(0);
v_isShared_3823_ = v_isSharedCheck_3827_;
goto v_resetjp_3821_;
}
v_resetjp_3821_:
{
lean_object* v___x_3825_; 
if (v_isShared_3823_ == 0)
{
v___x_3825_ = v___x_3822_;
goto v_reusejp_3824_;
}
else
{
lean_object* v_reuseFailAlloc_3826_; 
v_reuseFailAlloc_3826_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3826_, 0, v_a_3820_);
v___x_3825_ = v_reuseFailAlloc_3826_;
goto v_reusejp_3824_;
}
v_reusejp_3824_:
{
return v___x_3825_;
}
}
}
else
{
lean_object* v_a_3828_; lean_object* v_fst_3829_; lean_object* v_snd_3830_; lean_object* v_fst_3831_; lean_object* v_snd_3832_; 
v_a_3828_ = lean_ctor_get(v___x_3819_, 0);
lean_inc(v_a_3828_);
lean_dec_ref_known(v___x_3819_, 1);
v_fst_3829_ = lean_ctor_get(v_a_3828_, 0);
lean_inc(v_fst_3829_);
v_snd_3830_ = lean_ctor_get(v_a_3828_, 1);
lean_inc(v_snd_3830_);
lean_dec(v_a_3828_);
v_fst_3831_ = lean_ctor_get(v_fst_3829_, 0);
lean_inc(v_fst_3831_);
v_snd_3832_ = lean_ctor_get(v_fst_3829_, 1);
lean_inc(v_snd_3832_);
lean_dec(v_fst_3829_);
v_fst_3798_ = v_fst_3831_;
v_snd_3799_ = v_snd_3832_;
v___y_3800_ = v_snd_3830_;
goto v___jp_3797_;
}
}
}
v___jp_3836_:
{
lean_object* v___x_3839_; lean_object* v___x_3840_; uint8_t v___x_3841_; lean_object* v___x_3842_; lean_object* v_a_3843_; lean_object* v_fst_3844_; lean_object* v_snd_3845_; uint8_t v___x_3846_; 
lean_inc(v_interp_3796_);
lean_inc(v_fst_3813_);
v___x_3839_ = lean_apply_1(v_interp_3796_, v_fst_3813_);
lean_inc_ref(v___f_3783_);
v___x_3840_ = lean_apply_1(v___f_3783_, v___x_3839_);
v___x_3841_ = 0;
lean_inc_ref(v_cfg_3784_);
v___x_3842_ = lp_plausible_Plausible_Testable_runPropE___redArg(v___x_3840_, v_cfg_3784_, v___x_3841_, v___y_3837_, v___y_3838_);
v_a_3843_ = lean_ctor_get(v___x_3842_, 0);
lean_inc(v_a_3843_);
lean_dec_ref(v___x_3842_);
v_fst_3844_ = lean_ctor_get(v_a_3843_, 0);
lean_inc(v_fst_3844_);
v_snd_3845_ = lean_ctor_get(v_a_3843_, 1);
lean_inc(v_snd_3845_);
lean_dec(v_a_3843_);
v___x_3846_ = lp_plausible_Plausible_TestResult_isFailure___redArg(v_fst_3844_);
if (v___x_3846_ == 0)
{
lean_dec_ref(v_cfg_3784_);
lean_dec_ref(v___f_3783_);
lean_dec_ref(v_inst_3781_);
v_fst_3798_ = v_fst_3813_;
v_snd_3799_ = v_fst_3844_;
v___y_3800_ = v_snd_3845_;
goto v___jp_3797_;
}
else
{
if (v_traceSuccesses_3834_ == 0)
{
v___y_3816_ = v_fst_3844_;
v___y_3817_ = v_snd_3845_;
v___y_3818_ = v___y_3838_;
goto v___jp_3815_;
}
else
{
lean_object* v___x_3847_; lean_object* v___x_3848_; lean_object* v___x_3849_; lean_object* v___x_3850_; lean_object* v___x_3851_; lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_1306__overap_3856_; lean_object* v___x_3857_; 
v___x_3847_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
lean_inc_ref(v_var_3782_);
v___x_3848_ = lean_string_append(v_var_3782_, v___x_3847_);
v___x_3849_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3794_);
lean_inc(v_fst_3813_);
v___x_3850_ = lean_apply_2(v_proxyRepr_3794_, v_fst_3813_, v___x_3849_);
v___x_3851_ = l_Std_Format_defWidth;
v___x_3852_ = l_Std_Format_pretty(v___x_3850_, v___x_3851_, v___x_3849_, v___x_3849_);
v___x_3853_ = lean_string_append(v___x_3848_, v___x_3852_);
lean_dec_ref(v___x_3852_);
v___x_3854_ = ((lean_object*)(lp_plausible_Plausible_Testable_varTestable___redArg___lam__0___closed__0));
v___x_3855_ = lean_string_append(v___x_3853_, v___x_3854_);
v___x_1306__overap_3856_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3835_, v___x_3855_);
lean_dec_ref(v___x_3855_);
lean_inc(v___y_3838_);
v___x_3857_ = lean_apply_2(v___x_1306__overap_3856_, v_snd_3845_, v___y_3838_);
if (lean_obj_tag(v___x_3857_) == 0)
{
lean_object* v_a_3858_; lean_object* v___x_3860_; uint8_t v_isShared_3861_; uint8_t v_isSharedCheck_3865_; 
lean_dec(v_fst_3844_);
lean_dec(v_fst_3813_);
lean_dec_ref(v_proxyRepr_3794_);
lean_dec_ref(v_cfg_3784_);
lean_dec_ref(v___f_3783_);
lean_dec_ref(v_var_3782_);
lean_dec_ref(v_inst_3781_);
v_a_3858_ = lean_ctor_get(v___x_3857_, 0);
v_isSharedCheck_3865_ = !lean_is_exclusive(v___x_3857_);
if (v_isSharedCheck_3865_ == 0)
{
v___x_3860_ = v___x_3857_;
v_isShared_3861_ = v_isSharedCheck_3865_;
goto v_resetjp_3859_;
}
else
{
lean_inc(v_a_3858_);
lean_dec(v___x_3857_);
v___x_3860_ = lean_box(0);
v_isShared_3861_ = v_isSharedCheck_3865_;
goto v_resetjp_3859_;
}
v_resetjp_3859_:
{
lean_object* v___x_3863_; 
if (v_isShared_3861_ == 0)
{
v___x_3863_ = v___x_3860_;
goto v_reusejp_3862_;
}
else
{
lean_object* v_reuseFailAlloc_3864_; 
v_reuseFailAlloc_3864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3864_, 0, v_a_3858_);
v___x_3863_ = v_reuseFailAlloc_3864_;
goto v_reusejp_3862_;
}
v_reusejp_3862_:
{
return v___x_3863_;
}
}
}
else
{
lean_object* v_a_3866_; lean_object* v_snd_3867_; 
v_a_3866_ = lean_ctor_get(v___x_3857_, 0);
lean_inc(v_a_3866_);
lean_dec_ref_known(v___x_3857_, 1);
v_snd_3867_ = lean_ctor_get(v_a_3866_, 1);
lean_inc(v_snd_3867_);
lean_dec(v_a_3866_);
v___y_3816_ = v_fst_3844_;
v___y_3817_ = v_snd_3867_;
v___y_3818_ = v___y_3838_;
goto v___jp_3815_;
}
}
}
}
v___jp_3868_:
{
lean_object* v___x_3869_; lean_object* v___x_3870_; lean_object* v___x_3871_; lean_object* v___x_3872_; lean_object* v___x_3873_; lean_object* v___x_3874_; lean_object* v___x_3875_; lean_object* v___x_1320__overap_3876_; lean_object* v___x_3877_; 
v___x_3869_ = ((lean_object*)(lp_plausible_Plausible_TestResult_addVarInfo___redArg___closed__0));
lean_inc_ref(v_var_3782_);
v___x_3870_ = lean_string_append(v_var_3782_, v___x_3869_);
v___x_3871_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_proxyRepr_3794_);
lean_inc(v_fst_3813_);
v___x_3872_ = lean_apply_2(v_proxyRepr_3794_, v_fst_3813_, v___x_3871_);
v___x_3873_ = l_Std_Format_defWidth;
v___x_3874_ = l_Std_Format_pretty(v___x_3872_, v___x_3873_, v___x_3871_, v___x_3871_);
v___x_3875_ = lean_string_append(v___x_3870_, v___x_3874_);
lean_dec_ref(v___x_3874_);
v___x_1320__overap_3876_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_3835_, v___x_3875_);
lean_dec_ref(v___x_3875_);
lean_inc(v___y_3787_);
v___x_3877_ = lean_apply_2(v___x_1320__overap_3876_, v_snd_3814_, v___y_3787_);
if (lean_obj_tag(v___x_3877_) == 0)
{
lean_object* v_a_3878_; lean_object* v___x_3880_; uint8_t v_isShared_3881_; uint8_t v_isSharedCheck_3885_; 
lean_dec(v_fst_3813_);
lean_dec_ref(v_proxyRepr_3794_);
lean_dec_ref(v_cfg_3784_);
lean_dec_ref(v___f_3783_);
lean_dec_ref(v_var_3782_);
lean_dec_ref(v_inst_3781_);
v_a_3878_ = lean_ctor_get(v___x_3877_, 0);
v_isSharedCheck_3885_ = !lean_is_exclusive(v___x_3877_);
if (v_isSharedCheck_3885_ == 0)
{
v___x_3880_ = v___x_3877_;
v_isShared_3881_ = v_isSharedCheck_3885_;
goto v_resetjp_3879_;
}
else
{
lean_inc(v_a_3878_);
lean_dec(v___x_3877_);
v___x_3880_ = lean_box(0);
v_isShared_3881_ = v_isSharedCheck_3885_;
goto v_resetjp_3879_;
}
v_resetjp_3879_:
{
lean_object* v___x_3883_; 
if (v_isShared_3881_ == 0)
{
v___x_3883_ = v___x_3880_;
goto v_reusejp_3882_;
}
else
{
lean_object* v_reuseFailAlloc_3884_; 
v_reuseFailAlloc_3884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3884_, 0, v_a_3878_);
v___x_3883_ = v_reuseFailAlloc_3884_;
goto v_reusejp_3882_;
}
v_reusejp_3882_:
{
return v___x_3883_;
}
}
}
else
{
lean_object* v_a_3886_; lean_object* v_snd_3887_; 
v_a_3886_ = lean_ctor_get(v___x_3877_, 0);
lean_inc(v_a_3886_);
lean_dec_ref_known(v___x_3877_, 1);
v_snd_3887_ = lean_ctor_get(v_a_3886_, 1);
lean_inc(v_snd_3887_);
lean_dec(v_a_3886_);
v___y_3837_ = v_snd_3887_;
v___y_3838_ = v___y_3787_;
goto v___jp_3836_;
}
}
}
v___jp_3788_:
{
lean_object* v___x_3791_; lean_object* v___x_3792_; lean_object* v___x_3793_; 
v___x_3791_ = lp_plausible_Plausible_TestResult_iff___redArg(v_fst_3789_);
v___x_3792_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3792_, 0, v___x_3791_);
lean_ctor_set(v___x_3792_, 1, v_snd_3790_);
v___x_3793_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3793_, 0, v___x_3792_);
return v___x_3793_;
}
v___jp_3797_:
{
lean_object* v___x_3801_; lean_object* v___x_3802_; 
v___x_3801_ = ((lean_object*)(lp_plausible_Plausible_TestResult_combine___redArg___closed__0));
v___x_3802_ = lp_plausible_Plausible_TestResult_addVarInfo___redArg(v_proxyRepr_3794_, v_var_3782_, v_fst_3798_, v_snd_3799_, v___x_3801_);
v_fst_3789_ = v___x_3802_;
v_snd_3790_ = v___y_3800_;
goto v___jp_3788_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1___boxed(lean_object* v_inst_3888_, lean_object* v_var_3889_, lean_object* v___f_3890_, lean_object* v_cfg_3891_, lean_object* v_min_3892_, lean_object* v___y_3893_, lean_object* v___y_3894_){
_start:
{
uint8_t v_min_boxed_3895_; lean_object* v_res_3896_; 
v_min_boxed_3895_ = lean_unbox(v_min_3892_);
v_res_3896_ = lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1(v_inst_3888_, v_var_3889_, v___f_3890_, v_cfg_3891_, v_min_boxed_3895_, v___y_3893_, v___y_3894_);
lean_dec(v___y_3894_);
return v_res_3896_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___redArg(lean_object* v_var_3897_, lean_object* v_inst_3898_, lean_object* v_inst_3899_, lean_object* v_inst_3900_){
_start:
{
lean_object* v___f_3901_; lean_object* v___f_3902_; 
v___f_3901_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_3901_, 0, v_inst_3898_);
lean_closure_set(v___f_3901_, 1, v_inst_3899_);
v___f_3902_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_subtypeVarTestable___redArg___lam__1___boxed), 7, 3);
lean_closure_set(v___f_3902_, 0, v_inst_3900_);
lean_closure_set(v___f_3902_, 1, v_var_3897_);
lean_closure_set(v___f_3902_, 2, v___f_3901_);
return v___f_3902_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable(lean_object* v_var_3903_, lean_object* v_00_u03b1_3904_, lean_object* v_p_3905_, lean_object* v_00_u03b2_3906_, lean_object* v_inst_3907_, lean_object* v_inst_3908_, lean_object* v_inst_3909_, lean_object* v_var_x27_3910_){
_start:
{
lean_object* v___x_3911_; 
v___x_3911_ = lp_plausible_Plausible_Testable_subtypeVarTestable___redArg(v_var_3903_, v_inst_3907_, v_inst_3908_, v_inst_3909_);
return v___x_3911_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_subtypeVarTestable___boxed(lean_object* v_var_3912_, lean_object* v_00_u03b1_3913_, lean_object* v_p_3914_, lean_object* v_00_u03b2_3915_, lean_object* v_inst_3916_, lean_object* v_inst_3917_, lean_object* v_inst_3918_, lean_object* v_var_x27_3919_){
_start:
{
lean_object* v_res_3920_; 
v_res_3920_ = lp_plausible_Plausible_Testable_subtypeVarTestable(v_var_3912_, v_00_u03b1_3913_, v_p_3914_, v_00_u03b2_3915_, v_inst_3916_, v_inst_3917_, v_inst_3918_, v_var_x27_3919_);
lean_dec_ref(v_var_x27_3919_);
return v_res_3920_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0(uint8_t v_inst_3922_, lean_object* v_inst_3923_, lean_object* v_x_3924_, uint8_t v_x_3925_, lean_object* v___y_3926_, lean_object* v___y_3927_){
_start:
{
if (v_inst_3922_ == 0)
{
lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; lean_object* v___x_3935_; lean_object* v___x_3936_; lean_object* v___x_3937_; 
v___x_3928_ = ((lean_object*)(lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___closed__0));
v___x_3929_ = lean_string_append(v___x_3928_, v_inst_3923_);
v___x_3930_ = ((lean_object*)(lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__13));
v___x_3931_ = lean_string_append(v___x_3929_, v___x_3930_);
v___x_3932_ = lean_box(0);
v___x_3933_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3933_, 0, v___x_3931_);
lean_ctor_set(v___x_3933_, 1, v___x_3932_);
v___x_3934_ = lean_unsigned_to_nat(0u);
v___x_3935_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3935_, 0, v___x_3933_);
lean_ctor_set(v___x_3935_, 1, v___x_3934_);
v___x_3936_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3936_, 0, v___x_3935_);
lean_ctor_set(v___x_3936_, 1, v___y_3926_);
v___x_3937_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3937_, 0, v___x_3936_);
return v___x_3937_;
}
else
{
lean_object* v___x_3938_; lean_object* v___x_3939_; lean_object* v___x_3940_; 
v___x_3938_ = lean_obj_once(&lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0, &lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0_once, _init_lp_plausible_Plausible_Testable_orTestable___redArg___lam__0___closed__0);
v___x_3939_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3939_, 0, v___x_3938_);
lean_ctor_set(v___x_3939_, 1, v___y_3926_);
v___x_3940_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3940_, 0, v___x_3939_);
return v___x_3940_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___boxed(lean_object* v_inst_3941_, lean_object* v_inst_3942_, lean_object* v_x_3943_, lean_object* v_x_3944_, lean_object* v___y_3945_, lean_object* v___y_3946_){
_start:
{
uint8_t v_inst_503__boxed_3947_; uint8_t v_x_506__boxed_3948_; lean_object* v_res_3949_; 
v_inst_503__boxed_3947_ = lean_unbox(v_inst_3941_);
v_x_506__boxed_3948_ = lean_unbox(v_x_3944_);
v_res_3949_ = lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0(v_inst_503__boxed_3947_, v_inst_3942_, v_x_3943_, v_x_506__boxed_3948_, v___y_3945_, v___y_3946_);
lean_dec(v___y_3946_);
lean_dec_ref(v_x_3943_);
lean_dec_ref(v_inst_3942_);
return v_res_3949_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg(lean_object* v_inst_3950_, uint8_t v_inst_3951_){
_start:
{
lean_object* v___x_3952_; lean_object* v___f_3953_; 
v___x_3952_ = lean_box(v_inst_3951_);
v___f_3953_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_3953_, 0, v___x_3952_);
lean_closure_set(v___f_3953_, 1, v_inst_3950_);
return v___f_3953_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___redArg___boxed(lean_object* v_inst_3954_, lean_object* v_inst_3955_){
_start:
{
uint8_t v_inst_549__boxed_3956_; lean_object* v_res_3957_; 
v_inst_549__boxed_3956_ = lean_unbox(v_inst_3955_);
v_res_3957_ = lp_plausible_Plausible_Testable_decidableTestable___redArg(v_inst_3954_, v_inst_549__boxed_3956_);
return v_res_3957_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable(lean_object* v_p_3958_, lean_object* v_inst_3959_, uint8_t v_inst_3960_){
_start:
{
lean_object* v___x_3961_; lean_object* v___f_3962_; 
v___x_3961_ = lean_box(v_inst_3960_);
v___f_3962_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_decidableTestable___redArg___lam__0___boxed), 6, 2);
lean_closure_set(v___f_3962_, 0, v___x_3961_);
lean_closure_set(v___f_3962_, 1, v_inst_3959_);
return v___f_3962_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_decidableTestable___boxed(lean_object* v_p_3963_, lean_object* v_inst_3964_, lean_object* v_inst_3965_){
_start:
{
uint8_t v_inst_560__boxed_3966_; lean_object* v_res_3967_; 
v_inst_560__boxed_3966_ = lean_unbox(v_inst_3965_);
v_res_3967_ = lp_plausible_Plausible_Testable_decidableTestable(v_p_3963_, v_inst_3964_, v_inst_560__boxed_3966_);
return v_res_3967_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Eq_printableProp___redArg(lean_object* v_inst_3969_, lean_object* v_x_3970_, lean_object* v_y_3971_){
_start:
{
lean_object* v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; lean_object* v___x_3978_; lean_object* v___x_3979_; lean_object* v___x_3980_; 
v___x_3972_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_3969_);
v___x_3973_ = lean_apply_2(v_inst_3969_, v_x_3970_, v___x_3972_);
v___x_3974_ = l_Std_Format_defWidth;
v___x_3975_ = l_Std_Format_pretty(v___x_3973_, v___x_3974_, v___x_3972_, v___x_3972_);
v___x_3976_ = ((lean_object*)(lp_plausible_Plausible_Eq_printableProp___redArg___closed__0));
v___x_3977_ = lean_string_append(v___x_3975_, v___x_3976_);
v___x_3978_ = lean_apply_2(v_inst_3969_, v_y_3971_, v___x_3972_);
v___x_3979_ = l_Std_Format_pretty(v___x_3978_, v___x_3974_, v___x_3972_, v___x_3972_);
v___x_3980_ = lean_string_append(v___x_3977_, v___x_3979_);
lean_dec_ref(v___x_3979_);
return v___x_3980_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Eq_printableProp(lean_object* v_00_u03b1_3981_, lean_object* v_inst_3982_, lean_object* v_x_3983_, lean_object* v_y_3984_){
_start:
{
lean_object* v___x_3985_; 
v___x_3985_ = lp_plausible_Plausible_Eq_printableProp___redArg(v_inst_3982_, v_x_3983_, v_y_3984_);
return v___x_3985_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Ne_printableProp___redArg(lean_object* v_inst_3987_, lean_object* v_x_3988_, lean_object* v_y_3989_){
_start:
{
lean_object* v___x_3990_; lean_object* v___x_3991_; lean_object* v___x_3992_; lean_object* v___x_3993_; lean_object* v___x_3994_; lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; 
v___x_3990_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_3987_);
v___x_3991_ = lean_apply_2(v_inst_3987_, v_x_3988_, v___x_3990_);
v___x_3992_ = l_Std_Format_defWidth;
v___x_3993_ = l_Std_Format_pretty(v___x_3991_, v___x_3992_, v___x_3990_, v___x_3990_);
v___x_3994_ = ((lean_object*)(lp_plausible_Plausible_Ne_printableProp___redArg___closed__0));
v___x_3995_ = lean_string_append(v___x_3993_, v___x_3994_);
v___x_3996_ = lean_apply_2(v_inst_3987_, v_y_3989_, v___x_3990_);
v___x_3997_ = l_Std_Format_pretty(v___x_3996_, v___x_3992_, v___x_3990_, v___x_3990_);
v___x_3998_ = lean_string_append(v___x_3995_, v___x_3997_);
lean_dec_ref(v___x_3997_);
return v___x_3998_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Ne_printableProp(lean_object* v_00_u03b1_3999_, lean_object* v_inst_4000_, lean_object* v_x_4001_, lean_object* v_y_4002_){
_start:
{
lean_object* v___x_4003_; 
v___x_4003_ = lp_plausible_Plausible_Ne_printableProp___redArg(v_inst_4000_, v_x_4001_, v_y_4002_);
return v___x_4003_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_LE_printableProp___redArg(lean_object* v_inst_4005_, lean_object* v_x_4006_, lean_object* v_y_4007_){
_start:
{
lean_object* v___x_4008_; lean_object* v___x_4009_; lean_object* v___x_4010_; lean_object* v___x_4011_; lean_object* v___x_4012_; lean_object* v___x_4013_; lean_object* v___x_4014_; lean_object* v___x_4015_; lean_object* v___x_4016_; 
v___x_4008_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_4005_);
v___x_4009_ = lean_apply_2(v_inst_4005_, v_x_4006_, v___x_4008_);
v___x_4010_ = l_Std_Format_defWidth;
v___x_4011_ = l_Std_Format_pretty(v___x_4009_, v___x_4010_, v___x_4008_, v___x_4008_);
v___x_4012_ = ((lean_object*)(lp_plausible_Plausible_LE_printableProp___redArg___closed__0));
v___x_4013_ = lean_string_append(v___x_4011_, v___x_4012_);
v___x_4014_ = lean_apply_2(v_inst_4005_, v_y_4007_, v___x_4008_);
v___x_4015_ = l_Std_Format_pretty(v___x_4014_, v___x_4010_, v___x_4008_, v___x_4008_);
v___x_4016_ = lean_string_append(v___x_4013_, v___x_4015_);
lean_dec_ref(v___x_4015_);
return v___x_4016_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_LE_printableProp(lean_object* v_00_u03b1_4017_, lean_object* v_inst_4018_, lean_object* v_inst_4019_, lean_object* v_x_4020_, lean_object* v_y_4021_){
_start:
{
lean_object* v___x_4022_; 
v___x_4022_ = lp_plausible_Plausible_LE_printableProp___redArg(v_inst_4018_, v_x_4020_, v_y_4021_);
return v___x_4022_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_LT_printableProp___redArg(lean_object* v_inst_4024_, lean_object* v_x_4025_, lean_object* v_y_4026_){
_start:
{
lean_object* v___x_4027_; lean_object* v___x_4028_; lean_object* v___x_4029_; lean_object* v___x_4030_; lean_object* v___x_4031_; lean_object* v___x_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; 
v___x_4027_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_inst_4024_);
v___x_4028_ = lean_apply_2(v_inst_4024_, v_x_4025_, v___x_4027_);
v___x_4029_ = l_Std_Format_defWidth;
v___x_4030_ = l_Std_Format_pretty(v___x_4028_, v___x_4029_, v___x_4027_, v___x_4027_);
v___x_4031_ = ((lean_object*)(lp_plausible_Plausible_LT_printableProp___redArg___closed__0));
v___x_4032_ = lean_string_append(v___x_4030_, v___x_4031_);
v___x_4033_ = lean_apply_2(v_inst_4024_, v_y_4026_, v___x_4027_);
v___x_4034_ = l_Std_Format_pretty(v___x_4033_, v___x_4029_, v___x_4027_, v___x_4027_);
v___x_4035_ = lean_string_append(v___x_4032_, v___x_4034_);
lean_dec_ref(v___x_4034_);
return v___x_4035_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_LT_printableProp(lean_object* v_00_u03b1_4036_, lean_object* v_inst_4037_, lean_object* v_inst_4038_, lean_object* v_x_4039_, lean_object* v_y_4040_){
_start:
{
lean_object* v___x_4041_; 
v___x_4041_ = lp_plausible_Plausible_LT_printableProp___redArg(v_inst_4037_, v_x_4039_, v_y_4040_);
return v___x_4041_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___redArg(lean_object* v_inst_4043_, lean_object* v_inst_4044_){
_start:
{
lean_object* v___x_4045_; lean_object* v___x_4046_; lean_object* v___x_4047_; 
v___x_4045_ = ((lean_object*)(lp_plausible_Plausible_And_printableProp___redArg___closed__0));
v___x_4046_ = lean_string_append(v_inst_4043_, v___x_4045_);
v___x_4047_ = lean_string_append(v___x_4046_, v_inst_4044_);
return v___x_4047_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___redArg___boxed(lean_object* v_inst_4048_, lean_object* v_inst_4049_){
_start:
{
lean_object* v_res_4050_; 
v_res_4050_ = lp_plausible_Plausible_And_printableProp___redArg(v_inst_4048_, v_inst_4049_);
lean_dec_ref(v_inst_4049_);
return v_res_4050_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp(lean_object* v_x_4051_, lean_object* v_y_4052_, lean_object* v_inst_4053_, lean_object* v_inst_4054_){
_start:
{
lean_object* v___x_4055_; 
v___x_4055_ = lp_plausible_Plausible_And_printableProp___redArg(v_inst_4053_, v_inst_4054_);
return v___x_4055_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_And_printableProp___boxed(lean_object* v_x_4056_, lean_object* v_y_4057_, lean_object* v_inst_4058_, lean_object* v_inst_4059_){
_start:
{
lean_object* v_res_4060_; 
v_res_4060_ = lp_plausible_Plausible_And_printableProp(v_x_4056_, v_y_4057_, v_inst_4058_, v_inst_4059_);
lean_dec_ref(v_inst_4059_);
return v_res_4060_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___redArg(lean_object* v_inst_4062_, lean_object* v_inst_4063_){
_start:
{
lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4066_; 
v___x_4064_ = ((lean_object*)(lp_plausible_Plausible_Or_printableProp___redArg___closed__0));
v___x_4065_ = lean_string_append(v_inst_4062_, v___x_4064_);
v___x_4066_ = lean_string_append(v___x_4065_, v_inst_4063_);
return v___x_4066_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___redArg___boxed(lean_object* v_inst_4067_, lean_object* v_inst_4068_){
_start:
{
lean_object* v_res_4069_; 
v_res_4069_ = lp_plausible_Plausible_Or_printableProp___redArg(v_inst_4067_, v_inst_4068_);
lean_dec_ref(v_inst_4068_);
return v_res_4069_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp(lean_object* v_x_4070_, lean_object* v_y_4071_, lean_object* v_inst_4072_, lean_object* v_inst_4073_){
_start:
{
lean_object* v___x_4074_; 
v___x_4074_ = lp_plausible_Plausible_Or_printableProp___redArg(v_inst_4072_, v_inst_4073_);
return v___x_4074_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Or_printableProp___boxed(lean_object* v_x_4075_, lean_object* v_y_4076_, lean_object* v_inst_4077_, lean_object* v_inst_4078_){
_start:
{
lean_object* v_res_4079_; 
v_res_4079_ = lp_plausible_Plausible_Or_printableProp(v_x_4075_, v_y_4076_, v_inst_4077_, v_inst_4078_);
lean_dec_ref(v_inst_4078_);
return v_res_4079_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___redArg(lean_object* v_inst_4081_, lean_object* v_inst_4082_){
_start:
{
lean_object* v___x_4083_; lean_object* v___x_4084_; lean_object* v___x_4085_; 
v___x_4083_ = ((lean_object*)(lp_plausible_Plausible_Iff_printableProp___redArg___closed__0));
v___x_4084_ = lean_string_append(v_inst_4081_, v___x_4083_);
v___x_4085_ = lean_string_append(v___x_4084_, v_inst_4082_);
return v___x_4085_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___redArg___boxed(lean_object* v_inst_4086_, lean_object* v_inst_4087_){
_start:
{
lean_object* v_res_4088_; 
v_res_4088_ = lp_plausible_Plausible_Iff_printableProp___redArg(v_inst_4086_, v_inst_4087_);
lean_dec_ref(v_inst_4087_);
return v_res_4088_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp(lean_object* v_x_4089_, lean_object* v_y_4090_, lean_object* v_inst_4091_, lean_object* v_inst_4092_){
_start:
{
lean_object* v___x_4093_; 
v___x_4093_ = lp_plausible_Plausible_Iff_printableProp___redArg(v_inst_4091_, v_inst_4092_);
return v___x_4093_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Iff_printableProp___boxed(lean_object* v_x_4094_, lean_object* v_y_4095_, lean_object* v_inst_4096_, lean_object* v_inst_4097_){
_start:
{
lean_object* v_res_4098_; 
v_res_4098_ = lp_plausible_Plausible_Iff_printableProp(v_x_4094_, v_y_4095_, v_inst_4096_, v_inst_4097_);
lean_dec_ref(v_inst_4097_);
return v_res_4098_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___redArg(lean_object* v_inst_4100_, lean_object* v_inst_4101_){
_start:
{
lean_object* v___x_4102_; lean_object* v___x_4103_; lean_object* v___x_4104_; 
v___x_4102_ = ((lean_object*)(lp_plausible_Plausible_Imp_printableProp___redArg___closed__0));
v___x_4103_ = lean_string_append(v_inst_4100_, v___x_4102_);
v___x_4104_ = lean_string_append(v___x_4103_, v_inst_4101_);
return v___x_4104_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___redArg___boxed(lean_object* v_inst_4105_, lean_object* v_inst_4106_){
_start:
{
lean_object* v_res_4107_; 
v_res_4107_ = lp_plausible_Plausible_Imp_printableProp___redArg(v_inst_4105_, v_inst_4106_);
lean_dec_ref(v_inst_4106_);
return v_res_4107_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp(lean_object* v_x_4108_, lean_object* v_y_4109_, lean_object* v_inst_4110_, lean_object* v_inst_4111_){
_start:
{
lean_object* v___x_4112_; 
v___x_4112_ = lp_plausible_Plausible_Imp_printableProp___redArg(v_inst_4110_, v_inst_4111_);
return v___x_4112_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Imp_printableProp___boxed(lean_object* v_x_4113_, lean_object* v_y_4114_, lean_object* v_inst_4115_, lean_object* v_inst_4116_){
_start:
{
lean_object* v_res_4117_; 
v_res_4117_ = lp_plausible_Plausible_Imp_printableProp(v_x_4113_, v_y_4114_, v_inst_4115_, v_inst_4116_);
lean_dec_ref(v_inst_4116_);
return v_res_4117_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___redArg(lean_object* v_inst_4119_){
_start:
{
lean_object* v___x_4120_; lean_object* v___x_4121_; 
v___x_4120_ = ((lean_object*)(lp_plausible_Plausible_Not_printableProp___redArg___closed__0));
v___x_4121_ = lean_string_append(v___x_4120_, v_inst_4119_);
return v___x_4121_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___redArg___boxed(lean_object* v_inst_4122_){
_start:
{
lean_object* v_res_4123_; 
v_res_4123_ = lp_plausible_Plausible_Not_printableProp___redArg(v_inst_4122_);
lean_dec_ref(v_inst_4122_);
return v_res_4123_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp(lean_object* v_x_4124_, lean_object* v_inst_4125_){
_start:
{
lean_object* v___x_4126_; 
v___x_4126_ = lp_plausible_Plausible_Not_printableProp___redArg(v_inst_4125_);
return v___x_4126_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Not_printableProp___boxed(lean_object* v_x_4127_, lean_object* v_inst_4128_){
_start:
{
lean_object* v_res_4129_; 
v_res_4129_ = lp_plausible_Plausible_Not_printableProp(v_x_4127_, v_inst_4128_);
lean_dec_ref(v_inst_4128_);
return v_res_4129_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_printableProp(uint8_t v_b_4134_){
_start:
{
if (v_b_4134_ == 0)
{
lean_object* v___x_4135_; 
v___x_4135_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__6));
return v___x_4135_;
}
else
{
lean_object* v___x_4136_; 
v___x_4136_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__9));
return v___x_4136_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Bool_printableProp___boxed(lean_object* v_b_4137_){
_start:
{
uint8_t v_b_boxed_4138_; lean_object* v_res_4139_; 
v_b_boxed_4138_ = lean_unbox(v_b_4137_);
v_res_4139_ = lp_plausible_Plausible_Bool_printableProp(v_b_boxed_4138_);
return v_res_4139_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___redArg(lean_object* v_cmd_4140_, lean_object* v_x_4141_, lean_object* v_a_4142_, lean_object* v_a_4143_){
_start:
{
lean_object* v_zero_4144_; uint8_t v_isZero_4145_; 
v_zero_4144_ = lean_unsigned_to_nat(0u);
v_isZero_4145_ = lean_nat_dec_eq(v_x_4141_, v_zero_4144_);
if (v_isZero_4145_ == 1)
{
lean_object* v___x_4146_; lean_object* v___x_4147_; lean_object* v___x_4148_; 
lean_dec(v_x_4141_);
lean_dec_ref(v_cmd_4140_);
v___x_4146_ = ((lean_object*)(lp_plausible_Plausible_Testable_runPropE___redArg___closed__0));
v___x_4147_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4147_, 0, v___x_4146_);
lean_ctor_set(v___x_4147_, 1, v_a_4142_);
v___x_4148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4148_, 0, v___x_4147_);
return v___x_4148_;
}
else
{
lean_object* v___x_4149_; 
lean_inc_ref(v_cmd_4140_);
lean_inc(v_a_4143_);
v___x_4149_ = lean_apply_2(v_cmd_4140_, v_a_4142_, v_a_4143_);
if (lean_obj_tag(v___x_4149_) == 0)
{
lean_dec(v_x_4141_);
lean_dec_ref(v_cmd_4140_);
return v___x_4149_;
}
else
{
lean_object* v_a_4150_; lean_object* v_fst_4151_; 
v_a_4150_ = lean_ctor_get(v___x_4149_, 0);
lean_inc(v_a_4150_);
v_fst_4151_ = lean_ctor_get(v_a_4150_, 0);
lean_inc(v_fst_4151_);
switch(lean_obj_tag(v_fst_4151_))
{
case 0:
{
lean_dec_ref_known(v_fst_4151_, 1);
lean_dec(v_a_4150_);
lean_dec(v_x_4141_);
lean_dec_ref(v_cmd_4140_);
return v___x_4149_;
}
case 1:
{
lean_object* v_snd_4152_; lean_object* v_one_4153_; lean_object* v_n_4154_; 
lean_dec_ref_known(v_fst_4151_, 1);
lean_dec_ref_known(v___x_4149_, 1);
v_snd_4152_ = lean_ctor_get(v_a_4150_, 1);
lean_inc(v_snd_4152_);
lean_dec(v_a_4150_);
v_one_4153_ = lean_unsigned_to_nat(1u);
v_n_4154_ = lean_nat_sub(v_x_4141_, v_one_4153_);
lean_dec(v_x_4141_);
v_x_4141_ = v_n_4154_;
v_a_4142_ = v_snd_4152_;
goto _start;
}
default: 
{
lean_object* v___x_4157_; uint8_t v_isShared_4158_; uint8_t v_isSharedCheck_4180_; 
lean_dec(v_x_4141_);
lean_dec_ref(v_cmd_4140_);
v_isSharedCheck_4180_ = !lean_is_exclusive(v___x_4149_);
if (v_isSharedCheck_4180_ == 0)
{
lean_object* v_unused_4181_; 
v_unused_4181_ = lean_ctor_get(v___x_4149_, 0);
lean_dec(v_unused_4181_);
v___x_4157_ = v___x_4149_;
v_isShared_4158_ = v_isSharedCheck_4180_;
goto v_resetjp_4156_;
}
else
{
lean_dec(v___x_4149_);
v___x_4157_ = lean_box(0);
v_isShared_4158_ = v_isSharedCheck_4180_;
goto v_resetjp_4156_;
}
v_resetjp_4156_:
{
lean_object* v_snd_4159_; lean_object* v___x_4161_; uint8_t v_isShared_4162_; uint8_t v_isSharedCheck_4178_; 
v_snd_4159_ = lean_ctor_get(v_a_4150_, 1);
v_isSharedCheck_4178_ = !lean_is_exclusive(v_a_4150_);
if (v_isSharedCheck_4178_ == 0)
{
lean_object* v_unused_4179_; 
v_unused_4179_ = lean_ctor_get(v_a_4150_, 0);
lean_dec(v_unused_4179_);
v___x_4161_ = v_a_4150_;
v_isShared_4162_ = v_isSharedCheck_4178_;
goto v_resetjp_4160_;
}
else
{
lean_inc(v_snd_4159_);
lean_dec(v_a_4150_);
v___x_4161_ = lean_box(0);
v_isShared_4162_ = v_isSharedCheck_4178_;
goto v_resetjp_4160_;
}
v_resetjp_4160_:
{
lean_object* v_a_4163_; lean_object* v_a_4164_; lean_object* v___x_4166_; uint8_t v_isShared_4167_; uint8_t v_isSharedCheck_4177_; 
v_a_4163_ = lean_ctor_get(v_fst_4151_, 0);
v_a_4164_ = lean_ctor_get(v_fst_4151_, 1);
v_isSharedCheck_4177_ = !lean_is_exclusive(v_fst_4151_);
if (v_isSharedCheck_4177_ == 0)
{
v___x_4166_ = v_fst_4151_;
v_isShared_4167_ = v_isSharedCheck_4177_;
goto v_resetjp_4165_;
}
else
{
lean_inc(v_a_4164_);
lean_inc(v_a_4163_);
lean_dec(v_fst_4151_);
v___x_4166_ = lean_box(0);
v_isShared_4167_ = v_isSharedCheck_4177_;
goto v_resetjp_4165_;
}
v_resetjp_4165_:
{
lean_object* v___x_4169_; 
if (v_isShared_4167_ == 0)
{
v___x_4169_ = v___x_4166_;
goto v_reusejp_4168_;
}
else
{
lean_object* v_reuseFailAlloc_4176_; 
v_reuseFailAlloc_4176_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4176_, 0, v_a_4163_);
lean_ctor_set(v_reuseFailAlloc_4176_, 1, v_a_4164_);
v___x_4169_ = v_reuseFailAlloc_4176_;
goto v_reusejp_4168_;
}
v_reusejp_4168_:
{
lean_object* v___x_4171_; 
if (v_isShared_4162_ == 0)
{
lean_ctor_set(v___x_4161_, 0, v___x_4169_);
v___x_4171_ = v___x_4161_;
goto v_reusejp_4170_;
}
else
{
lean_object* v_reuseFailAlloc_4175_; 
v_reuseFailAlloc_4175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4175_, 0, v___x_4169_);
lean_ctor_set(v_reuseFailAlloc_4175_, 1, v_snd_4159_);
v___x_4171_ = v_reuseFailAlloc_4175_;
goto v_reusejp_4170_;
}
v_reusejp_4170_:
{
lean_object* v___x_4173_; 
if (v_isShared_4158_ == 0)
{
lean_ctor_set(v___x_4157_, 0, v___x_4171_);
v___x_4173_ = v___x_4157_;
goto v_reusejp_4172_;
}
else
{
lean_object* v_reuseFailAlloc_4174_; 
v_reuseFailAlloc_4174_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4174_, 0, v___x_4171_);
v___x_4173_ = v_reuseFailAlloc_4174_;
goto v_reusejp_4172_;
}
v_reusejp_4172_:
{
return v___x_4173_;
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
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___redArg___boxed(lean_object* v_cmd_4182_, lean_object* v_x_4183_, lean_object* v_a_4184_, lean_object* v_a_4185_){
_start:
{
lean_object* v_res_4186_; 
v_res_4186_ = lp_plausible_Plausible_retry___redArg(v_cmd_4182_, v_x_4183_, v_a_4184_, v_a_4185_);
lean_dec(v_a_4185_);
return v_res_4186_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry(lean_object* v_p_4187_, lean_object* v_cmd_4188_, lean_object* v_x_4189_, lean_object* v_a_4190_, lean_object* v_a_4191_){
_start:
{
lean_object* v___x_4192_; 
v___x_4192_ = lp_plausible_Plausible_retry___redArg(v_cmd_4188_, v_x_4189_, v_a_4190_, v_a_4191_);
return v___x_4192_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_retry___boxed(lean_object* v_p_4193_, lean_object* v_cmd_4194_, lean_object* v_x_4195_, lean_object* v_a_4196_, lean_object* v_a_4197_){
_start:
{
lean_object* v_res_4198_; 
v_res_4198_ = lp_plausible_Plausible_retry(v_p_4193_, v_cmd_4194_, v_x_4195_, v_a_4196_, v_a_4197_);
lean_dec(v_a_4197_);
return v_res_4198_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___redArg(lean_object* v_x_4199_, lean_object* v_x_4200_){
_start:
{
if (lean_obj_tag(v_x_4200_) == 1)
{
lean_object* v_a_4201_; lean_object* v___x_4203_; uint8_t v_isShared_4204_; uint8_t v_isSharedCheck_4209_; 
v_a_4201_ = lean_ctor_get(v_x_4200_, 0);
v_isSharedCheck_4209_ = !lean_is_exclusive(v_x_4200_);
if (v_isSharedCheck_4209_ == 0)
{
v___x_4203_ = v_x_4200_;
v_isShared_4204_ = v_isSharedCheck_4209_;
goto v_resetjp_4202_;
}
else
{
lean_inc(v_a_4201_);
lean_dec(v_x_4200_);
v___x_4203_ = lean_box(0);
v_isShared_4204_ = v_isSharedCheck_4209_;
goto v_resetjp_4202_;
}
v_resetjp_4202_:
{
lean_object* v___x_4205_; lean_object* v___x_4207_; 
v___x_4205_ = lean_nat_add(v_a_4201_, v_x_4199_);
lean_dec(v_a_4201_);
if (v_isShared_4204_ == 0)
{
lean_ctor_set(v___x_4203_, 0, v___x_4205_);
v___x_4207_ = v___x_4203_;
goto v_reusejp_4206_;
}
else
{
lean_object* v_reuseFailAlloc_4208_; 
v_reuseFailAlloc_4208_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4208_, 0, v___x_4205_);
v___x_4207_ = v_reuseFailAlloc_4208_;
goto v_reusejp_4206_;
}
v_reusejp_4206_:
{
return v___x_4207_;
}
}
}
else
{
return v_x_4200_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___redArg___boxed(lean_object* v_x_4210_, lean_object* v_x_4211_){
_start:
{
lean_object* v_res_4212_; 
v_res_4212_ = lp_plausible_Plausible_giveUp___redArg(v_x_4210_, v_x_4211_);
lean_dec(v_x_4210_);
return v_res_4212_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp(lean_object* v_p_4213_, lean_object* v_x_4214_, lean_object* v_x_4215_){
_start:
{
lean_object* v___x_4216_; 
v___x_4216_ = lp_plausible_Plausible_giveUp___redArg(v_x_4214_, v_x_4215_);
return v___x_4216_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_giveUp___boxed(lean_object* v_p_4217_, lean_object* v_x_4218_, lean_object* v_x_4219_){
_start:
{
lean_object* v_res_4220_; 
v_res_4220_ = lp_plausible_Plausible_giveUp(v_p_4217_, v_x_4218_, v_x_4219_);
lean_dec(v_x_4218_);
return v_res_4220_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0(lean_object* v_numInst_4221_, lean_object* v_n_4222_, lean_object* v_maxSize_4223_, lean_object* v_x_4224_){
_start:
{
lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v___x_4228_; lean_object* v___x_4229_; 
v___x_4225_ = lean_nat_sub(v_numInst_4221_, v_n_4222_);
v___x_4226_ = lean_unsigned_to_nat(1u);
v___x_4227_ = lean_nat_sub(v___x_4225_, v___x_4226_);
lean_dec(v___x_4225_);
v___x_4228_ = lean_nat_mul(v___x_4227_, v_maxSize_4223_);
lean_dec(v___x_4227_);
v___x_4229_ = lean_nat_div(v___x_4228_, v_numInst_4221_);
lean_dec(v___x_4228_);
return v___x_4229_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0___boxed(lean_object* v_numInst_4230_, lean_object* v_n_4231_, lean_object* v_maxSize_4232_, lean_object* v_x_4233_){
_start:
{
lean_object* v_res_4234_; 
v_res_4234_ = lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0(v_numInst_4230_, v_n_4231_, v_maxSize_4232_, v_x_4233_);
lean_dec(v_x_4233_);
lean_dec(v_maxSize_4232_);
lean_dec(v_n_4231_);
lean_dec(v_numInst_4230_);
return v_res_4234_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg(lean_object* v_inst_4238_, lean_object* v_cfg_4239_, lean_object* v_x_4240_, lean_object* v_x_4241_, lean_object* v_a_4242_, lean_object* v_a_4243_){
_start:
{
lean_object* v___y_4245_; lean_object* v___y_4246_; lean_object* v_zero_4249_; uint8_t v_isZero_4250_; 
v_zero_4249_ = lean_unsigned_to_nat(0u);
v_isZero_4250_ = lean_nat_dec_eq(v_x_4241_, v_zero_4249_);
if (v_isZero_4250_ == 1)
{
lean_object* v___x_4251_; lean_object* v___x_4252_; 
lean_dec(v_x_4241_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
v___x_4251_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4251_, 0, v_x_4240_);
lean_ctor_set(v___x_4251_, 1, v_a_4242_);
v___x_4252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4252_, 0, v___x_4251_);
return v___x_4252_;
}
else
{
lean_object* v_numInst_4253_; lean_object* v_maxSize_4254_; lean_object* v_numRetries_4255_; uint8_t v_traceSuccesses_4256_; lean_object* v_one_4257_; lean_object* v_n_4258_; lean_object* v_size_4259_; lean_object* v___y_4261_; lean_object* v___y_4262_; 
v_numInst_4253_ = lean_ctor_get(v_cfg_4239_, 0);
v_maxSize_4254_ = lean_ctor_get(v_cfg_4239_, 1);
v_numRetries_4255_ = lean_ctor_get(v_cfg_4239_, 2);
v_traceSuccesses_4256_ = lean_ctor_get_uint8(v_cfg_4239_, sizeof(void*)*4 + 1);
v_one_4257_ = lean_unsigned_to_nat(1u);
v_n_4258_ = lean_nat_sub(v_x_4241_, v_one_4257_);
lean_dec(v_x_4241_);
lean_inc(v_maxSize_4254_);
lean_inc(v_n_4258_);
lean_inc(v_numInst_4253_);
v_size_4259_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_runSuiteAux___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v_size_4259_, 0, v_numInst_4253_);
lean_closure_set(v_size_4259_, 1, v_n_4258_);
lean_closure_set(v_size_4259_, 2, v_maxSize_4254_);
if (v_traceSuccesses_4256_ == 0)
{
v___y_4261_ = v_a_4242_;
v___y_4262_ = v_a_4243_;
goto v___jp_4260_;
}
else
{
lean_object* v___x_4279_; lean_object* v___x_4280_; lean_object* v___x_1472__overap_4281_; lean_object* v___x_4282_; 
v___x_4279_ = lean_obj_once(&lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11, &lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11_once, _init_lp_plausible_Plausible_Testable_decGuardTestable___redArg___lam__0___closed__11);
v___x_4280_ = ((lean_object*)(lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__0));
v___x_1472__overap_4281_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_4279_, v___x_4280_);
lean_inc(v_a_4243_);
v___x_4282_ = lean_apply_2(v___x_1472__overap_4281_, v_a_4242_, v_a_4243_);
if (lean_obj_tag(v___x_4282_) == 0)
{
lean_object* v_a_4283_; lean_object* v___x_4285_; uint8_t v_isShared_4286_; uint8_t v_isSharedCheck_4290_; 
lean_dec_ref(v_size_4259_);
lean_dec(v_n_4258_);
lean_dec_ref(v_x_4240_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
v_a_4283_ = lean_ctor_get(v___x_4282_, 0);
v_isSharedCheck_4290_ = !lean_is_exclusive(v___x_4282_);
if (v_isSharedCheck_4290_ == 0)
{
v___x_4285_ = v___x_4282_;
v_isShared_4286_ = v_isSharedCheck_4290_;
goto v_resetjp_4284_;
}
else
{
lean_inc(v_a_4283_);
lean_dec(v___x_4282_);
v___x_4285_ = lean_box(0);
v_isShared_4286_ = v_isSharedCheck_4290_;
goto v_resetjp_4284_;
}
v_resetjp_4284_:
{
lean_object* v___x_4288_; 
if (v_isShared_4286_ == 0)
{
v___x_4288_ = v___x_4285_;
goto v_reusejp_4287_;
}
else
{
lean_object* v_reuseFailAlloc_4289_; 
v_reuseFailAlloc_4289_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4289_, 0, v_a_4283_);
v___x_4288_ = v_reuseFailAlloc_4289_;
goto v_reusejp_4287_;
}
v_reusejp_4287_:
{
return v___x_4288_;
}
}
}
else
{
lean_object* v_a_4291_; lean_object* v_snd_4292_; lean_object* v___x_4293_; lean_object* v___x_4294_; lean_object* v___x_4295_; lean_object* v___x_4296_; lean_object* v___x_4297_; lean_object* v___x_1484__overap_4298_; lean_object* v___x_4299_; 
v_a_4291_ = lean_ctor_get(v___x_4282_, 0);
lean_inc(v_a_4291_);
lean_dec_ref_known(v___x_4282_, 1);
v_snd_4292_ = lean_ctor_get(v_a_4291_, 1);
lean_inc(v_snd_4292_);
lean_dec(v_a_4291_);
v___x_4293_ = ((lean_object*)(lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__1));
lean_inc(v_numRetries_4255_);
v___x_4294_ = l_Nat_reprFast(v_numRetries_4255_);
v___x_4295_ = lean_string_append(v___x_4293_, v___x_4294_);
lean_dec_ref(v___x_4294_);
v___x_4296_ = ((lean_object*)(lp_plausible_Plausible_Testable_runSuiteAux___redArg___closed__2));
v___x_4297_ = lean_string_append(v___x_4295_, v___x_4296_);
v___x_1484__overap_4298_ = lp_plausible_Plausible_Testable_slimTrace___redArg(v___x_4279_, v___x_4297_);
lean_dec_ref(v___x_4297_);
lean_inc(v_a_4243_);
v___x_4299_ = lean_apply_2(v___x_1484__overap_4298_, v_snd_4292_, v_a_4243_);
if (lean_obj_tag(v___x_4299_) == 0)
{
lean_object* v_a_4300_; lean_object* v___x_4302_; uint8_t v_isShared_4303_; uint8_t v_isSharedCheck_4307_; 
lean_dec_ref(v_size_4259_);
lean_dec(v_n_4258_);
lean_dec_ref(v_x_4240_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
v_a_4300_ = lean_ctor_get(v___x_4299_, 0);
v_isSharedCheck_4307_ = !lean_is_exclusive(v___x_4299_);
if (v_isSharedCheck_4307_ == 0)
{
v___x_4302_ = v___x_4299_;
v_isShared_4303_ = v_isSharedCheck_4307_;
goto v_resetjp_4301_;
}
else
{
lean_inc(v_a_4300_);
lean_dec(v___x_4299_);
v___x_4302_ = lean_box(0);
v_isShared_4303_ = v_isSharedCheck_4307_;
goto v_resetjp_4301_;
}
v_resetjp_4301_:
{
lean_object* v___x_4305_; 
if (v_isShared_4303_ == 0)
{
v___x_4305_ = v___x_4302_;
goto v_reusejp_4304_;
}
else
{
lean_object* v_reuseFailAlloc_4306_; 
v_reuseFailAlloc_4306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4306_, 0, v_a_4300_);
v___x_4305_ = v_reuseFailAlloc_4306_;
goto v_reusejp_4304_;
}
v_reusejp_4304_:
{
return v___x_4305_;
}
}
}
else
{
lean_object* v_a_4308_; lean_object* v_snd_4309_; 
v_a_4308_ = lean_ctor_get(v___x_4299_, 0);
lean_inc(v_a_4308_);
lean_dec_ref_known(v___x_4299_, 1);
v_snd_4309_ = lean_ctor_get(v_a_4308_, 1);
lean_inc(v_snd_4309_);
lean_dec(v_a_4308_);
v___y_4261_ = v_snd_4309_;
v___y_4262_ = v_a_4243_;
goto v___jp_4260_;
}
}
}
v___jp_4260_:
{
uint8_t v___x_4263_; lean_object* v___x_4264_; lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4263_ = 1;
v___x_4264_ = lean_box(v___x_4263_);
lean_inc_ref(v_cfg_4239_);
lean_inc_ref(v_inst_4238_);
v___x_4265_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_runProp___boxed), 6, 4);
lean_closure_set(v___x_4265_, 0, lean_box(0));
lean_closure_set(v___x_4265_, 1, v_inst_4238_);
lean_closure_set(v___x_4265_, 2, v_cfg_4239_);
lean_closure_set(v___x_4265_, 3, v___x_4264_);
v___x_4266_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Gen_resize___boxed), 5, 3);
lean_closure_set(v___x_4266_, 0, lean_box(0));
lean_closure_set(v___x_4266_, 1, v_size_4259_);
lean_closure_set(v___x_4266_, 2, v___x_4265_);
lean_inc(v_numRetries_4255_);
v___x_4267_ = lp_plausible_Plausible_retry___redArg(v___x_4266_, v_numRetries_4255_, v___y_4261_, v___y_4262_);
if (lean_obj_tag(v___x_4267_) == 0)
{
lean_dec(v_n_4258_);
lean_dec_ref(v_x_4240_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
return v___x_4267_;
}
else
{
lean_object* v_a_4268_; lean_object* v_fst_4269_; 
v_a_4268_ = lean_ctor_get(v___x_4267_, 0);
lean_inc(v_a_4268_);
lean_dec_ref_known(v___x_4267_, 1);
v_fst_4269_ = lean_ctor_get(v_a_4268_, 0);
lean_inc(v_fst_4269_);
switch(lean_obj_tag(v_fst_4269_))
{
case 0:
{
lean_object* v_a_4270_; 
lean_dec_ref(v_x_4240_);
v_a_4270_ = lean_ctor_get(v_fst_4269_, 0);
if (lean_obj_tag(v_a_4270_) == 0)
{
lean_object* v_snd_4271_; 
v_snd_4271_ = lean_ctor_get(v_a_4268_, 1);
lean_inc(v_snd_4271_);
lean_dec(v_a_4268_);
v_x_4240_ = v_fst_4269_;
v_x_4241_ = v_n_4258_;
v_a_4242_ = v_snd_4271_;
v_a_4243_ = v___y_4262_;
goto _start;
}
else
{
lean_object* v_snd_4273_; 
lean_dec(v_n_4258_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
v_snd_4273_ = lean_ctor_get(v_a_4268_, 1);
lean_inc(v_snd_4273_);
lean_dec(v_a_4268_);
v___y_4245_ = v_fst_4269_;
v___y_4246_ = v_snd_4273_;
goto v___jp_4244_;
}
}
case 1:
{
lean_object* v_snd_4274_; lean_object* v_a_4275_; lean_object* v___x_4276_; 
v_snd_4274_ = lean_ctor_get(v_a_4268_, 1);
lean_inc(v_snd_4274_);
lean_dec(v_a_4268_);
v_a_4275_ = lean_ctor_get(v_fst_4269_, 0);
lean_inc(v_a_4275_);
lean_dec_ref_known(v_fst_4269_, 1);
v___x_4276_ = lp_plausible_Plausible_giveUp___redArg(v_a_4275_, v_x_4240_);
lean_dec(v_a_4275_);
v_x_4240_ = v___x_4276_;
v_x_4241_ = v_n_4258_;
v_a_4242_ = v_snd_4274_;
v_a_4243_ = v___y_4262_;
goto _start;
}
default: 
{
lean_object* v_snd_4278_; 
lean_dec(v_n_4258_);
lean_dec_ref(v_x_4240_);
lean_dec_ref(v_cfg_4239_);
lean_dec_ref(v_inst_4238_);
v_snd_4278_ = lean_ctor_get(v_a_4268_, 1);
lean_inc(v_snd_4278_);
lean_dec(v_a_4268_);
v___y_4245_ = v_fst_4269_;
v___y_4246_ = v_snd_4278_;
goto v___jp_4244_;
}
}
}
}
}
v___jp_4244_:
{
lean_object* v___x_4247_; lean_object* v___x_4248_; 
v___x_4247_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4247_, 0, v___y_4245_);
lean_ctor_set(v___x_4247_, 1, v___y_4246_);
v___x_4248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4248_, 0, v___x_4247_);
return v___x_4248_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___redArg___boxed(lean_object* v_inst_4310_, lean_object* v_cfg_4311_, lean_object* v_x_4312_, lean_object* v_x_4313_, lean_object* v_a_4314_, lean_object* v_a_4315_){
_start:
{
lean_object* v_res_4316_; 
v_res_4316_ = lp_plausible_Plausible_Testable_runSuiteAux___redArg(v_inst_4310_, v_cfg_4311_, v_x_4312_, v_x_4313_, v_a_4314_, v_a_4315_);
lean_dec(v_a_4315_);
return v_res_4316_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux(lean_object* v_p_4317_, lean_object* v_inst_4318_, lean_object* v_cfg_4319_, lean_object* v_x_4320_, lean_object* v_x_4321_, lean_object* v_a_4322_, lean_object* v_a_4323_){
_start:
{
lean_object* v___x_4324_; 
v___x_4324_ = lp_plausible_Plausible_Testable_runSuiteAux___redArg(v_inst_4318_, v_cfg_4319_, v_x_4320_, v_x_4321_, v_a_4322_, v_a_4323_);
return v___x_4324_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuiteAux___boxed(lean_object* v_p_4325_, lean_object* v_inst_4326_, lean_object* v_cfg_4327_, lean_object* v_x_4328_, lean_object* v_x_4329_, lean_object* v_a_4330_, lean_object* v_a_4331_){
_start:
{
lean_object* v_res_4332_; 
v_res_4332_ = lp_plausible_Plausible_Testable_runSuiteAux(v_p_4325_, v_inst_4326_, v_cfg_4327_, v_x_4328_, v_x_4329_, v_a_4330_, v_a_4331_);
lean_dec(v_a_4331_);
return v_res_4332_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___redArg(lean_object* v_inst_4333_, lean_object* v_cfg_4334_, lean_object* v_a_4335_, lean_object* v_a_4336_){
_start:
{
lean_object* v_numInst_4337_; lean_object* v___x_4338_; lean_object* v___x_4339_; 
v_numInst_4337_ = lean_ctor_get(v_cfg_4334_, 0);
lean_inc(v_numInst_4337_);
v___x_4338_ = ((lean_object*)(lp_plausible_Plausible_instInhabitedTestResult_default___closed__0));
v___x_4339_ = lp_plausible_Plausible_Testable_runSuiteAux___redArg(v_inst_4333_, v_cfg_4334_, v___x_4338_, v_numInst_4337_, v_a_4335_, v_a_4336_);
return v___x_4339_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___redArg___boxed(lean_object* v_inst_4340_, lean_object* v_cfg_4341_, lean_object* v_a_4342_, lean_object* v_a_4343_){
_start:
{
lean_object* v_res_4344_; 
v_res_4344_ = lp_plausible_Plausible_Testable_runSuite___redArg(v_inst_4340_, v_cfg_4341_, v_a_4342_, v_a_4343_);
lean_dec(v_a_4343_);
return v_res_4344_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite(lean_object* v_p_4345_, lean_object* v_inst_4346_, lean_object* v_cfg_4347_, lean_object* v_a_4348_, lean_object* v_a_4349_){
_start:
{
lean_object* v___x_4350_; 
v___x_4350_ = lp_plausible_Plausible_Testable_runSuite___redArg(v_inst_4346_, v_cfg_4347_, v_a_4348_, v_a_4349_);
return v___x_4350_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_runSuite___boxed(lean_object* v_p_4351_, lean_object* v_inst_4352_, lean_object* v_cfg_4353_, lean_object* v_a_4354_, lean_object* v_a_4355_){
_start:
{
lean_object* v_res_4356_; 
v_res_4356_ = lp_plausible_Plausible_Testable_runSuite(v_p_4351_, v_inst_4352_, v_cfg_4353_, v_a_4354_, v_a_4355_);
lean_dec(v_a_4355_);
return v_res_4356_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___lam__0(lean_object* v_inst_4357_, lean_object* v_cfg_4358_, lean_object* v___y_4359_){
_start:
{
lean_object* v___x_4361_; lean_object* v___x_4362_; lean_object* v___x_4363_; 
v___x_4361_ = lean_unsigned_to_nat(0u);
v___x_4362_ = lp_plausible_Plausible_Testable_runSuite___redArg(v_inst_4357_, v_cfg_4358_, v___y_4359_, v___x_4361_);
v___x_4363_ = lp_plausible___private_Plausible_Gen_0__Plausible_errorOfGenError(lean_box(0), v___x_4362_);
return v___x_4363_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___lam__0___boxed(lean_object* v_inst_4364_, lean_object* v_cfg_4365_, lean_object* v___y_4366_, lean_object* v___y_4367_){
_start:
{
lean_object* v_res_4368_; 
v_res_4368_ = lp_plausible_Plausible_Testable_checkIO___redArg___lam__0(v_inst_4364_, v_cfg_4365_, v___y_4366_);
return v_res_4368_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_checkIO___redArg___closed__0(void){
_start:
{
lean_object* v___x_4369_; 
v___x_4369_ = l_instMonadEIO(lean_box(0));
return v___x_4369_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg(lean_object* v_inst_4370_, lean_object* v_cfg_4371_){
_start:
{
lean_object* v___x_4373_; lean_object* v_randomSeed_4374_; 
v___x_4373_ = lean_obj_once(&lp_plausible_Plausible_Testable_checkIO___redArg___closed__0, &lp_plausible_Plausible_Testable_checkIO___redArg___closed__0_once, _init_lp_plausible_Plausible_Testable_checkIO___redArg___closed__0);
v_randomSeed_4374_ = lean_ctor_get(v_cfg_4371_, 3);
if (lean_obj_tag(v_randomSeed_4374_) == 0)
{
lean_object* v___x_4375_; lean_object* v___x_4376_; lean_object* v___x_4377_; 
v___x_4375_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_runSuite___boxed), 5, 3);
lean_closure_set(v___x_4375_, 0, lean_box(0));
lean_closure_set(v___x_4375_, 1, v_inst_4370_);
lean_closure_set(v___x_4375_, 2, v_cfg_4371_);
v___x_4376_ = lean_unsigned_to_nat(0u);
v___x_4377_ = lp_plausible_Plausible_Gen_run___redArg(v___x_4375_, v___x_4376_);
return v___x_4377_;
}
else
{
lean_object* v_val_4378_; lean_object* v___f_4379_; lean_object* v___x_117__overap_4380_; lean_object* v___x_4381_; 
v_val_4378_ = lean_ctor_get(v_randomSeed_4374_, 0);
lean_inc(v_val_4378_);
v___f_4379_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Testable_checkIO___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_4379_, 0, v_inst_4370_);
lean_closure_set(v___f_4379_, 1, v_cfg_4371_);
v___x_117__overap_4380_ = lp_plausible_Plausible_runRandWith___redArg(v___x_4373_, v_val_4378_, v___f_4379_);
lean_dec(v_val_4378_);
v___x_4381_ = lean_apply_1(v___x_117__overap_4380_, lean_box(0));
return v___x_4381_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___redArg___boxed(lean_object* v_inst_4382_, lean_object* v_cfg_4383_, lean_object* v_a_4384_){
_start:
{
lean_object* v_res_4385_; 
v_res_4385_ = lp_plausible_Plausible_Testable_checkIO___redArg(v_inst_4382_, v_cfg_4383_);
return v_res_4385_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO(lean_object* v_p_4386_, lean_object* v_inst_4387_, lean_object* v_cfg_4388_){
_start:
{
lean_object* v___x_4390_; 
v___x_4390_ = lp_plausible_Plausible_Testable_checkIO___redArg(v_inst_4387_, v_cfg_4388_);
return v___x_4390_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_checkIO___boxed(lean_object* v_p_4391_, lean_object* v_inst_4392_, lean_object* v_cfg_4393_, lean_object* v_a_4394_){
_start:
{
lean_object* v_res_4395_; 
v_res_4395_ = lp_plausible_Plausible_Testable_checkIO(v_p_4391_, v_inst_4392_, v_cfg_4393_);
return v_res_4395_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0(lean_object* v_k_4396_, lean_object* v_b_4397_, lean_object* v___y_4398_, lean_object* v___y_4399_, lean_object* v___y_4400_, lean_object* v___y_4401_){
_start:
{
lean_object* v___x_4403_; 
lean_inc(v___y_4401_);
lean_inc_ref(v___y_4400_);
lean_inc(v___y_4399_);
lean_inc_ref(v___y_4398_);
v___x_4403_ = lean_apply_6(v_k_4396_, v_b_4397_, v___y_4398_, v___y_4399_, v___y_4400_, v___y_4401_, lean_box(0));
return v___x_4403_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0___boxed(lean_object* v_k_4404_, lean_object* v_b_4405_, lean_object* v___y_4406_, lean_object* v___y_4407_, lean_object* v___y_4408_, lean_object* v___y_4409_, lean_object* v___y_4410_){
_start:
{
lean_object* v_res_4411_; 
v_res_4411_ = lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0(v_k_4404_, v_b_4405_, v___y_4406_, v___y_4407_, v___y_4408_, v___y_4409_);
lean_dec(v___y_4409_);
lean_dec_ref(v___y_4408_);
lean_dec(v___y_4407_);
lean_dec_ref(v___y_4406_);
return v_res_4411_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg(lean_object* v_name_4412_, uint8_t v_bi_4413_, lean_object* v_type_4414_, lean_object* v_k_4415_, uint8_t v_kind_4416_, lean_object* v___y_4417_, lean_object* v___y_4418_, lean_object* v___y_4419_, lean_object* v___y_4420_){
_start:
{
lean_object* v___f_4422_; lean_object* v___x_4423_; 
v___f_4422_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_4422_, 0, v_k_4415_);
v___x_4423_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_4412_, v_bi_4413_, v_type_4414_, v___f_4422_, v_kind_4416_, v___y_4417_, v___y_4418_, v___y_4419_, v___y_4420_);
if (lean_obj_tag(v___x_4423_) == 0)
{
lean_object* v_a_4424_; lean_object* v___x_4426_; uint8_t v_isShared_4427_; uint8_t v_isSharedCheck_4431_; 
v_a_4424_ = lean_ctor_get(v___x_4423_, 0);
v_isSharedCheck_4431_ = !lean_is_exclusive(v___x_4423_);
if (v_isSharedCheck_4431_ == 0)
{
v___x_4426_ = v___x_4423_;
v_isShared_4427_ = v_isSharedCheck_4431_;
goto v_resetjp_4425_;
}
else
{
lean_inc(v_a_4424_);
lean_dec(v___x_4423_);
v___x_4426_ = lean_box(0);
v_isShared_4427_ = v_isSharedCheck_4431_;
goto v_resetjp_4425_;
}
v_resetjp_4425_:
{
lean_object* v___x_4429_; 
if (v_isShared_4427_ == 0)
{
v___x_4429_ = v___x_4426_;
goto v_reusejp_4428_;
}
else
{
lean_object* v_reuseFailAlloc_4430_; 
v_reuseFailAlloc_4430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4430_, 0, v_a_4424_);
v___x_4429_ = v_reuseFailAlloc_4430_;
goto v_reusejp_4428_;
}
v_reusejp_4428_:
{
return v___x_4429_;
}
}
}
else
{
lean_object* v_a_4432_; lean_object* v___x_4434_; uint8_t v_isShared_4435_; uint8_t v_isSharedCheck_4439_; 
v_a_4432_ = lean_ctor_get(v___x_4423_, 0);
v_isSharedCheck_4439_ = !lean_is_exclusive(v___x_4423_);
if (v_isSharedCheck_4439_ == 0)
{
v___x_4434_ = v___x_4423_;
v_isShared_4435_ = v_isSharedCheck_4439_;
goto v_resetjp_4433_;
}
else
{
lean_inc(v_a_4432_);
lean_dec(v___x_4423_);
v___x_4434_ = lean_box(0);
v_isShared_4435_ = v_isSharedCheck_4439_;
goto v_resetjp_4433_;
}
v_resetjp_4433_:
{
lean_object* v___x_4437_; 
if (v_isShared_4435_ == 0)
{
v___x_4437_ = v___x_4434_;
goto v_reusejp_4436_;
}
else
{
lean_object* v_reuseFailAlloc_4438_; 
v_reuseFailAlloc_4438_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4438_, 0, v_a_4432_);
v___x_4437_ = v_reuseFailAlloc_4438_;
goto v_reusejp_4436_;
}
v_reusejp_4436_:
{
return v___x_4437_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg___boxed(lean_object* v_name_4440_, lean_object* v_bi_4441_, lean_object* v_type_4442_, lean_object* v_k_4443_, lean_object* v_kind_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_, lean_object* v___y_4447_, lean_object* v___y_4448_, lean_object* v___y_4449_){
_start:
{
uint8_t v_bi_boxed_4450_; uint8_t v_kind_boxed_4451_; lean_object* v_res_4452_; 
v_bi_boxed_4450_ = lean_unbox(v_bi_4441_);
v_kind_boxed_4451_ = lean_unbox(v_kind_4444_);
v_res_4452_ = lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg(v_name_4440_, v_bi_boxed_4450_, v_type_4442_, v_k_4443_, v_kind_boxed_4451_, v___y_4445_, v___y_4446_, v___y_4447_, v___y_4448_);
lean_dec(v___y_4448_);
lean_dec_ref(v___y_4447_);
lean_dec(v___y_4446_);
lean_dec_ref(v___y_4445_);
return v_res_4452_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0(lean_object* v_00_u03b1_4453_, lean_object* v_name_4454_, uint8_t v_bi_4455_, lean_object* v_type_4456_, lean_object* v_k_4457_, uint8_t v_kind_4458_, lean_object* v___y_4459_, lean_object* v___y_4460_, lean_object* v___y_4461_, lean_object* v___y_4462_){
_start:
{
lean_object* v___x_4464_; 
v___x_4464_ = lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg(v_name_4454_, v_bi_4455_, v_type_4456_, v_k_4457_, v_kind_4458_, v___y_4459_, v___y_4460_, v___y_4461_, v___y_4462_);
return v___x_4464_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___boxed(lean_object* v_00_u03b1_4465_, lean_object* v_name_4466_, lean_object* v_bi_4467_, lean_object* v_type_4468_, lean_object* v_k_4469_, lean_object* v_kind_4470_, lean_object* v___y_4471_, lean_object* v___y_4472_, lean_object* v___y_4473_, lean_object* v___y_4474_, lean_object* v___y_4475_){
_start:
{
uint8_t v_bi_boxed_4476_; uint8_t v_kind_boxed_4477_; lean_object* v_res_4478_; 
v_bi_boxed_4476_ = lean_unbox(v_bi_4467_);
v_kind_boxed_4477_ = lean_unbox(v_kind_4470_);
v_res_4478_ = lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0(v_00_u03b1_4465_, v_name_4466_, v_bi_boxed_4476_, v_type_4468_, v_k_4469_, v_kind_boxed_4477_, v___y_4471_, v___y_4472_, v___y_4473_, v___y_4474_);
lean_dec(v___y_4474_);
lean_dec_ref(v___y_4473_);
lean_dec(v___y_4472_);
lean_dec_ref(v___y_4471_);
return v_res_4478_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__0(lean_object* v_e_4479_, lean_object* v___y_4480_, lean_object* v___y_4481_, lean_object* v___y_4482_, lean_object* v___y_4483_){
_start:
{
lean_object* v___x_4485_; lean_object* v___x_4486_; 
v___x_4485_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4485_, 0, v_e_4479_);
v___x_4486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4486_, 0, v___x_4485_);
return v___x_4486_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__0___boxed(lean_object* v_e_4487_, lean_object* v___y_4488_, lean_object* v___y_4489_, lean_object* v___y_4490_, lean_object* v___y_4491_, lean_object* v___y_4492_){
_start:
{
lean_object* v_res_4493_; 
v_res_4493_ = lp_plausible_Plausible_Decorations_addDecorations___lam__0(v_e_4487_, v___y_4488_, v___y_4489_, v___y_4490_, v___y_4491_);
lean_dec(v___y_4491_);
lean_dec_ref(v___y_4490_);
lean_dec(v___y_4489_);
lean_dec_ref(v___y_4488_);
return v_res_4493_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0(lean_object* v_k_4494_, lean_object* v___y_4495_, lean_object* v_b_4496_, lean_object* v___y_4497_, lean_object* v___y_4498_, lean_object* v___y_4499_, lean_object* v___y_4500_){
_start:
{
lean_object* v___x_4502_; 
lean_inc(v___y_4500_);
lean_inc_ref(v___y_4499_);
lean_inc(v___y_4498_);
lean_inc_ref(v___y_4497_);
lean_inc(v___y_4495_);
v___x_4502_ = lean_apply_7(v_k_4494_, v_b_4496_, v___y_4495_, v___y_4497_, v___y_4498_, v___y_4499_, v___y_4500_, lean_box(0));
return v___x_4502_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0___boxed(lean_object* v_k_4503_, lean_object* v___y_4504_, lean_object* v_b_4505_, lean_object* v___y_4506_, lean_object* v___y_4507_, lean_object* v___y_4508_, lean_object* v___y_4509_, lean_object* v___y_4510_){
_start:
{
lean_object* v_res_4511_; 
v_res_4511_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0(v_k_4503_, v___y_4504_, v_b_4505_, v___y_4506_, v___y_4507_, v___y_4508_, v___y_4509_);
lean_dec(v___y_4509_);
lean_dec_ref(v___y_4508_);
lean_dec(v___y_4507_);
lean_dec_ref(v___y_4506_);
lean_dec(v___y_4504_);
return v_res_4511_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(lean_object* v_name_4512_, uint8_t v_bi_4513_, lean_object* v_type_4514_, lean_object* v_k_4515_, uint8_t v_kind_4516_, lean_object* v___y_4517_, lean_object* v___y_4518_, lean_object* v___y_4519_, lean_object* v___y_4520_, lean_object* v___y_4521_){
_start:
{
lean_object* v___f_4523_; lean_object* v___x_4524_; 
lean_inc(v___y_4517_);
v___f_4523_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_4523_, 0, v_k_4515_);
lean_closure_set(v___f_4523_, 1, v___y_4517_);
v___x_4524_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_4512_, v_bi_4513_, v_type_4514_, v___f_4523_, v_kind_4516_, v___y_4518_, v___y_4519_, v___y_4520_, v___y_4521_);
if (lean_obj_tag(v___x_4524_) == 0)
{
return v___x_4524_;
}
else
{
lean_object* v_a_4525_; lean_object* v___x_4527_; uint8_t v_isShared_4528_; uint8_t v_isSharedCheck_4532_; 
v_a_4525_ = lean_ctor_get(v___x_4524_, 0);
v_isSharedCheck_4532_ = !lean_is_exclusive(v___x_4524_);
if (v_isSharedCheck_4532_ == 0)
{
v___x_4527_ = v___x_4524_;
v_isShared_4528_ = v_isSharedCheck_4532_;
goto v_resetjp_4526_;
}
else
{
lean_inc(v_a_4525_);
lean_dec(v___x_4524_);
v___x_4527_ = lean_box(0);
v_isShared_4528_ = v_isSharedCheck_4532_;
goto v_resetjp_4526_;
}
v_resetjp_4526_:
{
lean_object* v___x_4530_; 
if (v_isShared_4528_ == 0)
{
v___x_4530_ = v___x_4527_;
goto v_reusejp_4529_;
}
else
{
lean_object* v_reuseFailAlloc_4531_; 
v_reuseFailAlloc_4531_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4531_, 0, v_a_4525_);
v___x_4530_ = v_reuseFailAlloc_4531_;
goto v_reusejp_4529_;
}
v_reusejp_4529_:
{
return v___x_4530_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___boxed(lean_object* v_name_4533_, lean_object* v_bi_4534_, lean_object* v_type_4535_, lean_object* v_k_4536_, lean_object* v_kind_4537_, lean_object* v___y_4538_, lean_object* v___y_4539_, lean_object* v___y_4540_, lean_object* v___y_4541_, lean_object* v___y_4542_, lean_object* v___y_4543_){
_start:
{
uint8_t v_bi_boxed_4544_; uint8_t v_kind_boxed_4545_; lean_object* v_res_4546_; 
v_bi_boxed_4544_ = lean_unbox(v_bi_4534_);
v_kind_boxed_4545_ = lean_unbox(v_kind_4537_);
v_res_4546_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(v_name_4533_, v_bi_boxed_4544_, v_type_4535_, v_k_4536_, v_kind_boxed_4545_, v___y_4538_, v___y_4539_, v___y_4540_, v___y_4541_, v___y_4542_);
lean_dec(v___y_4542_);
lean_dec_ref(v___y_4541_);
lean_dec(v___y_4540_);
lean_dec_ref(v___y_4539_);
lean_dec(v___y_4538_);
return v_res_4546_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2(lean_object* v___x_4547_, lean_object* v___y_4548_, lean_object* v___y_4549_, lean_object* v___y_4550_, lean_object* v___y_4551_){
_start:
{
lean_object* v___x_4553_; 
v___x_4553_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4553_, 0, v___x_4547_);
return v___x_4553_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2___boxed(lean_object* v___x_4554_, lean_object* v___y_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_){
_start:
{
lean_object* v_res_4560_; 
v_res_4560_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2(v___x_4554_, v___y_4555_, v___y_4556_, v___y_4557_, v___y_4558_);
lean_dec(v___y_4558_);
lean_dec_ref(v___y_4557_);
lean_dec(v___y_4556_);
lean_dec_ref(v___y_4555_);
return v_res_4560_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg(lean_object* v_name_4561_, lean_object* v_type_4562_, lean_object* v_val_4563_, lean_object* v_k_4564_, uint8_t v_nondep_4565_, uint8_t v_kind_4566_, lean_object* v___y_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_, lean_object* v___y_4571_){
_start:
{
lean_object* v___f_4573_; lean_object* v___x_4574_; 
lean_inc(v___y_4567_);
v___f_4573_ = lean_alloc_closure((void*)(lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg___lam__0___boxed), 8, 2);
lean_closure_set(v___f_4573_, 0, v_k_4564_);
lean_closure_set(v___f_4573_, 1, v___y_4567_);
v___x_4574_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_4561_, v_type_4562_, v_val_4563_, v___f_4573_, v_nondep_4565_, v_kind_4566_, v___y_4568_, v___y_4569_, v___y_4570_, v___y_4571_);
if (lean_obj_tag(v___x_4574_) == 0)
{
return v___x_4574_;
}
else
{
lean_object* v_a_4575_; lean_object* v___x_4577_; uint8_t v_isShared_4578_; uint8_t v_isSharedCheck_4582_; 
v_a_4575_ = lean_ctor_get(v___x_4574_, 0);
v_isSharedCheck_4582_ = !lean_is_exclusive(v___x_4574_);
if (v_isSharedCheck_4582_ == 0)
{
v___x_4577_ = v___x_4574_;
v_isShared_4578_ = v_isSharedCheck_4582_;
goto v_resetjp_4576_;
}
else
{
lean_inc(v_a_4575_);
lean_dec(v___x_4574_);
v___x_4577_ = lean_box(0);
v_isShared_4578_ = v_isSharedCheck_4582_;
goto v_resetjp_4576_;
}
v_resetjp_4576_:
{
lean_object* v___x_4580_; 
if (v_isShared_4578_ == 0)
{
v___x_4580_ = v___x_4577_;
goto v_reusejp_4579_;
}
else
{
lean_object* v_reuseFailAlloc_4581_; 
v_reuseFailAlloc_4581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4581_, 0, v_a_4575_);
v___x_4580_ = v_reuseFailAlloc_4581_;
goto v_reusejp_4579_;
}
v_reusejp_4579_:
{
return v___x_4580_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg___boxed(lean_object* v_name_4583_, lean_object* v_type_4584_, lean_object* v_val_4585_, lean_object* v_k_4586_, lean_object* v_nondep_4587_, lean_object* v_kind_4588_, lean_object* v___y_4589_, lean_object* v___y_4590_, lean_object* v___y_4591_, lean_object* v___y_4592_, lean_object* v___y_4593_, lean_object* v___y_4594_){
_start:
{
uint8_t v_nondep_boxed_4595_; uint8_t v_kind_boxed_4596_; lean_object* v_res_4597_; 
v_nondep_boxed_4595_ = lean_unbox(v_nondep_4587_);
v_kind_boxed_4596_ = lean_unbox(v_kind_4588_);
v_res_4597_ = lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg(v_name_4583_, v_type_4584_, v_val_4585_, v_k_4586_, v_nondep_boxed_4595_, v_kind_boxed_4596_, v___y_4589_, v___y_4590_, v___y_4591_, v___y_4592_, v___y_4593_);
lean_dec(v___y_4593_);
lean_dec_ref(v___y_4592_);
lean_dec(v___y_4591_);
lean_dec_ref(v___y_4590_);
lean_dec(v___y_4589_);
return v_res_4597_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg(lean_object* v_a_4598_, lean_object* v_x_4599_){
_start:
{
if (lean_obj_tag(v_x_4599_) == 0)
{
lean_object* v___x_4600_; 
v___x_4600_ = lean_box(0);
return v___x_4600_;
}
else
{
lean_object* v_key_4601_; lean_object* v_value_4602_; lean_object* v_tail_4603_; uint8_t v___x_4604_; 
v_key_4601_ = lean_ctor_get(v_x_4599_, 0);
v_value_4602_ = lean_ctor_get(v_x_4599_, 1);
v_tail_4603_ = lean_ctor_get(v_x_4599_, 2);
v___x_4604_ = l_Lean_ExprStructEq_beq(v_key_4601_, v_a_4598_);
if (v___x_4604_ == 0)
{
v_x_4599_ = v_tail_4603_;
goto _start;
}
else
{
lean_object* v___x_4606_; 
lean_inc(v_value_4602_);
v___x_4606_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4606_, 0, v_value_4602_);
return v___x_4606_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg___boxed(lean_object* v_a_4607_, lean_object* v_x_4608_){
_start:
{
lean_object* v_res_4609_; 
v_res_4609_ = lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg(v_a_4607_, v_x_4608_);
lean_dec(v_x_4608_);
lean_dec_ref(v_a_4607_);
return v_res_4609_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg(lean_object* v_m_4610_, lean_object* v_a_4611_){
_start:
{
lean_object* v_buckets_4612_; lean_object* v___x_4613_; uint64_t v___x_4614_; uint64_t v___x_4615_; uint64_t v___x_4616_; uint64_t v_fold_4617_; uint64_t v___x_4618_; uint64_t v___x_4619_; uint64_t v___x_4620_; size_t v___x_4621_; size_t v___x_4622_; size_t v___x_4623_; size_t v___x_4624_; size_t v___x_4625_; lean_object* v___x_4626_; lean_object* v___x_4627_; 
v_buckets_4612_ = lean_ctor_get(v_m_4610_, 1);
v___x_4613_ = lean_array_get_size(v_buckets_4612_);
v___x_4614_ = l_Lean_ExprStructEq_hash(v_a_4611_);
v___x_4615_ = 32ULL;
v___x_4616_ = lean_uint64_shift_right(v___x_4614_, v___x_4615_);
v_fold_4617_ = lean_uint64_xor(v___x_4614_, v___x_4616_);
v___x_4618_ = 16ULL;
v___x_4619_ = lean_uint64_shift_right(v_fold_4617_, v___x_4618_);
v___x_4620_ = lean_uint64_xor(v_fold_4617_, v___x_4619_);
v___x_4621_ = lean_uint64_to_usize(v___x_4620_);
v___x_4622_ = lean_usize_of_nat(v___x_4613_);
v___x_4623_ = ((size_t)1ULL);
v___x_4624_ = lean_usize_sub(v___x_4622_, v___x_4623_);
v___x_4625_ = lean_usize_land(v___x_4621_, v___x_4624_);
v___x_4626_ = lean_array_uget_borrowed(v_buckets_4612_, v___x_4625_);
v___x_4627_ = lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg(v_a_4611_, v___x_4626_);
return v___x_4627_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg___boxed(lean_object* v_m_4628_, lean_object* v_a_4629_){
_start:
{
lean_object* v_res_4630_; 
v_res_4630_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg(v_m_4628_, v_a_4629_);
lean_dec_ref(v_a_4629_);
lean_dec_ref(v_m_4628_);
return v_res_4630_;
}
}
static lean_object* _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3(void){
_start:
{
lean_object* v___x_4636_; lean_object* v___x_4637_; 
v___x_4636_ = l_Lean_maxRecDepthErrorMessage;
v___x_4637_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4637_, 0, v___x_4636_);
return v___x_4637_;
}
}
static lean_object* _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4(void){
_start:
{
lean_object* v___x_4638_; lean_object* v___x_4639_; 
v___x_4638_ = lean_obj_once(&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3, &lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3_once, _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__3);
v___x_4639_ = l_Lean_MessageData_ofFormat(v___x_4638_);
return v___x_4639_;
}
}
static lean_object* _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5(void){
_start:
{
lean_object* v___x_4640_; lean_object* v___x_4641_; lean_object* v___x_4642_; 
v___x_4640_ = lean_obj_once(&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4, &lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4_once, _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__4);
v___x_4641_ = ((lean_object*)(lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__2));
v___x_4642_ = lean_alloc_ctor(8, 2, 0);
lean_ctor_set(v___x_4642_, 0, v___x_4641_);
lean_ctor_set(v___x_4642_, 1, v___x_4640_);
return v___x_4642_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg(lean_object* v_ref_4643_){
_start:
{
lean_object* v___x_4645_; lean_object* v___x_4646_; lean_object* v___x_4647_; 
v___x_4645_ = lean_obj_once(&lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5, &lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5_once, _init_lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___closed__5);
v___x_4646_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4646_, 0, v_ref_4643_);
lean_ctor_set(v___x_4646_, 1, v___x_4645_);
v___x_4647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4647_, 0, v___x_4646_);
return v___x_4647_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg___boxed(lean_object* v_ref_4648_, lean_object* v___y_4649_){
_start:
{
lean_object* v_res_4650_; 
v_res_4650_ = lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg(v_ref_4648_);
return v_res_4650_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg(lean_object* v_x_4651_, lean_object* v___y_4652_, lean_object* v___y_4653_, lean_object* v___y_4654_, lean_object* v___y_4655_, lean_object* v___y_4656_){
_start:
{
lean_object* v___y_4659_; lean_object* v_fileName_4668_; lean_object* v_fileMap_4669_; lean_object* v_options_4670_; lean_object* v_currRecDepth_4671_; lean_object* v_maxRecDepth_4672_; lean_object* v_ref_4673_; lean_object* v_currNamespace_4674_; lean_object* v_openDecls_4675_; lean_object* v_initHeartbeats_4676_; lean_object* v_maxHeartbeats_4677_; lean_object* v_quotContext_4678_; lean_object* v_currMacroScope_4679_; uint8_t v_diag_4680_; lean_object* v_cancelTk_x3f_4681_; uint8_t v_suppressElabErrors_4682_; lean_object* v_inheritedTraceOptions_4683_; lean_object* v___x_4689_; uint8_t v___x_4690_; 
v_fileName_4668_ = lean_ctor_get(v___y_4655_, 0);
v_fileMap_4669_ = lean_ctor_get(v___y_4655_, 1);
v_options_4670_ = lean_ctor_get(v___y_4655_, 2);
v_currRecDepth_4671_ = lean_ctor_get(v___y_4655_, 3);
v_maxRecDepth_4672_ = lean_ctor_get(v___y_4655_, 4);
v_ref_4673_ = lean_ctor_get(v___y_4655_, 5);
v_currNamespace_4674_ = lean_ctor_get(v___y_4655_, 6);
v_openDecls_4675_ = lean_ctor_get(v___y_4655_, 7);
v_initHeartbeats_4676_ = lean_ctor_get(v___y_4655_, 8);
v_maxHeartbeats_4677_ = lean_ctor_get(v___y_4655_, 9);
v_quotContext_4678_ = lean_ctor_get(v___y_4655_, 10);
v_currMacroScope_4679_ = lean_ctor_get(v___y_4655_, 11);
v_diag_4680_ = lean_ctor_get_uint8(v___y_4655_, sizeof(void*)*14);
v_cancelTk_x3f_4681_ = lean_ctor_get(v___y_4655_, 12);
v_suppressElabErrors_4682_ = lean_ctor_get_uint8(v___y_4655_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_4683_ = lean_ctor_get(v___y_4655_, 13);
v___x_4689_ = lean_unsigned_to_nat(0u);
v___x_4690_ = lean_nat_dec_eq(v_maxRecDepth_4672_, v___x_4689_);
if (v___x_4690_ == 0)
{
uint8_t v___x_4691_; 
v___x_4691_ = lean_nat_dec_eq(v_currRecDepth_4671_, v_maxRecDepth_4672_);
if (v___x_4691_ == 0)
{
goto v___jp_4684_;
}
else
{
lean_object* v___x_4692_; 
lean_dec_ref(v_x_4651_);
lean_inc(v_ref_4673_);
v___x_4692_ = lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg(v_ref_4673_);
v___y_4659_ = v___x_4692_;
goto v___jp_4658_;
}
}
else
{
goto v___jp_4684_;
}
v___jp_4658_:
{
if (lean_obj_tag(v___y_4659_) == 0)
{
return v___y_4659_;
}
else
{
lean_object* v_a_4660_; lean_object* v___x_4662_; uint8_t v_isShared_4663_; uint8_t v_isSharedCheck_4667_; 
v_a_4660_ = lean_ctor_get(v___y_4659_, 0);
v_isSharedCheck_4667_ = !lean_is_exclusive(v___y_4659_);
if (v_isSharedCheck_4667_ == 0)
{
v___x_4662_ = v___y_4659_;
v_isShared_4663_ = v_isSharedCheck_4667_;
goto v_resetjp_4661_;
}
else
{
lean_inc(v_a_4660_);
lean_dec(v___y_4659_);
v___x_4662_ = lean_box(0);
v_isShared_4663_ = v_isSharedCheck_4667_;
goto v_resetjp_4661_;
}
v_resetjp_4661_:
{
lean_object* v___x_4665_; 
if (v_isShared_4663_ == 0)
{
v___x_4665_ = v___x_4662_;
goto v_reusejp_4664_;
}
else
{
lean_object* v_reuseFailAlloc_4666_; 
v_reuseFailAlloc_4666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4666_, 0, v_a_4660_);
v___x_4665_ = v_reuseFailAlloc_4666_;
goto v_reusejp_4664_;
}
v_reusejp_4664_:
{
return v___x_4665_;
}
}
}
}
v___jp_4684_:
{
lean_object* v___x_4685_; lean_object* v___x_4686_; lean_object* v___x_4687_; lean_object* v___x_4688_; 
v___x_4685_ = lean_unsigned_to_nat(1u);
v___x_4686_ = lean_nat_add(v_currRecDepth_4671_, v___x_4685_);
lean_inc_ref(v_inheritedTraceOptions_4683_);
lean_inc(v_cancelTk_x3f_4681_);
lean_inc(v_currMacroScope_4679_);
lean_inc(v_quotContext_4678_);
lean_inc(v_maxHeartbeats_4677_);
lean_inc(v_initHeartbeats_4676_);
lean_inc(v_openDecls_4675_);
lean_inc(v_currNamespace_4674_);
lean_inc(v_ref_4673_);
lean_inc(v_maxRecDepth_4672_);
lean_inc_ref(v_options_4670_);
lean_inc_ref(v_fileMap_4669_);
lean_inc_ref(v_fileName_4668_);
v___x_4687_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_4687_, 0, v_fileName_4668_);
lean_ctor_set(v___x_4687_, 1, v_fileMap_4669_);
lean_ctor_set(v___x_4687_, 2, v_options_4670_);
lean_ctor_set(v___x_4687_, 3, v___x_4686_);
lean_ctor_set(v___x_4687_, 4, v_maxRecDepth_4672_);
lean_ctor_set(v___x_4687_, 5, v_ref_4673_);
lean_ctor_set(v___x_4687_, 6, v_currNamespace_4674_);
lean_ctor_set(v___x_4687_, 7, v_openDecls_4675_);
lean_ctor_set(v___x_4687_, 8, v_initHeartbeats_4676_);
lean_ctor_set(v___x_4687_, 9, v_maxHeartbeats_4677_);
lean_ctor_set(v___x_4687_, 10, v_quotContext_4678_);
lean_ctor_set(v___x_4687_, 11, v_currMacroScope_4679_);
lean_ctor_set(v___x_4687_, 12, v_cancelTk_x3f_4681_);
lean_ctor_set(v___x_4687_, 13, v_inheritedTraceOptions_4683_);
lean_ctor_set_uint8(v___x_4687_, sizeof(void*)*14, v_diag_4680_);
lean_ctor_set_uint8(v___x_4687_, sizeof(void*)*14 + 1, v_suppressElabErrors_4682_);
lean_inc(v___y_4656_);
lean_inc(v___y_4654_);
lean_inc_ref(v___y_4653_);
lean_inc(v___y_4652_);
v___x_4688_ = lean_apply_6(v_x_4651_, v___y_4652_, v___y_4653_, v___y_4654_, v___x_4687_, v___y_4656_, lean_box(0));
v___y_4659_ = v___x_4688_;
goto v___jp_4658_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg___boxed(lean_object* v_x_4693_, lean_object* v___y_4694_, lean_object* v___y_4695_, lean_object* v___y_4696_, lean_object* v___y_4697_, lean_object* v___y_4698_, lean_object* v___y_4699_){
_start:
{
lean_object* v_res_4700_; 
v_res_4700_ = lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg(v_x_4693_, v___y_4694_, v___y_4695_, v___y_4696_, v___y_4697_, v___y_4698_);
lean_dec(v___y_4698_);
lean_dec_ref(v___y_4697_);
lean_dec(v___y_4696_);
lean_dec_ref(v___y_4695_);
lean_dec(v___y_4694_);
return v_res_4700_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0(lean_object* v_00_u03b1_4701_, lean_object* v_x_4702_, lean_object* v___y_4703_, lean_object* v___y_4704_, lean_object* v___y_4705_, lean_object* v___y_4706_){
_start:
{
lean_object* v___x_4708_; lean_object* v___x_4709_; 
v___x_4708_ = lean_apply_1(v_x_4702_, lean_box(0));
v___x_4709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4709_, 0, v___x_4708_);
return v___x_4709_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0___boxed(lean_object* v_00_u03b1_4710_, lean_object* v_x_4711_, lean_object* v___y_4712_, lean_object* v___y_4713_, lean_object* v___y_4714_, lean_object* v___y_4715_, lean_object* v___y_4716_){
_start:
{
lean_object* v_res_4717_; 
v_res_4717_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0(v_00_u03b1_4710_, v_x_4711_, v___y_4712_, v___y_4713_, v___y_4714_, v___y_4715_);
lean_dec(v___y_4715_);
lean_dec_ref(v___y_4714_);
lean_dec(v___y_4713_);
lean_dec_ref(v___y_4712_);
return v_res_4717_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18___redArg(lean_object* v_a_4718_, lean_object* v_b_4719_, lean_object* v_x_4720_){
_start:
{
if (lean_obj_tag(v_x_4720_) == 0)
{
lean_dec(v_b_4719_);
lean_dec_ref(v_a_4718_);
return v_x_4720_;
}
else
{
lean_object* v_key_4721_; lean_object* v_value_4722_; lean_object* v_tail_4723_; lean_object* v___x_4725_; uint8_t v_isShared_4726_; uint8_t v_isSharedCheck_4735_; 
v_key_4721_ = lean_ctor_get(v_x_4720_, 0);
v_value_4722_ = lean_ctor_get(v_x_4720_, 1);
v_tail_4723_ = lean_ctor_get(v_x_4720_, 2);
v_isSharedCheck_4735_ = !lean_is_exclusive(v_x_4720_);
if (v_isSharedCheck_4735_ == 0)
{
v___x_4725_ = v_x_4720_;
v_isShared_4726_ = v_isSharedCheck_4735_;
goto v_resetjp_4724_;
}
else
{
lean_inc(v_tail_4723_);
lean_inc(v_value_4722_);
lean_inc(v_key_4721_);
lean_dec(v_x_4720_);
v___x_4725_ = lean_box(0);
v_isShared_4726_ = v_isSharedCheck_4735_;
goto v_resetjp_4724_;
}
v_resetjp_4724_:
{
uint8_t v___x_4727_; 
v___x_4727_ = l_Lean_ExprStructEq_beq(v_key_4721_, v_a_4718_);
if (v___x_4727_ == 0)
{
lean_object* v___x_4728_; lean_object* v___x_4730_; 
v___x_4728_ = lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18___redArg(v_a_4718_, v_b_4719_, v_tail_4723_);
if (v_isShared_4726_ == 0)
{
lean_ctor_set(v___x_4725_, 2, v___x_4728_);
v___x_4730_ = v___x_4725_;
goto v_reusejp_4729_;
}
else
{
lean_object* v_reuseFailAlloc_4731_; 
v_reuseFailAlloc_4731_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4731_, 0, v_key_4721_);
lean_ctor_set(v_reuseFailAlloc_4731_, 1, v_value_4722_);
lean_ctor_set(v_reuseFailAlloc_4731_, 2, v___x_4728_);
v___x_4730_ = v_reuseFailAlloc_4731_;
goto v_reusejp_4729_;
}
v_reusejp_4729_:
{
return v___x_4730_;
}
}
else
{
lean_object* v___x_4733_; 
lean_dec(v_value_4722_);
lean_dec(v_key_4721_);
if (v_isShared_4726_ == 0)
{
lean_ctor_set(v___x_4725_, 1, v_b_4719_);
lean_ctor_set(v___x_4725_, 0, v_a_4718_);
v___x_4733_ = v___x_4725_;
goto v_reusejp_4732_;
}
else
{
lean_object* v_reuseFailAlloc_4734_; 
v_reuseFailAlloc_4734_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4734_, 0, v_a_4718_);
lean_ctor_set(v_reuseFailAlloc_4734_, 1, v_b_4719_);
lean_ctor_set(v_reuseFailAlloc_4734_, 2, v_tail_4723_);
v___x_4733_ = v_reuseFailAlloc_4734_;
goto v_reusejp_4732_;
}
v_reusejp_4732_:
{
return v___x_4733_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19___redArg(lean_object* v_x_4736_, lean_object* v_x_4737_){
_start:
{
if (lean_obj_tag(v_x_4737_) == 0)
{
return v_x_4736_;
}
else
{
lean_object* v_key_4738_; lean_object* v_value_4739_; lean_object* v_tail_4740_; lean_object* v___x_4742_; uint8_t v_isShared_4743_; uint8_t v_isSharedCheck_4763_; 
v_key_4738_ = lean_ctor_get(v_x_4737_, 0);
v_value_4739_ = lean_ctor_get(v_x_4737_, 1);
v_tail_4740_ = lean_ctor_get(v_x_4737_, 2);
v_isSharedCheck_4763_ = !lean_is_exclusive(v_x_4737_);
if (v_isSharedCheck_4763_ == 0)
{
v___x_4742_ = v_x_4737_;
v_isShared_4743_ = v_isSharedCheck_4763_;
goto v_resetjp_4741_;
}
else
{
lean_inc(v_tail_4740_);
lean_inc(v_value_4739_);
lean_inc(v_key_4738_);
lean_dec(v_x_4737_);
v___x_4742_ = lean_box(0);
v_isShared_4743_ = v_isSharedCheck_4763_;
goto v_resetjp_4741_;
}
v_resetjp_4741_:
{
lean_object* v___x_4744_; uint64_t v___x_4745_; uint64_t v___x_4746_; uint64_t v___x_4747_; uint64_t v_fold_4748_; uint64_t v___x_4749_; uint64_t v___x_4750_; uint64_t v___x_4751_; size_t v___x_4752_; size_t v___x_4753_; size_t v___x_4754_; size_t v___x_4755_; size_t v___x_4756_; lean_object* v___x_4757_; lean_object* v___x_4759_; 
v___x_4744_ = lean_array_get_size(v_x_4736_);
v___x_4745_ = l_Lean_ExprStructEq_hash(v_key_4738_);
v___x_4746_ = 32ULL;
v___x_4747_ = lean_uint64_shift_right(v___x_4745_, v___x_4746_);
v_fold_4748_ = lean_uint64_xor(v___x_4745_, v___x_4747_);
v___x_4749_ = 16ULL;
v___x_4750_ = lean_uint64_shift_right(v_fold_4748_, v___x_4749_);
v___x_4751_ = lean_uint64_xor(v_fold_4748_, v___x_4750_);
v___x_4752_ = lean_uint64_to_usize(v___x_4751_);
v___x_4753_ = lean_usize_of_nat(v___x_4744_);
v___x_4754_ = ((size_t)1ULL);
v___x_4755_ = lean_usize_sub(v___x_4753_, v___x_4754_);
v___x_4756_ = lean_usize_land(v___x_4752_, v___x_4755_);
v___x_4757_ = lean_array_uget_borrowed(v_x_4736_, v___x_4756_);
lean_inc(v___x_4757_);
if (v_isShared_4743_ == 0)
{
lean_ctor_set(v___x_4742_, 2, v___x_4757_);
v___x_4759_ = v___x_4742_;
goto v_reusejp_4758_;
}
else
{
lean_object* v_reuseFailAlloc_4762_; 
v_reuseFailAlloc_4762_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4762_, 0, v_key_4738_);
lean_ctor_set(v_reuseFailAlloc_4762_, 1, v_value_4739_);
lean_ctor_set(v_reuseFailAlloc_4762_, 2, v___x_4757_);
v___x_4759_ = v_reuseFailAlloc_4762_;
goto v_reusejp_4758_;
}
v_reusejp_4758_:
{
lean_object* v___x_4760_; 
v___x_4760_ = lean_array_uset(v_x_4736_, v___x_4756_, v___x_4759_);
v_x_4736_ = v___x_4760_;
v_x_4737_ = v_tail_4740_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18___redArg(lean_object* v_i_4764_, lean_object* v_source_4765_, lean_object* v_target_4766_){
_start:
{
lean_object* v___x_4767_; uint8_t v___x_4768_; 
v___x_4767_ = lean_array_get_size(v_source_4765_);
v___x_4768_ = lean_nat_dec_lt(v_i_4764_, v___x_4767_);
if (v___x_4768_ == 0)
{
lean_dec_ref(v_source_4765_);
lean_dec(v_i_4764_);
return v_target_4766_;
}
else
{
lean_object* v_es_4769_; lean_object* v___x_4770_; lean_object* v_source_4771_; lean_object* v_target_4772_; lean_object* v___x_4773_; lean_object* v___x_4774_; 
v_es_4769_ = lean_array_fget(v_source_4765_, v_i_4764_);
v___x_4770_ = lean_box(0);
v_source_4771_ = lean_array_fset(v_source_4765_, v_i_4764_, v___x_4770_);
v_target_4772_ = lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19___redArg(v_target_4766_, v_es_4769_);
v___x_4773_ = lean_unsigned_to_nat(1u);
v___x_4774_ = lean_nat_add(v_i_4764_, v___x_4773_);
lean_dec(v_i_4764_);
v_i_4764_ = v___x_4774_;
v_source_4765_ = v_source_4771_;
v_target_4766_ = v_target_4772_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17___redArg(lean_object* v_data_4776_){
_start:
{
lean_object* v___x_4777_; lean_object* v___x_4778_; lean_object* v_nbuckets_4779_; lean_object* v___x_4780_; lean_object* v___x_4781_; lean_object* v___x_4782_; lean_object* v___x_4783_; 
v___x_4777_ = lean_array_get_size(v_data_4776_);
v___x_4778_ = lean_unsigned_to_nat(2u);
v_nbuckets_4779_ = lean_nat_mul(v___x_4777_, v___x_4778_);
v___x_4780_ = lean_unsigned_to_nat(0u);
v___x_4781_ = lean_box(0);
v___x_4782_ = lean_mk_array(v_nbuckets_4779_, v___x_4781_);
v___x_4783_ = lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18___redArg(v___x_4780_, v_data_4776_, v___x_4782_);
return v___x_4783_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg(lean_object* v_a_4784_, lean_object* v_x_4785_){
_start:
{
if (lean_obj_tag(v_x_4785_) == 0)
{
uint8_t v___x_4786_; 
v___x_4786_ = 0;
return v___x_4786_;
}
else
{
lean_object* v_key_4787_; lean_object* v_tail_4788_; uint8_t v___x_4789_; 
v_key_4787_ = lean_ctor_get(v_x_4785_, 0);
v_tail_4788_ = lean_ctor_get(v_x_4785_, 2);
v___x_4789_ = l_Lean_ExprStructEq_beq(v_key_4787_, v_a_4784_);
if (v___x_4789_ == 0)
{
v_x_4785_ = v_tail_4788_;
goto _start;
}
else
{
return v___x_4789_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg___boxed(lean_object* v_a_4791_, lean_object* v_x_4792_){
_start:
{
uint8_t v_res_4793_; lean_object* v_r_4794_; 
v_res_4793_ = lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg(v_a_4791_, v_x_4792_);
lean_dec(v_x_4792_);
lean_dec_ref(v_a_4791_);
v_r_4794_ = lean_box(v_res_4793_);
return v_r_4794_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11___redArg(lean_object* v_m_4795_, lean_object* v_a_4796_, lean_object* v_b_4797_){
_start:
{
lean_object* v_size_4798_; lean_object* v_buckets_4799_; lean_object* v___x_4801_; uint8_t v_isShared_4802_; uint8_t v_isSharedCheck_4842_; 
v_size_4798_ = lean_ctor_get(v_m_4795_, 0);
v_buckets_4799_ = lean_ctor_get(v_m_4795_, 1);
v_isSharedCheck_4842_ = !lean_is_exclusive(v_m_4795_);
if (v_isSharedCheck_4842_ == 0)
{
v___x_4801_ = v_m_4795_;
v_isShared_4802_ = v_isSharedCheck_4842_;
goto v_resetjp_4800_;
}
else
{
lean_inc(v_buckets_4799_);
lean_inc(v_size_4798_);
lean_dec(v_m_4795_);
v___x_4801_ = lean_box(0);
v_isShared_4802_ = v_isSharedCheck_4842_;
goto v_resetjp_4800_;
}
v_resetjp_4800_:
{
lean_object* v___x_4803_; uint64_t v___x_4804_; uint64_t v___x_4805_; uint64_t v___x_4806_; uint64_t v_fold_4807_; uint64_t v___x_4808_; uint64_t v___x_4809_; uint64_t v___x_4810_; size_t v___x_4811_; size_t v___x_4812_; size_t v___x_4813_; size_t v___x_4814_; size_t v___x_4815_; lean_object* v_bkt_4816_; uint8_t v___x_4817_; 
v___x_4803_ = lean_array_get_size(v_buckets_4799_);
v___x_4804_ = l_Lean_ExprStructEq_hash(v_a_4796_);
v___x_4805_ = 32ULL;
v___x_4806_ = lean_uint64_shift_right(v___x_4804_, v___x_4805_);
v_fold_4807_ = lean_uint64_xor(v___x_4804_, v___x_4806_);
v___x_4808_ = 16ULL;
v___x_4809_ = lean_uint64_shift_right(v_fold_4807_, v___x_4808_);
v___x_4810_ = lean_uint64_xor(v_fold_4807_, v___x_4809_);
v___x_4811_ = lean_uint64_to_usize(v___x_4810_);
v___x_4812_ = lean_usize_of_nat(v___x_4803_);
v___x_4813_ = ((size_t)1ULL);
v___x_4814_ = lean_usize_sub(v___x_4812_, v___x_4813_);
v___x_4815_ = lean_usize_land(v___x_4811_, v___x_4814_);
v_bkt_4816_ = lean_array_uget_borrowed(v_buckets_4799_, v___x_4815_);
v___x_4817_ = lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg(v_a_4796_, v_bkt_4816_);
if (v___x_4817_ == 0)
{
lean_object* v___x_4818_; lean_object* v_size_x27_4819_; lean_object* v___x_4820_; lean_object* v_buckets_x27_4821_; lean_object* v___x_4822_; lean_object* v___x_4823_; lean_object* v___x_4824_; lean_object* v___x_4825_; lean_object* v___x_4826_; uint8_t v___x_4827_; 
v___x_4818_ = lean_unsigned_to_nat(1u);
v_size_x27_4819_ = lean_nat_add(v_size_4798_, v___x_4818_);
lean_dec(v_size_4798_);
lean_inc(v_bkt_4816_);
v___x_4820_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4820_, 0, v_a_4796_);
lean_ctor_set(v___x_4820_, 1, v_b_4797_);
lean_ctor_set(v___x_4820_, 2, v_bkt_4816_);
v_buckets_x27_4821_ = lean_array_uset(v_buckets_4799_, v___x_4815_, v___x_4820_);
v___x_4822_ = lean_unsigned_to_nat(4u);
v___x_4823_ = lean_nat_mul(v_size_x27_4819_, v___x_4822_);
v___x_4824_ = lean_unsigned_to_nat(3u);
v___x_4825_ = lean_nat_div(v___x_4823_, v___x_4824_);
lean_dec(v___x_4823_);
v___x_4826_ = lean_array_get_size(v_buckets_x27_4821_);
v___x_4827_ = lean_nat_dec_le(v___x_4825_, v___x_4826_);
lean_dec(v___x_4825_);
if (v___x_4827_ == 0)
{
lean_object* v_val_4828_; lean_object* v___x_4830_; 
v_val_4828_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17___redArg(v_buckets_x27_4821_);
if (v_isShared_4802_ == 0)
{
lean_ctor_set(v___x_4801_, 1, v_val_4828_);
lean_ctor_set(v___x_4801_, 0, v_size_x27_4819_);
v___x_4830_ = v___x_4801_;
goto v_reusejp_4829_;
}
else
{
lean_object* v_reuseFailAlloc_4831_; 
v_reuseFailAlloc_4831_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4831_, 0, v_size_x27_4819_);
lean_ctor_set(v_reuseFailAlloc_4831_, 1, v_val_4828_);
v___x_4830_ = v_reuseFailAlloc_4831_;
goto v_reusejp_4829_;
}
v_reusejp_4829_:
{
return v___x_4830_;
}
}
else
{
lean_object* v___x_4833_; 
if (v_isShared_4802_ == 0)
{
lean_ctor_set(v___x_4801_, 1, v_buckets_x27_4821_);
lean_ctor_set(v___x_4801_, 0, v_size_x27_4819_);
v___x_4833_ = v___x_4801_;
goto v_reusejp_4832_;
}
else
{
lean_object* v_reuseFailAlloc_4834_; 
v_reuseFailAlloc_4834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4834_, 0, v_size_x27_4819_);
lean_ctor_set(v_reuseFailAlloc_4834_, 1, v_buckets_x27_4821_);
v___x_4833_ = v_reuseFailAlloc_4834_;
goto v_reusejp_4832_;
}
v_reusejp_4832_:
{
return v___x_4833_;
}
}
}
else
{
lean_object* v___x_4835_; lean_object* v_buckets_x27_4836_; lean_object* v___x_4837_; lean_object* v___x_4838_; lean_object* v___x_4840_; 
lean_inc(v_bkt_4816_);
v___x_4835_ = lean_box(0);
v_buckets_x27_4836_ = lean_array_uset(v_buckets_4799_, v___x_4815_, v___x_4835_);
v___x_4837_ = lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18___redArg(v_a_4796_, v_b_4797_, v_bkt_4816_);
v___x_4838_ = lean_array_uset(v_buckets_x27_4836_, v___x_4815_, v___x_4837_);
if (v_isShared_4802_ == 0)
{
lean_ctor_set(v___x_4801_, 1, v___x_4838_);
v___x_4840_ = v___x_4801_;
goto v_reusejp_4839_;
}
else
{
lean_object* v_reuseFailAlloc_4841_; 
v_reuseFailAlloc_4841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4841_, 0, v_size_4798_);
lean_ctor_set(v_reuseFailAlloc_4841_, 1, v___x_4838_);
v___x_4840_ = v_reuseFailAlloc_4841_;
goto v_reusejp_4839_;
}
v_reusejp_4839_:
{
return v___x_4840_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2(lean_object* v_a_4843_, lean_object* v_e_4844_, lean_object* v_a_4845_){
_start:
{
lean_object* v___x_4847_; lean_object* v___x_4848_; lean_object* v___x_4849_; lean_object* v___x_4850_; 
v___x_4847_ = lean_st_ref_take(v_a_4843_);
v___x_4848_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11___redArg(v___x_4847_, v_e_4844_, v_a_4845_);
v___x_4849_ = lean_st_ref_set(v_a_4843_, v___x_4848_);
v___x_4850_ = lean_box(0);
return v___x_4850_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2___boxed(lean_object* v_a_4851_, lean_object* v_e_4852_, lean_object* v_a_4853_, lean_object* v___y_4854_){
_start:
{
lean_object* v_res_4855_; 
v_res_4855_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2(v_a_4851_, v_e_4852_, v_a_4853_);
lean_dec(v_a_4851_);
return v_res_4855_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0(lean_object* v_fvars_4859_, lean_object* v_pre_4860_, lean_object* v_post_4861_, uint8_t v_usedLetOnly_4862_, uint8_t v_skipConstInApp_4863_, uint8_t v_skipInstances_4864_, lean_object* v_body_4865_, lean_object* v_x_4866_, lean_object* v___y_4867_, lean_object* v___y_4868_, lean_object* v___y_4869_, lean_object* v___y_4870_, lean_object* v___y_4871_){
_start:
{
lean_object* v___x_4873_; lean_object* v___x_4874_; 
v___x_4873_ = lean_array_push(v_fvars_4859_, v_x_4866_);
v___x_4874_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7(v_pre_4860_, v_post_4861_, v_usedLetOnly_4862_, v_skipConstInApp_4863_, v_skipInstances_4864_, v___x_4873_, v_body_4865_, v___y_4867_, v___y_4868_, v___y_4869_, v___y_4870_, v___y_4871_);
return v___x_4874_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0___boxed(lean_object* v_fvars_4875_, lean_object* v_pre_4876_, lean_object* v_post_4877_, lean_object* v_usedLetOnly_4878_, lean_object* v_skipConstInApp_4879_, lean_object* v_skipInstances_4880_, lean_object* v_body_4881_, lean_object* v_x_4882_, lean_object* v___y_4883_, lean_object* v___y_4884_, lean_object* v___y_4885_, lean_object* v___y_4886_, lean_object* v___y_4887_, lean_object* v___y_4888_){
_start:
{
uint8_t v_usedLetOnly_boxed_4889_; uint8_t v_skipConstInApp_boxed_4890_; uint8_t v_skipInstances_boxed_4891_; lean_object* v_res_4892_; 
v_usedLetOnly_boxed_4889_ = lean_unbox(v_usedLetOnly_4878_);
v_skipConstInApp_boxed_4890_ = lean_unbox(v_skipConstInApp_4879_);
v_skipInstances_boxed_4891_ = lean_unbox(v_skipInstances_4880_);
v_res_4892_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0(v_fvars_4875_, v_pre_4876_, v_post_4877_, v_usedLetOnly_boxed_4889_, v_skipConstInApp_boxed_4890_, v_skipInstances_boxed_4891_, v_body_4881_, v_x_4882_, v___y_4883_, v___y_4884_, v___y_4885_, v___y_4886_, v___y_4887_);
lean_dec(v___y_4887_);
lean_dec_ref(v___y_4886_);
lean_dec(v___y_4885_);
lean_dec_ref(v___y_4884_);
lean_dec(v___y_4883_);
return v_res_4892_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(lean_object* v_pre_4893_, lean_object* v_post_4894_, uint8_t v_usedLetOnly_4895_, uint8_t v_skipConstInApp_4896_, uint8_t v_skipInstances_4897_, lean_object* v_e_4898_, lean_object* v_a_4899_, lean_object* v___y_4900_, lean_object* v___y_4901_, lean_object* v___y_4902_, lean_object* v___y_4903_){
_start:
{
lean_object* v___x_4905_; 
lean_inc_ref(v_post_4894_);
lean_inc(v___y_4903_);
lean_inc_ref(v___y_4902_);
lean_inc(v___y_4901_);
lean_inc_ref(v___y_4900_);
lean_inc_ref(v_e_4898_);
v___x_4905_ = lean_apply_6(v_post_4894_, v_e_4898_, v___y_4900_, v___y_4901_, v___y_4902_, v___y_4903_, lean_box(0));
if (lean_obj_tag(v___x_4905_) == 0)
{
lean_object* v_a_4906_; lean_object* v___x_4908_; uint8_t v_isShared_4909_; uint8_t v_isSharedCheck_4924_; 
v_a_4906_ = lean_ctor_get(v___x_4905_, 0);
v_isSharedCheck_4924_ = !lean_is_exclusive(v___x_4905_);
if (v_isSharedCheck_4924_ == 0)
{
v___x_4908_ = v___x_4905_;
v_isShared_4909_ = v_isSharedCheck_4924_;
goto v_resetjp_4907_;
}
else
{
lean_inc(v_a_4906_);
lean_dec(v___x_4905_);
v___x_4908_ = lean_box(0);
v_isShared_4909_ = v_isSharedCheck_4924_;
goto v_resetjp_4907_;
}
v_resetjp_4907_:
{
switch(lean_obj_tag(v_a_4906_))
{
case 0:
{
lean_object* v_e_4910_; lean_object* v___x_4912_; 
lean_dec_ref(v_e_4898_);
lean_dec_ref(v_post_4894_);
lean_dec_ref(v_pre_4893_);
v_e_4910_ = lean_ctor_get(v_a_4906_, 0);
lean_inc_ref(v_e_4910_);
lean_dec_ref_known(v_a_4906_, 1);
if (v_isShared_4909_ == 0)
{
lean_ctor_set(v___x_4908_, 0, v_e_4910_);
v___x_4912_ = v___x_4908_;
goto v_reusejp_4911_;
}
else
{
lean_object* v_reuseFailAlloc_4913_; 
v_reuseFailAlloc_4913_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4913_, 0, v_e_4910_);
v___x_4912_ = v_reuseFailAlloc_4913_;
goto v_reusejp_4911_;
}
v_reusejp_4911_:
{
return v___x_4912_;
}
}
case 1:
{
lean_object* v_e_4914_; lean_object* v___x_4915_; 
lean_del_object(v___x_4908_);
lean_dec_ref(v_e_4898_);
v_e_4914_ = lean_ctor_get(v_a_4906_, 0);
lean_inc_ref(v_e_4914_);
lean_dec_ref_known(v_a_4906_, 1);
v___x_4915_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_4893_, v_post_4894_, v_usedLetOnly_4895_, v_skipConstInApp_4896_, v_skipInstances_4897_, v_e_4914_, v_a_4899_, v___y_4900_, v___y_4901_, v___y_4902_, v___y_4903_);
return v___x_4915_;
}
default: 
{
lean_object* v_e_x3f_4916_; 
lean_dec_ref(v_post_4894_);
lean_dec_ref(v_pre_4893_);
v_e_x3f_4916_ = lean_ctor_get(v_a_4906_, 0);
lean_inc(v_e_x3f_4916_);
lean_dec_ref_known(v_a_4906_, 1);
if (lean_obj_tag(v_e_x3f_4916_) == 0)
{
lean_object* v___x_4918_; 
if (v_isShared_4909_ == 0)
{
lean_ctor_set(v___x_4908_, 0, v_e_4898_);
v___x_4918_ = v___x_4908_;
goto v_reusejp_4917_;
}
else
{
lean_object* v_reuseFailAlloc_4919_; 
v_reuseFailAlloc_4919_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4919_, 0, v_e_4898_);
v___x_4918_ = v_reuseFailAlloc_4919_;
goto v_reusejp_4917_;
}
v_reusejp_4917_:
{
return v___x_4918_;
}
}
else
{
lean_object* v_val_4920_; lean_object* v___x_4922_; 
lean_dec_ref(v_e_4898_);
v_val_4920_ = lean_ctor_get(v_e_x3f_4916_, 0);
lean_inc(v_val_4920_);
lean_dec_ref_known(v_e_x3f_4916_, 1);
if (v_isShared_4909_ == 0)
{
lean_ctor_set(v___x_4908_, 0, v_val_4920_);
v___x_4922_ = v___x_4908_;
goto v_reusejp_4921_;
}
else
{
lean_object* v_reuseFailAlloc_4923_; 
v_reuseFailAlloc_4923_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4923_, 0, v_val_4920_);
v___x_4922_ = v_reuseFailAlloc_4923_;
goto v_reusejp_4921_;
}
v_reusejp_4921_:
{
return v___x_4922_;
}
}
}
}
}
}
else
{
lean_object* v_a_4925_; lean_object* v___x_4927_; uint8_t v_isShared_4928_; uint8_t v_isSharedCheck_4932_; 
lean_dec_ref(v_e_4898_);
lean_dec_ref(v_post_4894_);
lean_dec_ref(v_pre_4893_);
v_a_4925_ = lean_ctor_get(v___x_4905_, 0);
v_isSharedCheck_4932_ = !lean_is_exclusive(v___x_4905_);
if (v_isSharedCheck_4932_ == 0)
{
v___x_4927_ = v___x_4905_;
v_isShared_4928_ = v_isSharedCheck_4932_;
goto v_resetjp_4926_;
}
else
{
lean_inc(v_a_4925_);
lean_dec(v___x_4905_);
v___x_4927_ = lean_box(0);
v_isShared_4928_ = v_isSharedCheck_4932_;
goto v_resetjp_4926_;
}
v_resetjp_4926_:
{
lean_object* v___x_4930_; 
if (v_isShared_4928_ == 0)
{
v___x_4930_ = v___x_4927_;
goto v_reusejp_4929_;
}
else
{
lean_object* v_reuseFailAlloc_4931_; 
v_reuseFailAlloc_4931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4931_, 0, v_a_4925_);
v___x_4930_ = v_reuseFailAlloc_4931_;
goto v_reusejp_4929_;
}
v_reusejp_4929_:
{
return v___x_4930_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7(lean_object* v_pre_4933_, lean_object* v_post_4934_, uint8_t v_usedLetOnly_4935_, uint8_t v_skipConstInApp_4936_, uint8_t v_skipInstances_4937_, lean_object* v_fvars_4938_, lean_object* v_e_4939_, lean_object* v_a_4940_, lean_object* v___y_4941_, lean_object* v___y_4942_, lean_object* v___y_4943_, lean_object* v___y_4944_){
_start:
{
if (lean_obj_tag(v_e_4939_) == 6)
{
lean_object* v_binderName_4946_; lean_object* v_binderType_4947_; lean_object* v_body_4948_; uint8_t v_binderInfo_4949_; lean_object* v___x_4950_; lean_object* v___x_4951_; 
v_binderName_4946_ = lean_ctor_get(v_e_4939_, 0);
lean_inc(v_binderName_4946_);
v_binderType_4947_ = lean_ctor_get(v_e_4939_, 1);
lean_inc_ref(v_binderType_4947_);
v_body_4948_ = lean_ctor_get(v_e_4939_, 2);
lean_inc_ref(v_body_4948_);
v_binderInfo_4949_ = lean_ctor_get_uint8(v_e_4939_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_4939_, 3);
v___x_4950_ = lean_expr_instantiate_rev(v_binderType_4947_, v_fvars_4938_);
lean_dec_ref(v_binderType_4947_);
lean_inc_ref(v_post_4934_);
lean_inc_ref(v_pre_4933_);
v___x_4951_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_4933_, v_post_4934_, v_usedLetOnly_4935_, v_skipConstInApp_4936_, v_skipInstances_4937_, v___x_4950_, v_a_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_);
if (lean_obj_tag(v___x_4951_) == 0)
{
lean_object* v_a_4952_; lean_object* v___x_4953_; lean_object* v___x_4954_; lean_object* v___x_4955_; lean_object* v___f_4956_; uint8_t v___x_4957_; lean_object* v___x_4958_; 
v_a_4952_ = lean_ctor_get(v___x_4951_, 0);
lean_inc(v_a_4952_);
lean_dec_ref_known(v___x_4951_, 1);
v___x_4953_ = lean_box(v_usedLetOnly_4935_);
v___x_4954_ = lean_box(v_skipConstInApp_4936_);
v___x_4955_ = lean_box(v_skipInstances_4937_);
v___f_4956_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___lam__0___boxed), 14, 7);
lean_closure_set(v___f_4956_, 0, v_fvars_4938_);
lean_closure_set(v___f_4956_, 1, v_pre_4933_);
lean_closure_set(v___f_4956_, 2, v_post_4934_);
lean_closure_set(v___f_4956_, 3, v___x_4953_);
lean_closure_set(v___f_4956_, 4, v___x_4954_);
lean_closure_set(v___f_4956_, 5, v___x_4955_);
lean_closure_set(v___f_4956_, 6, v_body_4948_);
v___x_4957_ = 0;
v___x_4958_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(v_binderName_4946_, v_binderInfo_4949_, v_a_4952_, v___f_4956_, v___x_4957_, v_a_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_);
return v___x_4958_;
}
else
{
lean_dec_ref(v_body_4948_);
lean_dec(v_binderName_4946_);
lean_dec_ref(v_fvars_4938_);
lean_dec_ref(v_post_4934_);
lean_dec_ref(v_pre_4933_);
return v___x_4951_;
}
}
else
{
lean_object* v___x_4959_; lean_object* v___x_4960_; 
v___x_4959_ = lean_expr_instantiate_rev(v_e_4939_, v_fvars_4938_);
lean_dec_ref(v_e_4939_);
lean_inc_ref(v_post_4934_);
lean_inc_ref(v_pre_4933_);
v___x_4960_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_4933_, v_post_4934_, v_usedLetOnly_4935_, v_skipConstInApp_4936_, v_skipInstances_4937_, v___x_4959_, v_a_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_);
if (lean_obj_tag(v___x_4960_) == 0)
{
lean_object* v_a_4961_; uint8_t v___x_4962_; uint8_t v___x_4963_; uint8_t v___x_4964_; lean_object* v___x_4965_; 
v_a_4961_ = lean_ctor_get(v___x_4960_, 0);
lean_inc(v_a_4961_);
lean_dec_ref_known(v___x_4960_, 1);
v___x_4962_ = 0;
v___x_4963_ = 1;
v___x_4964_ = 1;
v___x_4965_ = l_Lean_Meta_mkLambdaFVars(v_fvars_4938_, v_a_4961_, v___x_4962_, v_usedLetOnly_4935_, v___x_4962_, v___x_4963_, v___x_4964_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_);
lean_dec_ref(v_fvars_4938_);
if (lean_obj_tag(v___x_4965_) == 0)
{
lean_object* v_a_4966_; lean_object* v___x_4967_; 
v_a_4966_ = lean_ctor_get(v___x_4965_, 0);
lean_inc(v_a_4966_);
lean_dec_ref_known(v___x_4965_, 1);
v___x_4967_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_4933_, v_post_4934_, v_usedLetOnly_4935_, v_skipConstInApp_4936_, v_skipInstances_4937_, v_a_4966_, v_a_4940_, v___y_4941_, v___y_4942_, v___y_4943_, v___y_4944_);
return v___x_4967_;
}
else
{
lean_dec_ref(v_post_4934_);
lean_dec_ref(v_pre_4933_);
return v___x_4965_;
}
}
else
{
lean_dec_ref(v_fvars_4938_);
lean_dec_ref(v_post_4934_);
lean_dec_ref(v_pre_4933_);
return v___x_4960_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0(lean_object* v_fvars_4968_, lean_object* v_pre_4969_, lean_object* v_post_4970_, uint8_t v_usedLetOnly_4971_, uint8_t v_skipConstInApp_4972_, uint8_t v_skipInstances_4973_, lean_object* v_body_4974_, lean_object* v_x_4975_, lean_object* v___y_4976_, lean_object* v___y_4977_, lean_object* v___y_4978_, lean_object* v___y_4979_, lean_object* v___y_4980_){
_start:
{
lean_object* v___x_4982_; lean_object* v___x_4983_; 
v___x_4982_ = lean_array_push(v_fvars_4968_, v_x_4975_);
v___x_4983_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8(v_pre_4969_, v_post_4970_, v_usedLetOnly_4971_, v_skipConstInApp_4972_, v_skipInstances_4973_, v___x_4982_, v_body_4974_, v___y_4976_, v___y_4977_, v___y_4978_, v___y_4979_, v___y_4980_);
return v___x_4983_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0___boxed(lean_object* v_fvars_4984_, lean_object* v_pre_4985_, lean_object* v_post_4986_, lean_object* v_usedLetOnly_4987_, lean_object* v_skipConstInApp_4988_, lean_object* v_skipInstances_4989_, lean_object* v_body_4990_, lean_object* v_x_4991_, lean_object* v___y_4992_, lean_object* v___y_4993_, lean_object* v___y_4994_, lean_object* v___y_4995_, lean_object* v___y_4996_, lean_object* v___y_4997_){
_start:
{
uint8_t v_usedLetOnly_boxed_4998_; uint8_t v_skipConstInApp_boxed_4999_; uint8_t v_skipInstances_boxed_5000_; lean_object* v_res_5001_; 
v_usedLetOnly_boxed_4998_ = lean_unbox(v_usedLetOnly_4987_);
v_skipConstInApp_boxed_4999_ = lean_unbox(v_skipConstInApp_4988_);
v_skipInstances_boxed_5000_ = lean_unbox(v_skipInstances_4989_);
v_res_5001_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0(v_fvars_4984_, v_pre_4985_, v_post_4986_, v_usedLetOnly_boxed_4998_, v_skipConstInApp_boxed_4999_, v_skipInstances_boxed_5000_, v_body_4990_, v_x_4991_, v___y_4992_, v___y_4993_, v___y_4994_, v___y_4995_, v___y_4996_);
lean_dec(v___y_4996_);
lean_dec_ref(v___y_4995_);
lean_dec(v___y_4994_);
lean_dec_ref(v___y_4993_);
lean_dec(v___y_4992_);
return v_res_5001_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8(lean_object* v_pre_5002_, lean_object* v_post_5003_, uint8_t v_usedLetOnly_5004_, uint8_t v_skipConstInApp_5005_, uint8_t v_skipInstances_5006_, lean_object* v_fvars_5007_, lean_object* v_e_5008_, lean_object* v_a_5009_, lean_object* v___y_5010_, lean_object* v___y_5011_, lean_object* v___y_5012_, lean_object* v___y_5013_){
_start:
{
if (lean_obj_tag(v_e_5008_) == 8)
{
lean_object* v_declName_5015_; lean_object* v_type_5016_; lean_object* v_value_5017_; lean_object* v_body_5018_; uint8_t v_nondep_5019_; lean_object* v___x_5020_; lean_object* v___x_5021_; 
v_declName_5015_ = lean_ctor_get(v_e_5008_, 0);
lean_inc(v_declName_5015_);
v_type_5016_ = lean_ctor_get(v_e_5008_, 1);
lean_inc_ref(v_type_5016_);
v_value_5017_ = lean_ctor_get(v_e_5008_, 2);
lean_inc_ref(v_value_5017_);
v_body_5018_ = lean_ctor_get(v_e_5008_, 3);
lean_inc_ref(v_body_5018_);
v_nondep_5019_ = lean_ctor_get_uint8(v_e_5008_, sizeof(void*)*4 + 8);
lean_dec_ref_known(v_e_5008_, 4);
v___x_5020_ = lean_expr_instantiate_rev(v_type_5016_, v_fvars_5007_);
lean_dec_ref(v_type_5016_);
lean_inc_ref(v_post_5003_);
lean_inc_ref(v_pre_5002_);
v___x_5021_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5002_, v_post_5003_, v_usedLetOnly_5004_, v_skipConstInApp_5005_, v_skipInstances_5006_, v___x_5020_, v_a_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
if (lean_obj_tag(v___x_5021_) == 0)
{
lean_object* v_a_5022_; lean_object* v___x_5023_; lean_object* v___x_5024_; 
v_a_5022_ = lean_ctor_get(v___x_5021_, 0);
lean_inc(v_a_5022_);
lean_dec_ref_known(v___x_5021_, 1);
v___x_5023_ = lean_expr_instantiate_rev(v_value_5017_, v_fvars_5007_);
lean_dec_ref(v_value_5017_);
lean_inc_ref(v_post_5003_);
lean_inc_ref(v_pre_5002_);
v___x_5024_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5002_, v_post_5003_, v_usedLetOnly_5004_, v_skipConstInApp_5005_, v_skipInstances_5006_, v___x_5023_, v_a_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
if (lean_obj_tag(v___x_5024_) == 0)
{
lean_object* v_a_5025_; lean_object* v___x_5026_; lean_object* v___x_5027_; lean_object* v___x_5028_; lean_object* v___f_5029_; uint8_t v___x_5030_; lean_object* v___x_5031_; 
v_a_5025_ = lean_ctor_get(v___x_5024_, 0);
lean_inc(v_a_5025_);
lean_dec_ref_known(v___x_5024_, 1);
v___x_5026_ = lean_box(v_usedLetOnly_5004_);
v___x_5027_ = lean_box(v_skipConstInApp_5005_);
v___x_5028_ = lean_box(v_skipInstances_5006_);
v___f_5029_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___lam__0___boxed), 14, 7);
lean_closure_set(v___f_5029_, 0, v_fvars_5007_);
lean_closure_set(v___f_5029_, 1, v_pre_5002_);
lean_closure_set(v___f_5029_, 2, v_post_5003_);
lean_closure_set(v___f_5029_, 3, v___x_5026_);
lean_closure_set(v___f_5029_, 4, v___x_5027_);
lean_closure_set(v___f_5029_, 5, v___x_5028_);
lean_closure_set(v___f_5029_, 6, v_body_5018_);
v___x_5030_ = 0;
v___x_5031_ = lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg(v_declName_5015_, v_a_5022_, v_a_5025_, v___f_5029_, v_nondep_5019_, v___x_5030_, v_a_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
return v___x_5031_;
}
else
{
lean_dec(v_a_5022_);
lean_dec_ref(v_body_5018_);
lean_dec(v_declName_5015_);
lean_dec_ref(v_fvars_5007_);
lean_dec_ref(v_post_5003_);
lean_dec_ref(v_pre_5002_);
return v___x_5024_;
}
}
else
{
lean_dec_ref(v_body_5018_);
lean_dec_ref(v_value_5017_);
lean_dec(v_declName_5015_);
lean_dec_ref(v_fvars_5007_);
lean_dec_ref(v_post_5003_);
lean_dec_ref(v_pre_5002_);
return v___x_5021_;
}
}
else
{
lean_object* v___x_5032_; lean_object* v___x_5033_; 
v___x_5032_ = lean_expr_instantiate_rev(v_e_5008_, v_fvars_5007_);
lean_dec_ref(v_e_5008_);
lean_inc_ref(v_post_5003_);
lean_inc_ref(v_pre_5002_);
v___x_5033_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5002_, v_post_5003_, v_usedLetOnly_5004_, v_skipConstInApp_5005_, v_skipInstances_5006_, v___x_5032_, v_a_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
if (lean_obj_tag(v___x_5033_) == 0)
{
lean_object* v_a_5034_; uint8_t v___x_5035_; uint8_t v___x_5036_; lean_object* v___x_5037_; 
v_a_5034_ = lean_ctor_get(v___x_5033_, 0);
lean_inc(v_a_5034_);
lean_dec_ref_known(v___x_5033_, 1);
v___x_5035_ = 0;
v___x_5036_ = 1;
v___x_5037_ = l_Lean_Meta_mkLetFVars(v_fvars_5007_, v_a_5034_, v_usedLetOnly_5004_, v___x_5035_, v___x_5036_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
lean_dec_ref(v_fvars_5007_);
if (lean_obj_tag(v___x_5037_) == 0)
{
lean_object* v_a_5038_; lean_object* v___x_5039_; 
v_a_5038_ = lean_ctor_get(v___x_5037_, 0);
lean_inc(v_a_5038_);
lean_dec_ref_known(v___x_5037_, 1);
v___x_5039_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5002_, v_post_5003_, v_usedLetOnly_5004_, v_skipConstInApp_5005_, v_skipInstances_5006_, v_a_5038_, v_a_5009_, v___y_5010_, v___y_5011_, v___y_5012_, v___y_5013_);
return v___x_5039_;
}
else
{
lean_dec_ref(v_post_5003_);
lean_dec_ref(v_pre_5002_);
return v___x_5037_;
}
}
else
{
lean_dec_ref(v_fvars_5007_);
lean_dec_ref(v_post_5003_);
lean_dec_ref(v_pre_5002_);
return v___x_5033_;
}
}
}
}
static lean_object* _init_lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1(void){
_start:
{
lean_object* v___x_5040_; lean_object* v_dummy_5041_; 
v___x_5040_ = lean_box(0);
v_dummy_5041_ = l_Lean_Expr_sort___override(v___x_5040_);
return v_dummy_5041_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2(lean_object* v_pre_5042_, lean_object* v_post_5043_, uint8_t v_usedLetOnly_5044_, uint8_t v_skipConstInApp_5045_, uint8_t v_skipInstances_5046_, size_t v_sz_5047_, size_t v_i_5048_, lean_object* v_bs_5049_, lean_object* v___y_5050_, lean_object* v___y_5051_, lean_object* v___y_5052_, lean_object* v___y_5053_, lean_object* v___y_5054_){
_start:
{
uint8_t v___x_5056_; 
v___x_5056_ = lean_usize_dec_lt(v_i_5048_, v_sz_5047_);
if (v___x_5056_ == 0)
{
lean_object* v___x_5057_; 
lean_dec_ref(v_post_5043_);
lean_dec_ref(v_pre_5042_);
v___x_5057_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5057_, 0, v_bs_5049_);
return v___x_5057_;
}
else
{
lean_object* v_v_5058_; lean_object* v___x_5059_; 
v_v_5058_ = lean_array_uget_borrowed(v_bs_5049_, v_i_5048_);
lean_inc(v_v_5058_);
lean_inc_ref(v_post_5043_);
lean_inc_ref(v_pre_5042_);
v___x_5059_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5042_, v_post_5043_, v_usedLetOnly_5044_, v_skipConstInApp_5045_, v_skipInstances_5046_, v_v_5058_, v___y_5050_, v___y_5051_, v___y_5052_, v___y_5053_, v___y_5054_);
if (lean_obj_tag(v___x_5059_) == 0)
{
lean_object* v_a_5060_; lean_object* v___x_5061_; lean_object* v_bs_x27_5062_; size_t v___x_5063_; size_t v___x_5064_; lean_object* v___x_5065_; 
v_a_5060_ = lean_ctor_get(v___x_5059_, 0);
lean_inc(v_a_5060_);
lean_dec_ref_known(v___x_5059_, 1);
v___x_5061_ = lean_unsigned_to_nat(0u);
v_bs_x27_5062_ = lean_array_uset(v_bs_5049_, v_i_5048_, v___x_5061_);
v___x_5063_ = ((size_t)1ULL);
v___x_5064_ = lean_usize_add(v_i_5048_, v___x_5063_);
v___x_5065_ = lean_array_uset(v_bs_x27_5062_, v_i_5048_, v_a_5060_);
v_i_5048_ = v___x_5064_;
v_bs_5049_ = v___x_5065_;
goto _start;
}
else
{
lean_object* v_a_5067_; lean_object* v___x_5069_; uint8_t v_isShared_5070_; uint8_t v_isSharedCheck_5074_; 
lean_dec_ref(v_bs_5049_);
lean_dec_ref(v_post_5043_);
lean_dec_ref(v_pre_5042_);
v_a_5067_ = lean_ctor_get(v___x_5059_, 0);
v_isSharedCheck_5074_ = !lean_is_exclusive(v___x_5059_);
if (v_isSharedCheck_5074_ == 0)
{
v___x_5069_ = v___x_5059_;
v_isShared_5070_ = v_isSharedCheck_5074_;
goto v_resetjp_5068_;
}
else
{
lean_inc(v_a_5067_);
lean_dec(v___x_5059_);
v___x_5069_ = lean_box(0);
v_isShared_5070_ = v_isSharedCheck_5074_;
goto v_resetjp_5068_;
}
v_resetjp_5068_:
{
lean_object* v___x_5072_; 
if (v_isShared_5070_ == 0)
{
v___x_5072_ = v___x_5069_;
goto v_reusejp_5071_;
}
else
{
lean_object* v_reuseFailAlloc_5073_; 
v_reuseFailAlloc_5073_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5073_, 0, v_a_5067_);
v___x_5072_ = v_reuseFailAlloc_5073_;
goto v_reusejp_5071_;
}
v_reusejp_5071_:
{
return v___x_5072_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0(lean_object* v_pre_5075_, lean_object* v_post_5076_, uint8_t v_usedLetOnly_5077_, uint8_t v_skipConstInApp_5078_, uint8_t v_skipInstances_5079_, lean_object* v___x_5080_, lean_object* v___y_5081_, lean_object* v_b_5082_, lean_object* v_a_5083_, lean_object* v___y_5084_, lean_object* v___y_5085_, lean_object* v___y_5086_, lean_object* v___y_5087_){
_start:
{
lean_object* v___x_5089_; 
v___x_5089_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5075_, v_post_5076_, v_usedLetOnly_5077_, v_skipConstInApp_5078_, v_skipInstances_5079_, v___x_5080_, v___y_5081_, v___y_5084_, v___y_5085_, v___y_5086_, v___y_5087_);
if (lean_obj_tag(v___x_5089_) == 0)
{
lean_object* v_a_5090_; lean_object* v___x_5092_; uint8_t v_isShared_5093_; uint8_t v_isSharedCheck_5099_; 
v_a_5090_ = lean_ctor_get(v___x_5089_, 0);
v_isSharedCheck_5099_ = !lean_is_exclusive(v___x_5089_);
if (v_isSharedCheck_5099_ == 0)
{
v___x_5092_ = v___x_5089_;
v_isShared_5093_ = v_isSharedCheck_5099_;
goto v_resetjp_5091_;
}
else
{
lean_inc(v_a_5090_);
lean_dec(v___x_5089_);
v___x_5092_ = lean_box(0);
v_isShared_5093_ = v_isSharedCheck_5099_;
goto v_resetjp_5091_;
}
v_resetjp_5091_:
{
lean_object* v___x_5094_; lean_object* v___x_5095_; lean_object* v___x_5097_; 
v___x_5094_ = lean_array_fset(v_b_5082_, v_a_5083_, v_a_5090_);
v___x_5095_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5095_, 0, v___x_5094_);
if (v_isShared_5093_ == 0)
{
lean_ctor_set(v___x_5092_, 0, v___x_5095_);
v___x_5097_ = v___x_5092_;
goto v_reusejp_5096_;
}
else
{
lean_object* v_reuseFailAlloc_5098_; 
v_reuseFailAlloc_5098_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5098_, 0, v___x_5095_);
v___x_5097_ = v_reuseFailAlloc_5098_;
goto v_reusejp_5096_;
}
v_reusejp_5096_:
{
return v___x_5097_;
}
}
}
else
{
lean_object* v_a_5100_; lean_object* v___x_5102_; uint8_t v_isShared_5103_; uint8_t v_isSharedCheck_5107_; 
lean_dec_ref(v_b_5082_);
v_a_5100_ = lean_ctor_get(v___x_5089_, 0);
v_isSharedCheck_5107_ = !lean_is_exclusive(v___x_5089_);
if (v_isSharedCheck_5107_ == 0)
{
v___x_5102_ = v___x_5089_;
v_isShared_5103_ = v_isSharedCheck_5107_;
goto v_resetjp_5101_;
}
else
{
lean_inc(v_a_5100_);
lean_dec(v___x_5089_);
v___x_5102_ = lean_box(0);
v_isShared_5103_ = v_isSharedCheck_5107_;
goto v_resetjp_5101_;
}
v_resetjp_5101_:
{
lean_object* v___x_5105_; 
if (v_isShared_5103_ == 0)
{
v___x_5105_ = v___x_5102_;
goto v_reusejp_5104_;
}
else
{
lean_object* v_reuseFailAlloc_5106_; 
v_reuseFailAlloc_5106_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5106_, 0, v_a_5100_);
v___x_5105_ = v_reuseFailAlloc_5106_;
goto v_reusejp_5104_;
}
v_reusejp_5104_:
{
return v___x_5105_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0___boxed(lean_object* v_pre_5108_, lean_object* v_post_5109_, lean_object* v_usedLetOnly_5110_, lean_object* v_skipConstInApp_5111_, lean_object* v_skipInstances_5112_, lean_object* v___x_5113_, lean_object* v___y_5114_, lean_object* v_b_5115_, lean_object* v_a_5116_, lean_object* v___y_5117_, lean_object* v___y_5118_, lean_object* v___y_5119_, lean_object* v___y_5120_, lean_object* v___y_5121_){
_start:
{
uint8_t v_usedLetOnly_boxed_5122_; uint8_t v_skipConstInApp_boxed_5123_; uint8_t v_skipInstances_boxed_5124_; lean_object* v_res_5125_; 
v_usedLetOnly_boxed_5122_ = lean_unbox(v_usedLetOnly_5110_);
v_skipConstInApp_boxed_5123_ = lean_unbox(v_skipConstInApp_5111_);
v_skipInstances_boxed_5124_ = lean_unbox(v_skipInstances_5112_);
v_res_5125_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0(v_pre_5108_, v_post_5109_, v_usedLetOnly_boxed_5122_, v_skipConstInApp_boxed_5123_, v_skipInstances_boxed_5124_, v___x_5113_, v___y_5114_, v_b_5115_, v_a_5116_, v___y_5117_, v___y_5118_, v___y_5119_, v___y_5120_);
lean_dec(v___y_5120_);
lean_dec_ref(v___y_5119_);
lean_dec(v___y_5118_);
lean_dec_ref(v___y_5117_);
lean_dec(v_a_5116_);
lean_dec(v___y_5114_);
return v_res_5125_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg(lean_object* v_upperBound_5126_, lean_object* v___x_5127_, lean_object* v_pre_5128_, lean_object* v_post_5129_, uint8_t v_usedLetOnly_5130_, uint8_t v_skipConstInApp_5131_, uint8_t v_skipInstances_5132_, lean_object* v_a_5133_, lean_object* v_b_5134_, lean_object* v___y_5135_, lean_object* v___y_5136_, lean_object* v___y_5137_, lean_object* v___y_5138_, lean_object* v___y_5139_){
_start:
{
lean_object* v___y_5142_; uint8_t v___x_5165_; 
v___x_5165_ = lean_nat_dec_lt(v_a_5133_, v_upperBound_5126_);
if (v___x_5165_ == 0)
{
lean_object* v___x_5166_; 
lean_dec(v_a_5133_);
lean_dec_ref(v_post_5129_);
lean_dec_ref(v_pre_5128_);
v___x_5166_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5166_, 0, v_b_5134_);
return v___x_5166_;
}
else
{
lean_object* v___x_5167_; lean_object* v___x_5168_; uint8_t v___x_5169_; 
v___x_5167_ = lean_array_fget_borrowed(v_b_5134_, v_a_5133_);
v___x_5168_ = lean_array_get_size(v___x_5127_);
v___x_5169_ = lean_nat_dec_lt(v_a_5133_, v___x_5168_);
if (v___x_5169_ == 0)
{
lean_object* v___x_5170_; lean_object* v___x_5171_; lean_object* v___x_5172_; lean_object* v___f_5173_; 
lean_inc(v___x_5167_);
v___x_5170_ = lean_box(v_usedLetOnly_5130_);
v___x_5171_ = lean_box(v_skipConstInApp_5131_);
v___x_5172_ = lean_box(v_skipInstances_5132_);
lean_inc(v_a_5133_);
lean_inc(v___y_5135_);
lean_inc_ref(v_post_5129_);
lean_inc_ref(v_pre_5128_);
v___f_5173_ = lean_alloc_closure((void*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_5173_, 0, v_pre_5128_);
lean_closure_set(v___f_5173_, 1, v_post_5129_);
lean_closure_set(v___f_5173_, 2, v___x_5170_);
lean_closure_set(v___f_5173_, 3, v___x_5171_);
lean_closure_set(v___f_5173_, 4, v___x_5172_);
lean_closure_set(v___f_5173_, 5, v___x_5167_);
lean_closure_set(v___f_5173_, 6, v___y_5135_);
lean_closure_set(v___f_5173_, 7, v_b_5134_);
lean_closure_set(v___f_5173_, 8, v_a_5133_);
v___y_5142_ = v___f_5173_;
goto v___jp_5141_;
}
else
{
lean_object* v___x_5174_; uint8_t v_isInstance_5175_; 
v___x_5174_ = lean_array_fget_borrowed(v___x_5127_, v_a_5133_);
v_isInstance_5175_ = lean_ctor_get_uint8(v___x_5174_, sizeof(void*)*1 + 4);
if (v_isInstance_5175_ == 0)
{
lean_object* v___x_5176_; lean_object* v___x_5177_; lean_object* v___x_5178_; lean_object* v___f_5179_; 
lean_inc(v___x_5167_);
v___x_5176_ = lean_box(v_usedLetOnly_5130_);
v___x_5177_ = lean_box(v_skipConstInApp_5131_);
v___x_5178_ = lean_box(v_skipInstances_5132_);
lean_inc(v_a_5133_);
lean_inc(v___y_5135_);
lean_inc_ref(v_post_5129_);
lean_inc_ref(v_pre_5128_);
v___f_5179_ = lean_alloc_closure((void*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__0___boxed), 14, 9);
lean_closure_set(v___f_5179_, 0, v_pre_5128_);
lean_closure_set(v___f_5179_, 1, v_post_5129_);
lean_closure_set(v___f_5179_, 2, v___x_5176_);
lean_closure_set(v___f_5179_, 3, v___x_5177_);
lean_closure_set(v___f_5179_, 4, v___x_5178_);
lean_closure_set(v___f_5179_, 5, v___x_5167_);
lean_closure_set(v___f_5179_, 6, v___y_5135_);
lean_closure_set(v___f_5179_, 7, v_b_5134_);
lean_closure_set(v___f_5179_, 8, v_a_5133_);
v___y_5142_ = v___f_5179_;
goto v___jp_5141_;
}
else
{
lean_object* v___x_5180_; lean_object* v___f_5181_; 
v___x_5180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5180_, 0, v_b_5134_);
v___f_5181_ = lean_alloc_closure((void*)(lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___lam__2___boxed), 6, 1);
lean_closure_set(v___f_5181_, 0, v___x_5180_);
v___y_5142_ = v___f_5181_;
goto v___jp_5141_;
}
}
}
v___jp_5141_:
{
lean_object* v___x_5143_; 
lean_inc(v___y_5139_);
lean_inc_ref(v___y_5138_);
lean_inc(v___y_5137_);
lean_inc_ref(v___y_5136_);
v___x_5143_ = lean_apply_5(v___y_5142_, v___y_5136_, v___y_5137_, v___y_5138_, v___y_5139_, lean_box(0));
if (lean_obj_tag(v___x_5143_) == 0)
{
lean_object* v_a_5144_; lean_object* v___x_5146_; uint8_t v_isShared_5147_; uint8_t v_isSharedCheck_5156_; 
v_a_5144_ = lean_ctor_get(v___x_5143_, 0);
v_isSharedCheck_5156_ = !lean_is_exclusive(v___x_5143_);
if (v_isSharedCheck_5156_ == 0)
{
v___x_5146_ = v___x_5143_;
v_isShared_5147_ = v_isSharedCheck_5156_;
goto v_resetjp_5145_;
}
else
{
lean_inc(v_a_5144_);
lean_dec(v___x_5143_);
v___x_5146_ = lean_box(0);
v_isShared_5147_ = v_isSharedCheck_5156_;
goto v_resetjp_5145_;
}
v_resetjp_5145_:
{
if (lean_obj_tag(v_a_5144_) == 0)
{
lean_object* v_a_5148_; lean_object* v___x_5150_; 
lean_dec(v_a_5133_);
lean_dec_ref(v_post_5129_);
lean_dec_ref(v_pre_5128_);
v_a_5148_ = lean_ctor_get(v_a_5144_, 0);
lean_inc(v_a_5148_);
lean_dec_ref_known(v_a_5144_, 1);
if (v_isShared_5147_ == 0)
{
lean_ctor_set(v___x_5146_, 0, v_a_5148_);
v___x_5150_ = v___x_5146_;
goto v_reusejp_5149_;
}
else
{
lean_object* v_reuseFailAlloc_5151_; 
v_reuseFailAlloc_5151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5151_, 0, v_a_5148_);
v___x_5150_ = v_reuseFailAlloc_5151_;
goto v_reusejp_5149_;
}
v_reusejp_5149_:
{
return v___x_5150_;
}
}
else
{
lean_object* v_a_5152_; lean_object* v___x_5153_; lean_object* v___x_5154_; 
lean_del_object(v___x_5146_);
v_a_5152_ = lean_ctor_get(v_a_5144_, 0);
lean_inc(v_a_5152_);
lean_dec_ref_known(v_a_5144_, 1);
v___x_5153_ = lean_unsigned_to_nat(1u);
v___x_5154_ = lean_nat_add(v_a_5133_, v___x_5153_);
lean_dec(v_a_5133_);
v_a_5133_ = v___x_5154_;
v_b_5134_ = v_a_5152_;
goto _start;
}
}
}
else
{
lean_object* v_a_5157_; lean_object* v___x_5159_; uint8_t v_isShared_5160_; uint8_t v_isSharedCheck_5164_; 
lean_dec(v_a_5133_);
lean_dec_ref(v_post_5129_);
lean_dec_ref(v_pre_5128_);
v_a_5157_ = lean_ctor_get(v___x_5143_, 0);
v_isSharedCheck_5164_ = !lean_is_exclusive(v___x_5143_);
if (v_isSharedCheck_5164_ == 0)
{
v___x_5159_ = v___x_5143_;
v_isShared_5160_ = v_isSharedCheck_5164_;
goto v_resetjp_5158_;
}
else
{
lean_inc(v_a_5157_);
lean_dec(v___x_5143_);
v___x_5159_ = lean_box(0);
v_isShared_5160_ = v_isSharedCheck_5164_;
goto v_resetjp_5158_;
}
v_resetjp_5158_:
{
lean_object* v___x_5162_; 
if (v_isShared_5160_ == 0)
{
v___x_5162_ = v___x_5159_;
goto v_reusejp_5161_;
}
else
{
lean_object* v_reuseFailAlloc_5163_; 
v_reuseFailAlloc_5163_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5163_, 0, v_a_5157_);
v___x_5162_ = v_reuseFailAlloc_5163_;
goto v_reusejp_5161_;
}
v_reusejp_5161_:
{
return v___x_5162_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9(uint8_t v_skipInstances_5182_, lean_object* v_pre_5183_, lean_object* v_post_5184_, uint8_t v_usedLetOnly_5185_, uint8_t v_skipConstInApp_5186_, lean_object* v_x_5187_, lean_object* v_x_5188_, lean_object* v_x_5189_, lean_object* v___y_5190_, lean_object* v___y_5191_, lean_object* v___y_5192_, lean_object* v___y_5193_, lean_object* v___y_5194_){
_start:
{
lean_object* v_f_5197_; lean_object* v___y_5198_; lean_object* v___y_5199_; lean_object* v___y_5200_; lean_object* v___y_5201_; lean_object* v___y_5202_; 
if (lean_obj_tag(v_x_5187_) == 5)
{
lean_object* v_fn_5245_; lean_object* v_arg_5246_; lean_object* v___x_5247_; lean_object* v___x_5248_; lean_object* v___x_5249_; 
v_fn_5245_ = lean_ctor_get(v_x_5187_, 0);
lean_inc_ref(v_fn_5245_);
v_arg_5246_ = lean_ctor_get(v_x_5187_, 1);
lean_inc_ref(v_arg_5246_);
lean_dec_ref_known(v_x_5187_, 2);
v___x_5247_ = lean_array_set(v_x_5188_, v_x_5189_, v_arg_5246_);
v___x_5248_ = lean_unsigned_to_nat(1u);
v___x_5249_ = lean_nat_sub(v_x_5189_, v___x_5248_);
lean_dec(v_x_5189_);
v_x_5187_ = v_fn_5245_;
v_x_5188_ = v___x_5247_;
v_x_5189_ = v___x_5249_;
goto _start;
}
else
{
lean_dec(v_x_5189_);
if (v_skipConstInApp_5186_ == 0)
{
goto v___jp_5242_;
}
else
{
uint8_t v___x_5251_; 
v___x_5251_ = l_Lean_Expr_isConst(v_x_5187_);
if (v___x_5251_ == 0)
{
goto v___jp_5242_;
}
else
{
v_f_5197_ = v_x_5187_;
v___y_5198_ = v___y_5190_;
v___y_5199_ = v___y_5191_;
v___y_5200_ = v___y_5192_;
v___y_5201_ = v___y_5193_;
v___y_5202_ = v___y_5194_;
goto v___jp_5196_;
}
}
}
v___jp_5196_:
{
if (v_skipInstances_5182_ == 0)
{
size_t v_sz_5203_; size_t v___x_5204_; lean_object* v___x_5205_; 
v_sz_5203_ = lean_array_size(v_x_5188_);
v___x_5204_ = ((size_t)0ULL);
lean_inc_ref(v_post_5184_);
lean_inc_ref(v_pre_5183_);
v___x_5205_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2(v_pre_5183_, v_post_5184_, v_usedLetOnly_5185_, v_skipConstInApp_5186_, v_skipInstances_5182_, v_sz_5203_, v___x_5204_, v_x_5188_, v___y_5198_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_);
if (lean_obj_tag(v___x_5205_) == 0)
{
lean_object* v_a_5206_; lean_object* v___x_5207_; lean_object* v___x_5208_; 
v_a_5206_ = lean_ctor_get(v___x_5205_, 0);
lean_inc(v_a_5206_);
lean_dec_ref_known(v___x_5205_, 1);
v___x_5207_ = l_Lean_mkAppN(v_f_5197_, v_a_5206_);
lean_dec(v_a_5206_);
v___x_5208_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5183_, v_post_5184_, v_usedLetOnly_5185_, v_skipConstInApp_5186_, v_skipInstances_5182_, v___x_5207_, v___y_5198_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_);
return v___x_5208_;
}
else
{
lean_object* v_a_5209_; lean_object* v___x_5211_; uint8_t v_isShared_5212_; uint8_t v_isSharedCheck_5216_; 
lean_dec_ref(v_f_5197_);
lean_dec_ref(v_post_5184_);
lean_dec_ref(v_pre_5183_);
v_a_5209_ = lean_ctor_get(v___x_5205_, 0);
v_isSharedCheck_5216_ = !lean_is_exclusive(v___x_5205_);
if (v_isSharedCheck_5216_ == 0)
{
v___x_5211_ = v___x_5205_;
v_isShared_5212_ = v_isSharedCheck_5216_;
goto v_resetjp_5210_;
}
else
{
lean_inc(v_a_5209_);
lean_dec(v___x_5205_);
v___x_5211_ = lean_box(0);
v_isShared_5212_ = v_isSharedCheck_5216_;
goto v_resetjp_5210_;
}
v_resetjp_5210_:
{
lean_object* v___x_5214_; 
if (v_isShared_5212_ == 0)
{
v___x_5214_ = v___x_5211_;
goto v_reusejp_5213_;
}
else
{
lean_object* v_reuseFailAlloc_5215_; 
v_reuseFailAlloc_5215_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5215_, 0, v_a_5209_);
v___x_5214_ = v_reuseFailAlloc_5215_;
goto v_reusejp_5213_;
}
v_reusejp_5213_:
{
return v___x_5214_;
}
}
}
}
else
{
lean_object* v___x_5217_; lean_object* v___x_5218_; 
v___x_5217_ = lean_array_get_size(v_x_5188_);
lean_inc_ref(v_f_5197_);
v___x_5218_ = l_Lean_Meta_getFunInfoNArgs(v_f_5197_, v___x_5217_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_);
if (lean_obj_tag(v___x_5218_) == 0)
{
lean_object* v_a_5219_; lean_object* v_paramInfo_5220_; lean_object* v___x_5221_; lean_object* v___x_5222_; 
v_a_5219_ = lean_ctor_get(v___x_5218_, 0);
lean_inc(v_a_5219_);
lean_dec_ref_known(v___x_5218_, 1);
v_paramInfo_5220_ = lean_ctor_get(v_a_5219_, 0);
lean_inc_ref(v_paramInfo_5220_);
lean_dec(v_a_5219_);
v___x_5221_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_post_5184_);
lean_inc_ref(v_pre_5183_);
v___x_5222_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg(v___x_5217_, v_paramInfo_5220_, v_pre_5183_, v_post_5184_, v_usedLetOnly_5185_, v_skipConstInApp_5186_, v_skipInstances_5182_, v___x_5221_, v_x_5188_, v___y_5198_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_);
lean_dec_ref(v_paramInfo_5220_);
if (lean_obj_tag(v___x_5222_) == 0)
{
lean_object* v_a_5223_; lean_object* v___x_5224_; lean_object* v___x_5225_; 
v_a_5223_ = lean_ctor_get(v___x_5222_, 0);
lean_inc(v_a_5223_);
lean_dec_ref_known(v___x_5222_, 1);
v___x_5224_ = l_Lean_mkAppN(v_f_5197_, v_a_5223_);
lean_dec(v_a_5223_);
v___x_5225_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5183_, v_post_5184_, v_usedLetOnly_5185_, v_skipConstInApp_5186_, v_skipInstances_5182_, v___x_5224_, v___y_5198_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_);
return v___x_5225_;
}
else
{
lean_object* v_a_5226_; lean_object* v___x_5228_; uint8_t v_isShared_5229_; uint8_t v_isSharedCheck_5233_; 
lean_dec_ref(v_f_5197_);
lean_dec_ref(v_post_5184_);
lean_dec_ref(v_pre_5183_);
v_a_5226_ = lean_ctor_get(v___x_5222_, 0);
v_isSharedCheck_5233_ = !lean_is_exclusive(v___x_5222_);
if (v_isSharedCheck_5233_ == 0)
{
v___x_5228_ = v___x_5222_;
v_isShared_5229_ = v_isSharedCheck_5233_;
goto v_resetjp_5227_;
}
else
{
lean_inc(v_a_5226_);
lean_dec(v___x_5222_);
v___x_5228_ = lean_box(0);
v_isShared_5229_ = v_isSharedCheck_5233_;
goto v_resetjp_5227_;
}
v_resetjp_5227_:
{
lean_object* v___x_5231_; 
if (v_isShared_5229_ == 0)
{
v___x_5231_ = v___x_5228_;
goto v_reusejp_5230_;
}
else
{
lean_object* v_reuseFailAlloc_5232_; 
v_reuseFailAlloc_5232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5232_, 0, v_a_5226_);
v___x_5231_ = v_reuseFailAlloc_5232_;
goto v_reusejp_5230_;
}
v_reusejp_5230_:
{
return v___x_5231_;
}
}
}
}
else
{
lean_object* v_a_5234_; lean_object* v___x_5236_; uint8_t v_isShared_5237_; uint8_t v_isSharedCheck_5241_; 
lean_dec_ref(v_f_5197_);
lean_dec_ref(v_x_5188_);
lean_dec_ref(v_post_5184_);
lean_dec_ref(v_pre_5183_);
v_a_5234_ = lean_ctor_get(v___x_5218_, 0);
v_isSharedCheck_5241_ = !lean_is_exclusive(v___x_5218_);
if (v_isSharedCheck_5241_ == 0)
{
v___x_5236_ = v___x_5218_;
v_isShared_5237_ = v_isSharedCheck_5241_;
goto v_resetjp_5235_;
}
else
{
lean_inc(v_a_5234_);
lean_dec(v___x_5218_);
v___x_5236_ = lean_box(0);
v_isShared_5237_ = v_isSharedCheck_5241_;
goto v_resetjp_5235_;
}
v_resetjp_5235_:
{
lean_object* v___x_5239_; 
if (v_isShared_5237_ == 0)
{
v___x_5239_ = v___x_5236_;
goto v_reusejp_5238_;
}
else
{
lean_object* v_reuseFailAlloc_5240_; 
v_reuseFailAlloc_5240_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5240_, 0, v_a_5234_);
v___x_5239_ = v_reuseFailAlloc_5240_;
goto v_reusejp_5238_;
}
v_reusejp_5238_:
{
return v___x_5239_;
}
}
}
}
}
v___jp_5242_:
{
lean_object* v___x_5243_; 
lean_inc_ref(v_post_5184_);
lean_inc_ref(v_pre_5183_);
v___x_5243_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5183_, v_post_5184_, v_usedLetOnly_5185_, v_skipConstInApp_5186_, v_skipInstances_5182_, v_x_5187_, v___y_5190_, v___y_5191_, v___y_5192_, v___y_5193_, v___y_5194_);
if (lean_obj_tag(v___x_5243_) == 0)
{
lean_object* v_a_5244_; 
v_a_5244_ = lean_ctor_get(v___x_5243_, 0);
lean_inc(v_a_5244_);
lean_dec_ref_known(v___x_5243_, 1);
v_f_5197_ = v_a_5244_;
v___y_5198_ = v___y_5190_;
v___y_5199_ = v___y_5191_;
v___y_5200_ = v___y_5192_;
v___y_5201_ = v___y_5193_;
v___y_5202_ = v___y_5194_;
goto v___jp_5196_;
}
else
{
lean_dec_ref(v_x_5188_);
lean_dec_ref(v_post_5184_);
lean_dec_ref(v_pre_5183_);
return v___x_5243_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1(lean_object* v___x_5252_, lean_object* v_pre_5253_, lean_object* v_e_5254_, lean_object* v_post_5255_, uint8_t v_usedLetOnly_5256_, uint8_t v_skipConstInApp_5257_, uint8_t v_skipInstances_5258_, lean_object* v___y_5259_, lean_object* v___y_5260_, lean_object* v___y_5261_, lean_object* v___y_5262_, lean_object* v___y_5263_){
_start:
{
lean_object* v___x_5265_; 
v___x_5265_ = l_Lean_Core_checkSystem(v___x_5252_, v___y_5262_, v___y_5263_);
if (lean_obj_tag(v___x_5265_) == 0)
{
lean_object* v___x_5266_; 
lean_dec_ref_known(v___x_5265_, 1);
lean_inc_ref(v_pre_5253_);
lean_inc(v___y_5263_);
lean_inc_ref(v___y_5262_);
lean_inc(v___y_5261_);
lean_inc_ref(v___y_5260_);
lean_inc_ref(v_e_5254_);
v___x_5266_ = lean_apply_6(v_pre_5253_, v_e_5254_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_, lean_box(0));
if (lean_obj_tag(v___x_5266_) == 0)
{
lean_object* v_a_5267_; lean_object* v___x_5269_; uint8_t v_isShared_5270_; uint8_t v_isSharedCheck_5315_; 
v_a_5267_ = lean_ctor_get(v___x_5266_, 0);
v_isSharedCheck_5315_ = !lean_is_exclusive(v___x_5266_);
if (v_isSharedCheck_5315_ == 0)
{
v___x_5269_ = v___x_5266_;
v_isShared_5270_ = v_isSharedCheck_5315_;
goto v_resetjp_5268_;
}
else
{
lean_inc(v_a_5267_);
lean_dec(v___x_5266_);
v___x_5269_ = lean_box(0);
v_isShared_5270_ = v_isSharedCheck_5315_;
goto v_resetjp_5268_;
}
v_resetjp_5268_:
{
lean_object* v___y_5272_; 
switch(lean_obj_tag(v_a_5267_))
{
case 0:
{
lean_object* v_e_5307_; lean_object* v___x_5309_; 
lean_dec_ref(v_post_5255_);
lean_dec_ref(v_e_5254_);
lean_dec_ref(v_pre_5253_);
v_e_5307_ = lean_ctor_get(v_a_5267_, 0);
lean_inc_ref(v_e_5307_);
lean_dec_ref_known(v_a_5267_, 1);
if (v_isShared_5270_ == 0)
{
lean_ctor_set(v___x_5269_, 0, v_e_5307_);
v___x_5309_ = v___x_5269_;
goto v_reusejp_5308_;
}
else
{
lean_object* v_reuseFailAlloc_5310_; 
v_reuseFailAlloc_5310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5310_, 0, v_e_5307_);
v___x_5309_ = v_reuseFailAlloc_5310_;
goto v_reusejp_5308_;
}
v_reusejp_5308_:
{
return v___x_5309_;
}
}
case 1:
{
lean_object* v_e_5311_; lean_object* v___x_5312_; 
lean_del_object(v___x_5269_);
lean_dec_ref(v_e_5254_);
v_e_5311_ = lean_ctor_get(v_a_5267_, 0);
lean_inc_ref(v_e_5311_);
lean_dec_ref_known(v_a_5267_, 1);
v___x_5312_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v_e_5311_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5312_;
}
default: 
{
lean_object* v_e_x3f_5313_; 
lean_del_object(v___x_5269_);
v_e_x3f_5313_ = lean_ctor_get(v_a_5267_, 0);
lean_inc(v_e_x3f_5313_);
lean_dec_ref_known(v_a_5267_, 1);
if (lean_obj_tag(v_e_x3f_5313_) == 0)
{
v___y_5272_ = v_e_5254_;
goto v___jp_5271_;
}
else
{
lean_object* v_val_5314_; 
lean_dec_ref(v_e_5254_);
v_val_5314_ = lean_ctor_get(v_e_x3f_5313_, 0);
lean_inc(v_val_5314_);
lean_dec_ref_known(v_e_x3f_5313_, 1);
v___y_5272_ = v_val_5314_;
goto v___jp_5271_;
}
}
}
v___jp_5271_:
{
switch(lean_obj_tag(v___y_5272_))
{
case 7:
{
lean_object* v___x_5273_; lean_object* v___x_5274_; 
v___x_5273_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0));
v___x_5274_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___x_5273_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5274_;
}
case 6:
{
lean_object* v___x_5275_; lean_object* v___x_5276_; 
v___x_5275_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0));
v___x_5276_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___x_5275_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5276_;
}
case 8:
{
lean_object* v___x_5277_; lean_object* v___x_5278_; 
v___x_5277_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__0));
v___x_5278_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___x_5277_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5278_;
}
case 5:
{
lean_object* v_dummy_5279_; lean_object* v_nargs_5280_; lean_object* v___x_5281_; lean_object* v___x_5282_; lean_object* v___x_5283_; lean_object* v___x_5284_; 
v_dummy_5279_ = lean_obj_once(&lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1, &lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1_once, _init_lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___closed__1);
v_nargs_5280_ = l_Lean_Expr_getAppNumArgs(v___y_5272_);
lean_inc(v_nargs_5280_);
v___x_5281_ = lean_mk_array(v_nargs_5280_, v_dummy_5279_);
v___x_5282_ = lean_unsigned_to_nat(1u);
v___x_5283_ = lean_nat_sub(v_nargs_5280_, v___x_5282_);
lean_dec(v_nargs_5280_);
v___x_5284_ = lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9(v_skipInstances_5258_, v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v___y_5272_, v___x_5281_, v___x_5283_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5284_;
}
case 10:
{
lean_object* v_data_5285_; lean_object* v_expr_5286_; lean_object* v___x_5287_; 
v_data_5285_ = lean_ctor_get(v___y_5272_, 0);
v_expr_5286_ = lean_ctor_get(v___y_5272_, 1);
lean_inc_ref(v_expr_5286_);
lean_inc_ref(v_post_5255_);
lean_inc_ref(v_pre_5253_);
v___x_5287_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v_expr_5286_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
if (lean_obj_tag(v___x_5287_) == 0)
{
lean_object* v_a_5288_; size_t v___x_5289_; size_t v___x_5290_; uint8_t v___x_5291_; 
v_a_5288_ = lean_ctor_get(v___x_5287_, 0);
lean_inc(v_a_5288_);
lean_dec_ref_known(v___x_5287_, 1);
v___x_5289_ = lean_ptr_addr(v_expr_5286_);
v___x_5290_ = lean_ptr_addr(v_a_5288_);
v___x_5291_ = lean_usize_dec_eq(v___x_5289_, v___x_5290_);
if (v___x_5291_ == 0)
{
lean_object* v___x_5292_; lean_object* v___x_5293_; 
lean_inc(v_data_5285_);
lean_dec_ref_known(v___y_5272_, 2);
v___x_5292_ = l_Lean_Expr_mdata___override(v_data_5285_, v_a_5288_);
v___x_5293_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___x_5292_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5293_;
}
else
{
lean_object* v___x_5294_; 
lean_dec(v_a_5288_);
v___x_5294_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5294_;
}
}
else
{
lean_dec_ref_known(v___y_5272_, 2);
lean_dec_ref(v_post_5255_);
lean_dec_ref(v_pre_5253_);
return v___x_5287_;
}
}
case 11:
{
lean_object* v_typeName_5295_; lean_object* v_idx_5296_; lean_object* v_struct_5297_; lean_object* v___x_5298_; 
v_typeName_5295_ = lean_ctor_get(v___y_5272_, 0);
v_idx_5296_ = lean_ctor_get(v___y_5272_, 1);
v_struct_5297_ = lean_ctor_get(v___y_5272_, 2);
lean_inc_ref(v_struct_5297_);
lean_inc_ref(v_post_5255_);
lean_inc_ref(v_pre_5253_);
v___x_5298_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v_struct_5297_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
if (lean_obj_tag(v___x_5298_) == 0)
{
lean_object* v_a_5299_; size_t v___x_5300_; size_t v___x_5301_; uint8_t v___x_5302_; 
v_a_5299_ = lean_ctor_get(v___x_5298_, 0);
lean_inc(v_a_5299_);
lean_dec_ref_known(v___x_5298_, 1);
v___x_5300_ = lean_ptr_addr(v_struct_5297_);
v___x_5301_ = lean_ptr_addr(v_a_5299_);
v___x_5302_ = lean_usize_dec_eq(v___x_5300_, v___x_5301_);
if (v___x_5302_ == 0)
{
lean_object* v___x_5303_; lean_object* v___x_5304_; 
lean_inc(v_idx_5296_);
lean_inc(v_typeName_5295_);
lean_dec_ref_known(v___y_5272_, 3);
v___x_5303_ = l_Lean_Expr_proj___override(v_typeName_5295_, v_idx_5296_, v_a_5299_);
v___x_5304_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___x_5303_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5304_;
}
else
{
lean_object* v___x_5305_; 
lean_dec(v_a_5299_);
v___x_5305_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5305_;
}
}
else
{
lean_dec_ref_known(v___y_5272_, 3);
lean_dec_ref(v_post_5255_);
lean_dec_ref(v_pre_5253_);
return v___x_5298_;
}
}
default: 
{
lean_object* v___x_5306_; 
v___x_5306_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5253_, v_post_5255_, v_usedLetOnly_5256_, v_skipConstInApp_5257_, v_skipInstances_5258_, v___y_5272_, v___y_5259_, v___y_5260_, v___y_5261_, v___y_5262_, v___y_5263_);
return v___x_5306_;
}
}
}
}
}
else
{
lean_object* v_a_5316_; lean_object* v___x_5318_; uint8_t v_isShared_5319_; uint8_t v_isSharedCheck_5323_; 
lean_dec_ref(v_post_5255_);
lean_dec_ref(v_e_5254_);
lean_dec_ref(v_pre_5253_);
v_a_5316_ = lean_ctor_get(v___x_5266_, 0);
v_isSharedCheck_5323_ = !lean_is_exclusive(v___x_5266_);
if (v_isSharedCheck_5323_ == 0)
{
v___x_5318_ = v___x_5266_;
v_isShared_5319_ = v_isSharedCheck_5323_;
goto v_resetjp_5317_;
}
else
{
lean_inc(v_a_5316_);
lean_dec(v___x_5266_);
v___x_5318_ = lean_box(0);
v_isShared_5319_ = v_isSharedCheck_5323_;
goto v_resetjp_5317_;
}
v_resetjp_5317_:
{
lean_object* v___x_5321_; 
if (v_isShared_5319_ == 0)
{
v___x_5321_ = v___x_5318_;
goto v_reusejp_5320_;
}
else
{
lean_object* v_reuseFailAlloc_5322_; 
v_reuseFailAlloc_5322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5322_, 0, v_a_5316_);
v___x_5321_ = v_reuseFailAlloc_5322_;
goto v_reusejp_5320_;
}
v_reusejp_5320_:
{
return v___x_5321_;
}
}
}
}
else
{
lean_object* v_a_5324_; lean_object* v___x_5326_; uint8_t v_isShared_5327_; uint8_t v_isSharedCheck_5331_; 
lean_dec_ref(v_post_5255_);
lean_dec_ref(v_e_5254_);
lean_dec_ref(v_pre_5253_);
v_a_5324_ = lean_ctor_get(v___x_5265_, 0);
v_isSharedCheck_5331_ = !lean_is_exclusive(v___x_5265_);
if (v_isSharedCheck_5331_ == 0)
{
v___x_5326_ = v___x_5265_;
v_isShared_5327_ = v_isSharedCheck_5331_;
goto v_resetjp_5325_;
}
else
{
lean_inc(v_a_5324_);
lean_dec(v___x_5265_);
v___x_5326_ = lean_box(0);
v_isShared_5327_ = v_isSharedCheck_5331_;
goto v_resetjp_5325_;
}
v_resetjp_5325_:
{
lean_object* v___x_5329_; 
if (v_isShared_5327_ == 0)
{
v___x_5329_ = v___x_5326_;
goto v_reusejp_5328_;
}
else
{
lean_object* v_reuseFailAlloc_5330_; 
v_reuseFailAlloc_5330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5330_, 0, v_a_5324_);
v___x_5329_ = v_reuseFailAlloc_5330_;
goto v_reusejp_5328_;
}
v_reusejp_5328_:
{
return v___x_5329_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___boxed(lean_object* v___x_5332_, lean_object* v_pre_5333_, lean_object* v_e_5334_, lean_object* v_post_5335_, lean_object* v_usedLetOnly_5336_, lean_object* v_skipConstInApp_5337_, lean_object* v_skipInstances_5338_, lean_object* v___y_5339_, lean_object* v___y_5340_, lean_object* v___y_5341_, lean_object* v___y_5342_, lean_object* v___y_5343_, lean_object* v___y_5344_){
_start:
{
uint8_t v_usedLetOnly_boxed_5345_; uint8_t v_skipConstInApp_boxed_5346_; uint8_t v_skipInstances_boxed_5347_; lean_object* v_res_5348_; 
v_usedLetOnly_boxed_5345_ = lean_unbox(v_usedLetOnly_5336_);
v_skipConstInApp_boxed_5346_ = lean_unbox(v_skipConstInApp_5337_);
v_skipInstances_boxed_5347_ = lean_unbox(v_skipInstances_5338_);
v_res_5348_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1(v___x_5332_, v_pre_5333_, v_e_5334_, v_post_5335_, v_usedLetOnly_boxed_5345_, v_skipConstInApp_boxed_5346_, v_skipInstances_boxed_5347_, v___y_5339_, v___y_5340_, v___y_5341_, v___y_5342_, v___y_5343_);
lean_dec(v___y_5343_);
lean_dec_ref(v___y_5342_);
lean_dec(v___y_5341_);
lean_dec_ref(v___y_5340_);
lean_dec(v___y_5339_);
return v_res_5348_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(lean_object* v_pre_5349_, lean_object* v_post_5350_, uint8_t v_usedLetOnly_5351_, uint8_t v_skipConstInApp_5352_, uint8_t v_skipInstances_5353_, lean_object* v_e_5354_, lean_object* v_a_5355_, lean_object* v___y_5356_, lean_object* v___y_5357_, lean_object* v___y_5358_, lean_object* v___y_5359_){
_start:
{
lean_object* v___x_5361_; lean_object* v___x_5362_; 
lean_inc(v_a_5355_);
v___x_5361_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_5361_, 0, lean_box(0));
lean_closure_set(v___x_5361_, 1, lean_box(0));
lean_closure_set(v___x_5361_, 2, v_a_5355_);
v___x_5362_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0(lean_box(0), v___x_5361_, v___y_5356_, v___y_5357_, v___y_5358_, v___y_5359_);
if (lean_obj_tag(v___x_5362_) == 0)
{
lean_object* v_a_5363_; lean_object* v___x_5365_; uint8_t v_isShared_5366_; uint8_t v_isSharedCheck_5397_; 
v_a_5363_ = lean_ctor_get(v___x_5362_, 0);
v_isSharedCheck_5397_ = !lean_is_exclusive(v___x_5362_);
if (v_isSharedCheck_5397_ == 0)
{
v___x_5365_ = v___x_5362_;
v_isShared_5366_ = v_isSharedCheck_5397_;
goto v_resetjp_5364_;
}
else
{
lean_inc(v_a_5363_);
lean_dec(v___x_5362_);
v___x_5365_ = lean_box(0);
v_isShared_5366_ = v_isSharedCheck_5397_;
goto v_resetjp_5364_;
}
v_resetjp_5364_:
{
lean_object* v___x_5367_; 
v___x_5367_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg(v_a_5363_, v_e_5354_);
lean_dec(v_a_5363_);
if (lean_obj_tag(v___x_5367_) == 0)
{
lean_object* v___x_5368_; lean_object* v___x_5369_; lean_object* v___x_5370_; lean_object* v___x_5371_; lean_object* v___f_5372_; lean_object* v___x_5373_; 
lean_del_object(v___x_5365_);
v___x_5368_ = ((lean_object*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___closed__0));
v___x_5369_ = lean_box(v_usedLetOnly_5351_);
v___x_5370_ = lean_box(v_skipConstInApp_5352_);
v___x_5371_ = lean_box(v_skipInstances_5353_);
lean_inc_ref(v_e_5354_);
v___f_5372_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__1___boxed), 13, 7);
lean_closure_set(v___f_5372_, 0, v___x_5368_);
lean_closure_set(v___f_5372_, 1, v_pre_5349_);
lean_closure_set(v___f_5372_, 2, v_e_5354_);
lean_closure_set(v___f_5372_, 3, v_post_5350_);
lean_closure_set(v___f_5372_, 4, v___x_5369_);
lean_closure_set(v___f_5372_, 5, v___x_5370_);
lean_closure_set(v___f_5372_, 6, v___x_5371_);
v___x_5373_ = lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg(v___f_5372_, v_a_5355_, v___y_5356_, v___y_5357_, v___y_5358_, v___y_5359_);
if (lean_obj_tag(v___x_5373_) == 0)
{
lean_object* v_a_5374_; lean_object* v___f_5375_; lean_object* v___x_5376_; 
v_a_5374_ = lean_ctor_get(v___x_5373_, 0);
lean_inc_n(v_a_5374_, 2);
lean_dec_ref_known(v___x_5373_, 1);
lean_inc(v_a_5355_);
v___f_5375_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__2___boxed), 4, 3);
lean_closure_set(v___f_5375_, 0, v_a_5355_);
lean_closure_set(v___f_5375_, 1, v_e_5354_);
lean_closure_set(v___f_5375_, 2, v_a_5374_);
v___x_5376_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___lam__0(lean_box(0), v___f_5375_, v___y_5356_, v___y_5357_, v___y_5358_, v___y_5359_);
if (lean_obj_tag(v___x_5376_) == 0)
{
lean_object* v___x_5378_; uint8_t v_isShared_5379_; uint8_t v_isSharedCheck_5383_; 
v_isSharedCheck_5383_ = !lean_is_exclusive(v___x_5376_);
if (v_isSharedCheck_5383_ == 0)
{
lean_object* v_unused_5384_; 
v_unused_5384_ = lean_ctor_get(v___x_5376_, 0);
lean_dec(v_unused_5384_);
v___x_5378_ = v___x_5376_;
v_isShared_5379_ = v_isSharedCheck_5383_;
goto v_resetjp_5377_;
}
else
{
lean_dec(v___x_5376_);
v___x_5378_ = lean_box(0);
v_isShared_5379_ = v_isSharedCheck_5383_;
goto v_resetjp_5377_;
}
v_resetjp_5377_:
{
lean_object* v___x_5381_; 
if (v_isShared_5379_ == 0)
{
lean_ctor_set(v___x_5378_, 0, v_a_5374_);
v___x_5381_ = v___x_5378_;
goto v_reusejp_5380_;
}
else
{
lean_object* v_reuseFailAlloc_5382_; 
v_reuseFailAlloc_5382_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5382_, 0, v_a_5374_);
v___x_5381_ = v_reuseFailAlloc_5382_;
goto v_reusejp_5380_;
}
v_reusejp_5380_:
{
return v___x_5381_;
}
}
}
else
{
lean_object* v_a_5385_; lean_object* v___x_5387_; uint8_t v_isShared_5388_; uint8_t v_isSharedCheck_5392_; 
lean_dec(v_a_5374_);
v_a_5385_ = lean_ctor_get(v___x_5376_, 0);
v_isSharedCheck_5392_ = !lean_is_exclusive(v___x_5376_);
if (v_isSharedCheck_5392_ == 0)
{
v___x_5387_ = v___x_5376_;
v_isShared_5388_ = v_isSharedCheck_5392_;
goto v_resetjp_5386_;
}
else
{
lean_inc(v_a_5385_);
lean_dec(v___x_5376_);
v___x_5387_ = lean_box(0);
v_isShared_5388_ = v_isSharedCheck_5392_;
goto v_resetjp_5386_;
}
v_resetjp_5386_:
{
lean_object* v___x_5390_; 
if (v_isShared_5388_ == 0)
{
v___x_5390_ = v___x_5387_;
goto v_reusejp_5389_;
}
else
{
lean_object* v_reuseFailAlloc_5391_; 
v_reuseFailAlloc_5391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5391_, 0, v_a_5385_);
v___x_5390_ = v_reuseFailAlloc_5391_;
goto v_reusejp_5389_;
}
v_reusejp_5389_:
{
return v___x_5390_;
}
}
}
}
else
{
lean_dec_ref(v_e_5354_);
return v___x_5373_;
}
}
else
{
lean_object* v_val_5393_; lean_object* v___x_5395_; 
lean_dec_ref(v_e_5354_);
lean_dec_ref(v_post_5350_);
lean_dec_ref(v_pre_5349_);
v_val_5393_ = lean_ctor_get(v___x_5367_, 0);
lean_inc(v_val_5393_);
lean_dec_ref_known(v___x_5367_, 1);
if (v_isShared_5366_ == 0)
{
lean_ctor_set(v___x_5365_, 0, v_val_5393_);
v___x_5395_ = v___x_5365_;
goto v_reusejp_5394_;
}
else
{
lean_object* v_reuseFailAlloc_5396_; 
v_reuseFailAlloc_5396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5396_, 0, v_val_5393_);
v___x_5395_ = v_reuseFailAlloc_5396_;
goto v_reusejp_5394_;
}
v_reusejp_5394_:
{
return v___x_5395_;
}
}
}
}
else
{
lean_object* v_a_5398_; lean_object* v___x_5400_; uint8_t v_isShared_5401_; uint8_t v_isSharedCheck_5405_; 
lean_dec_ref(v_e_5354_);
lean_dec_ref(v_post_5350_);
lean_dec_ref(v_pre_5349_);
v_a_5398_ = lean_ctor_get(v___x_5362_, 0);
v_isSharedCheck_5405_ = !lean_is_exclusive(v___x_5362_);
if (v_isSharedCheck_5405_ == 0)
{
v___x_5400_ = v___x_5362_;
v_isShared_5401_ = v_isSharedCheck_5405_;
goto v_resetjp_5399_;
}
else
{
lean_inc(v_a_5398_);
lean_dec(v___x_5362_);
v___x_5400_ = lean_box(0);
v_isShared_5401_ = v_isSharedCheck_5405_;
goto v_resetjp_5399_;
}
v_resetjp_5399_:
{
lean_object* v___x_5403_; 
if (v_isShared_5401_ == 0)
{
v___x_5403_ = v___x_5400_;
goto v_reusejp_5402_;
}
else
{
lean_object* v_reuseFailAlloc_5404_; 
v_reuseFailAlloc_5404_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5404_, 0, v_a_5398_);
v___x_5403_ = v_reuseFailAlloc_5404_;
goto v_reusejp_5402_;
}
v_reusejp_5402_:
{
return v___x_5403_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0___boxed(lean_object* v_fvars_5406_, lean_object* v_pre_5407_, lean_object* v_post_5408_, lean_object* v_usedLetOnly_5409_, lean_object* v_skipConstInApp_5410_, lean_object* v_skipInstances_5411_, lean_object* v_body_5412_, lean_object* v_x_5413_, lean_object* v___y_5414_, lean_object* v___y_5415_, lean_object* v___y_5416_, lean_object* v___y_5417_, lean_object* v___y_5418_, lean_object* v___y_5419_){
_start:
{
uint8_t v_usedLetOnly_boxed_5420_; uint8_t v_skipConstInApp_boxed_5421_; uint8_t v_skipInstances_boxed_5422_; lean_object* v_res_5423_; 
v_usedLetOnly_boxed_5420_ = lean_unbox(v_usedLetOnly_5409_);
v_skipConstInApp_boxed_5421_ = lean_unbox(v_skipConstInApp_5410_);
v_skipInstances_boxed_5422_ = lean_unbox(v_skipInstances_5411_);
v_res_5423_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0(v_fvars_5406_, v_pre_5407_, v_post_5408_, v_usedLetOnly_boxed_5420_, v_skipConstInApp_boxed_5421_, v_skipInstances_boxed_5422_, v_body_5412_, v_x_5413_, v___y_5414_, v___y_5415_, v___y_5416_, v___y_5417_, v___y_5418_);
lean_dec(v___y_5418_);
lean_dec_ref(v___y_5417_);
lean_dec(v___y_5416_);
lean_dec_ref(v___y_5415_);
lean_dec(v___y_5414_);
return v_res_5423_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6(lean_object* v_pre_5424_, lean_object* v_post_5425_, uint8_t v_usedLetOnly_5426_, uint8_t v_skipConstInApp_5427_, uint8_t v_skipInstances_5428_, lean_object* v_fvars_5429_, lean_object* v_e_5430_, lean_object* v_a_5431_, lean_object* v___y_5432_, lean_object* v___y_5433_, lean_object* v___y_5434_, lean_object* v___y_5435_){
_start:
{
if (lean_obj_tag(v_e_5430_) == 7)
{
lean_object* v_binderName_5437_; lean_object* v_binderType_5438_; lean_object* v_body_5439_; uint8_t v_binderInfo_5440_; lean_object* v___x_5441_; lean_object* v___x_5442_; 
v_binderName_5437_ = lean_ctor_get(v_e_5430_, 0);
lean_inc(v_binderName_5437_);
v_binderType_5438_ = lean_ctor_get(v_e_5430_, 1);
lean_inc_ref(v_binderType_5438_);
v_body_5439_ = lean_ctor_get(v_e_5430_, 2);
lean_inc_ref(v_body_5439_);
v_binderInfo_5440_ = lean_ctor_get_uint8(v_e_5430_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_e_5430_, 3);
v___x_5441_ = lean_expr_instantiate_rev(v_binderType_5438_, v_fvars_5429_);
lean_dec_ref(v_binderType_5438_);
lean_inc_ref(v_post_5425_);
lean_inc_ref(v_pre_5424_);
v___x_5442_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5424_, v_post_5425_, v_usedLetOnly_5426_, v_skipConstInApp_5427_, v_skipInstances_5428_, v___x_5441_, v_a_5431_, v___y_5432_, v___y_5433_, v___y_5434_, v___y_5435_);
if (lean_obj_tag(v___x_5442_) == 0)
{
lean_object* v_a_5443_; lean_object* v___x_5444_; lean_object* v___x_5445_; lean_object* v___x_5446_; lean_object* v___f_5447_; uint8_t v___x_5448_; lean_object* v___x_5449_; 
v_a_5443_ = lean_ctor_get(v___x_5442_, 0);
lean_inc(v_a_5443_);
lean_dec_ref_known(v___x_5442_, 1);
v___x_5444_ = lean_box(v_usedLetOnly_5426_);
v___x_5445_ = lean_box(v_skipConstInApp_5427_);
v___x_5446_ = lean_box(v_skipInstances_5428_);
v___f_5447_ = lean_alloc_closure((void*)(lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0___boxed), 14, 7);
lean_closure_set(v___f_5447_, 0, v_fvars_5429_);
lean_closure_set(v___f_5447_, 1, v_pre_5424_);
lean_closure_set(v___f_5447_, 2, v_post_5425_);
lean_closure_set(v___f_5447_, 3, v___x_5444_);
lean_closure_set(v___f_5447_, 4, v___x_5445_);
lean_closure_set(v___f_5447_, 5, v___x_5446_);
lean_closure_set(v___f_5447_, 6, v_body_5439_);
v___x_5448_ = 0;
v___x_5449_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(v_binderName_5437_, v_binderInfo_5440_, v_a_5443_, v___f_5447_, v___x_5448_, v_a_5431_, v___y_5432_, v___y_5433_, v___y_5434_, v___y_5435_);
return v___x_5449_;
}
else
{
lean_dec_ref(v_body_5439_);
lean_dec(v_binderName_5437_);
lean_dec_ref(v_fvars_5429_);
lean_dec_ref(v_post_5425_);
lean_dec_ref(v_pre_5424_);
return v___x_5442_;
}
}
else
{
lean_object* v___x_5450_; lean_object* v___x_5451_; 
v___x_5450_ = lean_expr_instantiate_rev(v_e_5430_, v_fvars_5429_);
lean_dec_ref(v_e_5430_);
lean_inc_ref(v_post_5425_);
lean_inc_ref(v_pre_5424_);
v___x_5451_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5424_, v_post_5425_, v_usedLetOnly_5426_, v_skipConstInApp_5427_, v_skipInstances_5428_, v___x_5450_, v_a_5431_, v___y_5432_, v___y_5433_, v___y_5434_, v___y_5435_);
if (lean_obj_tag(v___x_5451_) == 0)
{
lean_object* v_a_5452_; uint8_t v___x_5453_; uint8_t v___x_5454_; uint8_t v___x_5455_; lean_object* v___x_5456_; 
v_a_5452_ = lean_ctor_get(v___x_5451_, 0);
lean_inc(v_a_5452_);
lean_dec_ref_known(v___x_5451_, 1);
v___x_5453_ = 0;
v___x_5454_ = 1;
v___x_5455_ = 1;
v___x_5456_ = l_Lean_Meta_mkForallFVars(v_fvars_5429_, v_a_5452_, v___x_5453_, v_usedLetOnly_5426_, v___x_5454_, v___x_5455_, v___y_5432_, v___y_5433_, v___y_5434_, v___y_5435_);
lean_dec_ref(v_fvars_5429_);
if (lean_obj_tag(v___x_5456_) == 0)
{
lean_object* v_a_5457_; lean_object* v___x_5458_; 
v_a_5457_ = lean_ctor_get(v___x_5456_, 0);
lean_inc(v_a_5457_);
lean_dec_ref_known(v___x_5456_, 1);
v___x_5458_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5424_, v_post_5425_, v_usedLetOnly_5426_, v_skipConstInApp_5427_, v_skipInstances_5428_, v_a_5457_, v_a_5431_, v___y_5432_, v___y_5433_, v___y_5434_, v___y_5435_);
return v___x_5458_;
}
else
{
lean_dec_ref(v_post_5425_);
lean_dec_ref(v_pre_5424_);
return v___x_5456_;
}
}
else
{
lean_dec_ref(v_fvars_5429_);
lean_dec_ref(v_post_5425_);
lean_dec_ref(v_pre_5424_);
return v___x_5451_;
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___lam__0(lean_object* v_fvars_5459_, lean_object* v_pre_5460_, lean_object* v_post_5461_, uint8_t v_usedLetOnly_5462_, uint8_t v_skipConstInApp_5463_, uint8_t v_skipInstances_5464_, lean_object* v_body_5465_, lean_object* v_x_5466_, lean_object* v___y_5467_, lean_object* v___y_5468_, lean_object* v___y_5469_, lean_object* v___y_5470_, lean_object* v___y_5471_){
_start:
{
lean_object* v___x_5473_; lean_object* v___x_5474_; 
v___x_5473_ = lean_array_push(v_fvars_5459_, v_x_5466_);
v___x_5474_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6(v_pre_5460_, v_post_5461_, v_usedLetOnly_5462_, v_skipConstInApp_5463_, v_skipInstances_5464_, v___x_5473_, v_body_5465_, v___y_5467_, v___y_5468_, v___y_5469_, v___y_5470_, v___y_5471_);
return v___x_5474_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3___boxed(lean_object* v_pre_5475_, lean_object* v_post_5476_, lean_object* v_usedLetOnly_5477_, lean_object* v_skipConstInApp_5478_, lean_object* v_skipInstances_5479_, lean_object* v_e_5480_, lean_object* v_a_5481_, lean_object* v___y_5482_, lean_object* v___y_5483_, lean_object* v___y_5484_, lean_object* v___y_5485_, lean_object* v___y_5486_){
_start:
{
uint8_t v_usedLetOnly_boxed_5487_; uint8_t v_skipConstInApp_boxed_5488_; uint8_t v_skipInstances_boxed_5489_; lean_object* v_res_5490_; 
v_usedLetOnly_boxed_5487_ = lean_unbox(v_usedLetOnly_5477_);
v_skipConstInApp_boxed_5488_ = lean_unbox(v_skipConstInApp_5478_);
v_skipInstances_boxed_5489_ = lean_unbox(v_skipInstances_5479_);
v_res_5490_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitPost___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__3(v_pre_5475_, v_post_5476_, v_usedLetOnly_boxed_5487_, v_skipConstInApp_boxed_5488_, v_skipInstances_boxed_5489_, v_e_5480_, v_a_5481_, v___y_5482_, v___y_5483_, v___y_5484_, v___y_5485_);
lean_dec(v___y_5485_);
lean_dec_ref(v___y_5484_);
lean_dec(v___y_5483_);
lean_dec_ref(v___y_5482_);
lean_dec(v_a_5481_);
return v_res_5490_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2___boxed(lean_object* v_pre_5491_, lean_object* v_post_5492_, lean_object* v_usedLetOnly_5493_, lean_object* v_skipConstInApp_5494_, lean_object* v_skipInstances_5495_, lean_object* v_sz_5496_, lean_object* v_i_5497_, lean_object* v_bs_5498_, lean_object* v___y_5499_, lean_object* v___y_5500_, lean_object* v___y_5501_, lean_object* v___y_5502_, lean_object* v___y_5503_, lean_object* v___y_5504_){
_start:
{
uint8_t v_usedLetOnly_boxed_5505_; uint8_t v_skipConstInApp_boxed_5506_; uint8_t v_skipInstances_boxed_5507_; size_t v_sz_boxed_5508_; size_t v_i_boxed_5509_; lean_object* v_res_5510_; 
v_usedLetOnly_boxed_5505_ = lean_unbox(v_usedLetOnly_5493_);
v_skipConstInApp_boxed_5506_ = lean_unbox(v_skipConstInApp_5494_);
v_skipInstances_boxed_5507_ = lean_unbox(v_skipInstances_5495_);
v_sz_boxed_5508_ = lean_unbox_usize(v_sz_5496_);
lean_dec(v_sz_5496_);
v_i_boxed_5509_ = lean_unbox_usize(v_i_5497_);
lean_dec(v_i_5497_);
v_res_5510_ = lp_plausible___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__2(v_pre_5491_, v_post_5492_, v_usedLetOnly_boxed_5505_, v_skipConstInApp_boxed_5506_, v_skipInstances_boxed_5507_, v_sz_boxed_5508_, v_i_boxed_5509_, v_bs_5498_, v___y_5499_, v___y_5500_, v___y_5501_, v___y_5502_, v___y_5503_);
lean_dec(v___y_5503_);
lean_dec_ref(v___y_5502_);
lean_dec(v___y_5501_);
lean_dec_ref(v___y_5500_);
lean_dec(v___y_5499_);
return v_res_5510_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1___boxed(lean_object* v_pre_5511_, lean_object* v_post_5512_, lean_object* v_usedLetOnly_5513_, lean_object* v_skipConstInApp_5514_, lean_object* v_skipInstances_5515_, lean_object* v_e_5516_, lean_object* v_a_5517_, lean_object* v___y_5518_, lean_object* v___y_5519_, lean_object* v___y_5520_, lean_object* v___y_5521_, lean_object* v___y_5522_){
_start:
{
uint8_t v_usedLetOnly_boxed_5523_; uint8_t v_skipConstInApp_boxed_5524_; uint8_t v_skipInstances_boxed_5525_; lean_object* v_res_5526_; 
v_usedLetOnly_boxed_5523_ = lean_unbox(v_usedLetOnly_5513_);
v_skipConstInApp_boxed_5524_ = lean_unbox(v_skipConstInApp_5514_);
v_skipInstances_boxed_5525_ = lean_unbox(v_skipInstances_5515_);
v_res_5526_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5511_, v_post_5512_, v_usedLetOnly_boxed_5523_, v_skipConstInApp_boxed_5524_, v_skipInstances_boxed_5525_, v_e_5516_, v_a_5517_, v___y_5518_, v___y_5519_, v___y_5520_, v___y_5521_);
lean_dec(v___y_5521_);
lean_dec_ref(v___y_5520_);
lean_dec(v___y_5519_);
lean_dec_ref(v___y_5518_);
lean_dec(v_a_5517_);
return v_res_5526_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6___boxed(lean_object* v_pre_5527_, lean_object* v_post_5528_, lean_object* v_usedLetOnly_5529_, lean_object* v_skipConstInApp_5530_, lean_object* v_skipInstances_5531_, lean_object* v_fvars_5532_, lean_object* v_e_5533_, lean_object* v_a_5534_, lean_object* v___y_5535_, lean_object* v___y_5536_, lean_object* v___y_5537_, lean_object* v___y_5538_, lean_object* v___y_5539_){
_start:
{
uint8_t v_usedLetOnly_boxed_5540_; uint8_t v_skipConstInApp_boxed_5541_; uint8_t v_skipInstances_boxed_5542_; lean_object* v_res_5543_; 
v_usedLetOnly_boxed_5540_ = lean_unbox(v_usedLetOnly_5529_);
v_skipConstInApp_boxed_5541_ = lean_unbox(v_skipConstInApp_5530_);
v_skipInstances_boxed_5542_ = lean_unbox(v_skipInstances_5531_);
v_res_5543_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6(v_pre_5527_, v_post_5528_, v_usedLetOnly_boxed_5540_, v_skipConstInApp_boxed_5541_, v_skipInstances_boxed_5542_, v_fvars_5532_, v_e_5533_, v_a_5534_, v___y_5535_, v___y_5536_, v___y_5537_, v___y_5538_);
lean_dec(v___y_5538_);
lean_dec_ref(v___y_5537_);
lean_dec(v___y_5536_);
lean_dec_ref(v___y_5535_);
lean_dec(v_a_5534_);
return v_res_5543_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7___boxed(lean_object* v_pre_5544_, lean_object* v_post_5545_, lean_object* v_usedLetOnly_5546_, lean_object* v_skipConstInApp_5547_, lean_object* v_skipInstances_5548_, lean_object* v_fvars_5549_, lean_object* v_e_5550_, lean_object* v_a_5551_, lean_object* v___y_5552_, lean_object* v___y_5553_, lean_object* v___y_5554_, lean_object* v___y_5555_, lean_object* v___y_5556_){
_start:
{
uint8_t v_usedLetOnly_boxed_5557_; uint8_t v_skipConstInApp_boxed_5558_; uint8_t v_skipInstances_boxed_5559_; lean_object* v_res_5560_; 
v_usedLetOnly_boxed_5557_ = lean_unbox(v_usedLetOnly_5546_);
v_skipConstInApp_boxed_5558_ = lean_unbox(v_skipConstInApp_5547_);
v_skipInstances_boxed_5559_ = lean_unbox(v_skipInstances_5548_);
v_res_5560_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLambda___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__7(v_pre_5544_, v_post_5545_, v_usedLetOnly_boxed_5557_, v_skipConstInApp_boxed_5558_, v_skipInstances_boxed_5559_, v_fvars_5549_, v_e_5550_, v_a_5551_, v___y_5552_, v___y_5553_, v___y_5554_, v___y_5555_);
lean_dec(v___y_5555_);
lean_dec_ref(v___y_5554_);
lean_dec(v___y_5553_);
lean_dec_ref(v___y_5552_);
lean_dec(v_a_5551_);
return v_res_5560_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8___boxed(lean_object* v_pre_5561_, lean_object* v_post_5562_, lean_object* v_usedLetOnly_5563_, lean_object* v_skipConstInApp_5564_, lean_object* v_skipInstances_5565_, lean_object* v_fvars_5566_, lean_object* v_e_5567_, lean_object* v_a_5568_, lean_object* v___y_5569_, lean_object* v___y_5570_, lean_object* v___y_5571_, lean_object* v___y_5572_, lean_object* v___y_5573_){
_start:
{
uint8_t v_usedLetOnly_boxed_5574_; uint8_t v_skipConstInApp_boxed_5575_; uint8_t v_skipInstances_boxed_5576_; lean_object* v_res_5577_; 
v_usedLetOnly_boxed_5574_ = lean_unbox(v_usedLetOnly_5563_);
v_skipConstInApp_boxed_5575_ = lean_unbox(v_skipConstInApp_5564_);
v_skipInstances_boxed_5576_ = lean_unbox(v_skipInstances_5565_);
v_res_5577_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8(v_pre_5561_, v_post_5562_, v_usedLetOnly_boxed_5574_, v_skipConstInApp_boxed_5575_, v_skipInstances_boxed_5576_, v_fvars_5566_, v_e_5567_, v_a_5568_, v___y_5569_, v___y_5570_, v___y_5571_, v___y_5572_);
lean_dec(v___y_5572_);
lean_dec_ref(v___y_5571_);
lean_dec(v___y_5570_);
lean_dec_ref(v___y_5569_);
lean_dec(v_a_5568_);
return v_res_5577_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg___boxed(lean_object* v_upperBound_5578_, lean_object* v___x_5579_, lean_object* v_pre_5580_, lean_object* v_post_5581_, lean_object* v_usedLetOnly_5582_, lean_object* v_skipConstInApp_5583_, lean_object* v_skipInstances_5584_, lean_object* v_a_5585_, lean_object* v_b_5586_, lean_object* v___y_5587_, lean_object* v___y_5588_, lean_object* v___y_5589_, lean_object* v___y_5590_, lean_object* v___y_5591_, lean_object* v___y_5592_){
_start:
{
uint8_t v_usedLetOnly_boxed_5593_; uint8_t v_skipConstInApp_boxed_5594_; uint8_t v_skipInstances_boxed_5595_; lean_object* v_res_5596_; 
v_usedLetOnly_boxed_5593_ = lean_unbox(v_usedLetOnly_5582_);
v_skipConstInApp_boxed_5594_ = lean_unbox(v_skipConstInApp_5583_);
v_skipInstances_boxed_5595_ = lean_unbox(v_skipInstances_5584_);
v_res_5596_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg(v_upperBound_5578_, v___x_5579_, v_pre_5580_, v_post_5581_, v_usedLetOnly_boxed_5593_, v_skipConstInApp_boxed_5594_, v_skipInstances_boxed_5595_, v_a_5585_, v_b_5586_, v___y_5587_, v___y_5588_, v___y_5589_, v___y_5590_, v___y_5591_);
lean_dec(v___y_5591_);
lean_dec_ref(v___y_5590_);
lean_dec(v___y_5589_);
lean_dec_ref(v___y_5588_);
lean_dec(v___y_5587_);
lean_dec_ref(v___x_5579_);
lean_dec(v_upperBound_5578_);
return v_res_5596_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9___boxed(lean_object* v_skipInstances_5597_, lean_object* v_pre_5598_, lean_object* v_post_5599_, lean_object* v_usedLetOnly_5600_, lean_object* v_skipConstInApp_5601_, lean_object* v_x_5602_, lean_object* v_x_5603_, lean_object* v_x_5604_, lean_object* v___y_5605_, lean_object* v___y_5606_, lean_object* v___y_5607_, lean_object* v___y_5608_, lean_object* v___y_5609_, lean_object* v___y_5610_){
_start:
{
uint8_t v_skipInstances_boxed_5611_; uint8_t v_usedLetOnly_boxed_5612_; uint8_t v_skipConstInApp_boxed_5613_; lean_object* v_res_5614_; 
v_skipInstances_boxed_5611_ = lean_unbox(v_skipInstances_5597_);
v_usedLetOnly_boxed_5612_ = lean_unbox(v_usedLetOnly_5600_);
v_skipConstInApp_boxed_5613_ = lean_unbox(v_skipConstInApp_5601_);
v_res_5614_ = lp_plausible_Lean_Expr_withAppAux___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__9(v_skipInstances_boxed_5611_, v_pre_5598_, v_post_5599_, v_usedLetOnly_boxed_5612_, v_skipConstInApp_boxed_5613_, v_x_5602_, v_x_5603_, v_x_5604_, v___y_5605_, v___y_5606_, v___y_5607_, v___y_5608_, v___y_5609_);
lean_dec(v___y_5609_);
lean_dec_ref(v___y_5608_);
lean_dec(v___y_5607_);
lean_dec_ref(v___y_5606_);
lean_dec(v___y_5605_);
return v_res_5614_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0(lean_object* v_00_u03b1_5615_, lean_object* v_x_5616_, lean_object* v___y_5617_, lean_object* v___y_5618_, lean_object* v___y_5619_, lean_object* v___y_5620_){
_start:
{
lean_object* v___x_5622_; lean_object* v___x_5623_; 
v___x_5622_ = lean_apply_1(v_x_5616_, lean_box(0));
v___x_5623_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5623_, 0, v___x_5622_);
return v___x_5623_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0___boxed(lean_object* v_00_u03b1_5624_, lean_object* v_x_5625_, lean_object* v___y_5626_, lean_object* v___y_5627_, lean_object* v___y_5628_, lean_object* v___y_5629_, lean_object* v___y_5630_){
_start:
{
lean_object* v_res_5631_; 
v_res_5631_ = lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0(v_00_u03b1_5624_, v_x_5625_, v___y_5626_, v___y_5627_, v___y_5628_, v___y_5629_);
lean_dec(v___y_5629_);
lean_dec_ref(v___y_5628_);
lean_dec(v___y_5627_);
lean_dec_ref(v___y_5626_);
return v_res_5631_;
}
}
static lean_object* _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0(void){
_start:
{
lean_object* v___x_5632_; lean_object* v___x_5633_; lean_object* v___x_5634_; 
v___x_5632_ = lean_box(0);
v___x_5633_ = lean_unsigned_to_nat(16u);
v___x_5634_ = lean_mk_array(v___x_5633_, v___x_5632_);
return v___x_5634_;
}
}
static lean_object* _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1(void){
_start:
{
lean_object* v___x_5635_; lean_object* v___x_5636_; lean_object* v___x_5637_; 
v___x_5635_ = lean_obj_once(&lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0, &lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0_once, _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__0);
v___x_5636_ = lean_unsigned_to_nat(0u);
v___x_5637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5637_, 0, v___x_5636_);
lean_ctor_set(v___x_5637_, 1, v___x_5635_);
return v___x_5637_;
}
}
static lean_object* _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2(void){
_start:
{
lean_object* v___x_5638_; lean_object* v___x_5639_; 
v___x_5638_ = lean_obj_once(&lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1, &lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1_once, _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__1);
v___x_5639_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_5639_, 0, lean_box(0));
lean_closure_set(v___x_5639_, 1, lean_box(0));
lean_closure_set(v___x_5639_, 2, v___x_5638_);
return v___x_5639_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1(lean_object* v_input_5640_, lean_object* v_pre_5641_, lean_object* v_post_5642_, uint8_t v_usedLetOnly_5643_, uint8_t v_skipConstInApp_5644_, lean_object* v___y_5645_, lean_object* v___y_5646_, lean_object* v___y_5647_, lean_object* v___y_5648_){
_start:
{
lean_object* v___x_5650_; lean_object* v___x_5651_; lean_object* v_a_5652_; uint8_t v___x_5653_; lean_object* v___x_5654_; 
v___x_5650_ = lean_obj_once(&lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2, &lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2_once, _init_lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___closed__2);
v___x_5651_ = lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0(lean_box(0), v___x_5650_, v___y_5645_, v___y_5646_, v___y_5647_, v___y_5648_);
v_a_5652_ = lean_ctor_get(v___x_5651_, 0);
lean_inc(v_a_5652_);
lean_dec_ref(v___x_5651_);
v___x_5653_ = 0;
v___x_5654_ = lp_plausible___private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1(v_pre_5641_, v_post_5642_, v_usedLetOnly_5643_, v_skipConstInApp_5644_, v___x_5653_, v_input_5640_, v_a_5652_, v___y_5645_, v___y_5646_, v___y_5647_, v___y_5648_);
if (lean_obj_tag(v___x_5654_) == 0)
{
lean_object* v_a_5655_; lean_object* v___x_5656_; lean_object* v___x_5657_; lean_object* v___x_5659_; uint8_t v_isShared_5660_; uint8_t v_isSharedCheck_5664_; 
v_a_5655_ = lean_ctor_get(v___x_5654_, 0);
lean_inc(v_a_5655_);
lean_dec_ref_known(v___x_5654_, 1);
v___x_5656_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_5656_, 0, lean_box(0));
lean_closure_set(v___x_5656_, 1, lean_box(0));
lean_closure_set(v___x_5656_, 2, v_a_5652_);
v___x_5657_ = lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___lam__0(lean_box(0), v___x_5656_, v___y_5645_, v___y_5646_, v___y_5647_, v___y_5648_);
v_isSharedCheck_5664_ = !lean_is_exclusive(v___x_5657_);
if (v_isSharedCheck_5664_ == 0)
{
lean_object* v_unused_5665_; 
v_unused_5665_ = lean_ctor_get(v___x_5657_, 0);
lean_dec(v_unused_5665_);
v___x_5659_ = v___x_5657_;
v_isShared_5660_ = v_isSharedCheck_5664_;
goto v_resetjp_5658_;
}
else
{
lean_dec(v___x_5657_);
v___x_5659_ = lean_box(0);
v_isShared_5660_ = v_isSharedCheck_5664_;
goto v_resetjp_5658_;
}
v_resetjp_5658_:
{
lean_object* v___x_5662_; 
if (v_isShared_5660_ == 0)
{
lean_ctor_set(v___x_5659_, 0, v_a_5655_);
v___x_5662_ = v___x_5659_;
goto v_reusejp_5661_;
}
else
{
lean_object* v_reuseFailAlloc_5663_; 
v_reuseFailAlloc_5663_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5663_, 0, v_a_5655_);
v___x_5662_ = v_reuseFailAlloc_5663_;
goto v_reusejp_5661_;
}
v_reusejp_5661_:
{
return v___x_5662_;
}
}
}
else
{
lean_dec(v_a_5652_);
return v___x_5654_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1___boxed(lean_object* v_input_5666_, lean_object* v_pre_5667_, lean_object* v_post_5668_, lean_object* v_usedLetOnly_5669_, lean_object* v_skipConstInApp_5670_, lean_object* v___y_5671_, lean_object* v___y_5672_, lean_object* v___y_5673_, lean_object* v___y_5674_, lean_object* v___y_5675_){
_start:
{
uint8_t v_usedLetOnly_boxed_5676_; uint8_t v_skipConstInApp_boxed_5677_; lean_object* v_res_5678_; 
v_usedLetOnly_boxed_5676_ = lean_unbox(v_usedLetOnly_5669_);
v_skipConstInApp_boxed_5677_ = lean_unbox(v_skipConstInApp_5670_);
v_res_5678_ = lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1(v_input_5666_, v_pre_5667_, v_post_5668_, v_usedLetOnly_boxed_5676_, v_skipConstInApp_boxed_5677_, v___y_5671_, v___y_5672_, v___y_5673_, v___y_5674_);
lean_dec(v___y_5674_);
lean_dec_ref(v___y_5673_);
lean_dec(v___y_5672_);
lean_dec_ref(v___y_5671_);
return v_res_5678_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__1(lean_object* v_body_5680_, lean_object* v_fvar_5681_, lean_object* v___y_5682_, lean_object* v___y_5683_, lean_object* v___y_5684_, lean_object* v___y_5685_){
_start:
{
lean_object* v___x_5687_; lean_object* v___x_5688_; 
v___x_5687_ = lean_expr_instantiate1(v_body_5680_, v_fvar_5681_);
v___x_5688_ = lp_plausible_Plausible_Decorations_addDecorations(v___x_5687_, v___y_5682_, v___y_5683_, v___y_5684_, v___y_5685_);
if (lean_obj_tag(v___x_5688_) == 0)
{
lean_object* v_a_5689_; lean_object* v___x_5691_; uint8_t v_isShared_5692_; uint8_t v_isSharedCheck_5700_; 
v_a_5689_ = lean_ctor_get(v___x_5688_, 0);
v_isSharedCheck_5700_ = !lean_is_exclusive(v___x_5688_);
if (v_isSharedCheck_5700_ == 0)
{
v___x_5691_ = v___x_5688_;
v_isShared_5692_ = v_isSharedCheck_5700_;
goto v_resetjp_5690_;
}
else
{
lean_inc(v_a_5689_);
lean_dec(v___x_5688_);
v___x_5691_ = lean_box(0);
v_isShared_5692_ = v_isSharedCheck_5700_;
goto v_resetjp_5690_;
}
v_resetjp_5690_:
{
lean_object* v___x_5693_; lean_object* v___x_5694_; lean_object* v___x_5695_; lean_object* v___x_5696_; lean_object* v___x_5698_; 
v___x_5693_ = lean_unsigned_to_nat(1u);
v___x_5694_ = lean_mk_empty_array_with_capacity(v___x_5693_);
v___x_5695_ = lean_array_push(v___x_5694_, v_fvar_5681_);
v___x_5696_ = lean_expr_abstract(v_a_5689_, v___x_5695_);
lean_dec_ref(v___x_5695_);
lean_dec(v_a_5689_);
if (v_isShared_5692_ == 0)
{
lean_ctor_set(v___x_5691_, 0, v___x_5696_);
v___x_5698_ = v___x_5691_;
goto v_reusejp_5697_;
}
else
{
lean_object* v_reuseFailAlloc_5699_; 
v_reuseFailAlloc_5699_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5699_, 0, v___x_5696_);
v___x_5698_ = v_reuseFailAlloc_5699_;
goto v_reusejp_5697_;
}
v_reusejp_5697_:
{
return v___x_5698_;
}
}
}
else
{
lean_dec_ref(v_fvar_5681_);
return v___x_5688_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__1___boxed(lean_object* v_body_5701_, lean_object* v_fvar_5702_, lean_object* v___y_5703_, lean_object* v___y_5704_, lean_object* v___y_5705_, lean_object* v___y_5706_, lean_object* v___y_5707_){
_start:
{
lean_object* v_res_5708_; 
v_res_5708_ = lp_plausible_Plausible_Decorations_addDecorations___lam__1(v_body_5701_, v_fvar_5702_, v___y_5703_, v___y_5704_, v___y_5705_, v___y_5706_);
lean_dec(v___y_5706_);
lean_dec_ref(v___y_5705_);
lean_dec(v___y_5704_);
lean_dec_ref(v___y_5703_);
lean_dec_ref(v_body_5701_);
return v_res_5708_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2(lean_object* v_expr_5715_, lean_object* v___y_5716_, lean_object* v___y_5717_, lean_object* v___y_5718_, lean_object* v___y_5719_){
_start:
{
lean_object* v___x_5721_; 
lean_inc(v___y_5719_);
lean_inc_ref(v___y_5718_);
lean_inc(v___y_5717_);
lean_inc_ref(v___y_5716_);
lean_inc_ref(v_expr_5715_);
v___x_5721_ = lean_infer_type(v_expr_5715_, v___y_5716_, v___y_5717_, v___y_5718_, v___y_5719_);
if (lean_obj_tag(v___x_5721_) == 0)
{
lean_object* v_a_5722_; lean_object* v___x_5724_; uint8_t v_isShared_5725_; uint8_t v_isSharedCheck_5787_; 
v_a_5722_ = lean_ctor_get(v___x_5721_, 0);
v_isSharedCheck_5787_ = !lean_is_exclusive(v___x_5721_);
if (v_isSharedCheck_5787_ == 0)
{
v___x_5724_ = v___x_5721_;
v_isShared_5725_ = v_isSharedCheck_5787_;
goto v_resetjp_5723_;
}
else
{
lean_inc(v_a_5722_);
lean_dec(v___x_5721_);
v___x_5724_ = lean_box(0);
v_isShared_5725_ = v_isSharedCheck_5787_;
goto v_resetjp_5723_;
}
v_resetjp_5723_:
{
uint8_t v___x_5726_; 
v___x_5726_ = l_Lean_Expr_isProp(v_a_5722_);
lean_dec(v_a_5722_);
if (v___x_5726_ == 0)
{
lean_object* v___x_5727_; lean_object* v___x_5729_; 
v___x_5727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5727_, 0, v_expr_5715_);
if (v_isShared_5725_ == 0)
{
lean_ctor_set(v___x_5724_, 0, v___x_5727_);
v___x_5729_ = v___x_5724_;
goto v_reusejp_5728_;
}
else
{
lean_object* v_reuseFailAlloc_5730_; 
v_reuseFailAlloc_5730_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5730_, 0, v___x_5727_);
v___x_5729_ = v_reuseFailAlloc_5730_;
goto v_reusejp_5728_;
}
v_reusejp_5728_:
{
return v___x_5729_;
}
}
else
{
if (lean_obj_tag(v_expr_5715_) == 7)
{
lean_object* v_binderName_5731_; lean_object* v_binderType_5732_; lean_object* v_body_5733_; uint8_t v_binderInfo_5734_; lean_object* v___x_5735_; 
lean_del_object(v___x_5724_);
v_binderName_5731_ = lean_ctor_get(v_expr_5715_, 0);
lean_inc(v_binderName_5731_);
v_binderType_5732_ = lean_ctor_get(v_expr_5715_, 1);
lean_inc_ref_n(v_binderType_5732_, 2);
v_body_5733_ = lean_ctor_get(v_expr_5715_, 2);
lean_inc_ref(v_body_5733_);
v_binderInfo_5734_ = lean_ctor_get_uint8(v_expr_5715_, sizeof(void*)*3 + 8);
lean_dec_ref_known(v_expr_5715_, 3);
v___x_5735_ = lp_plausible_Plausible_Decorations_addDecorations(v_binderType_5732_, v___y_5716_, v___y_5717_, v___y_5718_, v___y_5719_);
if (lean_obj_tag(v___x_5735_) == 0)
{
lean_object* v_a_5736_; lean_object* v___f_5737_; uint8_t v___x_5738_; lean_object* v___x_5739_; 
v_a_5736_ = lean_ctor_get(v___x_5735_, 0);
lean_inc(v_a_5736_);
lean_dec_ref_known(v___x_5735_, 1);
v___f_5737_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Decorations_addDecorations___lam__1___boxed), 7, 1);
lean_closure_set(v___f_5737_, 0, v_body_5733_);
v___x_5738_ = 0;
lean_inc(v_binderName_5731_);
v___x_5739_ = lp_plausible_Lean_Meta_withLocalDecl___at___00Plausible_Decorations_addDecorations_spec__0___redArg(v_binderName_5731_, v_binderInfo_5734_, v_binderType_5732_, v___f_5737_, v___x_5738_, v___y_5716_, v___y_5717_, v___y_5718_, v___y_5719_);
if (lean_obj_tag(v___x_5739_) == 0)
{
lean_object* v_a_5740_; lean_object* v___x_5741_; lean_object* v___x_5742_; lean_object* v___x_5743_; lean_object* v___x_5744_; lean_object* v___x_5745_; lean_object* v___x_5746_; lean_object* v___x_5747_; lean_object* v___x_5748_; lean_object* v___x_5749_; 
v_a_5740_ = lean_ctor_get(v___x_5739_, 0);
lean_inc(v_a_5740_);
lean_dec_ref_known(v___x_5739_, 1);
lean_inc(v_binderName_5731_);
v___x_5741_ = l_Lean_Expr_forallE___override(v_binderName_5731_, v_a_5736_, v_a_5740_, v_binderInfo_5734_);
v___x_5742_ = ((lean_object*)(lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__1));
v___x_5743_ = l_Lean_Name_toString(v_binderName_5731_, v___x_5726_);
v___x_5744_ = l_Lean_mkStrLit(v___x_5743_);
v___x_5745_ = lean_unsigned_to_nat(2u);
v___x_5746_ = lean_mk_empty_array_with_capacity(v___x_5745_);
v___x_5747_ = lean_array_push(v___x_5746_, v___x_5744_);
v___x_5748_ = lean_array_push(v___x_5747_, v___x_5741_);
v___x_5749_ = l_Lean_Meta_mkAppM(v___x_5742_, v___x_5748_, v___y_5716_, v___y_5717_, v___y_5718_, v___y_5719_);
if (lean_obj_tag(v___x_5749_) == 0)
{
lean_object* v_a_5750_; lean_object* v___x_5752_; uint8_t v_isShared_5753_; uint8_t v_isSharedCheck_5758_; 
v_a_5750_ = lean_ctor_get(v___x_5749_, 0);
v_isSharedCheck_5758_ = !lean_is_exclusive(v___x_5749_);
if (v_isSharedCheck_5758_ == 0)
{
v___x_5752_ = v___x_5749_;
v_isShared_5753_ = v_isSharedCheck_5758_;
goto v_resetjp_5751_;
}
else
{
lean_inc(v_a_5750_);
lean_dec(v___x_5749_);
v___x_5752_ = lean_box(0);
v_isShared_5753_ = v_isSharedCheck_5758_;
goto v_resetjp_5751_;
}
v_resetjp_5751_:
{
lean_object* v___x_5754_; lean_object* v___x_5756_; 
v___x_5754_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5754_, 0, v_a_5750_);
if (v_isShared_5753_ == 0)
{
lean_ctor_set(v___x_5752_, 0, v___x_5754_);
v___x_5756_ = v___x_5752_;
goto v_reusejp_5755_;
}
else
{
lean_object* v_reuseFailAlloc_5757_; 
v_reuseFailAlloc_5757_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5757_, 0, v___x_5754_);
v___x_5756_ = v_reuseFailAlloc_5757_;
goto v_reusejp_5755_;
}
v_reusejp_5755_:
{
return v___x_5756_;
}
}
}
else
{
lean_object* v_a_5759_; lean_object* v___x_5761_; uint8_t v_isShared_5762_; uint8_t v_isSharedCheck_5766_; 
v_a_5759_ = lean_ctor_get(v___x_5749_, 0);
v_isSharedCheck_5766_ = !lean_is_exclusive(v___x_5749_);
if (v_isSharedCheck_5766_ == 0)
{
v___x_5761_ = v___x_5749_;
v_isShared_5762_ = v_isSharedCheck_5766_;
goto v_resetjp_5760_;
}
else
{
lean_inc(v_a_5759_);
lean_dec(v___x_5749_);
v___x_5761_ = lean_box(0);
v_isShared_5762_ = v_isSharedCheck_5766_;
goto v_resetjp_5760_;
}
v_resetjp_5760_:
{
lean_object* v___x_5764_; 
if (v_isShared_5762_ == 0)
{
v___x_5764_ = v___x_5761_;
goto v_reusejp_5763_;
}
else
{
lean_object* v_reuseFailAlloc_5765_; 
v_reuseFailAlloc_5765_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5765_, 0, v_a_5759_);
v___x_5764_ = v_reuseFailAlloc_5765_;
goto v_reusejp_5763_;
}
v_reusejp_5763_:
{
return v___x_5764_;
}
}
}
}
else
{
lean_object* v_a_5767_; lean_object* v___x_5769_; uint8_t v_isShared_5770_; uint8_t v_isSharedCheck_5774_; 
lean_dec(v_a_5736_);
lean_dec(v_binderName_5731_);
v_a_5767_ = lean_ctor_get(v___x_5739_, 0);
v_isSharedCheck_5774_ = !lean_is_exclusive(v___x_5739_);
if (v_isSharedCheck_5774_ == 0)
{
v___x_5769_ = v___x_5739_;
v_isShared_5770_ = v_isSharedCheck_5774_;
goto v_resetjp_5768_;
}
else
{
lean_inc(v_a_5767_);
lean_dec(v___x_5739_);
v___x_5769_ = lean_box(0);
v_isShared_5770_ = v_isSharedCheck_5774_;
goto v_resetjp_5768_;
}
v_resetjp_5768_:
{
lean_object* v___x_5772_; 
if (v_isShared_5770_ == 0)
{
v___x_5772_ = v___x_5769_;
goto v_reusejp_5771_;
}
else
{
lean_object* v_reuseFailAlloc_5773_; 
v_reuseFailAlloc_5773_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5773_, 0, v_a_5767_);
v___x_5772_ = v_reuseFailAlloc_5773_;
goto v_reusejp_5771_;
}
v_reusejp_5771_:
{
return v___x_5772_;
}
}
}
}
else
{
lean_object* v_a_5775_; lean_object* v___x_5777_; uint8_t v_isShared_5778_; uint8_t v_isSharedCheck_5782_; 
lean_dec_ref(v_body_5733_);
lean_dec_ref(v_binderType_5732_);
lean_dec(v_binderName_5731_);
v_a_5775_ = lean_ctor_get(v___x_5735_, 0);
v_isSharedCheck_5782_ = !lean_is_exclusive(v___x_5735_);
if (v_isSharedCheck_5782_ == 0)
{
v___x_5777_ = v___x_5735_;
v_isShared_5778_ = v_isSharedCheck_5782_;
goto v_resetjp_5776_;
}
else
{
lean_inc(v_a_5775_);
lean_dec(v___x_5735_);
v___x_5777_ = lean_box(0);
v_isShared_5778_ = v_isSharedCheck_5782_;
goto v_resetjp_5776_;
}
v_resetjp_5776_:
{
lean_object* v___x_5780_; 
if (v_isShared_5778_ == 0)
{
v___x_5780_ = v___x_5777_;
goto v_reusejp_5779_;
}
else
{
lean_object* v_reuseFailAlloc_5781_; 
v_reuseFailAlloc_5781_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5781_, 0, v_a_5775_);
v___x_5780_ = v_reuseFailAlloc_5781_;
goto v_reusejp_5779_;
}
v_reusejp_5779_:
{
return v___x_5780_;
}
}
}
}
else
{
lean_object* v___x_5783_; lean_object* v___x_5785_; 
lean_dec_ref(v_expr_5715_);
v___x_5783_ = ((lean_object*)(lp_plausible_Plausible_Decorations_addDecorations___lam__2___closed__2));
if (v_isShared_5725_ == 0)
{
lean_ctor_set(v___x_5724_, 0, v___x_5783_);
v___x_5785_ = v___x_5724_;
goto v_reusejp_5784_;
}
else
{
lean_object* v_reuseFailAlloc_5786_; 
v_reuseFailAlloc_5786_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5786_, 0, v___x_5783_);
v___x_5785_ = v_reuseFailAlloc_5786_;
goto v_reusejp_5784_;
}
v_reusejp_5784_:
{
return v___x_5785_;
}
}
}
}
}
else
{
lean_object* v_a_5788_; lean_object* v___x_5790_; uint8_t v_isShared_5791_; uint8_t v_isSharedCheck_5795_; 
lean_dec_ref(v_expr_5715_);
v_a_5788_ = lean_ctor_get(v___x_5721_, 0);
v_isSharedCheck_5795_ = !lean_is_exclusive(v___x_5721_);
if (v_isSharedCheck_5795_ == 0)
{
v___x_5790_ = v___x_5721_;
v_isShared_5791_ = v_isSharedCheck_5795_;
goto v_resetjp_5789_;
}
else
{
lean_inc(v_a_5788_);
lean_dec(v___x_5721_);
v___x_5790_ = lean_box(0);
v_isShared_5791_ = v_isSharedCheck_5795_;
goto v_resetjp_5789_;
}
v_resetjp_5789_:
{
lean_object* v___x_5793_; 
if (v_isShared_5791_ == 0)
{
v___x_5793_ = v___x_5790_;
goto v_reusejp_5792_;
}
else
{
lean_object* v_reuseFailAlloc_5794_; 
v_reuseFailAlloc_5794_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5794_, 0, v_a_5788_);
v___x_5793_ = v_reuseFailAlloc_5794_;
goto v_reusejp_5792_;
}
v_reusejp_5792_:
{
return v___x_5793_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___lam__2___boxed(lean_object* v_expr_5796_, lean_object* v___y_5797_, lean_object* v___y_5798_, lean_object* v___y_5799_, lean_object* v___y_5800_, lean_object* v___y_5801_){
_start:
{
lean_object* v_res_5802_; 
v_res_5802_ = lp_plausible_Plausible_Decorations_addDecorations___lam__2(v_expr_5796_, v___y_5797_, v___y_5798_, v___y_5799_, v___y_5800_);
lean_dec(v___y_5800_);
lean_dec_ref(v___y_5799_);
lean_dec(v___y_5798_);
lean_dec_ref(v___y_5797_);
return v_res_5802_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations(lean_object* v_e_5803_, lean_object* v_a_5804_, lean_object* v_a_5805_, lean_object* v_a_5806_, lean_object* v_a_5807_){
_start:
{
lean_object* v___f_5809_; lean_object* v___f_5810_; uint8_t v___x_5811_; lean_object* v___x_5812_; 
v___f_5809_ = ((lean_object*)(lp_plausible_Plausible_Decorations_addDecorations___closed__0));
v___f_5810_ = lean_alloc_closure((void*)(lp_plausible_Plausible_Decorations_addDecorations___lam__2___boxed), 6, 0);
v___x_5811_ = 0;
v___x_5812_ = lp_plausible_Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1(v_e_5803_, v___f_5810_, v___f_5809_, v___x_5811_, v___x_5811_, v_a_5804_, v_a_5805_, v_a_5806_, v_a_5807_);
return v___x_5812_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations_addDecorations___boxed(lean_object* v_e_5813_, lean_object* v_a_5814_, lean_object* v_a_5815_, lean_object* v_a_5816_, lean_object* v_a_5817_, lean_object* v_a_5818_){
_start:
{
lean_object* v_res_5819_; 
v_res_5819_ = lp_plausible_Plausible_Decorations_addDecorations(v_e_5813_, v_a_5814_, v_a_5815_, v_a_5816_, v_a_5817_);
lean_dec(v_a_5817_);
lean_dec_ref(v_a_5816_);
lean_dec(v_a_5815_);
lean_dec_ref(v_a_5814_);
return v_res_5819_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4(lean_object* v_upperBound_5820_, lean_object* v___x_5821_, lean_object* v_pre_5822_, lean_object* v_post_5823_, uint8_t v_usedLetOnly_5824_, uint8_t v_skipConstInApp_5825_, uint8_t v_skipInstances_5826_, lean_object* v___x_5827_, lean_object* v_inst_5828_, lean_object* v_R_5829_, lean_object* v_a_5830_, lean_object* v_b_5831_, lean_object* v_c_5832_, lean_object* v___y_5833_, lean_object* v___y_5834_, lean_object* v___y_5835_, lean_object* v___y_5836_, lean_object* v___y_5837_){
_start:
{
lean_object* v___x_5839_; 
v___x_5839_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___redArg(v_upperBound_5820_, v___x_5821_, v_pre_5822_, v_post_5823_, v_usedLetOnly_5824_, v_skipConstInApp_5825_, v_skipInstances_5826_, v_a_5830_, v_b_5831_, v___y_5833_, v___y_5834_, v___y_5835_, v___y_5836_, v___y_5837_);
return v___x_5839_;
}
}
LEAN_EXPORT lean_object* lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4___boxed(lean_object** _args){
lean_object* v_upperBound_5840_ = _args[0];
lean_object* v___x_5841_ = _args[1];
lean_object* v_pre_5842_ = _args[2];
lean_object* v_post_5843_ = _args[3];
lean_object* v_usedLetOnly_5844_ = _args[4];
lean_object* v_skipConstInApp_5845_ = _args[5];
lean_object* v_skipInstances_5846_ = _args[6];
lean_object* v___x_5847_ = _args[7];
lean_object* v_inst_5848_ = _args[8];
lean_object* v_R_5849_ = _args[9];
lean_object* v_a_5850_ = _args[10];
lean_object* v_b_5851_ = _args[11];
lean_object* v_c_5852_ = _args[12];
lean_object* v___y_5853_ = _args[13];
lean_object* v___y_5854_ = _args[14];
lean_object* v___y_5855_ = _args[15];
lean_object* v___y_5856_ = _args[16];
lean_object* v___y_5857_ = _args[17];
lean_object* v___y_5858_ = _args[18];
_start:
{
uint8_t v_usedLetOnly_boxed_5859_; uint8_t v_skipConstInApp_boxed_5860_; uint8_t v_skipInstances_boxed_5861_; lean_object* v_res_5862_; 
v_usedLetOnly_boxed_5859_ = lean_unbox(v_usedLetOnly_5844_);
v_skipConstInApp_boxed_5860_ = lean_unbox(v_skipConstInApp_5845_);
v_skipInstances_boxed_5861_ = lean_unbox(v_skipInstances_5846_);
v_res_5862_ = lp_plausible_WellFounded_opaqueFix_u2083___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__4(v_upperBound_5840_, v___x_5841_, v_pre_5842_, v_post_5843_, v_usedLetOnly_boxed_5859_, v_skipConstInApp_boxed_5860_, v_skipInstances_boxed_5861_, v___x_5847_, v_inst_5848_, v_R_5849_, v_a_5850_, v_b_5851_, v_c_5852_, v___y_5853_, v___y_5854_, v___y_5855_, v___y_5856_, v___y_5857_);
lean_dec(v___y_5857_);
lean_dec_ref(v___y_5856_);
lean_dec(v___y_5855_);
lean_dec_ref(v___y_5854_);
lean_dec(v___y_5853_);
lean_dec(v___x_5847_);
lean_dec_ref(v___x_5841_);
lean_dec(v_upperBound_5840_);
return v_res_5862_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5(lean_object* v_00_u03b2_5863_, lean_object* v_m_5864_, lean_object* v_a_5865_){
_start:
{
lean_object* v___x_5866_; 
v___x_5866_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___redArg(v_m_5864_, v_a_5865_);
return v___x_5866_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5___boxed(lean_object* v_00_u03b2_5867_, lean_object* v_m_5868_, lean_object* v_a_5869_){
_start:
{
lean_object* v_res_5870_; 
v_res_5870_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5(v_00_u03b2_5867_, v_m_5868_, v_a_5869_);
lean_dec_ref(v_a_5869_);
lean_dec_ref(v_m_5868_);
return v_res_5870_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8(lean_object* v_00_u03b1_5871_, lean_object* v_name_5872_, uint8_t v_bi_5873_, lean_object* v_type_5874_, lean_object* v_k_5875_, uint8_t v_kind_5876_, lean_object* v___y_5877_, lean_object* v___y_5878_, lean_object* v___y_5879_, lean_object* v___y_5880_, lean_object* v___y_5881_){
_start:
{
lean_object* v___x_5883_; 
v___x_5883_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___redArg(v_name_5872_, v_bi_5873_, v_type_5874_, v_k_5875_, v_kind_5876_, v___y_5877_, v___y_5878_, v___y_5879_, v___y_5880_, v___y_5881_);
return v___x_5883_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8___boxed(lean_object* v_00_u03b1_5884_, lean_object* v_name_5885_, lean_object* v_bi_5886_, lean_object* v_type_5887_, lean_object* v_k_5888_, lean_object* v_kind_5889_, lean_object* v___y_5890_, lean_object* v___y_5891_, lean_object* v___y_5892_, lean_object* v___y_5893_, lean_object* v___y_5894_, lean_object* v___y_5895_){
_start:
{
uint8_t v_bi_boxed_5896_; uint8_t v_kind_boxed_5897_; lean_object* v_res_5898_; 
v_bi_boxed_5896_ = lean_unbox(v_bi_5886_);
v_kind_boxed_5897_ = lean_unbox(v_kind_5889_);
v_res_5898_ = lp_plausible_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitForall___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__6_spec__8(v_00_u03b1_5884_, v_name_5885_, v_bi_boxed_5896_, v_type_5887_, v_k_5888_, v_kind_boxed_5897_, v___y_5890_, v___y_5891_, v___y_5892_, v___y_5893_, v___y_5894_);
lean_dec(v___y_5894_);
lean_dec_ref(v___y_5893_);
lean_dec(v___y_5892_);
lean_dec_ref(v___y_5891_);
lean_dec(v___y_5890_);
return v_res_5898_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11(lean_object* v_00_u03b1_5899_, lean_object* v_name_5900_, lean_object* v_type_5901_, lean_object* v_val_5902_, lean_object* v_k_5903_, uint8_t v_nondep_5904_, uint8_t v_kind_5905_, lean_object* v___y_5906_, lean_object* v___y_5907_, lean_object* v___y_5908_, lean_object* v___y_5909_, lean_object* v___y_5910_){
_start:
{
lean_object* v___x_5912_; 
v___x_5912_ = lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___redArg(v_name_5900_, v_type_5901_, v_val_5902_, v_k_5903_, v_nondep_5904_, v_kind_5905_, v___y_5906_, v___y_5907_, v___y_5908_, v___y_5909_, v___y_5910_);
return v___x_5912_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11___boxed(lean_object* v_00_u03b1_5913_, lean_object* v_name_5914_, lean_object* v_type_5915_, lean_object* v_val_5916_, lean_object* v_k_5917_, lean_object* v_nondep_5918_, lean_object* v_kind_5919_, lean_object* v___y_5920_, lean_object* v___y_5921_, lean_object* v___y_5922_, lean_object* v___y_5923_, lean_object* v___y_5924_, lean_object* v___y_5925_){
_start:
{
uint8_t v_nondep_boxed_5926_; uint8_t v_kind_boxed_5927_; lean_object* v_res_5928_; 
v_nondep_boxed_5926_ = lean_unbox(v_nondep_5918_);
v_kind_boxed_5927_ = lean_unbox(v_kind_5919_);
v_res_5928_ = lp_plausible_Lean_Meta_withLetDecl___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit_visitLet___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__8_spec__11(v_00_u03b1_5913_, v_name_5914_, v_type_5915_, v_val_5916_, v_k_5917_, v_nondep_boxed_5926_, v_kind_boxed_5927_, v___y_5920_, v___y_5921_, v___y_5922_, v___y_5923_, v___y_5924_);
lean_dec(v___y_5924_);
lean_dec_ref(v___y_5923_);
lean_dec(v___y_5922_);
lean_dec_ref(v___y_5921_);
lean_dec(v___y_5920_);
return v_res_5928_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14(lean_object* v_00_u03b1_5929_, lean_object* v_ref_5930_, lean_object* v___y_5931_, lean_object* v___y_5932_, lean_object* v___y_5933_, lean_object* v___y_5934_){
_start:
{
lean_object* v___x_5936_; 
v___x_5936_ = lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___redArg(v_ref_5930_);
return v___x_5936_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14___boxed(lean_object* v_00_u03b1_5937_, lean_object* v_ref_5938_, lean_object* v___y_5939_, lean_object* v___y_5940_, lean_object* v___y_5941_, lean_object* v___y_5942_, lean_object* v___y_5943_){
_start:
{
lean_object* v_res_5944_; 
v_res_5944_ = lp_plausible_Lean_throwMaxRecDepthAt___at___00Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10_spec__14(v_00_u03b1_5937_, v_ref_5938_, v___y_5939_, v___y_5940_, v___y_5941_, v___y_5942_);
lean_dec(v___y_5942_);
lean_dec_ref(v___y_5941_);
lean_dec(v___y_5940_);
lean_dec_ref(v___y_5939_);
return v_res_5944_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10(lean_object* v_00_u03b1_5945_, lean_object* v_x_5946_, lean_object* v___y_5947_, lean_object* v___y_5948_, lean_object* v___y_5949_, lean_object* v___y_5950_, lean_object* v___y_5951_){
_start:
{
lean_object* v___x_5953_; 
v___x_5953_ = lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___redArg(v_x_5946_, v___y_5947_, v___y_5948_, v___y_5949_, v___y_5950_, v___y_5951_);
return v___x_5953_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10___boxed(lean_object* v_00_u03b1_5954_, lean_object* v_x_5955_, lean_object* v___y_5956_, lean_object* v___y_5957_, lean_object* v___y_5958_, lean_object* v___y_5959_, lean_object* v___y_5960_, lean_object* v___y_5961_){
_start:
{
lean_object* v_res_5962_; 
v_res_5962_ = lp_plausible_Lean_Meta_withIncRecDepth___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__10(v_00_u03b1_5954_, v_x_5955_, v___y_5956_, v___y_5957_, v___y_5958_, v___y_5959_, v___y_5960_);
lean_dec(v___y_5960_);
lean_dec_ref(v___y_5959_);
lean_dec(v___y_5958_);
lean_dec_ref(v___y_5957_);
lean_dec(v___y_5956_);
return v_res_5962_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11(lean_object* v_00_u03b2_5963_, lean_object* v_m_5964_, lean_object* v_a_5965_, lean_object* v_b_5966_){
_start:
{
lean_object* v___x_5967_; 
v___x_5967_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11___redArg(v_m_5964_, v_a_5965_, v_b_5966_);
return v___x_5967_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6(lean_object* v_00_u03b2_5968_, lean_object* v_a_5969_, lean_object* v_x_5970_){
_start:
{
lean_object* v___x_5971_; 
v___x_5971_ = lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___redArg(v_a_5969_, v_x_5970_);
return v___x_5971_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6___boxed(lean_object* v_00_u03b2_5972_, lean_object* v_a_5973_, lean_object* v_x_5974_){
_start:
{
lean_object* v_res_5975_; 
v_res_5975_ = lp_plausible_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__5_spec__6(v_00_u03b2_5972_, v_a_5973_, v_x_5974_);
lean_dec(v_x_5974_);
lean_dec_ref(v_a_5973_);
return v_res_5975_;
}
}
LEAN_EXPORT uint8_t lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16(lean_object* v_00_u03b2_5976_, lean_object* v_a_5977_, lean_object* v_x_5978_){
_start:
{
uint8_t v___x_5979_; 
v___x_5979_ = lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___redArg(v_a_5977_, v_x_5978_);
return v___x_5979_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16___boxed(lean_object* v_00_u03b2_5980_, lean_object* v_a_5981_, lean_object* v_x_5982_){
_start:
{
uint8_t v_res_5983_; lean_object* v_r_5984_; 
v_res_5983_ = lp_plausible_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__16(v_00_u03b2_5980_, v_a_5981_, v_x_5982_);
lean_dec(v_x_5982_);
lean_dec_ref(v_a_5981_);
v_r_5984_ = lean_box(v_res_5983_);
return v_r_5984_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17(lean_object* v_00_u03b2_5985_, lean_object* v_data_5986_){
_start:
{
lean_object* v___x_5987_; 
v___x_5987_ = lp_plausible_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17___redArg(v_data_5986_);
return v___x_5987_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18(lean_object* v_00_u03b2_5988_, lean_object* v_a_5989_, lean_object* v_b_5990_, lean_object* v_x_5991_){
_start:
{
lean_object* v___x_5992_; 
v___x_5992_ = lp_plausible_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__18___redArg(v_a_5989_, v_b_5990_, v_x_5991_);
return v___x_5992_;
}
}
LEAN_EXPORT lean_object* lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18(lean_object* v_00_u03b2_5993_, lean_object* v_i_5994_, lean_object* v_source_5995_, lean_object* v_target_5996_){
_start:
{
lean_object* v___x_5997_; 
v___x_5997_ = lp_plausible___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18___redArg(v_i_5994_, v_source_5995_, v_target_5996_);
return v___x_5997_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19(lean_object* v_00_u03b2_5998_, lean_object* v_x_5999_, lean_object* v_x_6000_){
_start:
{
lean_object* v___x_6001_; 
v___x_6001_ = lp_plausible_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Lean_Meta_Transform_0__Lean_Meta_transformWithCache_visit___at___00Lean_Meta_transform___at___00Plausible_Decorations_addDecorations_spec__1_spec__1_spec__11_spec__17_spec__18_spec__19___redArg(v_x_5999_, v_x_6000_);
return v___x_6001_;
}
}
static lean_object* _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_6017_; lean_object* v___x_6018_; lean_object* v___x_6019_; 
v___x_6017_ = lean_box(0);
v___x_6018_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_6019_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_6019_, 0, v___x_6018_);
lean_ctor_set(v___x_6019_, 1, v___x_6017_);
return v___x_6019_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg(){
_start:
{
lean_object* v___x_6021_; lean_object* v___x_6022_; 
v___x_6021_ = lean_obj_once(&lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0, &lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0_once, _init_lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___closed__0);
v___x_6022_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6022_, 0, v___x_6021_);
return v___x_6022_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg___boxed(lean_object* v___y_6023_){
_start:
{
lean_object* v_res_6024_; 
v_res_6024_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg();
return v_res_6024_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0(lean_object* v_00_u03b1_6025_, lean_object* v___y_6026_, lean_object* v___y_6027_, lean_object* v___y_6028_, lean_object* v___y_6029_, lean_object* v___y_6030_, lean_object* v___y_6031_, lean_object* v___y_6032_, lean_object* v___y_6033_){
_start:
{
lean_object* v___x_6035_; 
v___x_6035_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg();
return v___x_6035_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___boxed(lean_object* v_00_u03b1_6036_, lean_object* v___y_6037_, lean_object* v___y_6038_, lean_object* v___y_6039_, lean_object* v___y_6040_, lean_object* v___y_6041_, lean_object* v___y_6042_, lean_object* v___y_6043_, lean_object* v___y_6044_, lean_object* v___y_6045_){
_start:
{
lean_object* v_res_6046_; 
v_res_6046_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0(v_00_u03b1_6036_, v___y_6037_, v___y_6038_, v___y_6039_, v___y_6040_, v___y_6041_, v___y_6042_, v___y_6043_, v___y_6044_);
lean_dec(v___y_6044_);
lean_dec_ref(v___y_6043_);
lean_dec(v___y_6042_);
lean_dec_ref(v___y_6041_);
lean_dec(v___y_6040_);
lean_dec_ref(v___y_6039_);
lean_dec(v___y_6038_);
lean_dec_ref(v___y_6037_);
return v_res_6046_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1(lean_object* v_x_6050_, lean_object* v_a_6051_, lean_object* v_a_6052_, lean_object* v_a_6053_, lean_object* v_a_6054_, lean_object* v_a_6055_, lean_object* v_a_6056_, lean_object* v_a_6057_, lean_object* v_a_6058_){
_start:
{
lean_object* v___x_6060_; lean_object* v___x_6061_; lean_object* v___x_6062_; uint8_t v___x_6063_; 
v___x_6060_ = ((lean_object*)(lp_plausible_Plausible_instToExprConfiguration___lam__0___closed__0));
v___x_6061_ = ((lean_object*)(lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__0));
v___x_6062_ = ((lean_object*)(lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2));
v___x_6063_ = l_Lean_Syntax_isOfKind(v_x_6050_, v___x_6062_);
if (v___x_6063_ == 0)
{
lean_object* v___x_6064_; 
v___x_6064_ = lp_plausible_Lean_Elab_throwUnsupportedSyntax___at___00Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1_spec__0___redArg();
return v___x_6064_;
}
else
{
lean_object* v___x_6065_; 
v___x_6065_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v_a_6052_, v_a_6055_, v_a_6056_, v_a_6057_, v_a_6058_);
if (lean_obj_tag(v___x_6065_) == 0)
{
lean_object* v_a_6066_; lean_object* v___x_6067_; 
v_a_6066_ = lean_ctor_get(v___x_6065_, 0);
lean_inc(v_a_6066_);
lean_dec_ref_known(v___x_6065_, 1);
v___x_6067_ = l_Lean_MVarId_getType(v_a_6066_, v_a_6055_, v_a_6056_, v_a_6057_, v_a_6058_);
if (lean_obj_tag(v___x_6067_) == 0)
{
lean_object* v_a_6068_; lean_object* v___x_6070_; uint8_t v_isShared_6071_; uint8_t v_isSharedCheck_6102_; 
v_a_6068_ = lean_ctor_get(v___x_6067_, 0);
v_isSharedCheck_6102_ = !lean_is_exclusive(v___x_6067_);
if (v_isSharedCheck_6102_ == 0)
{
v___x_6070_ = v___x_6067_;
v_isShared_6071_ = v_isSharedCheck_6102_;
goto v_resetjp_6069_;
}
else
{
lean_inc(v_a_6068_);
lean_dec(v___x_6067_);
v___x_6070_ = lean_box(0);
v_isShared_6071_ = v_isSharedCheck_6102_;
goto v_resetjp_6069_;
}
v_resetjp_6069_:
{
if (lean_obj_tag(v_a_6068_) == 5)
{
lean_object* v_fn_6077_; 
v_fn_6077_ = lean_ctor_get(v_a_6068_, 0);
if (lean_obj_tag(v_fn_6077_) == 4)
{
lean_object* v_declName_6078_; 
v_declName_6078_ = lean_ctor_get(v_fn_6077_, 0);
lean_inc(v_declName_6078_);
if (lean_obj_tag(v_declName_6078_) == 1)
{
lean_object* v_pre_6079_; 
v_pre_6079_ = lean_ctor_get(v_declName_6078_, 0);
lean_inc(v_pre_6079_);
if (lean_obj_tag(v_pre_6079_) == 1)
{
lean_object* v_pre_6080_; 
v_pre_6080_ = lean_ctor_get(v_pre_6079_, 0);
lean_inc(v_pre_6080_);
if (lean_obj_tag(v_pre_6080_) == 1)
{
lean_object* v_pre_6081_; 
v_pre_6081_ = lean_ctor_get(v_pre_6080_, 0);
if (lean_obj_tag(v_pre_6081_) == 0)
{
lean_object* v_arg_6082_; lean_object* v_str_6083_; lean_object* v_str_6084_; lean_object* v_str_6085_; uint8_t v___x_6086_; 
v_arg_6082_ = lean_ctor_get(v_a_6068_, 1);
lean_inc_ref(v_arg_6082_);
lean_dec_ref_known(v_a_6068_, 2);
v_str_6083_ = lean_ctor_get(v_declName_6078_, 1);
lean_inc_ref(v_str_6083_);
lean_dec_ref_known(v_declName_6078_, 2);
v_str_6084_ = lean_ctor_get(v_pre_6079_, 1);
lean_inc_ref(v_str_6084_);
lean_dec_ref_known(v_pre_6079_, 2);
v_str_6085_ = lean_ctor_get(v_pre_6080_, 1);
lean_inc_ref(v_str_6085_);
lean_dec_ref_known(v_pre_6080_, 2);
v___x_6086_ = lean_string_dec_eq(v_str_6085_, v___x_6060_);
lean_dec_ref(v_str_6085_);
if (v___x_6086_ == 0)
{
lean_dec_ref(v_str_6084_);
lean_dec_ref(v_str_6083_);
lean_dec_ref(v_arg_6082_);
goto v___jp_6072_;
}
else
{
uint8_t v___x_6087_; 
v___x_6087_ = lean_string_dec_eq(v_str_6084_, v___x_6061_);
lean_dec_ref(v_str_6084_);
if (v___x_6087_ == 0)
{
lean_dec_ref(v_str_6083_);
lean_dec_ref(v_arg_6082_);
goto v___jp_6072_;
}
else
{
lean_object* v___x_6088_; uint8_t v___x_6089_; 
v___x_6088_ = ((lean_object*)(lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__0));
v___x_6089_ = lean_string_dec_eq(v_str_6083_, v___x_6088_);
lean_dec_ref(v_str_6083_);
if (v___x_6089_ == 0)
{
lean_dec_ref(v_arg_6082_);
goto v___jp_6072_;
}
else
{
lean_object* v___x_6090_; 
lean_del_object(v___x_6070_);
v___x_6090_ = lp_plausible_Plausible_Decorations_addDecorations(v_arg_6082_, v_a_6055_, v_a_6056_, v_a_6057_, v_a_6058_);
if (lean_obj_tag(v___x_6090_) == 0)
{
lean_object* v_a_6091_; lean_object* v___x_6092_; lean_object* v___x_6093_; 
v_a_6091_ = lean_ctor_get(v___x_6090_, 0);
lean_inc(v_a_6091_);
lean_dec_ref_known(v___x_6090_, 1);
v___x_6092_ = ((lean_object*)(lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___closed__1));
v___x_6093_ = l_Lean_Elab_Tactic_closeMainGoal___redArg(v___x_6092_, v_a_6091_, v___x_6063_, v_a_6052_, v_a_6053_, v_a_6054_, v_a_6055_, v_a_6056_, v_a_6057_, v_a_6058_);
return v___x_6093_;
}
else
{
lean_object* v_a_6094_; lean_object* v___x_6096_; uint8_t v_isShared_6097_; uint8_t v_isSharedCheck_6101_; 
v_a_6094_ = lean_ctor_get(v___x_6090_, 0);
v_isSharedCheck_6101_ = !lean_is_exclusive(v___x_6090_);
if (v_isSharedCheck_6101_ == 0)
{
v___x_6096_ = v___x_6090_;
v_isShared_6097_ = v_isSharedCheck_6101_;
goto v_resetjp_6095_;
}
else
{
lean_inc(v_a_6094_);
lean_dec(v___x_6090_);
v___x_6096_ = lean_box(0);
v_isShared_6097_ = v_isSharedCheck_6101_;
goto v_resetjp_6095_;
}
v_resetjp_6095_:
{
lean_object* v___x_6099_; 
if (v_isShared_6097_ == 0)
{
v___x_6099_ = v___x_6096_;
goto v_reusejp_6098_;
}
else
{
lean_object* v_reuseFailAlloc_6100_; 
v_reuseFailAlloc_6100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6100_, 0, v_a_6094_);
v___x_6099_ = v_reuseFailAlloc_6100_;
goto v_reusejp_6098_;
}
v_reusejp_6098_:
{
return v___x_6099_;
}
}
}
}
}
}
}
else
{
lean_dec_ref_known(v_pre_6080_, 2);
lean_dec_ref_known(v_pre_6079_, 2);
lean_dec_ref_known(v_declName_6078_, 2);
lean_dec_ref_known(v_a_6068_, 2);
goto v___jp_6072_;
}
}
else
{
lean_dec_ref_known(v_pre_6079_, 2);
lean_dec(v_pre_6080_);
lean_dec_ref_known(v_declName_6078_, 2);
lean_dec_ref_known(v_a_6068_, 2);
goto v___jp_6072_;
}
}
else
{
lean_dec(v_pre_6079_);
lean_dec_ref_known(v_declName_6078_, 2);
lean_dec_ref_known(v_a_6068_, 2);
goto v___jp_6072_;
}
}
else
{
lean_dec(v_declName_6078_);
lean_dec_ref_known(v_a_6068_, 2);
goto v___jp_6072_;
}
}
else
{
lean_dec_ref_known(v_a_6068_, 2);
goto v___jp_6072_;
}
}
else
{
lean_dec(v_a_6068_);
goto v___jp_6072_;
}
v___jp_6072_:
{
lean_object* v___x_6073_; lean_object* v___x_6075_; 
v___x_6073_ = lean_box(0);
if (v_isShared_6071_ == 0)
{
lean_ctor_set(v___x_6070_, 0, v___x_6073_);
v___x_6075_ = v___x_6070_;
goto v_reusejp_6074_;
}
else
{
lean_object* v_reuseFailAlloc_6076_; 
v_reuseFailAlloc_6076_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6076_, 0, v___x_6073_);
v___x_6075_ = v_reuseFailAlloc_6076_;
goto v_reusejp_6074_;
}
v_reusejp_6074_:
{
return v___x_6075_;
}
}
}
}
else
{
lean_object* v_a_6103_; lean_object* v___x_6105_; uint8_t v_isShared_6106_; uint8_t v_isSharedCheck_6110_; 
v_a_6103_ = lean_ctor_get(v___x_6067_, 0);
v_isSharedCheck_6110_ = !lean_is_exclusive(v___x_6067_);
if (v_isSharedCheck_6110_ == 0)
{
v___x_6105_ = v___x_6067_;
v_isShared_6106_ = v_isSharedCheck_6110_;
goto v_resetjp_6104_;
}
else
{
lean_inc(v_a_6103_);
lean_dec(v___x_6067_);
v___x_6105_ = lean_box(0);
v_isShared_6106_ = v_isSharedCheck_6110_;
goto v_resetjp_6104_;
}
v_resetjp_6104_:
{
lean_object* v___x_6108_; 
if (v_isShared_6106_ == 0)
{
v___x_6108_ = v___x_6105_;
goto v_reusejp_6107_;
}
else
{
lean_object* v_reuseFailAlloc_6109_; 
v_reuseFailAlloc_6109_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6109_, 0, v_a_6103_);
v___x_6108_ = v_reuseFailAlloc_6109_;
goto v_reusejp_6107_;
}
v_reusejp_6107_:
{
return v___x_6108_;
}
}
}
}
else
{
lean_object* v_a_6111_; lean_object* v___x_6113_; uint8_t v_isShared_6114_; uint8_t v_isSharedCheck_6118_; 
v_a_6111_ = lean_ctor_get(v___x_6065_, 0);
v_isSharedCheck_6118_ = !lean_is_exclusive(v___x_6065_);
if (v_isSharedCheck_6118_ == 0)
{
v___x_6113_ = v___x_6065_;
v_isShared_6114_ = v_isSharedCheck_6118_;
goto v_resetjp_6112_;
}
else
{
lean_inc(v_a_6111_);
lean_dec(v___x_6065_);
v___x_6113_ = lean_box(0);
v_isShared_6114_ = v_isSharedCheck_6118_;
goto v_resetjp_6112_;
}
v_resetjp_6112_:
{
lean_object* v___x_6116_; 
if (v_isShared_6114_ == 0)
{
v___x_6116_ = v___x_6113_;
goto v_reusejp_6115_;
}
else
{
lean_object* v_reuseFailAlloc_6117_; 
v_reuseFailAlloc_6117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6117_, 0, v_a_6111_);
v___x_6116_ = v_reuseFailAlloc_6117_;
goto v_reusejp_6115_;
}
v_reusejp_6115_:
{
return v___x_6116_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1___boxed(lean_object* v_x_6119_, lean_object* v_a_6120_, lean_object* v_a_6121_, lean_object* v_a_6122_, lean_object* v_a_6123_, lean_object* v_a_6124_, lean_object* v_a_6125_, lean_object* v_a_6126_, lean_object* v_a_6127_, lean_object* v_a_6128_){
_start:
{
lean_object* v_res_6129_; 
v_res_6129_ = lp_plausible_Plausible_Decorations___aux__Plausible__Testable______elabRules__Plausible__Decorations__tacticMk__decorations__1(v_x_6119_, v_a_6120_, v_a_6121_, v_a_6122_, v_a_6123_, v_a_6124_, v_a_6125_, v_a_6126_, v_a_6127_);
lean_dec(v_a_6127_);
lean_dec_ref(v_a_6126_);
lean_dec(v_a_6125_);
lean_dec_ref(v_a_6124_);
lean_dec(v_a_6123_);
lean_dec_ref(v_a_6122_);
lean_dec(v_a_6121_);
lean_dec_ref(v_a_6120_);
return v_res_6129_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__10(void){
_start:
{
lean_object* v___x_6150_; lean_object* v___x_6151_; 
v___x_6150_ = ((lean_object*)(lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__3));
v___x_6151_ = l_Lean_mkAtom(v___x_6150_);
return v___x_6151_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__11(void){
_start:
{
lean_object* v___x_6152_; lean_object* v___x_6153_; lean_object* v___x_6154_; 
v___x_6152_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__10, &lp_plausible_Plausible_Testable_check___auto__1___closed__10_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__10);
v___x_6153_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__5));
v___x_6154_ = lean_array_push(v___x_6153_, v___x_6152_);
return v___x_6154_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__12(void){
_start:
{
lean_object* v___x_6155_; lean_object* v___x_6156_; lean_object* v___x_6157_; lean_object* v___x_6158_; 
v___x_6155_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__11, &lp_plausible_Plausible_Testable_check___auto__1___closed__11_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__11);
v___x_6156_ = ((lean_object*)(lp_plausible_Plausible_Decorations_tacticMk__decorations___closed__2));
v___x_6157_ = lean_box(2);
v___x_6158_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6158_, 0, v___x_6157_);
lean_ctor_set(v___x_6158_, 1, v___x_6156_);
lean_ctor_set(v___x_6158_, 2, v___x_6155_);
return v___x_6158_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__13(void){
_start:
{
lean_object* v___x_6159_; lean_object* v___x_6160_; lean_object* v___x_6161_; 
v___x_6159_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__12, &lp_plausible_Plausible_Testable_check___auto__1___closed__12_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__12);
v___x_6160_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__5));
v___x_6161_ = lean_array_push(v___x_6160_, v___x_6159_);
return v___x_6161_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__14(void){
_start:
{
lean_object* v___x_6162_; lean_object* v___x_6163_; lean_object* v___x_6164_; lean_object* v___x_6165_; 
v___x_6162_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__13, &lp_plausible_Plausible_Testable_check___auto__1___closed__13_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__13);
v___x_6163_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__9));
v___x_6164_ = lean_box(2);
v___x_6165_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6165_, 0, v___x_6164_);
lean_ctor_set(v___x_6165_, 1, v___x_6163_);
lean_ctor_set(v___x_6165_, 2, v___x_6162_);
return v___x_6165_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__15(void){
_start:
{
lean_object* v___x_6166_; lean_object* v___x_6167_; lean_object* v___x_6168_; 
v___x_6166_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__14, &lp_plausible_Plausible_Testable_check___auto__1___closed__14_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__14);
v___x_6167_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__5));
v___x_6168_ = lean_array_push(v___x_6167_, v___x_6166_);
return v___x_6168_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__16(void){
_start:
{
lean_object* v___x_6169_; lean_object* v___x_6170_; lean_object* v___x_6171_; lean_object* v___x_6172_; 
v___x_6169_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__15, &lp_plausible_Plausible_Testable_check___auto__1___closed__15_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__15);
v___x_6170_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__7));
v___x_6171_ = lean_box(2);
v___x_6172_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6172_, 0, v___x_6171_);
lean_ctor_set(v___x_6172_, 1, v___x_6170_);
lean_ctor_set(v___x_6172_, 2, v___x_6169_);
return v___x_6172_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__17(void){
_start:
{
lean_object* v___x_6173_; lean_object* v___x_6174_; lean_object* v___x_6175_; 
v___x_6173_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__16, &lp_plausible_Plausible_Testable_check___auto__1___closed__16_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__16);
v___x_6174_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__5));
v___x_6175_ = lean_array_push(v___x_6174_, v___x_6173_);
return v___x_6175_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1___closed__18(void){
_start:
{
lean_object* v___x_6176_; lean_object* v___x_6177_; lean_object* v___x_6178_; lean_object* v___x_6179_; 
v___x_6176_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__17, &lp_plausible_Plausible_Testable_check___auto__1___closed__17_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__17);
v___x_6177_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__4));
v___x_6178_ = lean_box(2);
v___x_6179_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_6179_, 0, v___x_6178_);
lean_ctor_set(v___x_6179_, 1, v___x_6177_);
lean_ctor_set(v___x_6179_, 2, v___x_6176_);
return v___x_6179_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___auto__1(void){
_start:
{
lean_object* v___x_6180_; 
v___x_6180_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___auto__1___closed__18, &lp_plausible_Plausible_Testable_check___auto__1___closed__18_once, _init_lp_plausible_Plausible_Testable_check___auto__1___closed__18);
return v___x_6180_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___redArg___closed__0(void){
_start:
{
lean_object* v___x_6181_; lean_object* v___x_6182_; 
v___x_6181_ = lean_obj_once(&lp_plausible_Plausible_Testable_checkIO___redArg___closed__0, &lp_plausible_Plausible_Testable_checkIO___redArg___closed__0_once, _init_lp_plausible_Plausible_Testable_checkIO___redArg___closed__0);
v___x_6182_ = l_StateRefT_x27_instMonad___redArg(v___x_6181_);
return v___x_6182_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___redArg___closed__6(void){
_start:
{
lean_object* v___x_6189_; lean_object* v___x_6190_; 
v___x_6189_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__5));
v___x_6190_ = l_Lean_MessageData_ofFormat(v___x_6189_);
return v___x_6190_;
}
}
static lean_object* _init_lp_plausible_Plausible_Testable_check___redArg___closed__11(void){
_start:
{
lean_object* v___x_6196_; lean_object* v___x_6197_; 
v___x_6196_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__10));
v___x_6197_ = l_Lean_MessageData_ofFormat(v___x_6196_);
return v___x_6197_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___redArg(lean_object* v_cfg_6198_, lean_object* v_inst_6199_, lean_object* v_a_6200_, lean_object* v_a_6201_){
_start:
{
lean_object* v___x_6203_; lean_object* v_toApplicative_6204_; lean_object* v_toFunctor_6205_; lean_object* v_toSeq_6206_; lean_object* v_toSeqLeft_6207_; lean_object* v_toSeqRight_6208_; lean_object* v___f_6209_; lean_object* v___f_6210_; lean_object* v___f_6211_; lean_object* v___f_6212_; lean_object* v___x_6213_; lean_object* v___f_6214_; lean_object* v___f_6215_; lean_object* v___f_6216_; lean_object* v___x_6217_; lean_object* v___x_6218_; lean_object* v___x_6219_; 
v___x_6203_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___redArg___closed__0, &lp_plausible_Plausible_Testable_check___redArg___closed__0_once, _init_lp_plausible_Plausible_Testable_check___redArg___closed__0);
v_toApplicative_6204_ = lean_ctor_get(v___x_6203_, 0);
v_toFunctor_6205_ = lean_ctor_get(v_toApplicative_6204_, 0);
v_toSeq_6206_ = lean_ctor_get(v_toApplicative_6204_, 2);
v_toSeqLeft_6207_ = lean_ctor_get(v_toApplicative_6204_, 3);
v_toSeqRight_6208_ = lean_ctor_get(v_toApplicative_6204_, 4);
v___f_6209_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__1));
v___f_6210_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__2));
lean_inc_ref_n(v_toFunctor_6205_, 2);
v___f_6211_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_6211_, 0, v_toFunctor_6205_);
v___f_6212_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_6212_, 0, v_toFunctor_6205_);
v___x_6213_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6213_, 0, v___f_6211_);
lean_ctor_set(v___x_6213_, 1, v___f_6212_);
lean_inc(v_toSeqRight_6208_);
v___f_6214_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_6214_, 0, v_toSeqRight_6208_);
lean_inc(v_toSeqLeft_6207_);
v___f_6215_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_6215_, 0, v_toSeqLeft_6207_);
lean_inc(v_toSeq_6206_);
v___f_6216_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_6216_, 0, v_toSeq_6206_);
v___x_6217_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_6217_, 0, v___x_6213_);
lean_ctor_set(v___x_6217_, 1, v___f_6209_);
lean_ctor_set(v___x_6217_, 2, v___f_6216_);
lean_ctor_set(v___x_6217_, 3, v___f_6215_);
lean_ctor_set(v___x_6217_, 4, v___f_6214_);
v___x_6218_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6218_, 0, v___x_6217_);
lean_ctor_set(v___x_6218_, 1, v___f_6210_);
lean_inc_ref(v_cfg_6198_);
v___x_6219_ = lp_plausible_Plausible_Testable_checkIO___redArg(v_inst_6199_, v_cfg_6198_);
if (lean_obj_tag(v___x_6219_) == 0)
{
lean_object* v_a_6220_; lean_object* v___x_6222_; uint8_t v_isShared_6223_; uint8_t v_isSharedCheck_6281_; 
v_a_6220_ = lean_ctor_get(v___x_6219_, 0);
v_isSharedCheck_6281_ = !lean_is_exclusive(v___x_6219_);
if (v_isSharedCheck_6281_ == 0)
{
v___x_6222_ = v___x_6219_;
v_isShared_6223_ = v_isSharedCheck_6281_;
goto v_resetjp_6221_;
}
else
{
lean_inc(v_a_6220_);
lean_dec(v___x_6219_);
v___x_6222_ = lean_box(0);
v_isShared_6223_ = v_isSharedCheck_6281_;
goto v_resetjp_6221_;
}
v_resetjp_6221_:
{
switch(lean_obj_tag(v_a_6220_))
{
case 0:
{
uint8_t v_quiet_6224_; 
lean_dec_ref_known(v_a_6220_, 1);
v_quiet_6224_ = lean_ctor_get_uint8(v_cfg_6198_, sizeof(void*)*4 + 4);
lean_dec_ref(v_cfg_6198_);
if (v_quiet_6224_ == 0)
{
lean_object* v___x_6225_; lean_object* v___x_6226_; lean_object* v___f_6227_; lean_object* v___x_6228_; lean_object* v___x_901__overap_6229_; lean_object* v___x_6230_; 
lean_del_object(v___x_6222_);
v___x_6225_ = l_Lean_Core_instMonadLogCoreM;
v___x_6226_ = l_Lean_Core_instAddMessageContextCoreM;
v___f_6227_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__3));
v___x_6228_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___redArg___closed__6, &lp_plausible_Plausible_Testable_check___redArg___closed__6_once, _init_lp_plausible_Plausible_Testable_check___redArg___closed__6);
v___x_901__overap_6229_ = l_Lean_logInfo___redArg(v___x_6218_, v___x_6225_, v___x_6226_, v___f_6227_, v___x_6228_);
lean_inc(v_a_6201_);
lean_inc_ref(v_a_6200_);
v___x_6230_ = lean_apply_3(v___x_901__overap_6229_, v_a_6200_, v_a_6201_, lean_box(0));
return v___x_6230_;
}
else
{
lean_object* v___x_6231_; lean_object* v___x_6233_; 
lean_dec_ref_known(v___x_6218_, 2);
v___x_6231_ = lean_box(0);
if (v_isShared_6223_ == 0)
{
lean_ctor_set(v___x_6222_, 0, v___x_6231_);
v___x_6233_ = v___x_6222_;
goto v_reusejp_6232_;
}
else
{
lean_object* v_reuseFailAlloc_6234_; 
v_reuseFailAlloc_6234_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6234_, 0, v___x_6231_);
v___x_6233_ = v_reuseFailAlloc_6234_;
goto v_reusejp_6232_;
}
v_reusejp_6232_:
{
return v___x_6233_;
}
}
}
case 1:
{
uint8_t v_quiet_6235_; 
v_quiet_6235_ = lean_ctor_get_uint8(v_cfg_6198_, sizeof(void*)*4 + 4);
lean_dec_ref(v_cfg_6198_);
if (v_quiet_6235_ == 0)
{
lean_object* v_a_6236_; lean_object* v___x_6238_; uint8_t v_isShared_6239_; uint8_t v_isSharedCheck_6254_; 
lean_del_object(v___x_6222_);
v_a_6236_ = lean_ctor_get(v_a_6220_, 0);
v_isSharedCheck_6254_ = !lean_is_exclusive(v_a_6220_);
if (v_isSharedCheck_6254_ == 0)
{
v___x_6238_ = v_a_6220_;
v_isShared_6239_ = v_isSharedCheck_6254_;
goto v_resetjp_6237_;
}
else
{
lean_inc(v_a_6236_);
lean_dec(v_a_6220_);
v___x_6238_ = lean_box(0);
v_isShared_6239_ = v_isSharedCheck_6254_;
goto v_resetjp_6237_;
}
v_resetjp_6237_:
{
lean_object* v___x_6240_; lean_object* v___x_6241_; lean_object* v___x_6242_; lean_object* v___x_6243_; lean_object* v___x_6244_; lean_object* v___x_6245_; lean_object* v___x_6246_; lean_object* v___f_6247_; lean_object* v___x_6249_; 
v___x_6240_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__7));
v___x_6241_ = l_Nat_reprFast(v_a_6236_);
v___x_6242_ = lean_string_append(v___x_6240_, v___x_6241_);
lean_dec_ref(v___x_6241_);
v___x_6243_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__8));
v___x_6244_ = lean_string_append(v___x_6242_, v___x_6243_);
v___x_6245_ = l_Lean_Core_instMonadLogCoreM;
v___x_6246_ = l_Lean_Core_instAddMessageContextCoreM;
v___f_6247_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__3));
if (v_isShared_6239_ == 0)
{
lean_ctor_set_tag(v___x_6238_, 3);
lean_ctor_set(v___x_6238_, 0, v___x_6244_);
v___x_6249_ = v___x_6238_;
goto v_reusejp_6248_;
}
else
{
lean_object* v_reuseFailAlloc_6253_; 
v_reuseFailAlloc_6253_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6253_, 0, v___x_6244_);
v___x_6249_ = v_reuseFailAlloc_6253_;
goto v_reusejp_6248_;
}
v_reusejp_6248_:
{
lean_object* v___x_6250_; lean_object* v___x_914__overap_6251_; lean_object* v___x_6252_; 
v___x_6250_ = l_Lean_MessageData_ofFormat(v___x_6249_);
v___x_914__overap_6251_ = l_Lean_logWarning___redArg(v___x_6218_, v___x_6245_, v___x_6246_, v___f_6247_, v___x_6250_);
lean_inc(v_a_6201_);
lean_inc_ref(v_a_6200_);
v___x_6252_ = lean_apply_3(v___x_914__overap_6251_, v_a_6200_, v_a_6201_, lean_box(0));
return v___x_6252_;
}
}
}
else
{
lean_object* v___x_6255_; lean_object* v___x_6257_; 
lean_dec_ref_known(v_a_6220_, 1);
lean_dec_ref_known(v___x_6218_, 2);
v___x_6255_ = lean_box(0);
if (v_isShared_6223_ == 0)
{
lean_ctor_set(v___x_6222_, 0, v___x_6255_);
v___x_6257_ = v___x_6222_;
goto v_reusejp_6256_;
}
else
{
lean_object* v_reuseFailAlloc_6258_; 
v_reuseFailAlloc_6258_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6258_, 0, v___x_6255_);
v___x_6257_ = v_reuseFailAlloc_6258_;
goto v_reusejp_6256_;
}
v_reusejp_6256_:
{
return v___x_6257_;
}
}
}
default: 
{
lean_object* v_a_6259_; lean_object* v_a_6260_; uint8_t v_quiet_6261_; lean_object* v___x_6262_; 
lean_del_object(v___x_6222_);
v_a_6259_ = lean_ctor_get(v_a_6220_, 0);
lean_inc(v_a_6259_);
v_a_6260_ = lean_ctor_get(v_a_6220_, 1);
lean_inc(v_a_6260_);
lean_dec_ref_known(v_a_6220_, 2);
v_quiet_6261_ = lean_ctor_get_uint8(v_cfg_6198_, sizeof(void*)*4 + 4);
lean_dec_ref(v_cfg_6198_);
v___x_6262_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___redArg___closed__9));
if (v_quiet_6261_ == 0)
{
lean_object* v___x_6263_; lean_object* v___x_6264_; lean_object* v___x_6265_; lean_object* v___x_6266_; lean_object* v___x_6267_; lean_object* v___x_6268_; lean_object* v___x_6269_; lean_object* v___x_6270_; lean_object* v___x_860__overap_6271_; lean_object* v___x_6272_; 
v___x_6263_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___x_6264_ = l_Lean_Core_instMonadRefCoreM;
v___x_6265_ = l_Lean_Core_instAddMessageContextCoreM;
lean_inc_ref(v___x_6218_);
v___x_6266_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_6265_, v___x_6218_);
v___x_6267_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_6267_, 0, v___x_6263_);
lean_ctor_set(v___x_6267_, 1, v___x_6264_);
lean_ctor_set(v___x_6267_, 2, v___x_6266_);
v___x_6268_ = lp_plausible_Plausible_Testable_formatFailure(v___x_6262_, v_a_6259_, v_a_6260_);
v___x_6269_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_6269_, 0, v___x_6268_);
v___x_6270_ = l_Lean_MessageData_ofFormat(v___x_6269_);
v___x_860__overap_6271_ = l_Lean_throwError___redArg(v___x_6218_, v___x_6267_, v___x_6270_);
lean_inc(v_a_6201_);
lean_inc_ref(v_a_6200_);
v___x_6272_ = lean_apply_3(v___x_860__overap_6271_, v_a_6200_, v_a_6201_, lean_box(0));
return v___x_6272_;
}
else
{
lean_object* v___x_6273_; lean_object* v___x_6274_; lean_object* v___x_6275_; lean_object* v___x_6276_; lean_object* v___x_6277_; lean_object* v___x_6278_; lean_object* v___x_868__overap_6279_; lean_object* v___x_6280_; 
lean_dec(v_a_6260_);
lean_dec(v_a_6259_);
v___x_6273_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___x_6274_ = l_Lean_Core_instMonadRefCoreM;
v___x_6275_ = l_Lean_Core_instAddMessageContextCoreM;
lean_inc_ref(v___x_6218_);
v___x_6276_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___x_6275_, v___x_6218_);
v___x_6277_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_6277_, 0, v___x_6273_);
lean_ctor_set(v___x_6277_, 1, v___x_6274_);
lean_ctor_set(v___x_6277_, 2, v___x_6276_);
v___x_6278_ = lean_obj_once(&lp_plausible_Plausible_Testable_check___redArg___closed__11, &lp_plausible_Plausible_Testable_check___redArg___closed__11_once, _init_lp_plausible_Plausible_Testable_check___redArg___closed__11);
v___x_868__overap_6279_ = l_Lean_throwError___redArg(v___x_6218_, v___x_6277_, v___x_6278_);
lean_inc(v_a_6201_);
lean_inc_ref(v_a_6200_);
v___x_6280_ = lean_apply_3(v___x_868__overap_6279_, v_a_6200_, v_a_6201_, lean_box(0));
return v___x_6280_;
}
}
}
}
}
else
{
lean_object* v_a_6282_; lean_object* v___x_6284_; uint8_t v_isShared_6285_; uint8_t v_isSharedCheck_6294_; 
lean_dec_ref_known(v___x_6218_, 2);
lean_dec_ref(v_cfg_6198_);
v_a_6282_ = lean_ctor_get(v___x_6219_, 0);
v_isSharedCheck_6294_ = !lean_is_exclusive(v___x_6219_);
if (v_isSharedCheck_6294_ == 0)
{
v___x_6284_ = v___x_6219_;
v_isShared_6285_ = v_isSharedCheck_6294_;
goto v_resetjp_6283_;
}
else
{
lean_inc(v_a_6282_);
lean_dec(v___x_6219_);
v___x_6284_ = lean_box(0);
v_isShared_6285_ = v_isSharedCheck_6294_;
goto v_resetjp_6283_;
}
v_resetjp_6283_:
{
lean_object* v_ref_6286_; lean_object* v___x_6287_; lean_object* v___x_6288_; lean_object* v___x_6289_; lean_object* v___x_6290_; lean_object* v___x_6292_; 
v_ref_6286_ = lean_ctor_get(v_a_6200_, 5);
v___x_6287_ = lean_io_error_to_string(v_a_6282_);
v___x_6288_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_6288_, 0, v___x_6287_);
v___x_6289_ = l_Lean_MessageData_ofFormat(v___x_6288_);
lean_inc(v_ref_6286_);
v___x_6290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6290_, 0, v_ref_6286_);
lean_ctor_set(v___x_6290_, 1, v___x_6289_);
if (v_isShared_6285_ == 0)
{
lean_ctor_set(v___x_6284_, 0, v___x_6290_);
v___x_6292_ = v___x_6284_;
goto v_reusejp_6291_;
}
else
{
lean_object* v_reuseFailAlloc_6293_; 
v_reuseFailAlloc_6293_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_6293_, 0, v___x_6290_);
v___x_6292_ = v_reuseFailAlloc_6293_;
goto v_reusejp_6291_;
}
v_reusejp_6291_:
{
return v___x_6292_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___redArg___boxed(lean_object* v_cfg_6295_, lean_object* v_inst_6296_, lean_object* v_a_6297_, lean_object* v_a_6298_, lean_object* v_a_6299_){
_start:
{
lean_object* v_res_6300_; 
v_res_6300_ = lp_plausible_Plausible_Testable_check___redArg(v_cfg_6295_, v_inst_6296_, v_a_6297_, v_a_6298_);
lean_dec(v_a_6298_);
lean_dec_ref(v_a_6297_);
return v_res_6300_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check(lean_object* v_p_6301_, lean_object* v_cfg_6302_, lean_object* v_p_x27_6303_, lean_object* v_inst_6304_, lean_object* v_a_6305_, lean_object* v_a_6306_){
_start:
{
lean_object* v___x_6308_; 
v___x_6308_ = lp_plausible_Plausible_Testable_check___redArg(v_cfg_6302_, v_inst_6304_, v_a_6305_, v_a_6306_);
return v___x_6308_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible_Testable_check___boxed(lean_object* v_p_6309_, lean_object* v_cfg_6310_, lean_object* v_p_x27_6311_, lean_object* v_inst_6312_, lean_object* v_a_6313_, lean_object* v_a_6314_, lean_object* v_a_6315_){
_start:
{
lean_object* v_res_6316_; 
v_res_6316_ = lp_plausible_Plausible_Testable_check(v_p_6309_, v_cfg_6310_, v_p_x27_6311_, v_inst_6312_, v_a_6313_, v_a_6314_);
lean_dec(v_a_6314_);
lean_dec_ref(v_a_6313_);
return v_res_6316_;
}
}
static lean_object* _init_lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8(void){
_start:
{
lean_object* v___x_6358_; lean_object* v___x_6359_; 
v___x_6358_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__7));
v___x_6359_ = l_String_toRawSubstring_x27(v___x_6358_);
return v___x_6359_;
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1(lean_object* v_x_6380_, lean_object* v_a_6381_, lean_object* v_a_6382_){
_start:
{
lean_object* v___x_6383_; uint8_t v___x_6384_; 
v___x_6383_ = ((lean_object*)(lp_plausible_Plausible_command_x23test___00__closed__1));
lean_inc(v_x_6380_);
v___x_6384_ = l_Lean_Syntax_isOfKind(v_x_6380_, v___x_6383_);
if (v___x_6384_ == 0)
{
lean_object* v___x_6385_; lean_object* v___x_6386_; 
lean_dec(v_x_6380_);
v___x_6385_ = lean_box(1);
v___x_6386_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_6386_, 0, v___x_6385_);
lean_ctor_set(v___x_6386_, 1, v_a_6382_);
return v___x_6386_;
}
else
{
lean_object* v_quotContext_6387_; lean_object* v_currMacroScope_6388_; lean_object* v_ref_6389_; lean_object* v___x_6390_; lean_object* v_tk_6391_; lean_object* v___x_6392_; lean_object* v___x_6393_; uint8_t v___x_6394_; lean_object* v___x_6395_; lean_object* v___x_6396_; lean_object* v___x_6397_; lean_object* v___x_6398_; lean_object* v___x_6399_; lean_object* v___x_6400_; lean_object* v___x_6401_; lean_object* v___x_6402_; lean_object* v___x_6403_; lean_object* v___x_6404_; lean_object* v___x_6405_; lean_object* v___x_6406_; lean_object* v___x_6407_; lean_object* v___x_6408_; lean_object* v___x_6409_; lean_object* v___x_6410_; 
v_quotContext_6387_ = lean_ctor_get(v_a_6381_, 1);
v_currMacroScope_6388_ = lean_ctor_get(v_a_6381_, 2);
v_ref_6389_ = lean_ctor_get(v_a_6381_, 5);
v___x_6390_ = lean_unsigned_to_nat(0u);
v_tk_6391_ = l_Lean_Syntax_getArg(v_x_6380_, v___x_6390_);
v___x_6392_ = lean_unsigned_to_nat(1u);
v___x_6393_ = l_Lean_Syntax_getArg(v_x_6380_, v___x_6392_);
lean_dec(v_x_6380_);
v___x_6394_ = 0;
v___x_6395_ = l_Lean_SourceInfo_fromRef(v_ref_6389_, v___x_6394_);
v___x_6396_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__2));
v___x_6397_ = l_Lean_SourceInfo_fromRef(v_tk_6391_, v___x_6384_);
lean_dec(v_tk_6391_);
v___x_6398_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__3));
v___x_6399_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_6399_, 0, v___x_6397_);
lean_ctor_set(v___x_6399_, 1, v___x_6398_);
v___x_6400_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__6));
v___x_6401_ = lean_obj_once(&lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8, &lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8_once, _init_lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__8);
v___x_6402_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__11));
lean_inc(v_currMacroScope_6388_);
lean_inc(v_quotContext_6387_);
v___x_6403_ = l_Lean_addMacroScope(v_quotContext_6387_, v___x_6402_, v_currMacroScope_6388_);
v___x_6404_ = ((lean_object*)(lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___closed__16));
lean_inc_n(v___x_6395_, 3);
v___x_6405_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_6405_, 0, v___x_6395_);
lean_ctor_set(v___x_6405_, 1, v___x_6401_);
lean_ctor_set(v___x_6405_, 2, v___x_6403_);
lean_ctor_set(v___x_6405_, 3, v___x_6404_);
v___x_6406_ = ((lean_object*)(lp_plausible_Plausible_Testable_check___auto__1___closed__9));
v___x_6407_ = l_Lean_Syntax_node1(v___x_6395_, v___x_6406_, v___x_6393_);
v___x_6408_ = l_Lean_Syntax_node2(v___x_6395_, v___x_6400_, v___x_6405_, v___x_6407_);
v___x_6409_ = l_Lean_Syntax_node2(v___x_6395_, v___x_6396_, v___x_6399_, v___x_6408_);
v___x_6410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_6410_, 0, v___x_6409_);
lean_ctor_set(v___x_6410_, 1, v_a_6382_);
return v___x_6410_;
}
}
}
LEAN_EXPORT lean_object* lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1___boxed(lean_object* v_x_6411_, lean_object* v_a_6412_, lean_object* v_a_6413_){
_start:
{
lean_object* v_res_6414_; 
v_res_6414_ = lp_plausible_Plausible___aux__Plausible__Testable______macroRules__Plausible__command_x23test____1(v_x_6411_, v_a_6412_, v_a_6413_);
lean_dec_ref(v_a_6412_);
return v_res_6414_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_CoreM(uint8_t builtin);
lean_object* runtime_initialize_Lean_Exception(uint8_t builtin);
lean_object* runtime_initialize_Lean_Log(uint8_t builtin);
lean_object* runtime_initialize_plausible_Plausible_Sampleable(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_plausible_Plausible_Testable(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_CoreM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Sampleable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Config(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_plausible_Plausible_Testable(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_plausible_Plausible_instToExprConfiguration = _init_lp_plausible_Plausible_instToExprConfiguration();
lean_mark_persistent(lp_plausible_Plausible_instToExprConfiguration);
lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration = _init_lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration();
lean_mark_persistent(lp_plausible___private_Plausible_Testable_0__Plausible_instEvalExprConfiguration);
lp_plausible_Plausible_Testable_check___auto__1 = _init_lp_plausible_Plausible_Testable_check___auto__1();
lean_mark_persistent(lp_plausible_Plausible_Testable_check___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Config(uint8_t builtin);
lean_object* initialize_Lean_CoreM(uint8_t builtin);
lean_object* initialize_Lean_Exception(uint8_t builtin);
lean_object* initialize_Lean_Log(uint8_t builtin);
lean_object* initialize_plausible_Plausible_Sampleable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_plausible_Plausible_Testable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Config(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_CoreM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Log(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_plausible_Plausible_Sampleable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_plausible_Plausible_Testable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_plausible_Plausible_Testable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_plausible_Plausible_Testable(builtin);
}
#ifdef __cplusplus
}
#endif
