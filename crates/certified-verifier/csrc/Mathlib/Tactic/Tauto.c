// Lean compiler output
// Module: Mathlib.Tactic.Tauto
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Classical public meta import Mathlib.Lean.Meta public import Mathlib.Basic.Logic.Basic public import Mathlib.Tactic.CasesM public import Mathlib.Tactic.Core public import Lean.Elab.ConfigEval public meta import Lean.Elab.ConfigEval public import Qq
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_tryTactic___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_intros_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_FVarId_getDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* l_Lean_MVarId_assertAfter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_clear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lp_Qq_Qq_synthInstanceQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_toExpr(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_Meta_FVarSubst_apply(lean_object*, lean_object*);
lean_object* l_List_tail_x21___redArg(lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Tactic_constructorMatching(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_MVarId_casesMatching(lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lean_Elab_Tactic_allGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withoutRecover___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_isEmpty___redArg(lean_object*);
lean_object* l_Lean_Meta_throwTacticEx___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instanceExtension;
lean_object* l_Lean_ScopedEnvExtension_pushScope___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_addInstance(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_ScopedEnvExtension_popScope___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Elab_Command_getRef___redArg(lean_object*);
lean_object* l_Lean_Elab_Command_getCurrMacroScope___redArg(lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(lean_object*);
lean_object* l_Lean_Elab_ConfigEval_ConfigItem_shift(lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Meta_getMVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_logUnassignedUsingErrorInfos(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_abortTermExceptionId;
uint8_t l_Lean_Expr_hasSorry(lean_object*);
uint8_t l_Lean_Expr_hasSyntheticSorry(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lp_mathlib_Mathlib_TacticAnalysis_terminalReplacement(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, double);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Parser_Tactic_optConfig;
lean_object* lp_mathlib_Mathlib_TacticAnalysis_grindReplacementWith(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, double);
lean_object* l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_focus___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_registerTraceClass(lean_object*, uint8_t, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tauto"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(192, 3, 0, 253, 197, 134, 81, 132)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__7_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(191, 70, 156, 159, 11, 54, 216, 94)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__7_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__7_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Tauto"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__9_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__7_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(79, 236, 117, 24, 207, 13, 242, 83)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__9_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__9_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__10_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__9_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(226, 209, 196, 236, 110, 58, 134, 224)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__10_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__10_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__11_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__10_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(83, 197, 50, 36, 174, 199, 78, 39)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__11_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__11_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__12_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__11_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(66, 49, 40, 212, 154, 200, 174, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__12_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__12_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__13_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__12_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 37, 7, 175, 225, 55, 85, 30)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__13_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__13_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__14_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "initFn"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__14_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__14_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__15_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__13_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__14_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(11, 220, 136, 183, 20, 246, 120, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__15_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__15_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__16_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__16_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__16_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__17_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__15_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__16_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(206, 203, 121, 186, 228, 85, 229, 138)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__17_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__17_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__18_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__17_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(71, 112, 174, 114, 187, 88, 188, 220)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__18_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__18_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__19_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__18_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(70, 197, 238, 49, 131, 45, 138, 227)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__19_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__19_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__20_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__19_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(242, 98, 122, 251, 144, 122, 161, 163)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__20_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__20_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__22_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__22_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__22_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__24_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__24_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__24_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(185, 11, 203, 55, 27, 192, 137, 230)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "And"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 220, 212, 156, 122, 214, 55, 135)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Or"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(34, 237, 162, 225, 217, 98, 205, 196)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Ne"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__0_value),LEAN_SCALAR_PTR_LITERAL(161, 247, 70, 70, 118, 145, 235, 92)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "False"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__0_value),LEAN_SCALAR_PTR_LITERAL(227, 122, 176, 177, 50, 175, 152, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "distribNot found nothing to work on"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Decidable"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "not_or_of_imp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__7_value),LEAN_SCALAR_PTR_LITERAL(227, 121, 117, 236, 163, 238, 175, 219)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "distribNot found nothing to work on with negation"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__10_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mp"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__0_value),LEAN_SCALAR_PTR_LITERAL(19, 54, 203, 28, 77, 25, 163, 137)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__12_value),LEAN_SCALAR_PTR_LITERAL(147, 220, 216, 40, 239, 165, 44, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "iff_iff_and_or_not_and_not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__15_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__15_value),LEAN_SCALAR_PTR_LITERAL(23, 34, 20, 109, 246, 102, 9, 240)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "not_iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__18_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__18_value),LEAN_SCALAR_PTR_LITERAL(182, 247, 17, 97, 250, 253, 2, 142)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "not_imp_iff_and_not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__21_value),LEAN_SCALAR_PTR_LITERAL(23, 157, 185, 214, 204, 204, 132, 186)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "of_not_not"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__24_value),LEAN_SCALAR_PTR_LITERAL(47, 206, 222, 214, 181, 19, 26, 146)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "not_or"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__31_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__31_value),LEAN_SCALAR_PTR_LITERAL(112, 214, 26, 199, 216, 200, 97, 117)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__32 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__32_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "not_and_iff_not_or_not'"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__4_value),LEAN_SCALAR_PTR_LITERAL(87, 187, 205, 215, 218, 218, 68, 60)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__34_value),LEAN_SCALAR_PTR_LITERAL(126, 109, 223, 185, 153, 170, 148, 163)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "to_iff"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__37_value),LEAN_SCALAR_PTR_LITERAL(160, 11, 29, 231, 2, 163, 213, 65)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mt"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__40_value),LEAN_SCALAR_PTR_LITERAL(236, 40, 53, 218, 205, 101, 166, 74)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__41_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "propext"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__43_value),LEAN_SCALAR_PTR_LITERAL(53, 150, 49, 30, 125, 3, 39, 172)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__44_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__46_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "not fvar "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__48_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Tauto_distribNotAt_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_distribNot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_distribNot___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(243, 83, 25, 243, 21, 136, 46, 247)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__1_value),LEAN_SCALAR_PTR_LITERAL(64, 79, 53, 157, 12, 89, 102, 252)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__0_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nof type `"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "`"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__4_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "Could not evaluate the expression"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__7 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__7_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8;
static const lean_string_object lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Expression contains `sorry`:"};
static const lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "config"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Exists"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(65, 29, 48, 135, 199, 176, 149, 70)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term_<;>_"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__13_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(155, 25, 157, 68, 224, 95, 159, 143)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = " <;> "};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__4_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__7_value),((lean_object*)(((size_t)(51) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__8_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1_value),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)(((size_t)(50) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__9_value)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e__ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__10_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__2_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "andThenOnSubgoals"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(96, 51, 241, 130, 196, 103, 59, 150)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(161, 230, 229, 85, 182, 144, 182, 176)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(130, 204, 17, 2, 229, 228, 155, 120)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "contradiction"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(112, 219, 21, 122, 229, 107, 49, 36)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "or_iff_not_imp_left.mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "or_iff_not_imp_left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mpr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__5_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(139, 113, 8, 197, 77, 119, 12, 49)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__5_value),LEAN_SCALAR_PTR_LITERAL(230, 79, 236, 1, 92, 117, 39, 87)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Classical"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(252, 110, 199, 212, 121, 206, 243, 161)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__5_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__8_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__9_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__12_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__14_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "assumption"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(240, 50, 167, 190, 65, 82, 149, 231)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9(lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9___boxed, .m_arity = 12, .m_num_fixed = 3, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__3_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Failed"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1___boxed, .m_arity = 10, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__1_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__2_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__3_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__0_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__1_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__4_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__5_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6___boxed, .m_arity = 10, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__5_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__6_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___boxed, .m_arity = 13, .m_num_fixed = 4, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__6_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__2_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__3_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__4_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3;
static const lean_string_object lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "propDecidable"};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__4 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(40, 236, 220, 79, 38, 141, 161, 150)}};
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(166, 239, 88, 215, 135, 192, 113, 64)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 188, 173, 198, 169, 252, 183, 45)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "solveByElim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__3_value),LEAN_SCALAR_PTR_LITERAL(223, 81, 23, 77, 106, 178, 107, 255)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "solve_by_elim"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__5_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__7_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__7_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__0_value;
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___boxed, .m_arity = 11, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__0_value)} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__8_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(243, 83, 25, 243, 21, 136, 46, 247)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(40, 188, 227, 116, 33, 242, 140, 204)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__1_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tauto;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "tacticAnalysis"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "tautoToGrind"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(85, 245, 235, 249, 235, 40, 236, 31)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(203, 184, 72, 148, 130, 210, 20, 254)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__4_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_tacticAnalysis_tautoToGrind;
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrind___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrind___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tautoToGrind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "grind"};
static const lean_object* lp_mathlib_tautoToGrind___closed__0 = (const lean_object*)&lp_mathlib_tautoToGrind___closed__0_value;
static const lean_closure_object lp_mathlib_tautoToGrind___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_tautoToGrind___lam__0___boxed, .m_arity = 8, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__6_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__value),((lean_object*)&lp_mathlib_tautoToGrind___closed__0_value)} };
static const lean_object* lp_mathlib_tautoToGrind___closed__1 = (const lean_object*)&lp_mathlib_tautoToGrind___closed__1_value;
static lean_once_cell_t lp_mathlib_tautoToGrind___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_tautoToGrind___closed__2;
static lean_once_cell_t lp_mathlib_tautoToGrind___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tautoToGrind___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrind;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "regressions"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(85, 245, 235, 249, 235, 40, 236, 31)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(211, 180, 168, 101, 210, 79, 241, 195)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__2_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(205, 201, 193, 254, 71, 229, 148, 115)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_tacticAnalysis_regressions_tautoToGrind;
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrindRegressions___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrindRegressions___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_tautoToGrindRegressions___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_tautoToGrindRegressions___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_tautoToGrindRegressions___closed__0 = (const lean_object*)&lp_mathlib_tautoToGrindRegressions___closed__0_value;
static lean_once_cell_t lp_mathlib_tautoToGrindRegressions___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tautoToGrindRegressions___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrindRegressions;
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_49_ = lean_unsigned_to_nat(3330388757u);
v___x_50_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__20_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_51_ = l_Lean_Name_num___override(v___x_50_, v___x_49_);
return v___x_51_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_53_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__22_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_54_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__21_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_);
v___x_55_ = l_Lean_Name_str___override(v___x_54_, v___x_53_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__24_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_58_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__23_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_);
v___x_59_ = l_Lean_Name_str___override(v___x_58_, v___x_57_);
return v___x_59_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_(void){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_60_ = lean_unsigned_to_nat(2u);
v___x_61_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__25_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_);
v___x_62_ = l_Lean_Name_num___override(v___x_61_, v___x_60_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_64_; uint8_t v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_64_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_65_ = 0;
v___x_66_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2__once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__26_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_);
v___x_67_ = l_Lean_registerTraceClass(v___x_64_, v___x_65_, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2____boxed(lean_object* v_a_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_();
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg(lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = l_Lean_Meta_saveState___redArg(v___y_72_, v___y_74_);
if (lean_obj_tag(v___x_76_) == 0)
{
lean_object* v_a_77_; lean_object* v___x_78_; 
v_a_77_ = lean_ctor_get(v___x_76_, 0);
lean_inc(v_a_77_);
lean_dec_ref_known(v___x_76_, 1);
lean_inc(v___y_74_);
lean_inc_ref(v___y_73_);
lean_inc(v___y_72_);
lean_inc_ref(v___y_71_);
v___x_78_ = lean_apply_5(v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_, lean_box(0));
if (lean_obj_tag(v___x_78_) == 0)
{
lean_dec(v_a_77_);
return v___x_78_;
}
else
{
lean_object* v_a_79_; uint8_t v___y_81_; uint8_t v___x_99_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_a_79_);
v___x_99_ = l_Lean_Exception_isInterrupt(v_a_79_);
if (v___x_99_ == 0)
{
uint8_t v___x_100_; 
lean_inc(v_a_79_);
v___x_100_ = l_Lean_Exception_isRuntime(v_a_79_);
v___y_81_ = v___x_100_;
goto v___jp_80_;
}
else
{
v___y_81_ = v___x_99_;
goto v___jp_80_;
}
v___jp_80_:
{
if (v___y_81_ == 0)
{
lean_object* v___x_82_; 
lean_dec_ref_known(v___x_78_, 1);
v___x_82_ = l_Lean_Meta_SavedState_restore___redArg(v_a_77_, v___y_72_, v___y_74_);
lean_dec(v_a_77_);
if (lean_obj_tag(v___x_82_) == 0)
{
lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_89_ == 0)
{
lean_object* v_unused_90_; 
v_unused_90_ = lean_ctor_get(v___x_82_, 0);
lean_dec(v_unused_90_);
v___x_84_ = v___x_82_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_dec(v___x_82_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
lean_ctor_set_tag(v___x_84_, 1);
lean_ctor_set(v___x_84_, 0, v_a_79_);
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_79_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
else
{
lean_object* v_a_91_; lean_object* v___x_93_; uint8_t v_isShared_94_; uint8_t v_isSharedCheck_98_; 
lean_dec(v_a_79_);
v_a_91_ = lean_ctor_get(v___x_82_, 0);
v_isSharedCheck_98_ = !lean_is_exclusive(v___x_82_);
if (v_isSharedCheck_98_ == 0)
{
v___x_93_ = v___x_82_;
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
else
{
lean_inc(v_a_91_);
lean_dec(v___x_82_);
v___x_93_ = lean_box(0);
v_isShared_94_ = v_isSharedCheck_98_;
goto v_resetjp_92_;
}
v_resetjp_92_:
{
lean_object* v___x_96_; 
if (v_isShared_94_ == 0)
{
v___x_96_ = v___x_93_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v_a_91_);
v___x_96_ = v_reuseFailAlloc_97_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
return v___x_96_;
}
}
}
}
else
{
lean_dec(v_a_79_);
lean_dec(v_a_77_);
return v___x_78_;
}
}
}
}
else
{
lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_108_; 
lean_dec_ref(v_x_70_);
v_a_101_ = lean_ctor_get(v___x_76_, 0);
v_isSharedCheck_108_ = !lean_is_exclusive(v___x_76_);
if (v_isSharedCheck_108_ == 0)
{
v___x_103_ = v___x_76_;
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v___x_76_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_106_; 
if (v_isShared_104_ == 0)
{
v___x_106_ = v___x_103_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v_a_101_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg___boxed(lean_object* v_x_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg(v_x_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0(lean_object* v_00_u03b1_116_, lean_object* v_x_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v___x_123_; 
v___x_123_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg(v_x_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___boxed(lean_object* v_00_u03b1_124_, lean_object* v_x_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0(v_00_u03b1_124_, v_x_125_, v___y_126_, v___y_127_, v___y_128_, v___y_129_);
lean_dec(v___y_129_);
lean_dec_ref(v___y_128_);
lean_dec(v___y_127_);
lean_dec_ref(v___y_126_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(lean_object* v_e_132_, lean_object* v___y_133_){
_start:
{
uint8_t v___x_135_; 
v___x_135_ = l_Lean_Expr_hasMVar(v_e_132_);
if (v___x_135_ == 0)
{
lean_object* v___x_136_; 
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v_e_132_);
return v___x_136_;
}
else
{
lean_object* v___x_137_; lean_object* v_mctx_138_; lean_object* v___x_139_; lean_object* v_fst_140_; lean_object* v_snd_141_; lean_object* v___x_142_; lean_object* v_cache_143_; lean_object* v_zetaDeltaFVarIds_144_; lean_object* v_postponed_145_; lean_object* v_diag_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_155_; 
v___x_137_ = lean_st_ref_get(v___y_133_);
v_mctx_138_ = lean_ctor_get(v___x_137_, 0);
lean_inc_ref(v_mctx_138_);
lean_dec(v___x_137_);
v___x_139_ = l_Lean_instantiateMVarsCore(v_mctx_138_, v_e_132_);
v_fst_140_ = lean_ctor_get(v___x_139_, 0);
lean_inc(v_fst_140_);
v_snd_141_ = lean_ctor_get(v___x_139_, 1);
lean_inc(v_snd_141_);
lean_dec_ref(v___x_139_);
v___x_142_ = lean_st_ref_take(v___y_133_);
v_cache_143_ = lean_ctor_get(v___x_142_, 1);
v_zetaDeltaFVarIds_144_ = lean_ctor_get(v___x_142_, 2);
v_postponed_145_ = lean_ctor_get(v___x_142_, 3);
v_diag_146_ = lean_ctor_get(v___x_142_, 4);
v_isSharedCheck_155_ = !lean_is_exclusive(v___x_142_);
if (v_isSharedCheck_155_ == 0)
{
lean_object* v_unused_156_; 
v_unused_156_ = lean_ctor_get(v___x_142_, 0);
lean_dec(v_unused_156_);
v___x_148_ = v___x_142_;
v_isShared_149_ = v_isSharedCheck_155_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_diag_146_);
lean_inc(v_postponed_145_);
lean_inc(v_zetaDeltaFVarIds_144_);
lean_inc(v_cache_143_);
lean_dec(v___x_142_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_155_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_151_; 
if (v_isShared_149_ == 0)
{
lean_ctor_set(v___x_148_, 0, v_snd_141_);
v___x_151_ = v___x_148_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_154_; 
v_reuseFailAlloc_154_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_154_, 0, v_snd_141_);
lean_ctor_set(v_reuseFailAlloc_154_, 1, v_cache_143_);
lean_ctor_set(v_reuseFailAlloc_154_, 2, v_zetaDeltaFVarIds_144_);
lean_ctor_set(v_reuseFailAlloc_154_, 3, v_postponed_145_);
lean_ctor_set(v_reuseFailAlloc_154_, 4, v_diag_146_);
v___x_151_ = v_reuseFailAlloc_154_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_152_ = lean_st_ref_set(v___y_133_, v___x_151_);
v___x_153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_153_, 0, v_fst_140_);
return v___x_153_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg___boxed(lean_object* v_e_157_, lean_object* v___y_158_, lean_object* v___y_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_e_157_, v___y_158_);
lean_dec(v___y_158_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1(lean_object* v_e_161_, lean_object* v___y_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_e_161_, v___y_163_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___boxed(lean_object* v_e_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1(v_e_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(lean_object* v_k_175_, uint8_t v_allowLevelAssignments_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v___x_182_; 
v___x_182_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_176_, v_k_175_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
if (lean_obj_tag(v___x_182_) == 0)
{
lean_object* v_a_183_; lean_object* v___x_185_; uint8_t v_isShared_186_; uint8_t v_isSharedCheck_190_; 
v_a_183_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_190_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_190_ == 0)
{
v___x_185_ = v___x_182_;
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
else
{
lean_inc(v_a_183_);
lean_dec(v___x_182_);
v___x_185_ = lean_box(0);
v_isShared_186_ = v_isSharedCheck_190_;
goto v_resetjp_184_;
}
v_resetjp_184_:
{
lean_object* v___x_188_; 
if (v_isShared_186_ == 0)
{
v___x_188_ = v___x_185_;
goto v_reusejp_187_;
}
else
{
lean_object* v_reuseFailAlloc_189_; 
v_reuseFailAlloc_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_189_, 0, v_a_183_);
v___x_188_ = v_reuseFailAlloc_189_;
goto v_reusejp_187_;
}
v_reusejp_187_:
{
return v___x_188_;
}
}
}
else
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_198_; 
v_a_191_ = lean_ctor_get(v___x_182_, 0);
v_isSharedCheck_198_ = !lean_is_exclusive(v___x_182_);
if (v_isSharedCheck_198_ == 0)
{
v___x_193_ = v___x_182_;
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_182_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_198_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_197_; 
v_reuseFailAlloc_197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_197_, 0, v_a_191_);
v___x_196_ = v_reuseFailAlloc_197_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
return v___x_196_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg___boxed(lean_object* v_k_199_, lean_object* v_allowLevelAssignments_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_206_; lean_object* v_res_207_; 
v_allowLevelAssignments_boxed_206_ = lean_unbox(v_allowLevelAssignments_200_);
v_res_207_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v_k_199_, v_allowLevelAssignments_boxed_206_, v___y_201_, v___y_202_, v___y_203_, v___y_204_);
lean_dec(v___y_204_);
lean_dec_ref(v___y_203_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2(lean_object* v_00_u03b1_208_, lean_object* v_k_209_, uint8_t v_allowLevelAssignments_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; 
v___x_216_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v_k_209_, v_allowLevelAssignments_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
return v___x_216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___boxed(lean_object* v_00_u03b1_217_, lean_object* v_k_218_, lean_object* v_allowLevelAssignments_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_225_; lean_object* v_res_226_; 
v_allowLevelAssignments_boxed_225_ = lean_unbox(v_allowLevelAssignments_219_);
v_res_226_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2(v_00_u03b1_217_, v_k_218_, v_allowLevelAssignments_boxed_225_, v___y_220_, v___y_221_, v___y_222_, v___y_223_);
lean_dec(v___y_223_);
lean_dec_ref(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg(lean_object* v_mvarId_227_, lean_object* v_x_228_, lean_object* v___y_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_){
_start:
{
lean_object* v___x_234_; 
v___x_234_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_227_, v_x_228_, v___y_229_, v___y_230_, v___y_231_, v___y_232_);
if (lean_obj_tag(v___x_234_) == 0)
{
lean_object* v_a_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_242_; 
v_a_235_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_242_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_242_ == 0)
{
v___x_237_ = v___x_234_;
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_a_235_);
lean_dec(v___x_234_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_242_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
lean_object* v___x_240_; 
if (v_isShared_238_ == 0)
{
v___x_240_ = v___x_237_;
goto v_reusejp_239_;
}
else
{
lean_object* v_reuseFailAlloc_241_; 
v_reuseFailAlloc_241_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_241_, 0, v_a_235_);
v___x_240_ = v_reuseFailAlloc_241_;
goto v_reusejp_239_;
}
v_reusejp_239_:
{
return v___x_240_;
}
}
}
else
{
lean_object* v_a_243_; lean_object* v___x_245_; uint8_t v_isShared_246_; uint8_t v_isSharedCheck_250_; 
v_a_243_ = lean_ctor_get(v___x_234_, 0);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_234_);
if (v_isSharedCheck_250_ == 0)
{
v___x_245_ = v___x_234_;
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
else
{
lean_inc(v_a_243_);
lean_dec(v___x_234_);
v___x_245_ = lean_box(0);
v_isShared_246_ = v_isSharedCheck_250_;
goto v_resetjp_244_;
}
v_resetjp_244_:
{
lean_object* v___x_248_; 
if (v_isShared_246_ == 0)
{
v___x_248_ = v___x_245_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v_a_243_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg___boxed(lean_object* v_mvarId_251_, lean_object* v_x_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg(v_mvarId_251_, v_x_252_, v___y_253_, v___y_254_, v___y_255_, v___y_256_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
lean_dec(v___y_254_);
lean_dec_ref(v___y_253_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4(lean_object* v_00_u03b1_259_, lean_object* v_mvarId_260_, lean_object* v_x_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v___x_267_; 
v___x_267_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg(v_mvarId_260_, v_x_261_, v___y_262_, v___y_263_, v___y_264_, v___y_265_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___boxed(lean_object* v_00_u03b1_268_, lean_object* v_mvarId_269_, lean_object* v_x_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4(v_00_u03b1_268_, v_mvarId_269_, v_x_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_);
lean_dec(v___y_274_);
lean_dec_ref(v___y_273_);
lean_dec(v___y_272_);
lean_dec_ref(v___y_271_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0(lean_object* v_p_277_, lean_object* v_a_278_, lean_object* v_g_279_, lean_object* v_fvarId_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_){
_start:
{
lean_object* v___x_286_; 
lean_inc(v___y_284_);
lean_inc_ref(v___y_283_);
lean_inc(v___y_282_);
lean_inc_ref(v___y_281_);
lean_inc_ref(v_p_277_);
v___x_286_ = lean_infer_type(v_p_277_, v___y_281_, v___y_282_, v___y_283_, v___y_284_);
if (lean_obj_tag(v___x_286_) == 0)
{
lean_object* v_a_287_; lean_object* v___x_288_; lean_object* v___x_289_; 
v_a_287_ = lean_ctor_get(v___x_286_, 0);
lean_inc(v_a_287_);
lean_dec_ref_known(v___x_286_, 1);
v___x_288_ = l_Lean_LocalDecl_userName(v_a_278_);
lean_inc(v_fvarId_280_);
v___x_289_ = l_Lean_MVarId_assertAfter(v_g_279_, v_fvarId_280_, v___x_288_, v_a_287_, v_p_277_, v___y_281_, v___y_282_, v___y_283_, v___y_284_);
if (lean_obj_tag(v___x_289_) == 0)
{
lean_object* v_a_290_; lean_object* v_fvarId_291_; lean_object* v_mvarId_292_; lean_object* v_subst_293_; lean_object* v___x_295_; uint8_t v_isShared_296_; uint8_t v_isSharedCheck_317_; 
v_a_290_ = lean_ctor_get(v___x_289_, 0);
lean_inc(v_a_290_);
lean_dec_ref_known(v___x_289_, 1);
v_fvarId_291_ = lean_ctor_get(v_a_290_, 0);
v_mvarId_292_ = lean_ctor_get(v_a_290_, 1);
v_subst_293_ = lean_ctor_get(v_a_290_, 2);
v_isSharedCheck_317_ = !lean_is_exclusive(v_a_290_);
if (v_isSharedCheck_317_ == 0)
{
v___x_295_ = v_a_290_;
v_isShared_296_ = v_isSharedCheck_317_;
goto v_resetjp_294_;
}
else
{
lean_inc(v_subst_293_);
lean_inc(v_mvarId_292_);
lean_inc(v_fvarId_291_);
lean_dec(v_a_290_);
v___x_295_ = lean_box(0);
v_isShared_296_ = v_isSharedCheck_317_;
goto v_resetjp_294_;
}
v_resetjp_294_:
{
lean_object* v___x_297_; 
v___x_297_ = l_Lean_MVarId_clear(v_mvarId_292_, v_fvarId_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v_a_298_; lean_object* v___x_300_; uint8_t v_isShared_301_; uint8_t v_isSharedCheck_308_; 
v_a_298_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_308_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_308_ == 0)
{
v___x_300_ = v___x_297_;
v_isShared_301_ = v_isSharedCheck_308_;
goto v_resetjp_299_;
}
else
{
lean_inc(v_a_298_);
lean_dec(v___x_297_);
v___x_300_ = lean_box(0);
v_isShared_301_ = v_isSharedCheck_308_;
goto v_resetjp_299_;
}
v_resetjp_299_:
{
lean_object* v___x_303_; 
if (v_isShared_296_ == 0)
{
lean_ctor_set(v___x_295_, 1, v_a_298_);
v___x_303_ = v___x_295_;
goto v_reusejp_302_;
}
else
{
lean_object* v_reuseFailAlloc_307_; 
v_reuseFailAlloc_307_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_307_, 0, v_fvarId_291_);
lean_ctor_set(v_reuseFailAlloc_307_, 1, v_a_298_);
lean_ctor_set(v_reuseFailAlloc_307_, 2, v_subst_293_);
v___x_303_ = v_reuseFailAlloc_307_;
goto v_reusejp_302_;
}
v_reusejp_302_:
{
lean_object* v___x_305_; 
if (v_isShared_301_ == 0)
{
lean_ctor_set(v___x_300_, 0, v___x_303_);
v___x_305_ = v___x_300_;
goto v_reusejp_304_;
}
else
{
lean_object* v_reuseFailAlloc_306_; 
v_reuseFailAlloc_306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_306_, 0, v___x_303_);
v___x_305_ = v_reuseFailAlloc_306_;
goto v_reusejp_304_;
}
v_reusejp_304_:
{
return v___x_305_;
}
}
}
}
else
{
lean_object* v_a_309_; lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_316_; 
lean_del_object(v___x_295_);
lean_dec(v_subst_293_);
lean_dec(v_fvarId_291_);
v_a_309_ = lean_ctor_get(v___x_297_, 0);
v_isSharedCheck_316_ = !lean_is_exclusive(v___x_297_);
if (v_isSharedCheck_316_ == 0)
{
v___x_311_ = v___x_297_;
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
else
{
lean_inc(v_a_309_);
lean_dec(v___x_297_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_316_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_314_; 
if (v_isShared_312_ == 0)
{
v___x_314_ = v___x_311_;
goto v_reusejp_313_;
}
else
{
lean_object* v_reuseFailAlloc_315_; 
v_reuseFailAlloc_315_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_315_, 0, v_a_309_);
v___x_314_ = v_reuseFailAlloc_315_;
goto v_reusejp_313_;
}
v_reusejp_313_:
{
return v___x_314_;
}
}
}
}
}
else
{
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
lean_dec(v_fvarId_280_);
return v___x_289_;
}
}
else
{
lean_object* v_a_318_; lean_object* v___x_320_; uint8_t v_isShared_321_; uint8_t v_isSharedCheck_325_; 
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
lean_dec(v_fvarId_280_);
lean_dec(v_g_279_);
lean_dec_ref(v_p_277_);
v_a_318_ = lean_ctor_get(v___x_286_, 0);
v_isSharedCheck_325_ = !lean_is_exclusive(v___x_286_);
if (v_isSharedCheck_325_ == 0)
{
v___x_320_ = v___x_286_;
v_isShared_321_ = v_isSharedCheck_325_;
goto v_resetjp_319_;
}
else
{
lean_inc(v_a_318_);
lean_dec(v___x_286_);
v___x_320_ = lean_box(0);
v_isShared_321_ = v_isSharedCheck_325_;
goto v_resetjp_319_;
}
v_resetjp_319_:
{
lean_object* v___x_323_; 
if (v_isShared_321_ == 0)
{
v___x_323_ = v___x_320_;
goto v_reusejp_322_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v_a_318_);
v___x_323_ = v_reuseFailAlloc_324_;
goto v_reusejp_322_;
}
v_reusejp_322_:
{
return v___x_323_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0___boxed(lean_object* v_p_326_, lean_object* v_a_327_, lean_object* v_g_328_, lean_object* v_fvarId_329_, lean_object* v___y_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_){
_start:
{
lean_object* v_res_335_; 
v_res_335_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0(v_p_326_, v_a_327_, v_g_328_, v_fvarId_329_, v___y_330_, v___y_331_, v___y_332_, v___y_333_);
lean_dec_ref(v_a_327_);
return v_res_335_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2(void){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_339_ = lean_box(0);
v___x_340_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__1));
v___x_341_ = l_Lean_Expr_const___override(v___x_340_, v___x_339_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1(lean_object* v___x_345_, uint8_t v___x_346_, lean_object* v___x_347_, lean_object* v___x_348_, lean_object* v___x_349_, lean_object* v_a_350_, uint8_t v___x_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_){
_start:
{
lean_object* v___x_357_; 
lean_inc(v___x_347_);
lean_inc(v___x_345_);
v___x_357_ = l_Lean_Meta_mkFreshExprMVar(v___x_345_, v___x_346_, v___x_347_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
if (lean_obj_tag(v___x_357_) == 0)
{
lean_object* v_a_358_; lean_object* v___x_359_; 
v_a_358_ = lean_ctor_get(v___x_357_, 0);
lean_inc(v_a_358_);
lean_dec_ref_known(v___x_357_, 1);
v___x_359_ = l_Lean_Meta_mkFreshExprMVar(v___x_345_, v___x_346_, v___x_347_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
if (lean_obj_tag(v___x_359_) == 0)
{
lean_object* v_a_360_; lean_object* v_keyedConfig_361_; uint8_t v_trackZetaDelta_362_; lean_object* v_zetaDeltaSet_363_; lean_object* v_lctx_364_; lean_object* v_localInstances_365_; lean_object* v_defEqCtx_x3f_366_; lean_object* v_synthPendingDepth_367_; lean_object* v_customCanUnfoldPredicate_x3f_368_; uint8_t v_univApprox_369_; uint8_t v_inTypeClassResolution_370_; uint8_t v_cacheInferType_371_; lean_object* v___x_373_; uint8_t v_isShared_374_; uint8_t v_isSharedCheck_424_; 
v_a_360_ = lean_ctor_get(v___x_359_, 0);
lean_inc(v_a_360_);
lean_dec_ref_known(v___x_359_, 1);
v_keyedConfig_361_ = lean_ctor_get(v___y_352_, 0);
v_trackZetaDelta_362_ = lean_ctor_get_uint8(v___y_352_, sizeof(void*)*7);
v_zetaDeltaSet_363_ = lean_ctor_get(v___y_352_, 1);
v_lctx_364_ = lean_ctor_get(v___y_352_, 2);
v_localInstances_365_ = lean_ctor_get(v___y_352_, 3);
v_defEqCtx_x3f_366_ = lean_ctor_get(v___y_352_, 4);
v_synthPendingDepth_367_ = lean_ctor_get(v___y_352_, 5);
v_customCanUnfoldPredicate_x3f_368_ = lean_ctor_get(v___y_352_, 6);
v_univApprox_369_ = lean_ctor_get_uint8(v___y_352_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_370_ = lean_ctor_get_uint8(v___y_352_, sizeof(void*)*7 + 2);
v_cacheInferType_371_ = lean_ctor_get_uint8(v___y_352_, sizeof(void*)*7 + 3);
v_isSharedCheck_424_ = !lean_is_exclusive(v___y_352_);
if (v_isSharedCheck_424_ == 0)
{
v___x_373_ = v___y_352_;
v_isShared_374_ = v_isSharedCheck_424_;
goto v_resetjp_372_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_368_);
lean_inc(v_synthPendingDepth_367_);
lean_inc(v_defEqCtx_x3f_366_);
lean_inc(v_localInstances_365_);
lean_inc(v_lctx_364_);
lean_inc(v_zetaDeltaSet_363_);
lean_inc(v_keyedConfig_361_);
lean_dec(v___y_352_);
v___x_373_ = lean_box(0);
v_isShared_374_ = v_isSharedCheck_424_;
goto v_resetjp_372_;
}
v_resetjp_372_:
{
lean_object* v___x_375_; lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___x_383_; lean_object* v___x_384_; uint8_t v___x_385_; lean_object* v___x_386_; lean_object* v___x_388_; 
v___x_375_ = lean_box(0);
v___x_376_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_377_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4));
v___x_378_ = l_Lean_Level_succ___override(v___x_348_);
v___x_379_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_378_);
lean_ctor_set(v___x_379_, 1, v___x_375_);
v___x_380_ = l_Lean_Expr_const___override(v___x_377_, v___x_379_);
v___x_381_ = l_Lean_Expr_app___override(v___x_380_, v___x_349_);
lean_inc(v_a_358_);
v___x_382_ = l_Lean_Expr_app___override(v___x_381_, v_a_358_);
lean_inc(v_a_360_);
v___x_383_ = l_Lean_Expr_app___override(v___x_382_, v_a_360_);
v___x_384_ = l_Lean_Expr_app___override(v___x_376_, v___x_383_);
v___x_385_ = 2;
v___x_386_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_385_, v_keyedConfig_361_);
if (v_isShared_374_ == 0)
{
lean_ctor_set(v___x_373_, 0, v___x_386_);
v___x_388_ = v___x_373_;
goto v_reusejp_387_;
}
else
{
lean_object* v_reuseFailAlloc_423_; 
v_reuseFailAlloc_423_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_423_, 0, v___x_386_);
lean_ctor_set(v_reuseFailAlloc_423_, 1, v_zetaDeltaSet_363_);
lean_ctor_set(v_reuseFailAlloc_423_, 2, v_lctx_364_);
lean_ctor_set(v_reuseFailAlloc_423_, 3, v_localInstances_365_);
lean_ctor_set(v_reuseFailAlloc_423_, 4, v_defEqCtx_x3f_366_);
lean_ctor_set(v_reuseFailAlloc_423_, 5, v_synthPendingDepth_367_);
lean_ctor_set(v_reuseFailAlloc_423_, 6, v_customCanUnfoldPredicate_x3f_368_);
lean_ctor_set_uint8(v_reuseFailAlloc_423_, sizeof(void*)*7, v_trackZetaDelta_362_);
lean_ctor_set_uint8(v_reuseFailAlloc_423_, sizeof(void*)*7 + 1, v_univApprox_369_);
lean_ctor_set_uint8(v_reuseFailAlloc_423_, sizeof(void*)*7 + 2, v_inTypeClassResolution_370_);
lean_ctor_set_uint8(v_reuseFailAlloc_423_, sizeof(void*)*7 + 3, v_cacheInferType_371_);
v___x_388_ = v_reuseFailAlloc_423_;
goto v_reusejp_387_;
}
v_reusejp_387_:
{
lean_object* v___x_389_; 
v___x_389_ = l_Lean_Meta_isExprDefEq(v___x_384_, v_a_350_, v___x_388_, v___y_353_, v___y_354_, v___y_355_);
lean_dec_ref(v___x_388_);
if (lean_obj_tag(v___x_389_) == 0)
{
lean_object* v_a_390_; lean_object* v___x_392_; uint8_t v_isShared_393_; uint8_t v_isSharedCheck_414_; 
v_a_390_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_414_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_414_ == 0)
{
v___x_392_ = v___x_389_;
v_isShared_393_ = v_isSharedCheck_414_;
goto v_resetjp_391_;
}
else
{
lean_inc(v_a_390_);
lean_dec(v___x_389_);
v___x_392_ = lean_box(0);
v_isShared_393_ = v_isSharedCheck_414_;
goto v_resetjp_391_;
}
v_resetjp_391_:
{
uint8_t v___x_394_; 
v___x_394_ = lean_unbox(v_a_390_);
if (v___x_394_ == 0)
{
lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_399_; 
lean_dec(v_a_390_);
v___x_395_ = lean_box(v___x_351_);
v___x_396_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_396_, 0, v_a_360_);
lean_ctor_set(v___x_396_, 1, v___x_395_);
v___x_397_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_397_, 0, v_a_358_);
lean_ctor_set(v___x_397_, 1, v___x_396_);
if (v_isShared_393_ == 0)
{
lean_ctor_set(v___x_392_, 0, v___x_397_);
v___x_399_ = v___x_392_;
goto v_reusejp_398_;
}
else
{
lean_object* v_reuseFailAlloc_400_; 
v_reuseFailAlloc_400_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_400_, 0, v___x_397_);
v___x_399_ = v_reuseFailAlloc_400_;
goto v_reusejp_398_;
}
v_reusejp_398_:
{
return v___x_399_;
}
}
else
{
lean_object* v___x_401_; lean_object* v_a_402_; lean_object* v___x_403_; lean_object* v_a_404_; lean_object* v___x_406_; uint8_t v_isShared_407_; uint8_t v_isSharedCheck_413_; 
lean_del_object(v___x_392_);
v___x_401_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_358_, v___y_353_);
v_a_402_ = lean_ctor_get(v___x_401_, 0);
lean_inc(v_a_402_);
lean_dec_ref(v___x_401_);
v___x_403_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_360_, v___y_353_);
v_a_404_ = lean_ctor_get(v___x_403_, 0);
v_isSharedCheck_413_ = !lean_is_exclusive(v___x_403_);
if (v_isSharedCheck_413_ == 0)
{
v___x_406_ = v___x_403_;
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
else
{
lean_inc(v_a_404_);
lean_dec(v___x_403_);
v___x_406_ = lean_box(0);
v_isShared_407_ = v_isSharedCheck_413_;
goto v_resetjp_405_;
}
v_resetjp_405_:
{
lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_411_; 
v___x_408_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_408_, 0, v_a_404_);
lean_ctor_set(v___x_408_, 1, v_a_390_);
v___x_409_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_409_, 0, v_a_402_);
lean_ctor_set(v___x_409_, 1, v___x_408_);
if (v_isShared_407_ == 0)
{
lean_ctor_set(v___x_406_, 0, v___x_409_);
v___x_411_ = v___x_406_;
goto v_reusejp_410_;
}
else
{
lean_object* v_reuseFailAlloc_412_; 
v_reuseFailAlloc_412_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_412_, 0, v___x_409_);
v___x_411_ = v_reuseFailAlloc_412_;
goto v_reusejp_410_;
}
v_reusejp_410_:
{
return v___x_411_;
}
}
}
}
}
else
{
lean_object* v_a_415_; lean_object* v___x_417_; uint8_t v_isShared_418_; uint8_t v_isSharedCheck_422_; 
lean_dec(v_a_360_);
lean_dec(v_a_358_);
v_a_415_ = lean_ctor_get(v___x_389_, 0);
v_isSharedCheck_422_ = !lean_is_exclusive(v___x_389_);
if (v_isSharedCheck_422_ == 0)
{
v___x_417_ = v___x_389_;
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
else
{
lean_inc(v_a_415_);
lean_dec(v___x_389_);
v___x_417_ = lean_box(0);
v_isShared_418_ = v_isSharedCheck_422_;
goto v_resetjp_416_;
}
v_resetjp_416_:
{
lean_object* v___x_420_; 
if (v_isShared_418_ == 0)
{
v___x_420_ = v___x_417_;
goto v_reusejp_419_;
}
else
{
lean_object* v_reuseFailAlloc_421_; 
v_reuseFailAlloc_421_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_421_, 0, v_a_415_);
v___x_420_ = v_reuseFailAlloc_421_;
goto v_reusejp_419_;
}
v_reusejp_419_:
{
return v___x_420_;
}
}
}
}
}
}
else
{
lean_object* v_a_425_; lean_object* v___x_427_; uint8_t v_isShared_428_; uint8_t v_isSharedCheck_432_; 
lean_dec(v_a_358_);
lean_dec_ref(v___y_352_);
lean_dec_ref(v_a_350_);
lean_dec_ref(v___x_349_);
lean_dec(v___x_348_);
v_a_425_ = lean_ctor_get(v___x_359_, 0);
v_isSharedCheck_432_ = !lean_is_exclusive(v___x_359_);
if (v_isSharedCheck_432_ == 0)
{
v___x_427_ = v___x_359_;
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
else
{
lean_inc(v_a_425_);
lean_dec(v___x_359_);
v___x_427_ = lean_box(0);
v_isShared_428_ = v_isSharedCheck_432_;
goto v_resetjp_426_;
}
v_resetjp_426_:
{
lean_object* v___x_430_; 
if (v_isShared_428_ == 0)
{
v___x_430_ = v___x_427_;
goto v_reusejp_429_;
}
else
{
lean_object* v_reuseFailAlloc_431_; 
v_reuseFailAlloc_431_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_431_, 0, v_a_425_);
v___x_430_ = v_reuseFailAlloc_431_;
goto v_reusejp_429_;
}
v_reusejp_429_:
{
return v___x_430_;
}
}
}
}
else
{
lean_object* v_a_433_; lean_object* v___x_435_; uint8_t v_isShared_436_; uint8_t v_isSharedCheck_440_; 
lean_dec_ref(v___y_352_);
lean_dec_ref(v_a_350_);
lean_dec_ref(v___x_349_);
lean_dec(v___x_348_);
lean_dec(v___x_347_);
lean_dec(v___x_345_);
v_a_433_ = lean_ctor_get(v___x_357_, 0);
v_isSharedCheck_440_ = !lean_is_exclusive(v___x_357_);
if (v_isSharedCheck_440_ == 0)
{
v___x_435_ = v___x_357_;
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
else
{
lean_inc(v_a_433_);
lean_dec(v___x_357_);
v___x_435_ = lean_box(0);
v_isShared_436_ = v_isSharedCheck_440_;
goto v_resetjp_434_;
}
v_resetjp_434_:
{
lean_object* v___x_438_; 
if (v_isShared_436_ == 0)
{
v___x_438_ = v___x_435_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_a_433_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___boxed(lean_object* v___x_441_, lean_object* v___x_442_, lean_object* v___x_443_, lean_object* v___x_444_, lean_object* v___x_445_, lean_object* v_a_446_, lean_object* v___x_447_, lean_object* v___y_448_, lean_object* v___y_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_){
_start:
{
uint8_t v___x_36714__boxed_453_; uint8_t v___x_36719__boxed_454_; lean_object* v_res_455_; 
v___x_36714__boxed_453_ = lean_unbox(v___x_442_);
v___x_36719__boxed_454_ = lean_unbox(v___x_447_);
v_res_455_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1(v___x_441_, v___x_36714__boxed_453_, v___x_443_, v___x_444_, v___x_445_, v_a_446_, v___x_36719__boxed_454_, v___y_448_, v___y_449_, v___y_450_, v___y_451_);
lean_dec(v___y_451_);
lean_dec_ref(v___y_450_);
lean_dec(v___y_449_);
return v_res_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2(lean_object* v___x_456_, uint8_t v___x_457_, lean_object* v___x_458_, lean_object* v___x_459_, lean_object* v___x_460_, lean_object* v_a_461_, uint8_t v___x_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_){
_start:
{
lean_object* v___x_468_; 
lean_inc(v___x_458_);
lean_inc(v___x_456_);
v___x_468_ = l_Lean_Meta_mkFreshExprMVar(v___x_456_, v___x_457_, v___x_458_, v___y_463_, v___y_464_, v___y_465_, v___y_466_);
if (lean_obj_tag(v___x_468_) == 0)
{
lean_object* v_a_469_; lean_object* v___x_470_; 
v_a_469_ = lean_ctor_get(v___x_468_, 0);
lean_inc(v_a_469_);
lean_dec_ref_known(v___x_468_, 1);
v___x_470_ = l_Lean_Meta_mkFreshExprMVar(v___x_456_, v___x_457_, v___x_458_, v___y_463_, v___y_464_, v___y_465_, v___y_466_);
if (lean_obj_tag(v___x_470_) == 0)
{
lean_object* v_a_471_; lean_object* v_keyedConfig_472_; uint8_t v_trackZetaDelta_473_; lean_object* v_zetaDeltaSet_474_; lean_object* v_lctx_475_; lean_object* v_localInstances_476_; lean_object* v_defEqCtx_x3f_477_; lean_object* v_synthPendingDepth_478_; lean_object* v_customCanUnfoldPredicate_x3f_479_; uint8_t v_univApprox_480_; uint8_t v_inTypeClassResolution_481_; uint8_t v_cacheInferType_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_533_; 
v_a_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_a_471_);
lean_dec_ref_known(v___x_470_, 1);
v_keyedConfig_472_ = lean_ctor_get(v___y_463_, 0);
v_trackZetaDelta_473_ = lean_ctor_get_uint8(v___y_463_, sizeof(void*)*7);
v_zetaDeltaSet_474_ = lean_ctor_get(v___y_463_, 1);
v_lctx_475_ = lean_ctor_get(v___y_463_, 2);
v_localInstances_476_ = lean_ctor_get(v___y_463_, 3);
v_defEqCtx_x3f_477_ = lean_ctor_get(v___y_463_, 4);
v_synthPendingDepth_478_ = lean_ctor_get(v___y_463_, 5);
v_customCanUnfoldPredicate_x3f_479_ = lean_ctor_get(v___y_463_, 6);
v_univApprox_480_ = lean_ctor_get_uint8(v___y_463_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_481_ = lean_ctor_get_uint8(v___y_463_, sizeof(void*)*7 + 2);
v_cacheInferType_482_ = lean_ctor_get_uint8(v___y_463_, sizeof(void*)*7 + 3);
v_isSharedCheck_533_ = !lean_is_exclusive(v___y_463_);
if (v_isSharedCheck_533_ == 0)
{
v___x_484_ = v___y_463_;
v_isShared_485_ = v_isSharedCheck_533_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_479_);
lean_inc(v_synthPendingDepth_478_);
lean_inc(v_defEqCtx_x3f_477_);
lean_inc(v_localInstances_476_);
lean_inc(v_lctx_475_);
lean_inc(v_zetaDeltaSet_474_);
lean_inc(v_keyedConfig_472_);
lean_dec(v___y_463_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_533_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; uint8_t v___x_494_; lean_object* v___x_495_; lean_object* v___x_497_; 
v___x_486_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4));
v___x_487_ = l_Lean_Level_succ___override(v___x_459_);
v___x_488_ = lean_box(0);
v___x_489_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_487_);
lean_ctor_set(v___x_489_, 1, v___x_488_);
v___x_490_ = l_Lean_Expr_const___override(v___x_486_, v___x_489_);
v___x_491_ = l_Lean_Expr_app___override(v___x_490_, v___x_460_);
lean_inc(v_a_469_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_a_469_);
lean_inc(v_a_471_);
v___x_493_ = l_Lean_Expr_app___override(v___x_492_, v_a_471_);
v___x_494_ = 2;
v___x_495_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_494_, v_keyedConfig_472_);
if (v_isShared_485_ == 0)
{
lean_ctor_set(v___x_484_, 0, v___x_495_);
v___x_497_ = v___x_484_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_532_; 
v_reuseFailAlloc_532_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_532_, 0, v___x_495_);
lean_ctor_set(v_reuseFailAlloc_532_, 1, v_zetaDeltaSet_474_);
lean_ctor_set(v_reuseFailAlloc_532_, 2, v_lctx_475_);
lean_ctor_set(v_reuseFailAlloc_532_, 3, v_localInstances_476_);
lean_ctor_set(v_reuseFailAlloc_532_, 4, v_defEqCtx_x3f_477_);
lean_ctor_set(v_reuseFailAlloc_532_, 5, v_synthPendingDepth_478_);
lean_ctor_set(v_reuseFailAlloc_532_, 6, v_customCanUnfoldPredicate_x3f_479_);
lean_ctor_set_uint8(v_reuseFailAlloc_532_, sizeof(void*)*7, v_trackZetaDelta_473_);
lean_ctor_set_uint8(v_reuseFailAlloc_532_, sizeof(void*)*7 + 1, v_univApprox_480_);
lean_ctor_set_uint8(v_reuseFailAlloc_532_, sizeof(void*)*7 + 2, v_inTypeClassResolution_481_);
lean_ctor_set_uint8(v_reuseFailAlloc_532_, sizeof(void*)*7 + 3, v_cacheInferType_482_);
v___x_497_ = v_reuseFailAlloc_532_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
lean_object* v___x_498_; 
v___x_498_ = l_Lean_Meta_isExprDefEq(v___x_493_, v_a_461_, v___x_497_, v___y_464_, v___y_465_, v___y_466_);
lean_dec_ref(v___x_497_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_523_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_523_ == 0)
{
v___x_501_ = v___x_498_;
v_isShared_502_ = v_isSharedCheck_523_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_a_499_);
lean_dec(v___x_498_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_523_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
uint8_t v___x_503_; 
v___x_503_ = lean_unbox(v_a_499_);
if (v___x_503_ == 0)
{
lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_508_; 
lean_dec(v_a_499_);
v___x_504_ = lean_box(v___x_462_);
v___x_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_505_, 0, v_a_471_);
lean_ctor_set(v___x_505_, 1, v___x_504_);
v___x_506_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_506_, 0, v_a_469_);
lean_ctor_set(v___x_506_, 1, v___x_505_);
if (v_isShared_502_ == 0)
{
lean_ctor_set(v___x_501_, 0, v___x_506_);
v___x_508_ = v___x_501_;
goto v_reusejp_507_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_506_);
v___x_508_ = v_reuseFailAlloc_509_;
goto v_reusejp_507_;
}
v_reusejp_507_:
{
return v___x_508_;
}
}
else
{
lean_object* v___x_510_; lean_object* v_a_511_; lean_object* v___x_512_; lean_object* v_a_513_; lean_object* v___x_515_; uint8_t v_isShared_516_; uint8_t v_isSharedCheck_522_; 
lean_del_object(v___x_501_);
v___x_510_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_469_, v___y_464_);
v_a_511_ = lean_ctor_get(v___x_510_, 0);
lean_inc(v_a_511_);
lean_dec_ref(v___x_510_);
v___x_512_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_471_, v___y_464_);
v_a_513_ = lean_ctor_get(v___x_512_, 0);
v_isSharedCheck_522_ = !lean_is_exclusive(v___x_512_);
if (v_isSharedCheck_522_ == 0)
{
v___x_515_ = v___x_512_;
v_isShared_516_ = v_isSharedCheck_522_;
goto v_resetjp_514_;
}
else
{
lean_inc(v_a_513_);
lean_dec(v___x_512_);
v___x_515_ = lean_box(0);
v_isShared_516_ = v_isSharedCheck_522_;
goto v_resetjp_514_;
}
v_resetjp_514_:
{
lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v___x_520_; 
v___x_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_517_, 0, v_a_513_);
lean_ctor_set(v___x_517_, 1, v_a_499_);
v___x_518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_518_, 0, v_a_511_);
lean_ctor_set(v___x_518_, 1, v___x_517_);
if (v_isShared_516_ == 0)
{
lean_ctor_set(v___x_515_, 0, v___x_518_);
v___x_520_ = v___x_515_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v___x_518_);
v___x_520_ = v_reuseFailAlloc_521_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
return v___x_520_;
}
}
}
}
}
else
{
lean_object* v_a_524_; lean_object* v___x_526_; uint8_t v_isShared_527_; uint8_t v_isSharedCheck_531_; 
lean_dec(v_a_471_);
lean_dec(v_a_469_);
v_a_524_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_531_ == 0)
{
v___x_526_ = v___x_498_;
v_isShared_527_ = v_isSharedCheck_531_;
goto v_resetjp_525_;
}
else
{
lean_inc(v_a_524_);
lean_dec(v___x_498_);
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
else
{
lean_object* v_a_534_; lean_object* v___x_536_; uint8_t v_isShared_537_; uint8_t v_isSharedCheck_541_; 
lean_dec(v_a_469_);
lean_dec_ref(v___y_463_);
lean_dec_ref(v_a_461_);
lean_dec_ref(v___x_460_);
lean_dec(v___x_459_);
v_a_534_ = lean_ctor_get(v___x_470_, 0);
v_isSharedCheck_541_ = !lean_is_exclusive(v___x_470_);
if (v_isSharedCheck_541_ == 0)
{
v___x_536_ = v___x_470_;
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
else
{
lean_inc(v_a_534_);
lean_dec(v___x_470_);
v___x_536_ = lean_box(0);
v_isShared_537_ = v_isSharedCheck_541_;
goto v_resetjp_535_;
}
v_resetjp_535_:
{
lean_object* v___x_539_; 
if (v_isShared_537_ == 0)
{
v___x_539_ = v___x_536_;
goto v_reusejp_538_;
}
else
{
lean_object* v_reuseFailAlloc_540_; 
v_reuseFailAlloc_540_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_540_, 0, v_a_534_);
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
else
{
lean_object* v_a_542_; lean_object* v___x_544_; uint8_t v_isShared_545_; uint8_t v_isSharedCheck_549_; 
lean_dec_ref(v___y_463_);
lean_dec_ref(v_a_461_);
lean_dec_ref(v___x_460_);
lean_dec(v___x_459_);
lean_dec(v___x_458_);
lean_dec(v___x_456_);
v_a_542_ = lean_ctor_get(v___x_468_, 0);
v_isSharedCheck_549_ = !lean_is_exclusive(v___x_468_);
if (v_isSharedCheck_549_ == 0)
{
v___x_544_ = v___x_468_;
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
else
{
lean_inc(v_a_542_);
lean_dec(v___x_468_);
v___x_544_ = lean_box(0);
v_isShared_545_ = v_isSharedCheck_549_;
goto v_resetjp_543_;
}
v_resetjp_543_:
{
lean_object* v___x_547_; 
if (v_isShared_545_ == 0)
{
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
return v___x_547_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2___boxed(lean_object* v___x_550_, lean_object* v___x_551_, lean_object* v___x_552_, lean_object* v___x_553_, lean_object* v___x_554_, lean_object* v_a_555_, lean_object* v___x_556_, lean_object* v___y_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_){
_start:
{
uint8_t v___x_36910__boxed_562_; uint8_t v___x_36915__boxed_563_; lean_object* v_res_564_; 
v___x_36910__boxed_562_ = lean_unbox(v___x_551_);
v___x_36915__boxed_563_ = lean_unbox(v___x_556_);
v_res_564_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2(v___x_550_, v___x_36910__boxed_562_, v___x_552_, v___x_553_, v___x_554_, v_a_555_, v___x_36915__boxed_563_, v___y_557_, v___y_558_, v___y_559_, v___y_560_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
lean_dec(v___y_558_);
return v_res_564_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2(void){
_start:
{
lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; 
v___x_568_ = lean_box(0);
v___x_569_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__1));
v___x_570_ = l_Lean_Expr_const___override(v___x_569_, v___x_568_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3(lean_object* v___x_571_, uint8_t v___x_572_, lean_object* v___x_573_, lean_object* v_a_574_, uint8_t v___x_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_){
_start:
{
lean_object* v___x_581_; 
lean_inc(v___x_573_);
lean_inc(v___x_571_);
v___x_581_ = l_Lean_Meta_mkFreshExprMVar(v___x_571_, v___x_572_, v___x_573_, v___y_576_, v___y_577_, v___y_578_, v___y_579_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v_a_582_; lean_object* v___x_583_; 
v_a_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_a_582_);
lean_dec_ref_known(v___x_581_, 1);
v___x_583_ = l_Lean_Meta_mkFreshExprMVar(v___x_571_, v___x_572_, v___x_573_, v___y_576_, v___y_577_, v___y_578_, v___y_579_);
if (lean_obj_tag(v___x_583_) == 0)
{
lean_object* v_a_584_; lean_object* v_keyedConfig_585_; uint8_t v_trackZetaDelta_586_; lean_object* v_zetaDeltaSet_587_; lean_object* v_lctx_588_; lean_object* v_localInstances_589_; lean_object* v_defEqCtx_x3f_590_; lean_object* v_synthPendingDepth_591_; lean_object* v_customCanUnfoldPredicate_x3f_592_; uint8_t v_univApprox_593_; uint8_t v_inTypeClassResolution_594_; uint8_t v_cacheInferType_595_; lean_object* v___x_597_; uint8_t v_isShared_598_; uint8_t v_isSharedCheck_643_; 
v_a_584_ = lean_ctor_get(v___x_583_, 0);
lean_inc(v_a_584_);
lean_dec_ref_known(v___x_583_, 1);
v_keyedConfig_585_ = lean_ctor_get(v___y_576_, 0);
v_trackZetaDelta_586_ = lean_ctor_get_uint8(v___y_576_, sizeof(void*)*7);
v_zetaDeltaSet_587_ = lean_ctor_get(v___y_576_, 1);
v_lctx_588_ = lean_ctor_get(v___y_576_, 2);
v_localInstances_589_ = lean_ctor_get(v___y_576_, 3);
v_defEqCtx_x3f_590_ = lean_ctor_get(v___y_576_, 4);
v_synthPendingDepth_591_ = lean_ctor_get(v___y_576_, 5);
v_customCanUnfoldPredicate_x3f_592_ = lean_ctor_get(v___y_576_, 6);
v_univApprox_593_ = lean_ctor_get_uint8(v___y_576_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_594_ = lean_ctor_get_uint8(v___y_576_, sizeof(void*)*7 + 2);
v_cacheInferType_595_ = lean_ctor_get_uint8(v___y_576_, sizeof(void*)*7 + 3);
v_isSharedCheck_643_ = !lean_is_exclusive(v___y_576_);
if (v_isSharedCheck_643_ == 0)
{
v___x_597_ = v___y_576_;
v_isShared_598_ = v_isSharedCheck_643_;
goto v_resetjp_596_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_592_);
lean_inc(v_synthPendingDepth_591_);
lean_inc(v_defEqCtx_x3f_590_);
lean_inc(v_localInstances_589_);
lean_inc(v_lctx_588_);
lean_inc(v_zetaDeltaSet_587_);
lean_inc(v_keyedConfig_585_);
lean_dec(v___y_576_);
v___x_597_ = lean_box(0);
v_isShared_598_ = v_isSharedCheck_643_;
goto v_resetjp_596_;
}
v_resetjp_596_:
{
lean_object* v___x_599_; lean_object* v___x_600_; lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v___x_603_; uint8_t v___x_604_; lean_object* v___x_605_; lean_object* v___x_607_; 
v___x_599_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_600_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2);
lean_inc(v_a_582_);
v___x_601_ = l_Lean_Expr_app___override(v___x_600_, v_a_582_);
lean_inc(v_a_584_);
v___x_602_ = l_Lean_Expr_app___override(v___x_601_, v_a_584_);
v___x_603_ = l_Lean_Expr_app___override(v___x_599_, v___x_602_);
v___x_604_ = 2;
v___x_605_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_604_, v_keyedConfig_585_);
if (v_isShared_598_ == 0)
{
lean_ctor_set(v___x_597_, 0, v___x_605_);
v___x_607_ = v___x_597_;
goto v_reusejp_606_;
}
else
{
lean_object* v_reuseFailAlloc_642_; 
v_reuseFailAlloc_642_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_642_, 0, v___x_605_);
lean_ctor_set(v_reuseFailAlloc_642_, 1, v_zetaDeltaSet_587_);
lean_ctor_set(v_reuseFailAlloc_642_, 2, v_lctx_588_);
lean_ctor_set(v_reuseFailAlloc_642_, 3, v_localInstances_589_);
lean_ctor_set(v_reuseFailAlloc_642_, 4, v_defEqCtx_x3f_590_);
lean_ctor_set(v_reuseFailAlloc_642_, 5, v_synthPendingDepth_591_);
lean_ctor_set(v_reuseFailAlloc_642_, 6, v_customCanUnfoldPredicate_x3f_592_);
lean_ctor_set_uint8(v_reuseFailAlloc_642_, sizeof(void*)*7, v_trackZetaDelta_586_);
lean_ctor_set_uint8(v_reuseFailAlloc_642_, sizeof(void*)*7 + 1, v_univApprox_593_);
lean_ctor_set_uint8(v_reuseFailAlloc_642_, sizeof(void*)*7 + 2, v_inTypeClassResolution_594_);
lean_ctor_set_uint8(v_reuseFailAlloc_642_, sizeof(void*)*7 + 3, v_cacheInferType_595_);
v___x_607_ = v_reuseFailAlloc_642_;
goto v_reusejp_606_;
}
v_reusejp_606_:
{
lean_object* v___x_608_; 
v___x_608_ = l_Lean_Meta_isExprDefEq(v___x_603_, v_a_574_, v___x_607_, v___y_577_, v___y_578_, v___y_579_);
lean_dec_ref(v___x_607_);
if (lean_obj_tag(v___x_608_) == 0)
{
lean_object* v_a_609_; lean_object* v___x_611_; uint8_t v_isShared_612_; uint8_t v_isSharedCheck_633_; 
v_a_609_ = lean_ctor_get(v___x_608_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_608_);
if (v_isSharedCheck_633_ == 0)
{
v___x_611_ = v___x_608_;
v_isShared_612_ = v_isSharedCheck_633_;
goto v_resetjp_610_;
}
else
{
lean_inc(v_a_609_);
lean_dec(v___x_608_);
v___x_611_ = lean_box(0);
v_isShared_612_ = v_isSharedCheck_633_;
goto v_resetjp_610_;
}
v_resetjp_610_:
{
uint8_t v___x_613_; 
v___x_613_ = lean_unbox(v_a_609_);
if (v___x_613_ == 0)
{
lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_618_; 
lean_dec(v_a_609_);
v___x_614_ = lean_box(v___x_575_);
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v_a_584_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
v___x_616_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_616_, 0, v_a_582_);
lean_ctor_set(v___x_616_, 1, v___x_615_);
if (v_isShared_612_ == 0)
{
lean_ctor_set(v___x_611_, 0, v___x_616_);
v___x_618_ = v___x_611_;
goto v_reusejp_617_;
}
else
{
lean_object* v_reuseFailAlloc_619_; 
v_reuseFailAlloc_619_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_619_, 0, v___x_616_);
v___x_618_ = v_reuseFailAlloc_619_;
goto v_reusejp_617_;
}
v_reusejp_617_:
{
return v___x_618_;
}
}
else
{
lean_object* v___x_620_; lean_object* v_a_621_; lean_object* v___x_622_; lean_object* v_a_623_; lean_object* v___x_625_; uint8_t v_isShared_626_; uint8_t v_isSharedCheck_632_; 
lean_del_object(v___x_611_);
v___x_620_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_582_, v___y_577_);
v_a_621_ = lean_ctor_get(v___x_620_, 0);
lean_inc(v_a_621_);
lean_dec_ref(v___x_620_);
v___x_622_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_584_, v___y_577_);
v_a_623_ = lean_ctor_get(v___x_622_, 0);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_622_);
if (v_isSharedCheck_632_ == 0)
{
v___x_625_ = v___x_622_;
v_isShared_626_ = v_isSharedCheck_632_;
goto v_resetjp_624_;
}
else
{
lean_inc(v_a_623_);
lean_dec(v___x_622_);
v___x_625_ = lean_box(0);
v_isShared_626_ = v_isSharedCheck_632_;
goto v_resetjp_624_;
}
v_resetjp_624_:
{
lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_630_; 
v___x_627_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_627_, 0, v_a_623_);
lean_ctor_set(v___x_627_, 1, v_a_609_);
v___x_628_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_628_, 0, v_a_621_);
lean_ctor_set(v___x_628_, 1, v___x_627_);
if (v_isShared_626_ == 0)
{
lean_ctor_set(v___x_625_, 0, v___x_628_);
v___x_630_ = v___x_625_;
goto v_reusejp_629_;
}
else
{
lean_object* v_reuseFailAlloc_631_; 
v_reuseFailAlloc_631_ = lean_alloc_ctor(0, 1, 0);
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
}
else
{
lean_object* v_a_634_; lean_object* v___x_636_; uint8_t v_isShared_637_; uint8_t v_isSharedCheck_641_; 
lean_dec(v_a_584_);
lean_dec(v_a_582_);
v_a_634_ = lean_ctor_get(v___x_608_, 0);
v_isSharedCheck_641_ = !lean_is_exclusive(v___x_608_);
if (v_isSharedCheck_641_ == 0)
{
v___x_636_ = v___x_608_;
v_isShared_637_ = v_isSharedCheck_641_;
goto v_resetjp_635_;
}
else
{
lean_inc(v_a_634_);
lean_dec(v___x_608_);
v___x_636_ = lean_box(0);
v_isShared_637_ = v_isSharedCheck_641_;
goto v_resetjp_635_;
}
v_resetjp_635_:
{
lean_object* v___x_639_; 
if (v_isShared_637_ == 0)
{
v___x_639_ = v___x_636_;
goto v_reusejp_638_;
}
else
{
lean_object* v_reuseFailAlloc_640_; 
v_reuseFailAlloc_640_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_640_, 0, v_a_634_);
v___x_639_ = v_reuseFailAlloc_640_;
goto v_reusejp_638_;
}
v_reusejp_638_:
{
return v___x_639_;
}
}
}
}
}
}
else
{
lean_object* v_a_644_; lean_object* v___x_646_; uint8_t v_isShared_647_; uint8_t v_isSharedCheck_651_; 
lean_dec(v_a_582_);
lean_dec_ref(v___y_576_);
lean_dec_ref(v_a_574_);
v_a_644_ = lean_ctor_get(v___x_583_, 0);
v_isSharedCheck_651_ = !lean_is_exclusive(v___x_583_);
if (v_isSharedCheck_651_ == 0)
{
v___x_646_ = v___x_583_;
v_isShared_647_ = v_isSharedCheck_651_;
goto v_resetjp_645_;
}
else
{
lean_inc(v_a_644_);
lean_dec(v___x_583_);
v___x_646_ = lean_box(0);
v_isShared_647_ = v_isSharedCheck_651_;
goto v_resetjp_645_;
}
v_resetjp_645_:
{
lean_object* v___x_649_; 
if (v_isShared_647_ == 0)
{
v___x_649_ = v___x_646_;
goto v_reusejp_648_;
}
else
{
lean_object* v_reuseFailAlloc_650_; 
v_reuseFailAlloc_650_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_650_, 0, v_a_644_);
v___x_649_ = v_reuseFailAlloc_650_;
goto v_reusejp_648_;
}
v_reusejp_648_:
{
return v___x_649_;
}
}
}
}
else
{
lean_object* v_a_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_659_; 
lean_dec_ref(v___y_576_);
lean_dec_ref(v_a_574_);
lean_dec(v___x_573_);
lean_dec(v___x_571_);
v_a_652_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_659_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_659_ == 0)
{
v___x_654_ = v___x_581_;
v_isShared_655_ = v_isSharedCheck_659_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_a_652_);
lean_dec(v___x_581_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_659_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___x_657_; 
if (v_isShared_655_ == 0)
{
v___x_657_ = v___x_654_;
goto v_reusejp_656_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v_a_652_);
v___x_657_ = v_reuseFailAlloc_658_;
goto v_reusejp_656_;
}
v_reusejp_656_:
{
return v___x_657_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___boxed(lean_object* v___x_660_, lean_object* v___x_661_, lean_object* v___x_662_, lean_object* v_a_663_, lean_object* v___x_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_){
_start:
{
uint8_t v___x_37109__boxed_670_; uint8_t v___x_37112__boxed_671_; lean_object* v_res_672_; 
v___x_37109__boxed_670_ = lean_unbox(v___x_661_);
v___x_37112__boxed_671_ = lean_unbox(v___x_664_);
v_res_672_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3(v___x_660_, v___x_37109__boxed_670_, v___x_662_, v_a_663_, v___x_37112__boxed_671_, v___y_665_, v___y_666_, v___y_667_, v___y_668_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
return v_res_672_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2(void){
_start:
{
lean_object* v___x_676_; lean_object* v___x_677_; lean_object* v___x_678_; 
v___x_676_ = lean_box(0);
v___x_677_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__1));
v___x_678_ = l_Lean_Expr_const___override(v___x_677_, v___x_676_);
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4(lean_object* v___x_679_, uint8_t v___x_680_, lean_object* v___x_681_, lean_object* v_a_682_, uint8_t v___x_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_, lean_object* v___y_687_){
_start:
{
lean_object* v___x_689_; 
lean_inc(v___x_681_);
lean_inc(v___x_679_);
v___x_689_ = l_Lean_Meta_mkFreshExprMVar(v___x_679_, v___x_680_, v___x_681_, v___y_684_, v___y_685_, v___y_686_, v___y_687_);
if (lean_obj_tag(v___x_689_) == 0)
{
lean_object* v_a_690_; lean_object* v___x_691_; 
v_a_690_ = lean_ctor_get(v___x_689_, 0);
lean_inc(v_a_690_);
lean_dec_ref_known(v___x_689_, 1);
v___x_691_ = l_Lean_Meta_mkFreshExprMVar(v___x_679_, v___x_680_, v___x_681_, v___y_684_, v___y_685_, v___y_686_, v___y_687_);
if (lean_obj_tag(v___x_691_) == 0)
{
lean_object* v_a_692_; lean_object* v_keyedConfig_693_; uint8_t v_trackZetaDelta_694_; lean_object* v_zetaDeltaSet_695_; lean_object* v_lctx_696_; lean_object* v_localInstances_697_; lean_object* v_defEqCtx_x3f_698_; lean_object* v_synthPendingDepth_699_; lean_object* v_customCanUnfoldPredicate_x3f_700_; uint8_t v_univApprox_701_; uint8_t v_inTypeClassResolution_702_; uint8_t v_cacheInferType_703_; lean_object* v___x_705_; uint8_t v_isShared_706_; uint8_t v_isSharedCheck_751_; 
v_a_692_ = lean_ctor_get(v___x_691_, 0);
lean_inc(v_a_692_);
lean_dec_ref_known(v___x_691_, 1);
v_keyedConfig_693_ = lean_ctor_get(v___y_684_, 0);
v_trackZetaDelta_694_ = lean_ctor_get_uint8(v___y_684_, sizeof(void*)*7);
v_zetaDeltaSet_695_ = lean_ctor_get(v___y_684_, 1);
v_lctx_696_ = lean_ctor_get(v___y_684_, 2);
v_localInstances_697_ = lean_ctor_get(v___y_684_, 3);
v_defEqCtx_x3f_698_ = lean_ctor_get(v___y_684_, 4);
v_synthPendingDepth_699_ = lean_ctor_get(v___y_684_, 5);
v_customCanUnfoldPredicate_x3f_700_ = lean_ctor_get(v___y_684_, 6);
v_univApprox_701_ = lean_ctor_get_uint8(v___y_684_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_702_ = lean_ctor_get_uint8(v___y_684_, sizeof(void*)*7 + 2);
v_cacheInferType_703_ = lean_ctor_get_uint8(v___y_684_, sizeof(void*)*7 + 3);
v_isSharedCheck_751_ = !lean_is_exclusive(v___y_684_);
if (v_isSharedCheck_751_ == 0)
{
v___x_705_ = v___y_684_;
v_isShared_706_ = v_isSharedCheck_751_;
goto v_resetjp_704_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_700_);
lean_inc(v_synthPendingDepth_699_);
lean_inc(v_defEqCtx_x3f_698_);
lean_inc(v_localInstances_697_);
lean_inc(v_lctx_696_);
lean_inc(v_zetaDeltaSet_695_);
lean_inc(v_keyedConfig_693_);
lean_dec(v___y_684_);
v___x_705_ = lean_box(0);
v_isShared_706_ = v_isSharedCheck_751_;
goto v_resetjp_704_;
}
v_resetjp_704_:
{
lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; uint8_t v___x_712_; lean_object* v___x_713_; lean_object* v___x_715_; 
v___x_707_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_708_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2);
lean_inc(v_a_690_);
v___x_709_ = l_Lean_Expr_app___override(v___x_708_, v_a_690_);
lean_inc(v_a_692_);
v___x_710_ = l_Lean_Expr_app___override(v___x_709_, v_a_692_);
v___x_711_ = l_Lean_Expr_app___override(v___x_707_, v___x_710_);
v___x_712_ = 2;
v___x_713_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_712_, v_keyedConfig_693_);
if (v_isShared_706_ == 0)
{
lean_ctor_set(v___x_705_, 0, v___x_713_);
v___x_715_ = v___x_705_;
goto v_reusejp_714_;
}
else
{
lean_object* v_reuseFailAlloc_750_; 
v_reuseFailAlloc_750_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_750_, 0, v___x_713_);
lean_ctor_set(v_reuseFailAlloc_750_, 1, v_zetaDeltaSet_695_);
lean_ctor_set(v_reuseFailAlloc_750_, 2, v_lctx_696_);
lean_ctor_set(v_reuseFailAlloc_750_, 3, v_localInstances_697_);
lean_ctor_set(v_reuseFailAlloc_750_, 4, v_defEqCtx_x3f_698_);
lean_ctor_set(v_reuseFailAlloc_750_, 5, v_synthPendingDepth_699_);
lean_ctor_set(v_reuseFailAlloc_750_, 6, v_customCanUnfoldPredicate_x3f_700_);
lean_ctor_set_uint8(v_reuseFailAlloc_750_, sizeof(void*)*7, v_trackZetaDelta_694_);
lean_ctor_set_uint8(v_reuseFailAlloc_750_, sizeof(void*)*7 + 1, v_univApprox_701_);
lean_ctor_set_uint8(v_reuseFailAlloc_750_, sizeof(void*)*7 + 2, v_inTypeClassResolution_702_);
lean_ctor_set_uint8(v_reuseFailAlloc_750_, sizeof(void*)*7 + 3, v_cacheInferType_703_);
v___x_715_ = v_reuseFailAlloc_750_;
goto v_reusejp_714_;
}
v_reusejp_714_:
{
lean_object* v___x_716_; 
v___x_716_ = l_Lean_Meta_isExprDefEq(v___x_711_, v_a_682_, v___x_715_, v___y_685_, v___y_686_, v___y_687_);
lean_dec_ref(v___x_715_);
if (lean_obj_tag(v___x_716_) == 0)
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_741_; 
v_a_717_ = lean_ctor_get(v___x_716_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_716_);
if (v_isSharedCheck_741_ == 0)
{
v___x_719_ = v___x_716_;
v_isShared_720_ = v_isSharedCheck_741_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_716_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_741_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
uint8_t v___x_721_; 
v___x_721_ = lean_unbox(v_a_717_);
if (v___x_721_ == 0)
{
lean_object* v___x_722_; lean_object* v___x_723_; lean_object* v___x_724_; lean_object* v___x_726_; 
lean_dec(v_a_717_);
v___x_722_ = lean_box(v___x_683_);
v___x_723_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_723_, 0, v_a_692_);
lean_ctor_set(v___x_723_, 1, v___x_722_);
v___x_724_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_724_, 0, v_a_690_);
lean_ctor_set(v___x_724_, 1, v___x_723_);
if (v_isShared_720_ == 0)
{
lean_ctor_set(v___x_719_, 0, v___x_724_);
v___x_726_ = v___x_719_;
goto v_reusejp_725_;
}
else
{
lean_object* v_reuseFailAlloc_727_; 
v_reuseFailAlloc_727_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_727_, 0, v___x_724_);
v___x_726_ = v_reuseFailAlloc_727_;
goto v_reusejp_725_;
}
v_reusejp_725_:
{
return v___x_726_;
}
}
else
{
lean_object* v___x_728_; lean_object* v_a_729_; lean_object* v___x_730_; lean_object* v_a_731_; lean_object* v___x_733_; uint8_t v_isShared_734_; uint8_t v_isSharedCheck_740_; 
lean_del_object(v___x_719_);
v___x_728_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_690_, v___y_685_);
v_a_729_ = lean_ctor_get(v___x_728_, 0);
lean_inc(v_a_729_);
lean_dec_ref(v___x_728_);
v___x_730_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_692_, v___y_685_);
v_a_731_ = lean_ctor_get(v___x_730_, 0);
v_isSharedCheck_740_ = !lean_is_exclusive(v___x_730_);
if (v_isSharedCheck_740_ == 0)
{
v___x_733_ = v___x_730_;
v_isShared_734_ = v_isSharedCheck_740_;
goto v_resetjp_732_;
}
else
{
lean_inc(v_a_731_);
lean_dec(v___x_730_);
v___x_733_ = lean_box(0);
v_isShared_734_ = v_isSharedCheck_740_;
goto v_resetjp_732_;
}
v_resetjp_732_:
{
lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_738_; 
v___x_735_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_735_, 0, v_a_731_);
lean_ctor_set(v___x_735_, 1, v_a_717_);
v___x_736_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_736_, 0, v_a_729_);
lean_ctor_set(v___x_736_, 1, v___x_735_);
if (v_isShared_734_ == 0)
{
lean_ctor_set(v___x_733_, 0, v___x_736_);
v___x_738_ = v___x_733_;
goto v_reusejp_737_;
}
else
{
lean_object* v_reuseFailAlloc_739_; 
v_reuseFailAlloc_739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_739_, 0, v___x_736_);
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
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_dec(v_a_692_);
lean_dec(v_a_690_);
v_a_742_ = lean_ctor_get(v___x_716_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_716_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_716_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_716_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
}
else
{
lean_object* v_a_752_; lean_object* v___x_754_; uint8_t v_isShared_755_; uint8_t v_isSharedCheck_759_; 
lean_dec(v_a_690_);
lean_dec_ref(v___y_684_);
lean_dec_ref(v_a_682_);
v_a_752_ = lean_ctor_get(v___x_691_, 0);
v_isSharedCheck_759_ = !lean_is_exclusive(v___x_691_);
if (v_isSharedCheck_759_ == 0)
{
v___x_754_ = v___x_691_;
v_isShared_755_ = v_isSharedCheck_759_;
goto v_resetjp_753_;
}
else
{
lean_inc(v_a_752_);
lean_dec(v___x_691_);
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
lean_dec_ref(v___y_684_);
lean_dec_ref(v_a_682_);
lean_dec(v___x_681_);
lean_dec(v___x_679_);
v_a_760_ = lean_ctor_get(v___x_689_, 0);
v_isSharedCheck_767_ = !lean_is_exclusive(v___x_689_);
if (v_isSharedCheck_767_ == 0)
{
v___x_762_ = v___x_689_;
v_isShared_763_ = v_isSharedCheck_767_;
goto v_resetjp_761_;
}
else
{
lean_inc(v_a_760_);
lean_dec(v___x_689_);
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
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___boxed(lean_object* v___x_768_, lean_object* v___x_769_, lean_object* v___x_770_, lean_object* v_a_771_, lean_object* v___x_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_, lean_object* v___y_777_){
_start:
{
uint8_t v___x_37301__boxed_778_; uint8_t v___x_37304__boxed_779_; lean_object* v_res_780_; 
v___x_37301__boxed_778_ = lean_unbox(v___x_769_);
v___x_37304__boxed_779_ = lean_unbox(v___x_772_);
v_res_780_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4(v___x_768_, v___x_37301__boxed_778_, v___x_770_, v_a_771_, v___x_37304__boxed_779_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
lean_dec(v___y_776_);
lean_dec_ref(v___y_775_);
lean_dec(v___y_774_);
return v_res_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5(lean_object* v___x_784_, uint8_t v___x_785_, lean_object* v___x_786_, lean_object* v___x_787_, lean_object* v___x_788_, lean_object* v_a_789_, uint8_t v___x_790_, lean_object* v___y_791_, lean_object* v___y_792_, lean_object* v___y_793_, lean_object* v___y_794_){
_start:
{
lean_object* v___x_796_; 
lean_inc(v___x_786_);
lean_inc(v___x_784_);
v___x_796_ = l_Lean_Meta_mkFreshExprMVar(v___x_784_, v___x_785_, v___x_786_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_796_) == 0)
{
lean_object* v_a_797_; lean_object* v___x_798_; 
v_a_797_ = lean_ctor_get(v___x_796_, 0);
lean_inc(v_a_797_);
lean_dec_ref_known(v___x_796_, 1);
v___x_798_ = l_Lean_Meta_mkFreshExprMVar(v___x_784_, v___x_785_, v___x_786_, v___y_791_, v___y_792_, v___y_793_, v___y_794_);
if (lean_obj_tag(v___x_798_) == 0)
{
lean_object* v_a_799_; lean_object* v_keyedConfig_800_; uint8_t v_trackZetaDelta_801_; lean_object* v_zetaDeltaSet_802_; lean_object* v_lctx_803_; lean_object* v_localInstances_804_; lean_object* v_defEqCtx_x3f_805_; lean_object* v_synthPendingDepth_806_; lean_object* v_customCanUnfoldPredicate_x3f_807_; uint8_t v_univApprox_808_; uint8_t v_inTypeClassResolution_809_; uint8_t v_cacheInferType_810_; lean_object* v___x_812_; uint8_t v_isShared_813_; uint8_t v_isSharedCheck_863_; 
v_a_799_ = lean_ctor_get(v___x_798_, 0);
lean_inc(v_a_799_);
lean_dec_ref_known(v___x_798_, 1);
v_keyedConfig_800_ = lean_ctor_get(v___y_791_, 0);
v_trackZetaDelta_801_ = lean_ctor_get_uint8(v___y_791_, sizeof(void*)*7);
v_zetaDeltaSet_802_ = lean_ctor_get(v___y_791_, 1);
v_lctx_803_ = lean_ctor_get(v___y_791_, 2);
v_localInstances_804_ = lean_ctor_get(v___y_791_, 3);
v_defEqCtx_x3f_805_ = lean_ctor_get(v___y_791_, 4);
v_synthPendingDepth_806_ = lean_ctor_get(v___y_791_, 5);
v_customCanUnfoldPredicate_x3f_807_ = lean_ctor_get(v___y_791_, 6);
v_univApprox_808_ = lean_ctor_get_uint8(v___y_791_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_809_ = lean_ctor_get_uint8(v___y_791_, sizeof(void*)*7 + 2);
v_cacheInferType_810_ = lean_ctor_get_uint8(v___y_791_, sizeof(void*)*7 + 3);
v_isSharedCheck_863_ = !lean_is_exclusive(v___y_791_);
if (v_isSharedCheck_863_ == 0)
{
v___x_812_ = v___y_791_;
v_isShared_813_ = v_isSharedCheck_863_;
goto v_resetjp_811_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_807_);
lean_inc(v_synthPendingDepth_806_);
lean_inc(v_defEqCtx_x3f_805_);
lean_inc(v_localInstances_804_);
lean_inc(v_lctx_803_);
lean_inc(v_zetaDeltaSet_802_);
lean_inc(v_keyedConfig_800_);
lean_dec(v___y_791_);
v___x_812_ = lean_box(0);
v_isShared_813_ = v_isSharedCheck_863_;
goto v_resetjp_811_;
}
v_resetjp_811_:
{
lean_object* v___x_814_; lean_object* v___x_815_; lean_object* v___x_816_; lean_object* v___x_817_; lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___x_823_; uint8_t v___x_824_; lean_object* v___x_825_; lean_object* v___x_827_; 
v___x_814_ = lean_box(0);
v___x_815_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_816_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___closed__1));
v___x_817_ = l_Lean_Level_succ___override(v___x_787_);
v___x_818_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_818_, 0, v___x_817_);
lean_ctor_set(v___x_818_, 1, v___x_814_);
v___x_819_ = l_Lean_Expr_const___override(v___x_816_, v___x_818_);
v___x_820_ = l_Lean_Expr_app___override(v___x_819_, v___x_788_);
lean_inc(v_a_797_);
v___x_821_ = l_Lean_Expr_app___override(v___x_820_, v_a_797_);
lean_inc(v_a_799_);
v___x_822_ = l_Lean_Expr_app___override(v___x_821_, v_a_799_);
v___x_823_ = l_Lean_Expr_app___override(v___x_815_, v___x_822_);
v___x_824_ = 2;
v___x_825_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_824_, v_keyedConfig_800_);
if (v_isShared_813_ == 0)
{
lean_ctor_set(v___x_812_, 0, v___x_825_);
v___x_827_ = v___x_812_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_862_; 
v_reuseFailAlloc_862_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_862_, 0, v___x_825_);
lean_ctor_set(v_reuseFailAlloc_862_, 1, v_zetaDeltaSet_802_);
lean_ctor_set(v_reuseFailAlloc_862_, 2, v_lctx_803_);
lean_ctor_set(v_reuseFailAlloc_862_, 3, v_localInstances_804_);
lean_ctor_set(v_reuseFailAlloc_862_, 4, v_defEqCtx_x3f_805_);
lean_ctor_set(v_reuseFailAlloc_862_, 5, v_synthPendingDepth_806_);
lean_ctor_set(v_reuseFailAlloc_862_, 6, v_customCanUnfoldPredicate_x3f_807_);
lean_ctor_set_uint8(v_reuseFailAlloc_862_, sizeof(void*)*7, v_trackZetaDelta_801_);
lean_ctor_set_uint8(v_reuseFailAlloc_862_, sizeof(void*)*7 + 1, v_univApprox_808_);
lean_ctor_set_uint8(v_reuseFailAlloc_862_, sizeof(void*)*7 + 2, v_inTypeClassResolution_809_);
lean_ctor_set_uint8(v_reuseFailAlloc_862_, sizeof(void*)*7 + 3, v_cacheInferType_810_);
v___x_827_ = v_reuseFailAlloc_862_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
lean_object* v___x_828_; 
v___x_828_ = l_Lean_Meta_isExprDefEq(v___x_823_, v_a_789_, v___x_827_, v___y_792_, v___y_793_, v___y_794_);
lean_dec_ref(v___x_827_);
if (lean_obj_tag(v___x_828_) == 0)
{
lean_object* v_a_829_; lean_object* v___x_831_; uint8_t v_isShared_832_; uint8_t v_isSharedCheck_853_; 
v_a_829_ = lean_ctor_get(v___x_828_, 0);
v_isSharedCheck_853_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_853_ == 0)
{
v___x_831_ = v___x_828_;
v_isShared_832_ = v_isSharedCheck_853_;
goto v_resetjp_830_;
}
else
{
lean_inc(v_a_829_);
lean_dec(v___x_828_);
v___x_831_ = lean_box(0);
v_isShared_832_ = v_isSharedCheck_853_;
goto v_resetjp_830_;
}
v_resetjp_830_:
{
uint8_t v___x_833_; 
v___x_833_ = lean_unbox(v_a_829_);
if (v___x_833_ == 0)
{
lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v___x_836_; lean_object* v___x_838_; 
lean_dec(v_a_829_);
v___x_834_ = lean_box(v___x_790_);
v___x_835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_835_, 0, v_a_799_);
lean_ctor_set(v___x_835_, 1, v___x_834_);
v___x_836_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_836_, 0, v_a_797_);
lean_ctor_set(v___x_836_, 1, v___x_835_);
if (v_isShared_832_ == 0)
{
lean_ctor_set(v___x_831_, 0, v___x_836_);
v___x_838_ = v___x_831_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v___x_836_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
else
{
lean_object* v___x_840_; lean_object* v_a_841_; lean_object* v___x_842_; lean_object* v_a_843_; lean_object* v___x_845_; uint8_t v_isShared_846_; uint8_t v_isSharedCheck_852_; 
lean_del_object(v___x_831_);
v___x_840_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_797_, v___y_792_);
v_a_841_ = lean_ctor_get(v___x_840_, 0);
lean_inc(v_a_841_);
lean_dec_ref(v___x_840_);
v___x_842_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_799_, v___y_792_);
v_a_843_ = lean_ctor_get(v___x_842_, 0);
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_842_);
if (v_isSharedCheck_852_ == 0)
{
v___x_845_ = v___x_842_;
v_isShared_846_ = v_isSharedCheck_852_;
goto v_resetjp_844_;
}
else
{
lean_inc(v_a_843_);
lean_dec(v___x_842_);
v___x_845_ = lean_box(0);
v_isShared_846_ = v_isSharedCheck_852_;
goto v_resetjp_844_;
}
v_resetjp_844_:
{
lean_object* v___x_847_; lean_object* v___x_848_; lean_object* v___x_850_; 
v___x_847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_847_, 0, v_a_843_);
lean_ctor_set(v___x_847_, 1, v_a_829_);
v___x_848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_848_, 0, v_a_841_);
lean_ctor_set(v___x_848_, 1, v___x_847_);
if (v_isShared_846_ == 0)
{
lean_ctor_set(v___x_845_, 0, v___x_848_);
v___x_850_ = v___x_845_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v___x_848_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
}
}
else
{
lean_object* v_a_854_; lean_object* v___x_856_; uint8_t v_isShared_857_; uint8_t v_isSharedCheck_861_; 
lean_dec(v_a_799_);
lean_dec(v_a_797_);
v_a_854_ = lean_ctor_get(v___x_828_, 0);
v_isSharedCheck_861_ = !lean_is_exclusive(v___x_828_);
if (v_isSharedCheck_861_ == 0)
{
v___x_856_ = v___x_828_;
v_isShared_857_ = v_isSharedCheck_861_;
goto v_resetjp_855_;
}
else
{
lean_inc(v_a_854_);
lean_dec(v___x_828_);
v___x_856_ = lean_box(0);
v_isShared_857_ = v_isSharedCheck_861_;
goto v_resetjp_855_;
}
v_resetjp_855_:
{
lean_object* v___x_859_; 
if (v_isShared_857_ == 0)
{
v___x_859_ = v___x_856_;
goto v_reusejp_858_;
}
else
{
lean_object* v_reuseFailAlloc_860_; 
v_reuseFailAlloc_860_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_860_, 0, v_a_854_);
v___x_859_ = v_reuseFailAlloc_860_;
goto v_reusejp_858_;
}
v_reusejp_858_:
{
return v___x_859_;
}
}
}
}
}
}
else
{
lean_object* v_a_864_; lean_object* v___x_866_; uint8_t v_isShared_867_; uint8_t v_isSharedCheck_871_; 
lean_dec(v_a_797_);
lean_dec_ref(v___y_791_);
lean_dec_ref(v_a_789_);
lean_dec_ref(v___x_788_);
lean_dec(v___x_787_);
v_a_864_ = lean_ctor_get(v___x_798_, 0);
v_isSharedCheck_871_ = !lean_is_exclusive(v___x_798_);
if (v_isSharedCheck_871_ == 0)
{
v___x_866_ = v___x_798_;
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
else
{
lean_inc(v_a_864_);
lean_dec(v___x_798_);
v___x_866_ = lean_box(0);
v_isShared_867_ = v_isSharedCheck_871_;
goto v_resetjp_865_;
}
v_resetjp_865_:
{
lean_object* v___x_869_; 
if (v_isShared_867_ == 0)
{
v___x_869_ = v___x_866_;
goto v_reusejp_868_;
}
else
{
lean_object* v_reuseFailAlloc_870_; 
v_reuseFailAlloc_870_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_870_, 0, v_a_864_);
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
else
{
lean_object* v_a_872_; lean_object* v___x_874_; uint8_t v_isShared_875_; uint8_t v_isSharedCheck_879_; 
lean_dec_ref(v___y_791_);
lean_dec_ref(v_a_789_);
lean_dec_ref(v___x_788_);
lean_dec(v___x_787_);
lean_dec(v___x_786_);
lean_dec(v___x_784_);
v_a_872_ = lean_ctor_get(v___x_796_, 0);
v_isSharedCheck_879_ = !lean_is_exclusive(v___x_796_);
if (v_isSharedCheck_879_ == 0)
{
v___x_874_ = v___x_796_;
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
else
{
lean_inc(v_a_872_);
lean_dec(v___x_796_);
v___x_874_ = lean_box(0);
v_isShared_875_ = v_isSharedCheck_879_;
goto v_resetjp_873_;
}
v_resetjp_873_:
{
lean_object* v___x_877_; 
if (v_isShared_875_ == 0)
{
v___x_877_ = v___x_874_;
goto v_reusejp_876_;
}
else
{
lean_object* v_reuseFailAlloc_878_; 
v_reuseFailAlloc_878_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_878_, 0, v_a_872_);
v___x_877_ = v_reuseFailAlloc_878_;
goto v_reusejp_876_;
}
v_reusejp_876_:
{
return v___x_877_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___boxed(lean_object* v___x_880_, lean_object* v___x_881_, lean_object* v___x_882_, lean_object* v___x_883_, lean_object* v___x_884_, lean_object* v_a_885_, lean_object* v___x_886_, lean_object* v___y_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_){
_start:
{
uint8_t v___x_37487__boxed_892_; uint8_t v___x_37492__boxed_893_; lean_object* v_res_894_; 
v___x_37487__boxed_892_ = lean_unbox(v___x_881_);
v___x_37492__boxed_893_ = lean_unbox(v___x_886_);
v_res_894_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5(v___x_880_, v___x_37487__boxed_892_, v___x_882_, v___x_883_, v___x_884_, v_a_885_, v___x_37492__boxed_893_, v___y_887_, v___y_888_, v___y_889_, v___y_890_);
lean_dec(v___y_890_);
lean_dec_ref(v___y_889_);
lean_dec(v___y_888_);
return v_res_894_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6(lean_object* v___x_895_, uint8_t v___x_896_, lean_object* v___x_897_, lean_object* v_a_898_, uint8_t v___x_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
lean_object* v___x_905_; 
v___x_905_ = l_Lean_Meta_mkFreshExprMVar(v___x_895_, v___x_896_, v___x_897_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
if (lean_obj_tag(v___x_905_) == 0)
{
lean_object* v_a_906_; lean_object* v_keyedConfig_907_; uint8_t v_trackZetaDelta_908_; lean_object* v_zetaDeltaSet_909_; lean_object* v_lctx_910_; lean_object* v_localInstances_911_; lean_object* v_defEqCtx_x3f_912_; lean_object* v_synthPendingDepth_913_; lean_object* v_customCanUnfoldPredicate_x3f_914_; uint8_t v_univApprox_915_; uint8_t v_inTypeClassResolution_916_; uint8_t v_cacheInferType_917_; lean_object* v___x_919_; uint8_t v_isShared_920_; uint8_t v_isSharedCheck_959_; 
v_a_906_ = lean_ctor_get(v___x_905_, 0);
lean_inc(v_a_906_);
lean_dec_ref_known(v___x_905_, 1);
v_keyedConfig_907_ = lean_ctor_get(v___y_900_, 0);
v_trackZetaDelta_908_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*7);
v_zetaDeltaSet_909_ = lean_ctor_get(v___y_900_, 1);
v_lctx_910_ = lean_ctor_get(v___y_900_, 2);
v_localInstances_911_ = lean_ctor_get(v___y_900_, 3);
v_defEqCtx_x3f_912_ = lean_ctor_get(v___y_900_, 4);
v_synthPendingDepth_913_ = lean_ctor_get(v___y_900_, 5);
v_customCanUnfoldPredicate_x3f_914_ = lean_ctor_get(v___y_900_, 6);
v_univApprox_915_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_916_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*7 + 2);
v_cacheInferType_917_ = lean_ctor_get_uint8(v___y_900_, sizeof(void*)*7 + 3);
v_isSharedCheck_959_ = !lean_is_exclusive(v___y_900_);
if (v_isSharedCheck_959_ == 0)
{
v___x_919_ = v___y_900_;
v_isShared_920_ = v_isSharedCheck_959_;
goto v_resetjp_918_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_914_);
lean_inc(v_synthPendingDepth_913_);
lean_inc(v_defEqCtx_x3f_912_);
lean_inc(v_localInstances_911_);
lean_inc(v_lctx_910_);
lean_inc(v_zetaDeltaSet_909_);
lean_inc(v_keyedConfig_907_);
lean_dec(v___y_900_);
v___x_919_ = lean_box(0);
v_isShared_920_ = v_isSharedCheck_959_;
goto v_resetjp_918_;
}
v_resetjp_918_:
{
lean_object* v___x_921_; lean_object* v___x_922_; lean_object* v___x_923_; uint8_t v___x_924_; lean_object* v___x_925_; lean_object* v___x_927_; 
v___x_921_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
lean_inc(v_a_906_);
v___x_922_ = l_Lean_Expr_app___override(v___x_921_, v_a_906_);
v___x_923_ = l_Lean_Expr_app___override(v___x_921_, v___x_922_);
v___x_924_ = 2;
v___x_925_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_924_, v_keyedConfig_907_);
if (v_isShared_920_ == 0)
{
lean_ctor_set(v___x_919_, 0, v___x_925_);
v___x_927_ = v___x_919_;
goto v_reusejp_926_;
}
else
{
lean_object* v_reuseFailAlloc_958_; 
v_reuseFailAlloc_958_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_958_, 0, v___x_925_);
lean_ctor_set(v_reuseFailAlloc_958_, 1, v_zetaDeltaSet_909_);
lean_ctor_set(v_reuseFailAlloc_958_, 2, v_lctx_910_);
lean_ctor_set(v_reuseFailAlloc_958_, 3, v_localInstances_911_);
lean_ctor_set(v_reuseFailAlloc_958_, 4, v_defEqCtx_x3f_912_);
lean_ctor_set(v_reuseFailAlloc_958_, 5, v_synthPendingDepth_913_);
lean_ctor_set(v_reuseFailAlloc_958_, 6, v_customCanUnfoldPredicate_x3f_914_);
lean_ctor_set_uint8(v_reuseFailAlloc_958_, sizeof(void*)*7, v_trackZetaDelta_908_);
lean_ctor_set_uint8(v_reuseFailAlloc_958_, sizeof(void*)*7 + 1, v_univApprox_915_);
lean_ctor_set_uint8(v_reuseFailAlloc_958_, sizeof(void*)*7 + 2, v_inTypeClassResolution_916_);
lean_ctor_set_uint8(v_reuseFailAlloc_958_, sizeof(void*)*7 + 3, v_cacheInferType_917_);
v___x_927_ = v_reuseFailAlloc_958_;
goto v_reusejp_926_;
}
v_reusejp_926_:
{
lean_object* v___x_928_; 
v___x_928_ = l_Lean_Meta_isExprDefEq(v___x_923_, v_a_898_, v___x_927_, v___y_901_, v___y_902_, v___y_903_);
lean_dec_ref(v___x_927_);
if (lean_obj_tag(v___x_928_) == 0)
{
lean_object* v_a_929_; lean_object* v___x_931_; uint8_t v_isShared_932_; uint8_t v_isSharedCheck_949_; 
v_a_929_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_949_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_949_ == 0)
{
v___x_931_ = v___x_928_;
v_isShared_932_ = v_isSharedCheck_949_;
goto v_resetjp_930_;
}
else
{
lean_inc(v_a_929_);
lean_dec(v___x_928_);
v___x_931_ = lean_box(0);
v_isShared_932_ = v_isSharedCheck_949_;
goto v_resetjp_930_;
}
v_resetjp_930_:
{
uint8_t v___x_933_; 
v___x_933_ = lean_unbox(v_a_929_);
if (v___x_933_ == 0)
{
lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_937_; 
lean_dec(v_a_929_);
v___x_934_ = lean_box(v___x_899_);
v___x_935_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_935_, 0, v_a_906_);
lean_ctor_set(v___x_935_, 1, v___x_934_);
if (v_isShared_932_ == 0)
{
lean_ctor_set(v___x_931_, 0, v___x_935_);
v___x_937_ = v___x_931_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v___x_935_);
v___x_937_ = v_reuseFailAlloc_938_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
return v___x_937_;
}
}
else
{
lean_object* v___x_939_; lean_object* v_a_940_; lean_object* v___x_942_; uint8_t v_isShared_943_; uint8_t v_isSharedCheck_948_; 
lean_del_object(v___x_931_);
v___x_939_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_906_, v___y_901_);
v_a_940_ = lean_ctor_get(v___x_939_, 0);
v_isSharedCheck_948_ = !lean_is_exclusive(v___x_939_);
if (v_isSharedCheck_948_ == 0)
{
v___x_942_ = v___x_939_;
v_isShared_943_ = v_isSharedCheck_948_;
goto v_resetjp_941_;
}
else
{
lean_inc(v_a_940_);
lean_dec(v___x_939_);
v___x_942_ = lean_box(0);
v_isShared_943_ = v_isSharedCheck_948_;
goto v_resetjp_941_;
}
v_resetjp_941_:
{
lean_object* v___x_944_; lean_object* v___x_946_; 
v___x_944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_944_, 0, v_a_940_);
lean_ctor_set(v___x_944_, 1, v_a_929_);
if (v_isShared_943_ == 0)
{
lean_ctor_set(v___x_942_, 0, v___x_944_);
v___x_946_ = v___x_942_;
goto v_reusejp_945_;
}
else
{
lean_object* v_reuseFailAlloc_947_; 
v_reuseFailAlloc_947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_947_, 0, v___x_944_);
v___x_946_ = v_reuseFailAlloc_947_;
goto v_reusejp_945_;
}
v_reusejp_945_:
{
return v___x_946_;
}
}
}
}
}
else
{
lean_object* v_a_950_; lean_object* v___x_952_; uint8_t v_isShared_953_; uint8_t v_isSharedCheck_957_; 
lean_dec(v_a_906_);
v_a_950_ = lean_ctor_get(v___x_928_, 0);
v_isSharedCheck_957_ = !lean_is_exclusive(v___x_928_);
if (v_isSharedCheck_957_ == 0)
{
v___x_952_ = v___x_928_;
v_isShared_953_ = v_isSharedCheck_957_;
goto v_resetjp_951_;
}
else
{
lean_inc(v_a_950_);
lean_dec(v___x_928_);
v___x_952_ = lean_box(0);
v_isShared_953_ = v_isSharedCheck_957_;
goto v_resetjp_951_;
}
v_resetjp_951_:
{
lean_object* v___x_955_; 
if (v_isShared_953_ == 0)
{
v___x_955_ = v___x_952_;
goto v_reusejp_954_;
}
else
{
lean_object* v_reuseFailAlloc_956_; 
v_reuseFailAlloc_956_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_956_, 0, v_a_950_);
v___x_955_ = v_reuseFailAlloc_956_;
goto v_reusejp_954_;
}
v_reusejp_954_:
{
return v___x_955_;
}
}
}
}
}
}
else
{
lean_object* v_a_960_; lean_object* v___x_962_; uint8_t v_isShared_963_; uint8_t v_isSharedCheck_967_; 
lean_dec_ref(v___y_900_);
lean_dec_ref(v_a_898_);
v_a_960_ = lean_ctor_get(v___x_905_, 0);
v_isSharedCheck_967_ = !lean_is_exclusive(v___x_905_);
if (v_isSharedCheck_967_ == 0)
{
v___x_962_ = v___x_905_;
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
else
{
lean_inc(v_a_960_);
lean_dec(v___x_905_);
v___x_962_ = lean_box(0);
v_isShared_963_ = v_isSharedCheck_967_;
goto v_resetjp_961_;
}
v_resetjp_961_:
{
lean_object* v___x_965_; 
if (v_isShared_963_ == 0)
{
v___x_965_ = v___x_962_;
goto v_reusejp_964_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_a_960_);
v___x_965_ = v_reuseFailAlloc_966_;
goto v_reusejp_964_;
}
v_reusejp_964_:
{
return v___x_965_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6___boxed(lean_object* v___x_968_, lean_object* v___x_969_, lean_object* v___x_970_, lean_object* v_a_971_, lean_object* v___x_972_, lean_object* v___y_973_, lean_object* v___y_974_, lean_object* v___y_975_, lean_object* v___y_976_, lean_object* v___y_977_){
_start:
{
uint8_t v___x_37680__boxed_978_; uint8_t v___x_37683__boxed_979_; lean_object* v_res_980_; 
v___x_37680__boxed_978_ = lean_unbox(v___x_969_);
v___x_37683__boxed_979_ = lean_unbox(v___x_972_);
v_res_980_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6(v___x_968_, v___x_37680__boxed_978_, v___x_970_, v_a_971_, v___x_37683__boxed_979_, v___y_973_, v___y_974_, v___y_975_, v___y_976_);
lean_dec(v___y_976_);
lean_dec_ref(v___y_975_);
lean_dec(v___y_974_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7(lean_object* v___x_981_, uint8_t v___x_982_, lean_object* v___x_983_, lean_object* v_a_984_, uint8_t v___x_985_, lean_object* v___y_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_){
_start:
{
lean_object* v___x_991_; 
lean_inc(v___x_983_);
lean_inc(v___x_981_);
v___x_991_ = l_Lean_Meta_mkFreshExprMVar(v___x_981_, v___x_982_, v___x_983_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_991_) == 0)
{
lean_object* v_a_992_; lean_object* v___x_993_; 
v_a_992_ = lean_ctor_get(v___x_991_, 0);
lean_inc(v_a_992_);
lean_dec_ref_known(v___x_991_, 1);
lean_inc(v___x_983_);
v___x_993_ = l_Lean_Meta_mkFreshExprMVar(v___x_981_, v___x_982_, v___x_983_, v___y_986_, v___y_987_, v___y_988_, v___y_989_);
if (lean_obj_tag(v___x_993_) == 0)
{
lean_object* v_a_994_; lean_object* v_keyedConfig_995_; uint8_t v_trackZetaDelta_996_; lean_object* v_zetaDeltaSet_997_; lean_object* v_lctx_998_; lean_object* v_localInstances_999_; lean_object* v_defEqCtx_x3f_1000_; lean_object* v_synthPendingDepth_1001_; lean_object* v_customCanUnfoldPredicate_x3f_1002_; uint8_t v_univApprox_1003_; uint8_t v_inTypeClassResolution_1004_; uint8_t v_cacheInferType_1005_; lean_object* v___x_1007_; uint8_t v_isShared_1008_; uint8_t v_isSharedCheck_1052_; 
v_a_994_ = lean_ctor_get(v___x_993_, 0);
lean_inc(v_a_994_);
lean_dec_ref_known(v___x_993_, 1);
v_keyedConfig_995_ = lean_ctor_get(v___y_986_, 0);
v_trackZetaDelta_996_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*7);
v_zetaDeltaSet_997_ = lean_ctor_get(v___y_986_, 1);
v_lctx_998_ = lean_ctor_get(v___y_986_, 2);
v_localInstances_999_ = lean_ctor_get(v___y_986_, 3);
v_defEqCtx_x3f_1000_ = lean_ctor_get(v___y_986_, 4);
v_synthPendingDepth_1001_ = lean_ctor_get(v___y_986_, 5);
v_customCanUnfoldPredicate_x3f_1002_ = lean_ctor_get(v___y_986_, 6);
v_univApprox_1003_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1004_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*7 + 2);
v_cacheInferType_1005_ = lean_ctor_get_uint8(v___y_986_, sizeof(void*)*7 + 3);
v_isSharedCheck_1052_ = !lean_is_exclusive(v___y_986_);
if (v_isSharedCheck_1052_ == 0)
{
v___x_1007_ = v___y_986_;
v_isShared_1008_ = v_isSharedCheck_1052_;
goto v_resetjp_1006_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1002_);
lean_inc(v_synthPendingDepth_1001_);
lean_inc(v_defEqCtx_x3f_1000_);
lean_inc(v_localInstances_999_);
lean_inc(v_lctx_998_);
lean_inc(v_zetaDeltaSet_997_);
lean_inc(v_keyedConfig_995_);
lean_dec(v___y_986_);
v___x_1007_ = lean_box(0);
v_isShared_1008_ = v_isSharedCheck_1052_;
goto v_resetjp_1006_;
}
v_resetjp_1006_:
{
lean_object* v___x_1009_; uint8_t v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; uint8_t v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1016_; 
v___x_1009_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_1010_ = 0;
lean_inc(v_a_994_);
lean_inc(v_a_992_);
v___x_1011_ = l_Lean_Expr_forallE___override(v___x_983_, v_a_992_, v_a_994_, v___x_1010_);
v___x_1012_ = l_Lean_Expr_app___override(v___x_1009_, v___x_1011_);
v___x_1013_ = 2;
v___x_1014_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1013_, v_keyedConfig_995_);
if (v_isShared_1008_ == 0)
{
lean_ctor_set(v___x_1007_, 0, v___x_1014_);
v___x_1016_ = v___x_1007_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1051_; 
v_reuseFailAlloc_1051_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1051_, 0, v___x_1014_);
lean_ctor_set(v_reuseFailAlloc_1051_, 1, v_zetaDeltaSet_997_);
lean_ctor_set(v_reuseFailAlloc_1051_, 2, v_lctx_998_);
lean_ctor_set(v_reuseFailAlloc_1051_, 3, v_localInstances_999_);
lean_ctor_set(v_reuseFailAlloc_1051_, 4, v_defEqCtx_x3f_1000_);
lean_ctor_set(v_reuseFailAlloc_1051_, 5, v_synthPendingDepth_1001_);
lean_ctor_set(v_reuseFailAlloc_1051_, 6, v_customCanUnfoldPredicate_x3f_1002_);
lean_ctor_set_uint8(v_reuseFailAlloc_1051_, sizeof(void*)*7, v_trackZetaDelta_996_);
lean_ctor_set_uint8(v_reuseFailAlloc_1051_, sizeof(void*)*7 + 1, v_univApprox_1003_);
lean_ctor_set_uint8(v_reuseFailAlloc_1051_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1004_);
lean_ctor_set_uint8(v_reuseFailAlloc_1051_, sizeof(void*)*7 + 3, v_cacheInferType_1005_);
v___x_1016_ = v_reuseFailAlloc_1051_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
lean_object* v___x_1017_; 
v___x_1017_ = l_Lean_Meta_isExprDefEq(v___x_1012_, v_a_984_, v___x_1016_, v___y_987_, v___y_988_, v___y_989_);
lean_dec_ref(v___x_1016_);
if (lean_obj_tag(v___x_1017_) == 0)
{
lean_object* v_a_1018_; lean_object* v___x_1020_; uint8_t v_isShared_1021_; uint8_t v_isSharedCheck_1042_; 
v_a_1018_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1042_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1042_ == 0)
{
v___x_1020_ = v___x_1017_;
v_isShared_1021_ = v_isSharedCheck_1042_;
goto v_resetjp_1019_;
}
else
{
lean_inc(v_a_1018_);
lean_dec(v___x_1017_);
v___x_1020_ = lean_box(0);
v_isShared_1021_ = v_isSharedCheck_1042_;
goto v_resetjp_1019_;
}
v_resetjp_1019_:
{
uint8_t v___x_1022_; 
v___x_1022_ = lean_unbox(v_a_1018_);
if (v___x_1022_ == 0)
{
lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1027_; 
lean_dec(v_a_1018_);
v___x_1023_ = lean_box(v___x_985_);
v___x_1024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1024_, 0, v_a_994_);
lean_ctor_set(v___x_1024_, 1, v___x_1023_);
v___x_1025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1025_, 0, v_a_992_);
lean_ctor_set(v___x_1025_, 1, v___x_1024_);
if (v_isShared_1021_ == 0)
{
lean_ctor_set(v___x_1020_, 0, v___x_1025_);
v___x_1027_ = v___x_1020_;
goto v_reusejp_1026_;
}
else
{
lean_object* v_reuseFailAlloc_1028_; 
v_reuseFailAlloc_1028_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1028_, 0, v___x_1025_);
v___x_1027_ = v_reuseFailAlloc_1028_;
goto v_reusejp_1026_;
}
v_reusejp_1026_:
{
return v___x_1027_;
}
}
else
{
lean_object* v___x_1029_; lean_object* v_a_1030_; lean_object* v___x_1031_; lean_object* v_a_1032_; lean_object* v___x_1034_; uint8_t v_isShared_1035_; uint8_t v_isSharedCheck_1041_; 
lean_del_object(v___x_1020_);
v___x_1029_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_992_, v___y_987_);
v_a_1030_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_a_1030_);
lean_dec_ref(v___x_1029_);
v___x_1031_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_994_, v___y_987_);
v_a_1032_ = lean_ctor_get(v___x_1031_, 0);
v_isSharedCheck_1041_ = !lean_is_exclusive(v___x_1031_);
if (v_isSharedCheck_1041_ == 0)
{
v___x_1034_ = v___x_1031_;
v_isShared_1035_ = v_isSharedCheck_1041_;
goto v_resetjp_1033_;
}
else
{
lean_inc(v_a_1032_);
lean_dec(v___x_1031_);
v___x_1034_ = lean_box(0);
v_isShared_1035_ = v_isSharedCheck_1041_;
goto v_resetjp_1033_;
}
v_resetjp_1033_:
{
lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1039_; 
v___x_1036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1036_, 0, v_a_1032_);
lean_ctor_set(v___x_1036_, 1, v_a_1018_);
v___x_1037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1037_, 0, v_a_1030_);
lean_ctor_set(v___x_1037_, 1, v___x_1036_);
if (v_isShared_1035_ == 0)
{
lean_ctor_set(v___x_1034_, 0, v___x_1037_);
v___x_1039_ = v___x_1034_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v___x_1037_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
return v___x_1039_;
}
}
}
}
}
else
{
lean_object* v_a_1043_; lean_object* v___x_1045_; uint8_t v_isShared_1046_; uint8_t v_isSharedCheck_1050_; 
lean_dec(v_a_994_);
lean_dec(v_a_992_);
v_a_1043_ = lean_ctor_get(v___x_1017_, 0);
v_isSharedCheck_1050_ = !lean_is_exclusive(v___x_1017_);
if (v_isSharedCheck_1050_ == 0)
{
v___x_1045_ = v___x_1017_;
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
else
{
lean_inc(v_a_1043_);
lean_dec(v___x_1017_);
v___x_1045_ = lean_box(0);
v_isShared_1046_ = v_isSharedCheck_1050_;
goto v_resetjp_1044_;
}
v_resetjp_1044_:
{
lean_object* v___x_1048_; 
if (v_isShared_1046_ == 0)
{
v___x_1048_ = v___x_1045_;
goto v_reusejp_1047_;
}
else
{
lean_object* v_reuseFailAlloc_1049_; 
v_reuseFailAlloc_1049_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1049_, 0, v_a_1043_);
v___x_1048_ = v_reuseFailAlloc_1049_;
goto v_reusejp_1047_;
}
v_reusejp_1047_:
{
return v___x_1048_;
}
}
}
}
}
}
else
{
lean_object* v_a_1053_; lean_object* v___x_1055_; uint8_t v_isShared_1056_; uint8_t v_isSharedCheck_1060_; 
lean_dec(v_a_992_);
lean_dec_ref(v___y_986_);
lean_dec_ref(v_a_984_);
lean_dec(v___x_983_);
v_a_1053_ = lean_ctor_get(v___x_993_, 0);
v_isSharedCheck_1060_ = !lean_is_exclusive(v___x_993_);
if (v_isSharedCheck_1060_ == 0)
{
v___x_1055_ = v___x_993_;
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
else
{
lean_inc(v_a_1053_);
lean_dec(v___x_993_);
v___x_1055_ = lean_box(0);
v_isShared_1056_ = v_isSharedCheck_1060_;
goto v_resetjp_1054_;
}
v_resetjp_1054_:
{
lean_object* v___x_1058_; 
if (v_isShared_1056_ == 0)
{
v___x_1058_ = v___x_1055_;
goto v_reusejp_1057_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v_a_1053_);
v___x_1058_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1057_;
}
v_reusejp_1057_:
{
return v___x_1058_;
}
}
}
}
else
{
lean_object* v_a_1061_; lean_object* v___x_1063_; uint8_t v_isShared_1064_; uint8_t v_isSharedCheck_1068_; 
lean_dec_ref(v___y_986_);
lean_dec_ref(v_a_984_);
lean_dec(v___x_983_);
lean_dec(v___x_981_);
v_a_1061_ = lean_ctor_get(v___x_991_, 0);
v_isSharedCheck_1068_ = !lean_is_exclusive(v___x_991_);
if (v_isSharedCheck_1068_ == 0)
{
v___x_1063_ = v___x_991_;
v_isShared_1064_ = v_isSharedCheck_1068_;
goto v_resetjp_1062_;
}
else
{
lean_inc(v_a_1061_);
lean_dec(v___x_991_);
v___x_1063_ = lean_box(0);
v_isShared_1064_ = v_isSharedCheck_1068_;
goto v_resetjp_1062_;
}
v_resetjp_1062_:
{
lean_object* v___x_1066_; 
if (v_isShared_1064_ == 0)
{
v___x_1066_ = v___x_1063_;
goto v_reusejp_1065_;
}
else
{
lean_object* v_reuseFailAlloc_1067_; 
v_reuseFailAlloc_1067_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1067_, 0, v_a_1061_);
v___x_1066_ = v_reuseFailAlloc_1067_;
goto v_reusejp_1065_;
}
v_reusejp_1065_:
{
return v___x_1066_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7___boxed(lean_object* v___x_1069_, lean_object* v___x_1070_, lean_object* v___x_1071_, lean_object* v_a_1072_, lean_object* v___x_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_){
_start:
{
uint8_t v___x_37824__boxed_1079_; uint8_t v___x_37827__boxed_1080_; lean_object* v_res_1081_; 
v___x_37824__boxed_1079_ = lean_unbox(v___x_1070_);
v___x_37827__boxed_1080_ = lean_unbox(v___x_1073_);
v_res_1081_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7(v___x_1069_, v___x_37824__boxed_1079_, v___x_1071_, v_a_1072_, v___x_37827__boxed_1080_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
lean_dec(v___y_1075_);
return v_res_1081_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2(void){
_start:
{
lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1085_ = lean_box(0);
v___x_1086_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__1));
v___x_1087_ = l_Lean_Expr_const___override(v___x_1086_, v___x_1085_);
return v___x_1087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8(lean_object* v___x_1088_, uint8_t v___x_1089_, lean_object* v___x_1090_, lean_object* v_a_1091_, uint8_t v___x_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_){
_start:
{
lean_object* v___x_1098_; 
lean_inc(v___x_1090_);
lean_inc(v___x_1088_);
v___x_1098_ = l_Lean_Meta_mkFreshExprMVar(v___x_1088_, v___x_1089_, v___x_1090_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
if (lean_obj_tag(v___x_1098_) == 0)
{
lean_object* v_a_1099_; lean_object* v___x_1100_; 
v_a_1099_ = lean_ctor_get(v___x_1098_, 0);
lean_inc(v_a_1099_);
lean_dec_ref_known(v___x_1098_, 1);
v___x_1100_ = l_Lean_Meta_mkFreshExprMVar(v___x_1088_, v___x_1089_, v___x_1090_, v___y_1093_, v___y_1094_, v___y_1095_, v___y_1096_);
if (lean_obj_tag(v___x_1100_) == 0)
{
lean_object* v_a_1101_; lean_object* v_keyedConfig_1102_; uint8_t v_trackZetaDelta_1103_; lean_object* v_zetaDeltaSet_1104_; lean_object* v_lctx_1105_; lean_object* v_localInstances_1106_; lean_object* v_defEqCtx_x3f_1107_; lean_object* v_synthPendingDepth_1108_; lean_object* v_customCanUnfoldPredicate_x3f_1109_; uint8_t v_univApprox_1110_; uint8_t v_inTypeClassResolution_1111_; uint8_t v_cacheInferType_1112_; lean_object* v___x_1114_; uint8_t v_isShared_1115_; uint8_t v_isSharedCheck_1160_; 
v_a_1101_ = lean_ctor_get(v___x_1100_, 0);
lean_inc(v_a_1101_);
lean_dec_ref_known(v___x_1100_, 1);
v_keyedConfig_1102_ = lean_ctor_get(v___y_1093_, 0);
v_trackZetaDelta_1103_ = lean_ctor_get_uint8(v___y_1093_, sizeof(void*)*7);
v_zetaDeltaSet_1104_ = lean_ctor_get(v___y_1093_, 1);
v_lctx_1105_ = lean_ctor_get(v___y_1093_, 2);
v_localInstances_1106_ = lean_ctor_get(v___y_1093_, 3);
v_defEqCtx_x3f_1107_ = lean_ctor_get(v___y_1093_, 4);
v_synthPendingDepth_1108_ = lean_ctor_get(v___y_1093_, 5);
v_customCanUnfoldPredicate_x3f_1109_ = lean_ctor_get(v___y_1093_, 6);
v_univApprox_1110_ = lean_ctor_get_uint8(v___y_1093_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1111_ = lean_ctor_get_uint8(v___y_1093_, sizeof(void*)*7 + 2);
v_cacheInferType_1112_ = lean_ctor_get_uint8(v___y_1093_, sizeof(void*)*7 + 3);
v_isSharedCheck_1160_ = !lean_is_exclusive(v___y_1093_);
if (v_isSharedCheck_1160_ == 0)
{
v___x_1114_ = v___y_1093_;
v_isShared_1115_ = v_isSharedCheck_1160_;
goto v_resetjp_1113_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1109_);
lean_inc(v_synthPendingDepth_1108_);
lean_inc(v_defEqCtx_x3f_1107_);
lean_inc(v_localInstances_1106_);
lean_inc(v_lctx_1105_);
lean_inc(v_zetaDeltaSet_1104_);
lean_inc(v_keyedConfig_1102_);
lean_dec(v___y_1093_);
v___x_1114_ = lean_box(0);
v_isShared_1115_ = v_isSharedCheck_1160_;
goto v_resetjp_1113_;
}
v_resetjp_1113_:
{
lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; uint8_t v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1124_; 
v___x_1116_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_1117_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2);
lean_inc(v_a_1099_);
v___x_1118_ = l_Lean_Expr_app___override(v___x_1117_, v_a_1099_);
lean_inc(v_a_1101_);
v___x_1119_ = l_Lean_Expr_app___override(v___x_1118_, v_a_1101_);
v___x_1120_ = l_Lean_Expr_app___override(v___x_1116_, v___x_1119_);
v___x_1121_ = 2;
v___x_1122_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1121_, v_keyedConfig_1102_);
if (v_isShared_1115_ == 0)
{
lean_ctor_set(v___x_1114_, 0, v___x_1122_);
v___x_1124_ = v___x_1114_;
goto v_reusejp_1123_;
}
else
{
lean_object* v_reuseFailAlloc_1159_; 
v_reuseFailAlloc_1159_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1159_, 0, v___x_1122_);
lean_ctor_set(v_reuseFailAlloc_1159_, 1, v_zetaDeltaSet_1104_);
lean_ctor_set(v_reuseFailAlloc_1159_, 2, v_lctx_1105_);
lean_ctor_set(v_reuseFailAlloc_1159_, 3, v_localInstances_1106_);
lean_ctor_set(v_reuseFailAlloc_1159_, 4, v_defEqCtx_x3f_1107_);
lean_ctor_set(v_reuseFailAlloc_1159_, 5, v_synthPendingDepth_1108_);
lean_ctor_set(v_reuseFailAlloc_1159_, 6, v_customCanUnfoldPredicate_x3f_1109_);
lean_ctor_set_uint8(v_reuseFailAlloc_1159_, sizeof(void*)*7, v_trackZetaDelta_1103_);
lean_ctor_set_uint8(v_reuseFailAlloc_1159_, sizeof(void*)*7 + 1, v_univApprox_1110_);
lean_ctor_set_uint8(v_reuseFailAlloc_1159_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1111_);
lean_ctor_set_uint8(v_reuseFailAlloc_1159_, sizeof(void*)*7 + 3, v_cacheInferType_1112_);
v___x_1124_ = v_reuseFailAlloc_1159_;
goto v_reusejp_1123_;
}
v_reusejp_1123_:
{
lean_object* v___x_1125_; 
v___x_1125_ = l_Lean_Meta_isExprDefEq(v___x_1120_, v_a_1091_, v___x_1124_, v___y_1094_, v___y_1095_, v___y_1096_);
lean_dec_ref(v___x_1124_);
if (lean_obj_tag(v___x_1125_) == 0)
{
lean_object* v_a_1126_; lean_object* v___x_1128_; uint8_t v_isShared_1129_; uint8_t v_isSharedCheck_1150_; 
v_a_1126_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1150_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1150_ == 0)
{
v___x_1128_ = v___x_1125_;
v_isShared_1129_ = v_isSharedCheck_1150_;
goto v_resetjp_1127_;
}
else
{
lean_inc(v_a_1126_);
lean_dec(v___x_1125_);
v___x_1128_ = lean_box(0);
v_isShared_1129_ = v_isSharedCheck_1150_;
goto v_resetjp_1127_;
}
v_resetjp_1127_:
{
uint8_t v___x_1130_; 
v___x_1130_ = lean_unbox(v_a_1126_);
if (v___x_1130_ == 0)
{
lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1135_; 
lean_dec(v_a_1126_);
v___x_1131_ = lean_box(v___x_1092_);
v___x_1132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1132_, 0, v_a_1101_);
lean_ctor_set(v___x_1132_, 1, v___x_1131_);
v___x_1133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1133_, 0, v_a_1099_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
if (v_isShared_1129_ == 0)
{
lean_ctor_set(v___x_1128_, 0, v___x_1133_);
v___x_1135_ = v___x_1128_;
goto v_reusejp_1134_;
}
else
{
lean_object* v_reuseFailAlloc_1136_; 
v_reuseFailAlloc_1136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1136_, 0, v___x_1133_);
v___x_1135_ = v_reuseFailAlloc_1136_;
goto v_reusejp_1134_;
}
v_reusejp_1134_:
{
return v___x_1135_;
}
}
else
{
lean_object* v___x_1137_; lean_object* v_a_1138_; lean_object* v___x_1139_; lean_object* v_a_1140_; lean_object* v___x_1142_; uint8_t v_isShared_1143_; uint8_t v_isSharedCheck_1149_; 
lean_del_object(v___x_1128_);
v___x_1137_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1099_, v___y_1094_);
v_a_1138_ = lean_ctor_get(v___x_1137_, 0);
lean_inc(v_a_1138_);
lean_dec_ref(v___x_1137_);
v___x_1139_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1101_, v___y_1094_);
v_a_1140_ = lean_ctor_get(v___x_1139_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v___x_1139_);
if (v_isSharedCheck_1149_ == 0)
{
v___x_1142_ = v___x_1139_;
v_isShared_1143_ = v_isSharedCheck_1149_;
goto v_resetjp_1141_;
}
else
{
lean_inc(v_a_1140_);
lean_dec(v___x_1139_);
v___x_1142_ = lean_box(0);
v_isShared_1143_ = v_isSharedCheck_1149_;
goto v_resetjp_1141_;
}
v_resetjp_1141_:
{
lean_object* v___x_1144_; lean_object* v___x_1145_; lean_object* v___x_1147_; 
v___x_1144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1144_, 0, v_a_1140_);
lean_ctor_set(v___x_1144_, 1, v_a_1126_);
v___x_1145_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1145_, 0, v_a_1138_);
lean_ctor_set(v___x_1145_, 1, v___x_1144_);
if (v_isShared_1143_ == 0)
{
lean_ctor_set(v___x_1142_, 0, v___x_1145_);
v___x_1147_ = v___x_1142_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1148_; 
v_reuseFailAlloc_1148_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1148_, 0, v___x_1145_);
v___x_1147_ = v_reuseFailAlloc_1148_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
return v___x_1147_;
}
}
}
}
}
else
{
lean_object* v_a_1151_; lean_object* v___x_1153_; uint8_t v_isShared_1154_; uint8_t v_isSharedCheck_1158_; 
lean_dec(v_a_1101_);
lean_dec(v_a_1099_);
v_a_1151_ = lean_ctor_get(v___x_1125_, 0);
v_isSharedCheck_1158_ = !lean_is_exclusive(v___x_1125_);
if (v_isSharedCheck_1158_ == 0)
{
v___x_1153_ = v___x_1125_;
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
else
{
lean_inc(v_a_1151_);
lean_dec(v___x_1125_);
v___x_1153_ = lean_box(0);
v_isShared_1154_ = v_isSharedCheck_1158_;
goto v_resetjp_1152_;
}
v_resetjp_1152_:
{
lean_object* v___x_1156_; 
if (v_isShared_1154_ == 0)
{
v___x_1156_ = v___x_1153_;
goto v_reusejp_1155_;
}
else
{
lean_object* v_reuseFailAlloc_1157_; 
v_reuseFailAlloc_1157_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1157_, 0, v_a_1151_);
v___x_1156_ = v_reuseFailAlloc_1157_;
goto v_reusejp_1155_;
}
v_reusejp_1155_:
{
return v___x_1156_;
}
}
}
}
}
}
else
{
lean_object* v_a_1161_; lean_object* v___x_1163_; uint8_t v_isShared_1164_; uint8_t v_isSharedCheck_1168_; 
lean_dec(v_a_1099_);
lean_dec_ref(v___y_1093_);
lean_dec_ref(v_a_1091_);
v_a_1161_ = lean_ctor_get(v___x_1100_, 0);
v_isSharedCheck_1168_ = !lean_is_exclusive(v___x_1100_);
if (v_isSharedCheck_1168_ == 0)
{
v___x_1163_ = v___x_1100_;
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
else
{
lean_inc(v_a_1161_);
lean_dec(v___x_1100_);
v___x_1163_ = lean_box(0);
v_isShared_1164_ = v_isSharedCheck_1168_;
goto v_resetjp_1162_;
}
v_resetjp_1162_:
{
lean_object* v___x_1166_; 
if (v_isShared_1164_ == 0)
{
v___x_1166_ = v___x_1163_;
goto v_reusejp_1165_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v_a_1161_);
v___x_1166_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1165_;
}
v_reusejp_1165_:
{
return v___x_1166_;
}
}
}
}
else
{
lean_object* v_a_1169_; lean_object* v___x_1171_; uint8_t v_isShared_1172_; uint8_t v_isSharedCheck_1176_; 
lean_dec_ref(v___y_1093_);
lean_dec_ref(v_a_1091_);
lean_dec(v___x_1090_);
lean_dec(v___x_1088_);
v_a_1169_ = lean_ctor_get(v___x_1098_, 0);
v_isSharedCheck_1176_ = !lean_is_exclusive(v___x_1098_);
if (v_isSharedCheck_1176_ == 0)
{
v___x_1171_ = v___x_1098_;
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
else
{
lean_inc(v_a_1169_);
lean_dec(v___x_1098_);
v___x_1171_ = lean_box(0);
v_isShared_1172_ = v_isSharedCheck_1176_;
goto v_resetjp_1170_;
}
v_resetjp_1170_:
{
lean_object* v___x_1174_; 
if (v_isShared_1172_ == 0)
{
v___x_1174_ = v___x_1171_;
goto v_reusejp_1173_;
}
else
{
lean_object* v_reuseFailAlloc_1175_; 
v_reuseFailAlloc_1175_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1175_, 0, v_a_1169_);
v___x_1174_ = v_reuseFailAlloc_1175_;
goto v_reusejp_1173_;
}
v_reusejp_1173_:
{
return v___x_1174_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___boxed(lean_object* v___x_1177_, lean_object* v___x_1178_, lean_object* v___x_1179_, lean_object* v_a_1180_, lean_object* v___x_1181_, lean_object* v___y_1182_, lean_object* v___y_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_, lean_object* v___y_1186_){
_start:
{
uint8_t v___x_38011__boxed_1187_; uint8_t v___x_38014__boxed_1188_; lean_object* v_res_1189_; 
v___x_38011__boxed_1187_ = lean_unbox(v___x_1178_);
v___x_38014__boxed_1188_ = lean_unbox(v___x_1181_);
v_res_1189_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8(v___x_1177_, v___x_38011__boxed_1187_, v___x_1179_, v_a_1180_, v___x_38014__boxed_1188_, v___y_1182_, v___y_1183_, v___y_1184_, v___y_1185_);
lean_dec(v___y_1185_);
lean_dec_ref(v___y_1184_);
lean_dec(v___y_1183_);
return v_res_1189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9(lean_object* v___x_1190_, uint8_t v___x_1191_, lean_object* v___x_1192_, lean_object* v_a_1193_, uint8_t v___x_1194_, lean_object* v___y_1195_, lean_object* v___y_1196_, lean_object* v___y_1197_, lean_object* v___y_1198_){
_start:
{
lean_object* v___x_1200_; 
lean_inc(v___x_1192_);
lean_inc(v___x_1190_);
v___x_1200_ = l_Lean_Meta_mkFreshExprMVar(v___x_1190_, v___x_1191_, v___x_1192_, v___y_1195_, v___y_1196_, v___y_1197_, v___y_1198_);
if (lean_obj_tag(v___x_1200_) == 0)
{
lean_object* v_a_1201_; lean_object* v___x_1202_; 
v_a_1201_ = lean_ctor_get(v___x_1200_, 0);
lean_inc(v_a_1201_);
lean_dec_ref_known(v___x_1200_, 1);
v___x_1202_ = l_Lean_Meta_mkFreshExprMVar(v___x_1190_, v___x_1191_, v___x_1192_, v___y_1195_, v___y_1196_, v___y_1197_, v___y_1198_);
if (lean_obj_tag(v___x_1202_) == 0)
{
lean_object* v_a_1203_; lean_object* v_keyedConfig_1204_; uint8_t v_trackZetaDelta_1205_; lean_object* v_zetaDeltaSet_1206_; lean_object* v_lctx_1207_; lean_object* v_localInstances_1208_; lean_object* v_defEqCtx_x3f_1209_; lean_object* v_synthPendingDepth_1210_; lean_object* v_customCanUnfoldPredicate_x3f_1211_; uint8_t v_univApprox_1212_; uint8_t v_inTypeClassResolution_1213_; uint8_t v_cacheInferType_1214_; lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1260_; 
v_a_1203_ = lean_ctor_get(v___x_1202_, 0);
lean_inc(v_a_1203_);
lean_dec_ref_known(v___x_1202_, 1);
v_keyedConfig_1204_ = lean_ctor_get(v___y_1195_, 0);
v_trackZetaDelta_1205_ = lean_ctor_get_uint8(v___y_1195_, sizeof(void*)*7);
v_zetaDeltaSet_1206_ = lean_ctor_get(v___y_1195_, 1);
v_lctx_1207_ = lean_ctor_get(v___y_1195_, 2);
v_localInstances_1208_ = lean_ctor_get(v___y_1195_, 3);
v_defEqCtx_x3f_1209_ = lean_ctor_get(v___y_1195_, 4);
v_synthPendingDepth_1210_ = lean_ctor_get(v___y_1195_, 5);
v_customCanUnfoldPredicate_x3f_1211_ = lean_ctor_get(v___y_1195_, 6);
v_univApprox_1212_ = lean_ctor_get_uint8(v___y_1195_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1213_ = lean_ctor_get_uint8(v___y_1195_, sizeof(void*)*7 + 2);
v_cacheInferType_1214_ = lean_ctor_get_uint8(v___y_1195_, sizeof(void*)*7 + 3);
v_isSharedCheck_1260_ = !lean_is_exclusive(v___y_1195_);
if (v_isSharedCheck_1260_ == 0)
{
v___x_1216_ = v___y_1195_;
v_isShared_1217_ = v_isSharedCheck_1260_;
goto v_resetjp_1215_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1211_);
lean_inc(v_synthPendingDepth_1210_);
lean_inc(v_defEqCtx_x3f_1209_);
lean_inc(v_localInstances_1208_);
lean_inc(v_lctx_1207_);
lean_inc(v_zetaDeltaSet_1206_);
lean_inc(v_keyedConfig_1204_);
lean_dec(v___y_1195_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1260_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
lean_object* v___x_1218_; lean_object* v___x_1219_; lean_object* v___x_1220_; uint8_t v___x_1221_; lean_object* v___x_1222_; lean_object* v___x_1224_; 
v___x_1218_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2);
lean_inc(v_a_1201_);
v___x_1219_ = l_Lean_Expr_app___override(v___x_1218_, v_a_1201_);
lean_inc(v_a_1203_);
v___x_1220_ = l_Lean_Expr_app___override(v___x_1219_, v_a_1203_);
v___x_1221_ = 2;
v___x_1222_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1221_, v_keyedConfig_1204_);
if (v_isShared_1217_ == 0)
{
lean_ctor_set(v___x_1216_, 0, v___x_1222_);
v___x_1224_ = v___x_1216_;
goto v_reusejp_1223_;
}
else
{
lean_object* v_reuseFailAlloc_1259_; 
v_reuseFailAlloc_1259_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1259_, 0, v___x_1222_);
lean_ctor_set(v_reuseFailAlloc_1259_, 1, v_zetaDeltaSet_1206_);
lean_ctor_set(v_reuseFailAlloc_1259_, 2, v_lctx_1207_);
lean_ctor_set(v_reuseFailAlloc_1259_, 3, v_localInstances_1208_);
lean_ctor_set(v_reuseFailAlloc_1259_, 4, v_defEqCtx_x3f_1209_);
lean_ctor_set(v_reuseFailAlloc_1259_, 5, v_synthPendingDepth_1210_);
lean_ctor_set(v_reuseFailAlloc_1259_, 6, v_customCanUnfoldPredicate_x3f_1211_);
lean_ctor_set_uint8(v_reuseFailAlloc_1259_, sizeof(void*)*7, v_trackZetaDelta_1205_);
lean_ctor_set_uint8(v_reuseFailAlloc_1259_, sizeof(void*)*7 + 1, v_univApprox_1212_);
lean_ctor_set_uint8(v_reuseFailAlloc_1259_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1213_);
lean_ctor_set_uint8(v_reuseFailAlloc_1259_, sizeof(void*)*7 + 3, v_cacheInferType_1214_);
v___x_1224_ = v_reuseFailAlloc_1259_;
goto v_reusejp_1223_;
}
v_reusejp_1223_:
{
lean_object* v___x_1225_; 
v___x_1225_ = l_Lean_Meta_isExprDefEq(v___x_1220_, v_a_1193_, v___x_1224_, v___y_1196_, v___y_1197_, v___y_1198_);
lean_dec_ref(v___x_1224_);
if (lean_obj_tag(v___x_1225_) == 0)
{
lean_object* v_a_1226_; lean_object* v___x_1228_; uint8_t v_isShared_1229_; uint8_t v_isSharedCheck_1250_; 
v_a_1226_ = lean_ctor_get(v___x_1225_, 0);
v_isSharedCheck_1250_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1250_ == 0)
{
v___x_1228_ = v___x_1225_;
v_isShared_1229_ = v_isSharedCheck_1250_;
goto v_resetjp_1227_;
}
else
{
lean_inc(v_a_1226_);
lean_dec(v___x_1225_);
v___x_1228_ = lean_box(0);
v_isShared_1229_ = v_isSharedCheck_1250_;
goto v_resetjp_1227_;
}
v_resetjp_1227_:
{
uint8_t v___x_1230_; 
v___x_1230_ = lean_unbox(v_a_1226_);
if (v___x_1230_ == 0)
{
lean_object* v___x_1231_; lean_object* v___x_1232_; lean_object* v___x_1233_; lean_object* v___x_1235_; 
lean_dec(v_a_1226_);
v___x_1231_ = lean_box(v___x_1194_);
v___x_1232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1232_, 0, v_a_1203_);
lean_ctor_set(v___x_1232_, 1, v___x_1231_);
v___x_1233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1233_, 0, v_a_1201_);
lean_ctor_set(v___x_1233_, 1, v___x_1232_);
if (v_isShared_1229_ == 0)
{
lean_ctor_set(v___x_1228_, 0, v___x_1233_);
v___x_1235_ = v___x_1228_;
goto v_reusejp_1234_;
}
else
{
lean_object* v_reuseFailAlloc_1236_; 
v_reuseFailAlloc_1236_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1236_, 0, v___x_1233_);
v___x_1235_ = v_reuseFailAlloc_1236_;
goto v_reusejp_1234_;
}
v_reusejp_1234_:
{
return v___x_1235_;
}
}
else
{
lean_object* v___x_1237_; lean_object* v_a_1238_; lean_object* v___x_1239_; lean_object* v_a_1240_; lean_object* v___x_1242_; uint8_t v_isShared_1243_; uint8_t v_isSharedCheck_1249_; 
lean_del_object(v___x_1228_);
v___x_1237_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1201_, v___y_1196_);
v_a_1238_ = lean_ctor_get(v___x_1237_, 0);
lean_inc(v_a_1238_);
lean_dec_ref(v___x_1237_);
v___x_1239_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1203_, v___y_1196_);
v_a_1240_ = lean_ctor_get(v___x_1239_, 0);
v_isSharedCheck_1249_ = !lean_is_exclusive(v___x_1239_);
if (v_isSharedCheck_1249_ == 0)
{
v___x_1242_ = v___x_1239_;
v_isShared_1243_ = v_isSharedCheck_1249_;
goto v_resetjp_1241_;
}
else
{
lean_inc(v_a_1240_);
lean_dec(v___x_1239_);
v___x_1242_ = lean_box(0);
v_isShared_1243_ = v_isSharedCheck_1249_;
goto v_resetjp_1241_;
}
v_resetjp_1241_:
{
lean_object* v___x_1244_; lean_object* v___x_1245_; lean_object* v___x_1247_; 
v___x_1244_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1244_, 0, v_a_1240_);
lean_ctor_set(v___x_1244_, 1, v_a_1226_);
v___x_1245_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1245_, 0, v_a_1238_);
lean_ctor_set(v___x_1245_, 1, v___x_1244_);
if (v_isShared_1243_ == 0)
{
lean_ctor_set(v___x_1242_, 0, v___x_1245_);
v___x_1247_ = v___x_1242_;
goto v_reusejp_1246_;
}
else
{
lean_object* v_reuseFailAlloc_1248_; 
v_reuseFailAlloc_1248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1248_, 0, v___x_1245_);
v___x_1247_ = v_reuseFailAlloc_1248_;
goto v_reusejp_1246_;
}
v_reusejp_1246_:
{
return v___x_1247_;
}
}
}
}
}
else
{
lean_object* v_a_1251_; lean_object* v___x_1253_; uint8_t v_isShared_1254_; uint8_t v_isSharedCheck_1258_; 
lean_dec(v_a_1203_);
lean_dec(v_a_1201_);
v_a_1251_ = lean_ctor_get(v___x_1225_, 0);
v_isSharedCheck_1258_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1258_ == 0)
{
v___x_1253_ = v___x_1225_;
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
else
{
lean_inc(v_a_1251_);
lean_dec(v___x_1225_);
v___x_1253_ = lean_box(0);
v_isShared_1254_ = v_isSharedCheck_1258_;
goto v_resetjp_1252_;
}
v_resetjp_1252_:
{
lean_object* v___x_1256_; 
if (v_isShared_1254_ == 0)
{
v___x_1256_ = v___x_1253_;
goto v_reusejp_1255_;
}
else
{
lean_object* v_reuseFailAlloc_1257_; 
v_reuseFailAlloc_1257_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1257_, 0, v_a_1251_);
v___x_1256_ = v_reuseFailAlloc_1257_;
goto v_reusejp_1255_;
}
v_reusejp_1255_:
{
return v___x_1256_;
}
}
}
}
}
}
else
{
lean_object* v_a_1261_; lean_object* v___x_1263_; uint8_t v_isShared_1264_; uint8_t v_isSharedCheck_1268_; 
lean_dec(v_a_1201_);
lean_dec_ref(v___y_1195_);
lean_dec_ref(v_a_1193_);
v_a_1261_ = lean_ctor_get(v___x_1202_, 0);
v_isSharedCheck_1268_ = !lean_is_exclusive(v___x_1202_);
if (v_isSharedCheck_1268_ == 0)
{
v___x_1263_ = v___x_1202_;
v_isShared_1264_ = v_isSharedCheck_1268_;
goto v_resetjp_1262_;
}
else
{
lean_inc(v_a_1261_);
lean_dec(v___x_1202_);
v___x_1263_ = lean_box(0);
v_isShared_1264_ = v_isSharedCheck_1268_;
goto v_resetjp_1262_;
}
v_resetjp_1262_:
{
lean_object* v___x_1266_; 
if (v_isShared_1264_ == 0)
{
v___x_1266_ = v___x_1263_;
goto v_reusejp_1265_;
}
else
{
lean_object* v_reuseFailAlloc_1267_; 
v_reuseFailAlloc_1267_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1267_, 0, v_a_1261_);
v___x_1266_ = v_reuseFailAlloc_1267_;
goto v_reusejp_1265_;
}
v_reusejp_1265_:
{
return v___x_1266_;
}
}
}
}
else
{
lean_object* v_a_1269_; lean_object* v___x_1271_; uint8_t v_isShared_1272_; uint8_t v_isSharedCheck_1276_; 
lean_dec_ref(v___y_1195_);
lean_dec_ref(v_a_1193_);
lean_dec(v___x_1192_);
lean_dec(v___x_1190_);
v_a_1269_ = lean_ctor_get(v___x_1200_, 0);
v_isSharedCheck_1276_ = !lean_is_exclusive(v___x_1200_);
if (v_isSharedCheck_1276_ == 0)
{
v___x_1271_ = v___x_1200_;
v_isShared_1272_ = v_isSharedCheck_1276_;
goto v_resetjp_1270_;
}
else
{
lean_inc(v_a_1269_);
lean_dec(v___x_1200_);
v___x_1271_ = lean_box(0);
v_isShared_1272_ = v_isSharedCheck_1276_;
goto v_resetjp_1270_;
}
v_resetjp_1270_:
{
lean_object* v___x_1274_; 
if (v_isShared_1272_ == 0)
{
v___x_1274_ = v___x_1271_;
goto v_reusejp_1273_;
}
else
{
lean_object* v_reuseFailAlloc_1275_; 
v_reuseFailAlloc_1275_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1275_, 0, v_a_1269_);
v___x_1274_ = v_reuseFailAlloc_1275_;
goto v_reusejp_1273_;
}
v_reusejp_1273_:
{
return v___x_1274_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9___boxed(lean_object* v___x_1277_, lean_object* v___x_1278_, lean_object* v___x_1279_, lean_object* v_a_1280_, lean_object* v___x_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_, lean_object* v___y_1286_){
_start:
{
uint8_t v___x_38191__boxed_1287_; uint8_t v___x_38194__boxed_1288_; lean_object* v_res_1289_; 
v___x_38191__boxed_1287_ = lean_unbox(v___x_1278_);
v___x_38194__boxed_1288_ = lean_unbox(v___x_1281_);
v_res_1289_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9(v___x_1277_, v___x_38191__boxed_1287_, v___x_1279_, v_a_1280_, v___x_38194__boxed_1288_, v___y_1282_, v___y_1283_, v___y_1284_, v___y_1285_);
lean_dec(v___y_1285_);
lean_dec_ref(v___y_1284_);
lean_dec(v___y_1283_);
return v_res_1289_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2(void){
_start:
{
lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___x_1293_ = lean_box(0);
v___x_1294_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__1));
v___x_1295_ = l_Lean_Expr_const___override(v___x_1294_, v___x_1293_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10(lean_object* v___x_1296_, uint8_t v___x_1297_, lean_object* v___x_1298_, lean_object* v_a_1299_, uint8_t v___x_1300_, lean_object* v___y_1301_, lean_object* v___y_1302_, lean_object* v___y_1303_, lean_object* v___y_1304_){
_start:
{
lean_object* v___x_1306_; 
lean_inc(v___x_1298_);
v___x_1306_ = l_Lean_Meta_mkFreshExprMVar(v___x_1296_, v___x_1297_, v___x_1298_, v___y_1301_, v___y_1302_, v___y_1303_, v___y_1304_);
if (lean_obj_tag(v___x_1306_) == 0)
{
lean_object* v_a_1307_; lean_object* v_keyedConfig_1308_; uint8_t v_trackZetaDelta_1309_; lean_object* v_zetaDeltaSet_1310_; lean_object* v_lctx_1311_; lean_object* v_localInstances_1312_; lean_object* v_defEqCtx_x3f_1313_; lean_object* v_synthPendingDepth_1314_; lean_object* v_customCanUnfoldPredicate_x3f_1315_; uint8_t v_univApprox_1316_; uint8_t v_inTypeClassResolution_1317_; uint8_t v_cacheInferType_1318_; lean_object* v___x_1320_; uint8_t v_isShared_1321_; uint8_t v_isSharedCheck_1360_; 
v_a_1307_ = lean_ctor_get(v___x_1306_, 0);
lean_inc(v_a_1307_);
lean_dec_ref_known(v___x_1306_, 1);
v_keyedConfig_1308_ = lean_ctor_get(v___y_1301_, 0);
v_trackZetaDelta_1309_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7);
v_zetaDeltaSet_1310_ = lean_ctor_get(v___y_1301_, 1);
v_lctx_1311_ = lean_ctor_get(v___y_1301_, 2);
v_localInstances_1312_ = lean_ctor_get(v___y_1301_, 3);
v_defEqCtx_x3f_1313_ = lean_ctor_get(v___y_1301_, 4);
v_synthPendingDepth_1314_ = lean_ctor_get(v___y_1301_, 5);
v_customCanUnfoldPredicate_x3f_1315_ = lean_ctor_get(v___y_1301_, 6);
v_univApprox_1316_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1317_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 2);
v_cacheInferType_1318_ = lean_ctor_get_uint8(v___y_1301_, sizeof(void*)*7 + 3);
v_isSharedCheck_1360_ = !lean_is_exclusive(v___y_1301_);
if (v_isSharedCheck_1360_ == 0)
{
v___x_1320_ = v___y_1301_;
v_isShared_1321_ = v_isSharedCheck_1360_;
goto v_resetjp_1319_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1315_);
lean_inc(v_synthPendingDepth_1314_);
lean_inc(v_defEqCtx_x3f_1313_);
lean_inc(v_localInstances_1312_);
lean_inc(v_lctx_1311_);
lean_inc(v_zetaDeltaSet_1310_);
lean_inc(v_keyedConfig_1308_);
lean_dec(v___y_1301_);
v___x_1320_ = lean_box(0);
v_isShared_1321_ = v_isSharedCheck_1360_;
goto v_resetjp_1319_;
}
v_resetjp_1319_:
{
lean_object* v___x_1322_; uint8_t v___x_1323_; lean_object* v___x_1324_; uint8_t v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1328_; 
v___x_1322_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2);
v___x_1323_ = 0;
lean_inc(v_a_1307_);
v___x_1324_ = l_Lean_Expr_forallE___override(v___x_1298_, v_a_1307_, v___x_1322_, v___x_1323_);
v___x_1325_ = 2;
v___x_1326_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1325_, v_keyedConfig_1308_);
if (v_isShared_1321_ == 0)
{
lean_ctor_set(v___x_1320_, 0, v___x_1326_);
v___x_1328_ = v___x_1320_;
goto v_reusejp_1327_;
}
else
{
lean_object* v_reuseFailAlloc_1359_; 
v_reuseFailAlloc_1359_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1359_, 0, v___x_1326_);
lean_ctor_set(v_reuseFailAlloc_1359_, 1, v_zetaDeltaSet_1310_);
lean_ctor_set(v_reuseFailAlloc_1359_, 2, v_lctx_1311_);
lean_ctor_set(v_reuseFailAlloc_1359_, 3, v_localInstances_1312_);
lean_ctor_set(v_reuseFailAlloc_1359_, 4, v_defEqCtx_x3f_1313_);
lean_ctor_set(v_reuseFailAlloc_1359_, 5, v_synthPendingDepth_1314_);
lean_ctor_set(v_reuseFailAlloc_1359_, 6, v_customCanUnfoldPredicate_x3f_1315_);
lean_ctor_set_uint8(v_reuseFailAlloc_1359_, sizeof(void*)*7, v_trackZetaDelta_1309_);
lean_ctor_set_uint8(v_reuseFailAlloc_1359_, sizeof(void*)*7 + 1, v_univApprox_1316_);
lean_ctor_set_uint8(v_reuseFailAlloc_1359_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1317_);
lean_ctor_set_uint8(v_reuseFailAlloc_1359_, sizeof(void*)*7 + 3, v_cacheInferType_1318_);
v___x_1328_ = v_reuseFailAlloc_1359_;
goto v_reusejp_1327_;
}
v_reusejp_1327_:
{
lean_object* v___x_1329_; 
v___x_1329_ = l_Lean_Meta_isExprDefEq(v___x_1324_, v_a_1299_, v___x_1328_, v___y_1302_, v___y_1303_, v___y_1304_);
lean_dec_ref(v___x_1328_);
if (lean_obj_tag(v___x_1329_) == 0)
{
lean_object* v_a_1330_; lean_object* v___x_1332_; uint8_t v_isShared_1333_; uint8_t v_isSharedCheck_1350_; 
v_a_1330_ = lean_ctor_get(v___x_1329_, 0);
v_isSharedCheck_1350_ = !lean_is_exclusive(v___x_1329_);
if (v_isSharedCheck_1350_ == 0)
{
v___x_1332_ = v___x_1329_;
v_isShared_1333_ = v_isSharedCheck_1350_;
goto v_resetjp_1331_;
}
else
{
lean_inc(v_a_1330_);
lean_dec(v___x_1329_);
v___x_1332_ = lean_box(0);
v_isShared_1333_ = v_isSharedCheck_1350_;
goto v_resetjp_1331_;
}
v_resetjp_1331_:
{
uint8_t v___x_1334_; 
v___x_1334_ = lean_unbox(v_a_1330_);
if (v___x_1334_ == 0)
{
lean_object* v___x_1335_; lean_object* v___x_1336_; lean_object* v___x_1338_; 
lean_dec(v_a_1330_);
v___x_1335_ = lean_box(v___x_1300_);
v___x_1336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1336_, 0, v_a_1307_);
lean_ctor_set(v___x_1336_, 1, v___x_1335_);
if (v_isShared_1333_ == 0)
{
lean_ctor_set(v___x_1332_, 0, v___x_1336_);
v___x_1338_ = v___x_1332_;
goto v_reusejp_1337_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v___x_1336_);
v___x_1338_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1337_;
}
v_reusejp_1337_:
{
return v___x_1338_;
}
}
else
{
lean_object* v___x_1340_; lean_object* v_a_1341_; lean_object* v___x_1343_; uint8_t v_isShared_1344_; uint8_t v_isSharedCheck_1349_; 
lean_del_object(v___x_1332_);
v___x_1340_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1307_, v___y_1302_);
v_a_1341_ = lean_ctor_get(v___x_1340_, 0);
v_isSharedCheck_1349_ = !lean_is_exclusive(v___x_1340_);
if (v_isSharedCheck_1349_ == 0)
{
v___x_1343_ = v___x_1340_;
v_isShared_1344_ = v_isSharedCheck_1349_;
goto v_resetjp_1342_;
}
else
{
lean_inc(v_a_1341_);
lean_dec(v___x_1340_);
v___x_1343_ = lean_box(0);
v_isShared_1344_ = v_isSharedCheck_1349_;
goto v_resetjp_1342_;
}
v_resetjp_1342_:
{
lean_object* v___x_1345_; lean_object* v___x_1347_; 
v___x_1345_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1345_, 0, v_a_1341_);
lean_ctor_set(v___x_1345_, 1, v_a_1330_);
if (v_isShared_1344_ == 0)
{
lean_ctor_set(v___x_1343_, 0, v___x_1345_);
v___x_1347_ = v___x_1343_;
goto v_reusejp_1346_;
}
else
{
lean_object* v_reuseFailAlloc_1348_; 
v_reuseFailAlloc_1348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1348_, 0, v___x_1345_);
v___x_1347_ = v_reuseFailAlloc_1348_;
goto v_reusejp_1346_;
}
v_reusejp_1346_:
{
return v___x_1347_;
}
}
}
}
}
else
{
lean_object* v_a_1351_; lean_object* v___x_1353_; uint8_t v_isShared_1354_; uint8_t v_isSharedCheck_1358_; 
lean_dec(v_a_1307_);
v_a_1351_ = lean_ctor_get(v___x_1329_, 0);
v_isSharedCheck_1358_ = !lean_is_exclusive(v___x_1329_);
if (v_isSharedCheck_1358_ == 0)
{
v___x_1353_ = v___x_1329_;
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
else
{
lean_inc(v_a_1351_);
lean_dec(v___x_1329_);
v___x_1353_ = lean_box(0);
v_isShared_1354_ = v_isSharedCheck_1358_;
goto v_resetjp_1352_;
}
v_resetjp_1352_:
{
lean_object* v___x_1356_; 
if (v_isShared_1354_ == 0)
{
v___x_1356_ = v___x_1353_;
goto v_reusejp_1355_;
}
else
{
lean_object* v_reuseFailAlloc_1357_; 
v_reuseFailAlloc_1357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1357_, 0, v_a_1351_);
v___x_1356_ = v_reuseFailAlloc_1357_;
goto v_reusejp_1355_;
}
v_reusejp_1355_:
{
return v___x_1356_;
}
}
}
}
}
}
else
{
lean_object* v_a_1361_; lean_object* v___x_1363_; uint8_t v_isShared_1364_; uint8_t v_isSharedCheck_1368_; 
lean_dec_ref(v___y_1301_);
lean_dec_ref(v_a_1299_);
lean_dec(v___x_1298_);
v_a_1361_ = lean_ctor_get(v___x_1306_, 0);
v_isSharedCheck_1368_ = !lean_is_exclusive(v___x_1306_);
if (v_isSharedCheck_1368_ == 0)
{
v___x_1363_ = v___x_1306_;
v_isShared_1364_ = v_isSharedCheck_1368_;
goto v_resetjp_1362_;
}
else
{
lean_inc(v_a_1361_);
lean_dec(v___x_1306_);
v___x_1363_ = lean_box(0);
v_isShared_1364_ = v_isSharedCheck_1368_;
goto v_resetjp_1362_;
}
v_resetjp_1362_:
{
lean_object* v___x_1366_; 
if (v_isShared_1364_ == 0)
{
v___x_1366_ = v___x_1363_;
goto v_reusejp_1365_;
}
else
{
lean_object* v_reuseFailAlloc_1367_; 
v_reuseFailAlloc_1367_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1367_, 0, v_a_1361_);
v___x_1366_ = v_reuseFailAlloc_1367_;
goto v_reusejp_1365_;
}
v_reusejp_1365_:
{
return v___x_1366_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___boxed(lean_object* v___x_1369_, lean_object* v___x_1370_, lean_object* v___x_1371_, lean_object* v_a_1372_, lean_object* v___x_1373_, lean_object* v___y_1374_, lean_object* v___y_1375_, lean_object* v___y_1376_, lean_object* v___y_1377_, lean_object* v___y_1378_){
_start:
{
uint8_t v___x_38370__boxed_1379_; uint8_t v___x_38373__boxed_1380_; lean_object* v_res_1381_; 
v___x_38370__boxed_1379_ = lean_unbox(v___x_1370_);
v___x_38373__boxed_1380_ = lean_unbox(v___x_1373_);
v_res_1381_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10(v___x_1369_, v___x_38370__boxed_1379_, v___x_1371_, v_a_1372_, v___x_38373__boxed_1380_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_);
lean_dec(v___y_1377_);
lean_dec_ref(v___y_1376_);
lean_dec(v___y_1375_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11(lean_object* v___x_1382_, uint8_t v___x_1383_, lean_object* v___x_1384_, lean_object* v_a_1385_, lean_object* v___y_1386_, lean_object* v___y_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_){
_start:
{
lean_object* v___x_1391_; 
lean_inc(v___x_1384_);
lean_inc(v___x_1382_);
v___x_1391_ = l_Lean_Meta_mkFreshExprMVar(v___x_1382_, v___x_1383_, v___x_1384_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
if (lean_obj_tag(v___x_1391_) == 0)
{
lean_object* v_a_1392_; lean_object* v___x_1393_; 
v_a_1392_ = lean_ctor_get(v___x_1391_, 0);
lean_inc(v_a_1392_);
lean_dec_ref_known(v___x_1391_, 1);
lean_inc(v___x_1384_);
v___x_1393_ = l_Lean_Meta_mkFreshExprMVar(v___x_1382_, v___x_1383_, v___x_1384_, v___y_1386_, v___y_1387_, v___y_1388_, v___y_1389_);
if (lean_obj_tag(v___x_1393_) == 0)
{
lean_object* v_a_1394_; lean_object* v_keyedConfig_1395_; uint8_t v_trackZetaDelta_1396_; lean_object* v_zetaDeltaSet_1397_; lean_object* v_lctx_1398_; lean_object* v_localInstances_1399_; lean_object* v_defEqCtx_x3f_1400_; lean_object* v_synthPendingDepth_1401_; lean_object* v_customCanUnfoldPredicate_x3f_1402_; uint8_t v_univApprox_1403_; uint8_t v_inTypeClassResolution_1404_; uint8_t v_cacheInferType_1405_; lean_object* v___x_1407_; uint8_t v_isShared_1408_; uint8_t v_isSharedCheck_1449_; 
v_a_1394_ = lean_ctor_get(v___x_1393_, 0);
lean_inc(v_a_1394_);
lean_dec_ref_known(v___x_1393_, 1);
v_keyedConfig_1395_ = lean_ctor_get(v___y_1386_, 0);
v_trackZetaDelta_1396_ = lean_ctor_get_uint8(v___y_1386_, sizeof(void*)*7);
v_zetaDeltaSet_1397_ = lean_ctor_get(v___y_1386_, 1);
v_lctx_1398_ = lean_ctor_get(v___y_1386_, 2);
v_localInstances_1399_ = lean_ctor_get(v___y_1386_, 3);
v_defEqCtx_x3f_1400_ = lean_ctor_get(v___y_1386_, 4);
v_synthPendingDepth_1401_ = lean_ctor_get(v___y_1386_, 5);
v_customCanUnfoldPredicate_x3f_1402_ = lean_ctor_get(v___y_1386_, 6);
v_univApprox_1403_ = lean_ctor_get_uint8(v___y_1386_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_1404_ = lean_ctor_get_uint8(v___y_1386_, sizeof(void*)*7 + 2);
v_cacheInferType_1405_ = lean_ctor_get_uint8(v___y_1386_, sizeof(void*)*7 + 3);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___y_1386_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1407_ = v___y_1386_;
v_isShared_1408_ = v_isSharedCheck_1449_;
goto v_resetjp_1406_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_1402_);
lean_inc(v_synthPendingDepth_1401_);
lean_inc(v_defEqCtx_x3f_1400_);
lean_inc(v_localInstances_1399_);
lean_inc(v_lctx_1398_);
lean_inc(v_zetaDeltaSet_1397_);
lean_inc(v_keyedConfig_1395_);
lean_dec(v___y_1386_);
v___x_1407_ = lean_box(0);
v_isShared_1408_ = v_isSharedCheck_1449_;
goto v_resetjp_1406_;
}
v_resetjp_1406_:
{
uint8_t v___x_1409_; lean_object* v___x_1410_; uint8_t v___x_1411_; lean_object* v___x_1412_; lean_object* v___x_1414_; 
v___x_1409_ = 0;
lean_inc(v_a_1394_);
lean_inc(v_a_1392_);
v___x_1410_ = l_Lean_Expr_forallE___override(v___x_1384_, v_a_1392_, v_a_1394_, v___x_1409_);
v___x_1411_ = 2;
v___x_1412_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_1411_, v_keyedConfig_1395_);
if (v_isShared_1408_ == 0)
{
lean_ctor_set(v___x_1407_, 0, v___x_1412_);
v___x_1414_ = v___x_1407_;
goto v_reusejp_1413_;
}
else
{
lean_object* v_reuseFailAlloc_1448_; 
v_reuseFailAlloc_1448_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_1448_, 0, v___x_1412_);
lean_ctor_set(v_reuseFailAlloc_1448_, 1, v_zetaDeltaSet_1397_);
lean_ctor_set(v_reuseFailAlloc_1448_, 2, v_lctx_1398_);
lean_ctor_set(v_reuseFailAlloc_1448_, 3, v_localInstances_1399_);
lean_ctor_set(v_reuseFailAlloc_1448_, 4, v_defEqCtx_x3f_1400_);
lean_ctor_set(v_reuseFailAlloc_1448_, 5, v_synthPendingDepth_1401_);
lean_ctor_set(v_reuseFailAlloc_1448_, 6, v_customCanUnfoldPredicate_x3f_1402_);
lean_ctor_set_uint8(v_reuseFailAlloc_1448_, sizeof(void*)*7, v_trackZetaDelta_1396_);
lean_ctor_set_uint8(v_reuseFailAlloc_1448_, sizeof(void*)*7 + 1, v_univApprox_1403_);
lean_ctor_set_uint8(v_reuseFailAlloc_1448_, sizeof(void*)*7 + 2, v_inTypeClassResolution_1404_);
lean_ctor_set_uint8(v_reuseFailAlloc_1448_, sizeof(void*)*7 + 3, v_cacheInferType_1405_);
v___x_1414_ = v_reuseFailAlloc_1448_;
goto v_reusejp_1413_;
}
v_reusejp_1413_:
{
lean_object* v___x_1415_; 
v___x_1415_ = l_Lean_Meta_isExprDefEq(v___x_1410_, v_a_1385_, v___x_1414_, v___y_1387_, v___y_1388_, v___y_1389_);
lean_dec_ref(v___x_1414_);
if (lean_obj_tag(v___x_1415_) == 0)
{
lean_object* v_a_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1439_; 
v_a_1416_ = lean_ctor_get(v___x_1415_, 0);
v_isSharedCheck_1439_ = !lean_is_exclusive(v___x_1415_);
if (v_isSharedCheck_1439_ == 0)
{
v___x_1418_ = v___x_1415_;
v_isShared_1419_ = v_isSharedCheck_1439_;
goto v_resetjp_1417_;
}
else
{
lean_inc(v_a_1416_);
lean_dec(v___x_1415_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1439_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
uint8_t v___x_1420_; 
v___x_1420_ = lean_unbox(v_a_1416_);
if (v___x_1420_ == 0)
{
lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1424_; 
v___x_1421_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1421_, 0, v_a_1394_);
lean_ctor_set(v___x_1421_, 1, v_a_1416_);
v___x_1422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1422_, 0, v_a_1392_);
lean_ctor_set(v___x_1422_, 1, v___x_1421_);
if (v_isShared_1419_ == 0)
{
lean_ctor_set(v___x_1418_, 0, v___x_1422_);
v___x_1424_ = v___x_1418_;
goto v_reusejp_1423_;
}
else
{
lean_object* v_reuseFailAlloc_1425_; 
v_reuseFailAlloc_1425_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1425_, 0, v___x_1422_);
v___x_1424_ = v_reuseFailAlloc_1425_;
goto v_reusejp_1423_;
}
v_reusejp_1423_:
{
return v___x_1424_;
}
}
else
{
lean_object* v___x_1426_; lean_object* v_a_1427_; lean_object* v___x_1428_; lean_object* v_a_1429_; lean_object* v___x_1431_; uint8_t v_isShared_1432_; uint8_t v_isSharedCheck_1438_; 
lean_del_object(v___x_1418_);
v___x_1426_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1392_, v___y_1387_);
v_a_1427_ = lean_ctor_get(v___x_1426_, 0);
lean_inc(v_a_1427_);
lean_dec_ref(v___x_1426_);
v___x_1428_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_1394_, v___y_1387_);
v_a_1429_ = lean_ctor_get(v___x_1428_, 0);
v_isSharedCheck_1438_ = !lean_is_exclusive(v___x_1428_);
if (v_isSharedCheck_1438_ == 0)
{
v___x_1431_ = v___x_1428_;
v_isShared_1432_ = v_isSharedCheck_1438_;
goto v_resetjp_1430_;
}
else
{
lean_inc(v_a_1429_);
lean_dec(v___x_1428_);
v___x_1431_ = lean_box(0);
v_isShared_1432_ = v_isSharedCheck_1438_;
goto v_resetjp_1430_;
}
v_resetjp_1430_:
{
lean_object* v___x_1433_; lean_object* v___x_1434_; lean_object* v___x_1436_; 
v___x_1433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1433_, 0, v_a_1429_);
lean_ctor_set(v___x_1433_, 1, v_a_1416_);
v___x_1434_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1434_, 0, v_a_1427_);
lean_ctor_set(v___x_1434_, 1, v___x_1433_);
if (v_isShared_1432_ == 0)
{
lean_ctor_set(v___x_1431_, 0, v___x_1434_);
v___x_1436_ = v___x_1431_;
goto v_reusejp_1435_;
}
else
{
lean_object* v_reuseFailAlloc_1437_; 
v_reuseFailAlloc_1437_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1437_, 0, v___x_1434_);
v___x_1436_ = v_reuseFailAlloc_1437_;
goto v_reusejp_1435_;
}
v_reusejp_1435_:
{
return v___x_1436_;
}
}
}
}
}
else
{
lean_object* v_a_1440_; lean_object* v___x_1442_; uint8_t v_isShared_1443_; uint8_t v_isSharedCheck_1447_; 
lean_dec(v_a_1394_);
lean_dec(v_a_1392_);
v_a_1440_ = lean_ctor_get(v___x_1415_, 0);
v_isSharedCheck_1447_ = !lean_is_exclusive(v___x_1415_);
if (v_isSharedCheck_1447_ == 0)
{
v___x_1442_ = v___x_1415_;
v_isShared_1443_ = v_isSharedCheck_1447_;
goto v_resetjp_1441_;
}
else
{
lean_inc(v_a_1440_);
lean_dec(v___x_1415_);
v___x_1442_ = lean_box(0);
v_isShared_1443_ = v_isSharedCheck_1447_;
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
lean_object* v_reuseFailAlloc_1446_; 
v_reuseFailAlloc_1446_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1446_, 0, v_a_1440_);
v___x_1445_ = v_reuseFailAlloc_1446_;
goto v_reusejp_1444_;
}
v_reusejp_1444_:
{
return v___x_1445_;
}
}
}
}
}
}
else
{
lean_object* v_a_1450_; lean_object* v___x_1452_; uint8_t v_isShared_1453_; uint8_t v_isSharedCheck_1457_; 
lean_dec(v_a_1392_);
lean_dec_ref(v___y_1386_);
lean_dec_ref(v_a_1385_);
lean_dec(v___x_1384_);
v_a_1450_ = lean_ctor_get(v___x_1393_, 0);
v_isSharedCheck_1457_ = !lean_is_exclusive(v___x_1393_);
if (v_isSharedCheck_1457_ == 0)
{
v___x_1452_ = v___x_1393_;
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
else
{
lean_inc(v_a_1450_);
lean_dec(v___x_1393_);
v___x_1452_ = lean_box(0);
v_isShared_1453_ = v_isSharedCheck_1457_;
goto v_resetjp_1451_;
}
v_resetjp_1451_:
{
lean_object* v___x_1455_; 
if (v_isShared_1453_ == 0)
{
v___x_1455_ = v___x_1452_;
goto v_reusejp_1454_;
}
else
{
lean_object* v_reuseFailAlloc_1456_; 
v_reuseFailAlloc_1456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1456_, 0, v_a_1450_);
v___x_1455_ = v_reuseFailAlloc_1456_;
goto v_reusejp_1454_;
}
v_reusejp_1454_:
{
return v___x_1455_;
}
}
}
}
else
{
lean_object* v_a_1458_; lean_object* v___x_1460_; uint8_t v_isShared_1461_; uint8_t v_isSharedCheck_1465_; 
lean_dec_ref(v___y_1386_);
lean_dec_ref(v_a_1385_);
lean_dec(v___x_1384_);
lean_dec(v___x_1382_);
v_a_1458_ = lean_ctor_get(v___x_1391_, 0);
v_isSharedCheck_1465_ = !lean_is_exclusive(v___x_1391_);
if (v_isSharedCheck_1465_ == 0)
{
v___x_1460_ = v___x_1391_;
v_isShared_1461_ = v_isSharedCheck_1465_;
goto v_resetjp_1459_;
}
else
{
lean_inc(v_a_1458_);
lean_dec(v___x_1391_);
v___x_1460_ = lean_box(0);
v_isShared_1461_ = v_isSharedCheck_1465_;
goto v_resetjp_1459_;
}
v_resetjp_1459_:
{
lean_object* v___x_1463_; 
if (v_isShared_1461_ == 0)
{
v___x_1463_ = v___x_1460_;
goto v_reusejp_1462_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v_a_1458_);
v___x_1463_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1462_;
}
v_reusejp_1462_:
{
return v___x_1463_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11___boxed(lean_object* v___x_1466_, lean_object* v___x_1467_, lean_object* v___x_1468_, lean_object* v_a_1469_, lean_object* v___y_1470_, lean_object* v___y_1471_, lean_object* v___y_1472_, lean_object* v___y_1473_, lean_object* v___y_1474_){
_start:
{
uint8_t v___x_38511__boxed_1475_; lean_object* v_res_1476_; 
v___x_38511__boxed_1475_ = lean_unbox(v___x_1467_);
v_res_1476_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11(v___x_1466_, v___x_38511__boxed_1475_, v___x_1468_, v_a_1469_, v___y_1470_, v___y_1471_, v___y_1472_, v___y_1473_);
lean_dec(v___y_1473_);
lean_dec_ref(v___y_1472_);
lean_dec(v___y_1471_);
return v_res_1476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(lean_object* v_msgData_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_, lean_object* v___y_1480_, lean_object* v___y_1481_){
_start:
{
lean_object* v___x_1483_; lean_object* v_env_1484_; lean_object* v___x_1485_; lean_object* v_mctx_1486_; lean_object* v_lctx_1487_; lean_object* v_options_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; 
v___x_1483_ = lean_st_ref_get(v___y_1481_);
v_env_1484_ = lean_ctor_get(v___x_1483_, 0);
lean_inc_ref(v_env_1484_);
lean_dec(v___x_1483_);
v___x_1485_ = lean_st_ref_get(v___y_1479_);
v_mctx_1486_ = lean_ctor_get(v___x_1485_, 0);
lean_inc_ref(v_mctx_1486_);
lean_dec(v___x_1485_);
v_lctx_1487_ = lean_ctor_get(v___y_1478_, 2);
v_options_1488_ = lean_ctor_get(v___y_1480_, 2);
lean_inc_ref(v_options_1488_);
lean_inc_ref(v_lctx_1487_);
v___x_1489_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1489_, 0, v_env_1484_);
lean_ctor_set(v___x_1489_, 1, v_mctx_1486_);
lean_ctor_set(v___x_1489_, 2, v_lctx_1487_);
lean_ctor_set(v___x_1489_, 3, v_options_1488_);
v___x_1490_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1490_, 0, v___x_1489_);
lean_ctor_set(v___x_1490_, 1, v_msgData_1477_);
v___x_1491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1491_, 0, v___x_1490_);
return v___x_1491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3___boxed(lean_object* v_msgData_1492_, lean_object* v___y_1493_, lean_object* v___y_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_){
_start:
{
lean_object* v_res_1498_; 
v_res_1498_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(v_msgData_1492_, v___y_1493_, v___y_1494_, v___y_1495_, v___y_1496_);
lean_dec(v___y_1496_);
lean_dec_ref(v___y_1495_);
lean_dec(v___y_1494_);
lean_dec_ref(v___y_1493_);
return v_res_1498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(lean_object* v_msg_1499_, lean_object* v___y_1500_, lean_object* v___y_1501_, lean_object* v___y_1502_, lean_object* v___y_1503_){
_start:
{
lean_object* v_ref_1505_; lean_object* v___x_1506_; lean_object* v_a_1507_; lean_object* v___x_1509_; uint8_t v_isShared_1510_; uint8_t v_isSharedCheck_1515_; 
v_ref_1505_ = lean_ctor_get(v___y_1502_, 5);
v___x_1506_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(v_msg_1499_, v___y_1500_, v___y_1501_, v___y_1502_, v___y_1503_);
v_a_1507_ = lean_ctor_get(v___x_1506_, 0);
v_isSharedCheck_1515_ = !lean_is_exclusive(v___x_1506_);
if (v_isSharedCheck_1515_ == 0)
{
v___x_1509_ = v___x_1506_;
v_isShared_1510_ = v_isSharedCheck_1515_;
goto v_resetjp_1508_;
}
else
{
lean_inc(v_a_1507_);
lean_dec(v___x_1506_);
v___x_1509_ = lean_box(0);
v_isShared_1510_ = v_isSharedCheck_1515_;
goto v_resetjp_1508_;
}
v_resetjp_1508_:
{
lean_object* v___x_1511_; lean_object* v___x_1513_; 
lean_inc(v_ref_1505_);
v___x_1511_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1511_, 0, v_ref_1505_);
lean_ctor_set(v___x_1511_, 1, v_a_1507_);
if (v_isShared_1510_ == 0)
{
lean_ctor_set_tag(v___x_1509_, 1);
lean_ctor_set(v___x_1509_, 0, v___x_1511_);
v___x_1513_ = v___x_1509_;
goto v_reusejp_1512_;
}
else
{
lean_object* v_reuseFailAlloc_1514_; 
v_reuseFailAlloc_1514_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1514_, 0, v___x_1511_);
v___x_1513_ = v_reuseFailAlloc_1514_;
goto v_reusejp_1512_;
}
v_reusejp_1512_:
{
return v___x_1513_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg___boxed(lean_object* v_msg_1516_, lean_object* v___y_1517_, lean_object* v___y_1518_, lean_object* v___y_1519_, lean_object* v___y_1520_, lean_object* v___y_1521_){
_start:
{
lean_object* v_res_1522_; 
v_res_1522_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v_msg_1516_, v___y_1517_, v___y_1518_, v___y_1519_, v___y_1520_);
lean_dec(v___y_1520_);
lean_dec_ref(v___y_1519_);
lean_dec(v___y_1518_);
lean_dec_ref(v___y_1517_);
return v_res_1522_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0(void){
_start:
{
lean_object* v___x_1523_; lean_object* v___x_1524_; 
v___x_1523_ = lean_box(0);
v___x_1524_ = l_Lean_Expr_sort___override(v___x_1523_);
return v___x_1524_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1(void){
_start:
{
lean_object* v___x_1525_; lean_object* v___x_1526_; 
v___x_1525_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0);
v___x_1526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1526_, 0, v___x_1525_);
return v___x_1526_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3(void){
_start:
{
lean_object* v___x_1528_; lean_object* v___x_1529_; 
v___x_1528_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__2));
v___x_1529_ = l_Lean_stringToMessageData(v___x_1528_);
return v___x_1529_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6(void){
_start:
{
lean_object* v___x_1533_; lean_object* v___x_1534_; lean_object* v___x_1535_; 
v___x_1533_ = lean_box(0);
v___x_1534_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__5));
v___x_1535_ = l_Lean_Expr_const___override(v___x_1534_, v___x_1533_);
return v___x_1535_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9(void){
_start:
{
lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1542_; 
v___x_1540_ = lean_box(0);
v___x_1541_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__8));
v___x_1542_ = l_Lean_Expr_const___override(v___x_1541_, v___x_1540_);
return v___x_1542_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11(void){
_start:
{
lean_object* v___x_1544_; lean_object* v___x_1545_; 
v___x_1544_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__10));
v___x_1545_ = l_Lean_stringToMessageData(v___x_1544_);
return v___x_1545_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14(void){
_start:
{
lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; 
v___x_1550_ = lean_box(0);
v___x_1551_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__13));
v___x_1552_ = l_Lean_Expr_const___override(v___x_1551_, v___x_1550_);
return v___x_1552_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17(void){
_start:
{
lean_object* v___x_1557_; lean_object* v___x_1558_; lean_object* v___x_1559_; 
v___x_1557_ = lean_box(0);
v___x_1558_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__16));
v___x_1559_ = l_Lean_Expr_const___override(v___x_1558_, v___x_1557_);
return v___x_1559_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20(void){
_start:
{
lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; 
v___x_1564_ = lean_box(0);
v___x_1565_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__19));
v___x_1566_ = l_Lean_Expr_const___override(v___x_1565_, v___x_1564_);
return v___x_1566_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23(void){
_start:
{
lean_object* v___x_1571_; lean_object* v___x_1572_; lean_object* v___x_1573_; 
v___x_1571_ = lean_box(0);
v___x_1572_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__22));
v___x_1573_ = l_Lean_Expr_const___override(v___x_1572_, v___x_1571_);
return v___x_1573_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26(void){
_start:
{
lean_object* v___x_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; 
v___x_1578_ = lean_box(0);
v___x_1579_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__25));
v___x_1580_ = l_Lean_Expr_const___override(v___x_1579_, v___x_1578_);
return v___x_1580_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27(void){
_start:
{
lean_object* v___x_1581_; lean_object* v___x_1582_; 
v___x_1581_ = lean_box(0);
v___x_1582_ = l_Lean_Level_succ___override(v___x_1581_);
return v___x_1582_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28(void){
_start:
{
lean_object* v___x_1583_; lean_object* v___x_1584_; lean_object* v___x_1585_; 
v___x_1583_ = lean_box(0);
v___x_1584_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__27);
v___x_1585_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1585_, 0, v___x_1584_);
lean_ctor_set(v___x_1585_, 1, v___x_1583_);
return v___x_1585_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29(void){
_start:
{
lean_object* v___x_1586_; lean_object* v___x_1587_; lean_object* v___x_1588_; 
v___x_1586_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__28);
v___x_1587_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__4));
v___x_1588_ = l_Lean_Expr_const___override(v___x_1587_, v___x_1586_);
return v___x_1588_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30(void){
_start:
{
lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___x_1591_; 
v___x_1589_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0);
v___x_1590_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__29);
v___x_1591_ = l_Lean_Expr_app___override(v___x_1590_, v___x_1589_);
return v___x_1591_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33(void){
_start:
{
lean_object* v___x_1595_; lean_object* v___x_1596_; lean_object* v___x_1597_; 
v___x_1595_ = lean_box(0);
v___x_1596_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__32));
v___x_1597_ = l_Lean_Expr_const___override(v___x_1596_, v___x_1595_);
return v___x_1597_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36(void){
_start:
{
lean_object* v___x_1602_; lean_object* v___x_1603_; lean_object* v___x_1604_; 
v___x_1602_ = lean_box(0);
v___x_1603_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__35));
v___x_1604_ = l_Lean_Expr_const___override(v___x_1603_, v___x_1602_);
return v___x_1604_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39(void){
_start:
{
lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v___x_1611_; 
v___x_1609_ = lean_box(0);
v___x_1610_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__38));
v___x_1611_ = l_Lean_Expr_const___override(v___x_1610_, v___x_1609_);
return v___x_1611_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42(void){
_start:
{
lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1617_; 
v___x_1615_ = lean_box(0);
v___x_1616_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__41));
v___x_1617_ = l_Lean_Expr_const___override(v___x_1616_, v___x_1615_);
return v___x_1617_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45(void){
_start:
{
lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; 
v___x_1621_ = lean_box(0);
v___x_1622_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__44));
v___x_1623_ = l_Lean_Expr_const___override(v___x_1622_, v___x_1621_);
return v___x_1623_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47(void){
_start:
{
lean_object* v___x_1625_; lean_object* v___x_1626_; 
v___x_1625_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__46));
v___x_1626_ = l_Lean_stringToMessageData(v___x_1625_);
return v___x_1626_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49(void){
_start:
{
lean_object* v___x_1628_; lean_object* v___x_1629_; 
v___x_1628_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__48));
v___x_1629_ = l_Lean_stringToMessageData(v___x_1628_);
return v___x_1629_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12(lean_object* v_hypFVar_1630_, lean_object* v_g_1631_, lean_object* v___y_1632_, lean_object* v___y_1633_, lean_object* v___y_1634_, lean_object* v___y_1635_){
_start:
{
if (lean_obj_tag(v_hypFVar_1630_) == 1)
{
lean_object* v_fvarId_1637_; lean_object* v___x_1638_; 
v_fvarId_1637_ = lean_ctor_get(v_hypFVar_1630_, 0);
lean_inc_n(v_fvarId_1637_, 2);
lean_dec_ref_known(v_hypFVar_1630_, 1);
v___x_1638_ = l_Lean_FVarId_getDecl___redArg(v_fvarId_1637_, v___y_1632_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1638_) == 0)
{
lean_object* v_a_1639_; lean_object* v_p_1641_; lean_object* v___y_1642_; lean_object* v___y_1643_; lean_object* v___y_1644_; lean_object* v___y_1645_; lean_object* v_a_1649_; lean_object* v___x_2074_; lean_object* v___x_2075_; 
v_a_1639_ = lean_ctor_get(v___x_1638_, 0);
lean_inc(v_a_1639_);
lean_dec_ref_known(v___x_1638_, 1);
v___x_2074_ = l_Lean_LocalDecl_type(v_a_1639_);
lean_inc_ref(v___x_2074_);
v___x_2075_ = l_Lean_Meta_isProp(v___x_2074_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_2075_) == 0)
{
lean_object* v_a_2076_; uint8_t v___x_2077_; 
v_a_2076_ = lean_ctor_get(v___x_2075_, 0);
lean_inc(v_a_2076_);
lean_dec_ref_known(v___x_2075_, 1);
v___x_2077_ = lean_unbox(v_a_2076_);
lean_dec(v_a_2076_);
if (v___x_2077_ == 0)
{
lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v_a_2080_; lean_object* v___x_2082_; uint8_t v_isShared_2083_; uint8_t v_isSharedCheck_2087_; 
lean_dec_ref(v___x_2074_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v___x_2078_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47);
v___x_2079_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v___x_2078_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
v_isSharedCheck_2087_ = !lean_is_exclusive(v___x_2079_);
if (v_isSharedCheck_2087_ == 0)
{
v___x_2082_ = v___x_2079_;
v_isShared_2083_ = v_isSharedCheck_2087_;
goto v_resetjp_2081_;
}
else
{
lean_inc(v_a_2080_);
lean_dec(v___x_2079_);
v___x_2082_ = lean_box(0);
v_isShared_2083_ = v_isSharedCheck_2087_;
goto v_resetjp_2081_;
}
v_resetjp_2081_:
{
lean_object* v___x_2085_; 
if (v_isShared_2083_ == 0)
{
v___x_2085_ = v___x_2082_;
goto v_reusejp_2084_;
}
else
{
lean_object* v_reuseFailAlloc_2086_; 
v_reuseFailAlloc_2086_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2086_, 0, v_a_2080_);
v___x_2085_ = v_reuseFailAlloc_2086_;
goto v_reusejp_2084_;
}
v_reusejp_2084_:
{
return v___x_2085_;
}
}
}
else
{
v_a_1649_ = v___x_2074_;
goto v___jp_1648_;
}
}
else
{
lean_object* v_a_2088_; lean_object* v___x_2090_; uint8_t v_isShared_2091_; uint8_t v_isSharedCheck_2095_; 
lean_dec_ref(v___x_2074_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2088_ = lean_ctor_get(v___x_2075_, 0);
v_isSharedCheck_2095_ = !lean_is_exclusive(v___x_2075_);
if (v_isSharedCheck_2095_ == 0)
{
v___x_2090_ = v___x_2075_;
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
else
{
lean_inc(v_a_2088_);
lean_dec(v___x_2075_);
v___x_2090_ = lean_box(0);
v_isShared_2091_ = v_isSharedCheck_2095_;
goto v_resetjp_2089_;
}
v_resetjp_2089_:
{
lean_object* v___x_2093_; 
if (v_isShared_2091_ == 0)
{
v___x_2093_ = v___x_2090_;
goto v_reusejp_2092_;
}
else
{
lean_object* v_reuseFailAlloc_2094_; 
v_reuseFailAlloc_2094_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2094_, 0, v_a_2088_);
v___x_2093_ = v_reuseFailAlloc_2094_;
goto v_reusejp_2092_;
}
v_reusejp_2092_:
{
return v___x_2093_;
}
}
}
v___jp_1640_:
{
lean_object* v___f_1646_; lean_object* v___x_1647_; 
v___f_1646_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__0___boxed), 9, 4);
lean_closure_set(v___f_1646_, 0, v_p_1641_);
lean_closure_set(v___f_1646_, 1, v_a_1639_);
lean_closure_set(v___f_1646_, 2, v_g_1631_);
lean_closure_set(v___f_1646_, 3, v_fvarId_1637_);
v___x_1647_ = lp_mathlib_Lean_commitIfNoEx___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__0___redArg(v___f_1646_, v___y_1642_, v___y_1643_, v___y_1644_, v___y_1645_);
return v___x_1647_;
}
v___jp_1648_:
{
lean_object* v___x_1650_; lean_object* v___x_1651_; lean_object* v___x_1652_; uint8_t v___x_1653_; lean_object* v___x_1654_; uint8_t v___x_1655_; lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___f_1658_; lean_object* v___x_1659_; 
v___x_1650_ = lean_box(0);
v___x_1651_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0);
v___x_1652_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1);
v___x_1653_ = 0;
v___x_1654_ = lean_box(0);
v___x_1655_ = 0;
v___x_1656_ = lean_box(v___x_1653_);
v___x_1657_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1658_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___boxed), 12, 7);
lean_closure_set(v___f_1658_, 0, v___x_1652_);
lean_closure_set(v___f_1658_, 1, v___x_1656_);
lean_closure_set(v___f_1658_, 2, v___x_1654_);
lean_closure_set(v___f_1658_, 3, v___x_1650_);
lean_closure_set(v___f_1658_, 4, v___x_1651_);
lean_closure_set(v___f_1658_, 5, v_a_1649_);
lean_closure_set(v___f_1658_, 6, v___x_1657_);
v___x_1659_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1658_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1659_) == 0)
{
lean_object* v_a_1660_; lean_object* v_snd_1661_; lean_object* v_snd_1662_; uint8_t v___x_1663_; 
v_a_1660_ = lean_ctor_get(v___x_1659_, 0);
lean_inc(v_a_1660_);
lean_dec_ref_known(v___x_1659_, 1);
v_snd_1661_ = lean_ctor_get(v_a_1660_, 1);
lean_inc(v_snd_1661_);
v_snd_1662_ = lean_ctor_get(v_snd_1661_, 1);
v___x_1663_ = lean_unbox(v_snd_1662_);
if (v___x_1663_ == 0)
{
lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___f_1666_; lean_object* v___x_1667_; 
lean_dec(v_snd_1661_);
lean_dec(v_a_1660_);
v___x_1664_ = lean_box(v___x_1653_);
v___x_1665_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1666_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__2___boxed), 12, 7);
lean_closure_set(v___f_1666_, 0, v___x_1652_);
lean_closure_set(v___f_1666_, 1, v___x_1664_);
lean_closure_set(v___f_1666_, 2, v___x_1654_);
lean_closure_set(v___f_1666_, 3, v___x_1650_);
lean_closure_set(v___f_1666_, 4, v___x_1651_);
lean_closure_set(v___f_1666_, 5, v_a_1649_);
lean_closure_set(v___f_1666_, 6, v___x_1665_);
v___x_1667_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1666_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1667_) == 0)
{
lean_object* v_a_1668_; lean_object* v_snd_1669_; lean_object* v_snd_1670_; uint8_t v___x_1671_; 
v_a_1668_ = lean_ctor_get(v___x_1667_, 0);
lean_inc(v_a_1668_);
lean_dec_ref_known(v___x_1667_, 1);
v_snd_1669_ = lean_ctor_get(v_a_1668_, 1);
lean_inc(v_snd_1669_);
v_snd_1670_ = lean_ctor_get(v_snd_1669_, 1);
v___x_1671_ = lean_unbox(v_snd_1670_);
if (v___x_1671_ == 0)
{
lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v___f_1674_; lean_object* v___x_1675_; 
lean_dec(v_snd_1669_);
lean_dec(v_a_1668_);
v___x_1672_ = lean_box(v___x_1653_);
v___x_1673_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1674_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___boxed), 10, 5);
lean_closure_set(v___f_1674_, 0, v___x_1652_);
lean_closure_set(v___f_1674_, 1, v___x_1672_);
lean_closure_set(v___f_1674_, 2, v___x_1654_);
lean_closure_set(v___f_1674_, 3, v_a_1649_);
lean_closure_set(v___f_1674_, 4, v___x_1673_);
v___x_1675_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1674_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1675_) == 0)
{
lean_object* v_a_1676_; lean_object* v_snd_1677_; lean_object* v_snd_1678_; uint8_t v___x_1679_; 
v_a_1676_ = lean_ctor_get(v___x_1675_, 0);
lean_inc(v_a_1676_);
lean_dec_ref_known(v___x_1675_, 1);
v_snd_1677_ = lean_ctor_get(v_a_1676_, 1);
lean_inc(v_snd_1677_);
v_snd_1678_ = lean_ctor_get(v_snd_1677_, 1);
v___x_1679_ = lean_unbox(v_snd_1678_);
if (v___x_1679_ == 0)
{
lean_object* v___x_1680_; lean_object* v___x_1681_; lean_object* v___f_1682_; lean_object* v___x_1683_; 
lean_dec(v_snd_1677_);
lean_dec(v_a_1676_);
v___x_1680_ = lean_box(v___x_1653_);
v___x_1681_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1682_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___boxed), 10, 5);
lean_closure_set(v___f_1682_, 0, v___x_1652_);
lean_closure_set(v___f_1682_, 1, v___x_1680_);
lean_closure_set(v___f_1682_, 2, v___x_1654_);
lean_closure_set(v___f_1682_, 3, v_a_1649_);
lean_closure_set(v___f_1682_, 4, v___x_1681_);
v___x_1683_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1682_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1683_) == 0)
{
lean_object* v_a_1684_; lean_object* v_snd_1685_; lean_object* v_snd_1686_; uint8_t v___x_1687_; 
v_a_1684_ = lean_ctor_get(v___x_1683_, 0);
lean_inc(v_a_1684_);
lean_dec_ref_known(v___x_1683_, 1);
v_snd_1685_ = lean_ctor_get(v_a_1684_, 1);
lean_inc(v_snd_1685_);
v_snd_1686_ = lean_ctor_get(v_snd_1685_, 1);
v___x_1687_ = lean_unbox(v_snd_1686_);
if (v___x_1687_ == 0)
{
lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v___f_1690_; lean_object* v___x_1691_; 
lean_dec(v_snd_1685_);
lean_dec(v_a_1684_);
v___x_1688_ = lean_box(v___x_1653_);
v___x_1689_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1690_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__5___boxed), 12, 7);
lean_closure_set(v___f_1690_, 0, v___x_1652_);
lean_closure_set(v___f_1690_, 1, v___x_1688_);
lean_closure_set(v___f_1690_, 2, v___x_1654_);
lean_closure_set(v___f_1690_, 3, v___x_1650_);
lean_closure_set(v___f_1690_, 4, v___x_1651_);
lean_closure_set(v___f_1690_, 5, v_a_1649_);
lean_closure_set(v___f_1690_, 6, v___x_1689_);
v___x_1691_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1690_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1691_) == 0)
{
lean_object* v_a_1692_; lean_object* v_snd_1693_; lean_object* v_snd_1694_; uint8_t v___x_1695_; 
v_a_1692_ = lean_ctor_get(v___x_1691_, 0);
lean_inc(v_a_1692_);
lean_dec_ref_known(v___x_1691_, 1);
v_snd_1693_ = lean_ctor_get(v_a_1692_, 1);
lean_inc(v_snd_1693_);
v_snd_1694_ = lean_ctor_get(v_snd_1693_, 1);
v___x_1695_ = lean_unbox(v_snd_1694_);
if (v___x_1695_ == 0)
{
lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___f_1698_; lean_object* v___x_1699_; 
lean_dec(v_snd_1693_);
lean_dec(v_a_1692_);
v___x_1696_ = lean_box(v___x_1653_);
v___x_1697_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1698_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__6___boxed), 10, 5);
lean_closure_set(v___f_1698_, 0, v___x_1652_);
lean_closure_set(v___f_1698_, 1, v___x_1696_);
lean_closure_set(v___f_1698_, 2, v___x_1654_);
lean_closure_set(v___f_1698_, 3, v_a_1649_);
lean_closure_set(v___f_1698_, 4, v___x_1697_);
v___x_1699_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1698_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1699_) == 0)
{
lean_object* v_a_1700_; lean_object* v_snd_1701_; uint8_t v___x_1702_; 
v_a_1700_ = lean_ctor_get(v___x_1699_, 0);
lean_inc(v_a_1700_);
lean_dec_ref_known(v___x_1699_, 1);
v_snd_1701_ = lean_ctor_get(v_a_1700_, 1);
v___x_1702_ = lean_unbox(v_snd_1701_);
if (v___x_1702_ == 0)
{
lean_object* v___x_1703_; lean_object* v___x_1704_; lean_object* v___f_1705_; lean_object* v___x_1706_; 
lean_dec(v_a_1700_);
v___x_1703_ = lean_box(v___x_1653_);
v___x_1704_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1705_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__7___boxed), 10, 5);
lean_closure_set(v___f_1705_, 0, v___x_1652_);
lean_closure_set(v___f_1705_, 1, v___x_1703_);
lean_closure_set(v___f_1705_, 2, v___x_1654_);
lean_closure_set(v___f_1705_, 3, v_a_1649_);
lean_closure_set(v___f_1705_, 4, v___x_1704_);
v___x_1706_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1705_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1706_) == 0)
{
lean_object* v_a_1707_; lean_object* v_snd_1708_; lean_object* v_snd_1709_; uint8_t v___x_1710_; 
v_a_1707_ = lean_ctor_get(v___x_1706_, 0);
lean_inc(v_a_1707_);
lean_dec_ref_known(v___x_1706_, 1);
v_snd_1708_ = lean_ctor_get(v_a_1707_, 1);
lean_inc(v_snd_1708_);
v_snd_1709_ = lean_ctor_get(v_snd_1708_, 1);
v___x_1710_ = lean_unbox(v_snd_1709_);
if (v___x_1710_ == 0)
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___f_1713_; lean_object* v___x_1714_; 
lean_dec(v_snd_1708_);
lean_dec(v_a_1707_);
v___x_1711_ = lean_box(v___x_1653_);
v___x_1712_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1713_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___boxed), 10, 5);
lean_closure_set(v___f_1713_, 0, v___x_1652_);
lean_closure_set(v___f_1713_, 1, v___x_1711_);
lean_closure_set(v___f_1713_, 2, v___x_1654_);
lean_closure_set(v___f_1713_, 3, v_a_1649_);
lean_closure_set(v___f_1713_, 4, v___x_1712_);
v___x_1714_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1713_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1714_) == 0)
{
lean_object* v_a_1715_; lean_object* v_snd_1716_; lean_object* v_snd_1717_; uint8_t v___x_1718_; 
v_a_1715_ = lean_ctor_get(v___x_1714_, 0);
lean_inc(v_a_1715_);
lean_dec_ref_known(v___x_1714_, 1);
v_snd_1716_ = lean_ctor_get(v_a_1715_, 1);
lean_inc(v_snd_1716_);
v_snd_1717_ = lean_ctor_get(v_snd_1716_, 1);
v___x_1718_ = lean_unbox(v_snd_1717_);
if (v___x_1718_ == 0)
{
lean_object* v___x_1719_; lean_object* v___x_1720_; lean_object* v___f_1721_; lean_object* v___x_1722_; 
lean_dec(v_snd_1716_);
lean_dec(v_a_1715_);
v___x_1719_ = lean_box(v___x_1653_);
v___x_1720_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1721_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__9___boxed), 10, 5);
lean_closure_set(v___f_1721_, 0, v___x_1652_);
lean_closure_set(v___f_1721_, 1, v___x_1719_);
lean_closure_set(v___f_1721_, 2, v___x_1654_);
lean_closure_set(v___f_1721_, 3, v_a_1649_);
lean_closure_set(v___f_1721_, 4, v___x_1720_);
v___x_1722_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1721_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1722_) == 0)
{
lean_object* v_a_1723_; lean_object* v_snd_1724_; lean_object* v_snd_1725_; uint8_t v___x_1726_; 
v_a_1723_ = lean_ctor_get(v___x_1722_, 0);
lean_inc(v_a_1723_);
lean_dec_ref_known(v___x_1722_, 1);
v_snd_1724_ = lean_ctor_get(v_a_1723_, 1);
lean_inc(v_snd_1724_);
v_snd_1725_ = lean_ctor_get(v_snd_1724_, 1);
v___x_1726_ = lean_unbox(v_snd_1725_);
if (v___x_1726_ == 0)
{
lean_object* v___x_1727_; lean_object* v___x_1728_; lean_object* v___f_1729_; lean_object* v___x_1730_; 
lean_dec(v_snd_1724_);
lean_dec(v_a_1723_);
v___x_1727_ = lean_box(v___x_1653_);
v___x_1728_ = lean_box(v___x_1655_);
lean_inc_ref(v_a_1649_);
v___f_1729_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___boxed), 10, 5);
lean_closure_set(v___f_1729_, 0, v___x_1652_);
lean_closure_set(v___f_1729_, 1, v___x_1727_);
lean_closure_set(v___f_1729_, 2, v___x_1654_);
lean_closure_set(v___f_1729_, 3, v_a_1649_);
lean_closure_set(v___f_1729_, 4, v___x_1728_);
v___x_1730_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1729_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1730_) == 0)
{
lean_object* v_a_1731_; lean_object* v_snd_1732_; uint8_t v___x_1733_; 
v_a_1731_ = lean_ctor_get(v___x_1730_, 0);
lean_inc(v_a_1731_);
lean_dec_ref_known(v___x_1730_, 1);
v_snd_1732_ = lean_ctor_get(v_a_1731_, 1);
lean_inc(v_snd_1732_);
lean_dec(v_a_1731_);
v___x_1733_ = lean_unbox(v_snd_1732_);
lean_dec(v_snd_1732_);
if (v___x_1733_ == 0)
{
lean_object* v___x_1734_; lean_object* v___f_1735_; lean_object* v___x_1736_; 
v___x_1734_ = lean_box(v___x_1653_);
v___f_1735_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__11___boxed), 9, 4);
lean_closure_set(v___f_1735_, 0, v___x_1652_);
lean_closure_set(v___f_1735_, 1, v___x_1734_);
lean_closure_set(v___f_1735_, 2, v___x_1654_);
lean_closure_set(v___f_1735_, 3, v_a_1649_);
v___x_1736_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_1735_, v___x_1655_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1736_) == 0)
{
lean_object* v_a_1737_; lean_object* v_snd_1738_; lean_object* v_snd_1739_; uint8_t v___x_1740_; 
v_a_1737_ = lean_ctor_get(v___x_1736_, 0);
lean_inc(v_a_1737_);
lean_dec_ref_known(v___x_1736_, 1);
v_snd_1738_ = lean_ctor_get(v_a_1737_, 1);
lean_inc(v_snd_1738_);
v_snd_1739_ = lean_ctor_get(v_snd_1738_, 1);
v___x_1740_ = lean_unbox(v_snd_1739_);
if (v___x_1740_ == 0)
{
lean_object* v___x_1741_; lean_object* v___x_1742_; 
lean_dec(v_snd_1738_);
lean_dec(v_a_1737_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v___x_1741_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__3);
v___x_1742_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v___x_1741_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
return v___x_1742_;
}
else
{
lean_object* v_fst_1743_; lean_object* v_fst_1744_; lean_object* v___x_1745_; lean_object* v___x_1746_; lean_object* v___x_1747_; 
v_fst_1743_ = lean_ctor_get(v_a_1737_, 0);
lean_inc_n(v_fst_1743_, 2);
lean_dec(v_a_1737_);
v_fst_1744_ = lean_ctor_get(v_snd_1738_, 0);
lean_inc(v_fst_1744_);
lean_dec(v_snd_1738_);
v___x_1745_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1746_ = l_Lean_Expr_app___override(v___x_1745_, v_fst_1743_);
v___x_1747_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1746_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1747_) == 0)
{
lean_object* v_a_1748_; lean_object* v___x_1749_; lean_object* v___x_1750_; lean_object* v___x_1751_; lean_object* v___x_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; 
v_a_1748_ = lean_ctor_get(v___x_1747_, 0);
lean_inc(v_a_1748_);
lean_dec_ref_known(v___x_1747_, 1);
lean_inc(v_a_1639_);
v___x_1749_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1750_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__9);
v___x_1751_ = l_Lean_Expr_app___override(v___x_1750_, v_fst_1743_);
v___x_1752_ = l_Lean_Expr_app___override(v___x_1751_, v_fst_1744_);
v___x_1753_ = l_Lean_Expr_app___override(v___x_1752_, v_a_1748_);
v___x_1754_ = l_Lean_Expr_app___override(v___x_1753_, v___x_1749_);
v_p_1641_ = v___x_1754_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1755_; lean_object* v___x_1757_; uint8_t v_isShared_1758_; uint8_t v_isSharedCheck_1762_; 
lean_dec(v_fst_1744_);
lean_dec(v_fst_1743_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1755_ = lean_ctor_get(v___x_1747_, 0);
v_isSharedCheck_1762_ = !lean_is_exclusive(v___x_1747_);
if (v_isSharedCheck_1762_ == 0)
{
v___x_1757_ = v___x_1747_;
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
else
{
lean_inc(v_a_1755_);
lean_dec(v___x_1747_);
v___x_1757_ = lean_box(0);
v_isShared_1758_ = v_isSharedCheck_1762_;
goto v_resetjp_1756_;
}
v_resetjp_1756_:
{
lean_object* v___x_1760_; 
if (v_isShared_1758_ == 0)
{
v___x_1760_ = v___x_1757_;
goto v_reusejp_1759_;
}
else
{
lean_object* v_reuseFailAlloc_1761_; 
v_reuseFailAlloc_1761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1761_, 0, v_a_1755_);
v___x_1760_ = v_reuseFailAlloc_1761_;
goto v_reusejp_1759_;
}
v_reusejp_1759_:
{
return v___x_1760_;
}
}
}
}
}
else
{
lean_object* v_a_1763_; lean_object* v___x_1765_; uint8_t v_isShared_1766_; uint8_t v_isSharedCheck_1770_; 
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1763_ = lean_ctor_get(v___x_1736_, 0);
v_isSharedCheck_1770_ = !lean_is_exclusive(v___x_1736_);
if (v_isSharedCheck_1770_ == 0)
{
v___x_1765_ = v___x_1736_;
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
else
{
lean_inc(v_a_1763_);
lean_dec(v___x_1736_);
v___x_1765_ = lean_box(0);
v_isShared_1766_ = v_isSharedCheck_1770_;
goto v_resetjp_1764_;
}
v_resetjp_1764_:
{
lean_object* v___x_1768_; 
if (v_isShared_1766_ == 0)
{
v___x_1768_ = v___x_1765_;
goto v_reusejp_1767_;
}
else
{
lean_object* v_reuseFailAlloc_1769_; 
v_reuseFailAlloc_1769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1769_, 0, v_a_1763_);
v___x_1768_ = v_reuseFailAlloc_1769_;
goto v_reusejp_1767_;
}
v_reusejp_1767_:
{
return v___x_1768_;
}
}
}
}
else
{
lean_object* v___x_1771_; lean_object* v___x_1772_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v___x_1771_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__11);
v___x_1772_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v___x_1771_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
return v___x_1772_;
}
}
else
{
lean_object* v_a_1773_; lean_object* v___x_1775_; uint8_t v_isShared_1776_; uint8_t v_isSharedCheck_1780_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1773_ = lean_ctor_get(v___x_1730_, 0);
v_isSharedCheck_1780_ = !lean_is_exclusive(v___x_1730_);
if (v_isSharedCheck_1780_ == 0)
{
v___x_1775_ = v___x_1730_;
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
else
{
lean_inc(v_a_1773_);
lean_dec(v___x_1730_);
v___x_1775_ = lean_box(0);
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
v_resetjp_1774_:
{
lean_object* v___x_1778_; 
if (v_isShared_1776_ == 0)
{
v___x_1778_ = v___x_1775_;
goto v_reusejp_1777_;
}
else
{
lean_object* v_reuseFailAlloc_1779_; 
v_reuseFailAlloc_1779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1779_, 0, v_a_1773_);
v___x_1778_ = v_reuseFailAlloc_1779_;
goto v_reusejp_1777_;
}
v_reusejp_1777_:
{
return v___x_1778_;
}
}
}
}
else
{
lean_object* v_fst_1781_; lean_object* v_fst_1782_; lean_object* v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; 
lean_dec_ref(v_a_1649_);
v_fst_1781_ = lean_ctor_get(v_a_1723_, 0);
lean_inc(v_fst_1781_);
lean_dec(v_a_1723_);
v_fst_1782_ = lean_ctor_get(v_snd_1724_, 0);
lean_inc_n(v_fst_1782_, 2);
lean_dec(v_snd_1724_);
v___x_1783_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1784_ = l_Lean_Expr_app___override(v___x_1783_, v_fst_1782_);
v___x_1785_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1784_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1785_) == 0)
{
lean_object* v_a_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; lean_object* v___x_1789_; lean_object* v___x_1790_; lean_object* v___x_1791_; lean_object* v___x_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; lean_object* v___x_1795_; lean_object* v___x_1796_; lean_object* v___x_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; lean_object* v___x_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; lean_object* v___x_1804_; lean_object* v___x_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; lean_object* v___x_1808_; lean_object* v___x_1809_; lean_object* v___x_1810_; 
v_a_1786_ = lean_ctor_get(v___x_1785_, 0);
lean_inc(v_a_1786_);
lean_dec_ref_known(v___x_1785_, 1);
lean_inc(v_a_1639_);
v___x_1787_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1788_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2);
v___x_1789_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2);
lean_inc_n(v_fst_1781_, 3);
v___x_1790_ = l_Lean_Expr_app___override(v___x_1789_, v_fst_1781_);
lean_inc_n(v_fst_1782_, 3);
v___x_1791_ = l_Lean_Expr_app___override(v___x_1790_, v_fst_1782_);
v___x_1792_ = l_Lean_Expr_app___override(v___x_1788_, v___x_1791_);
v___x_1793_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_1794_ = l_Lean_Expr_app___override(v___x_1793_, v_fst_1781_);
v___x_1795_ = l_Lean_Expr_app___override(v___x_1789_, v___x_1794_);
v___x_1796_ = l_Lean_Expr_app___override(v___x_1793_, v_fst_1782_);
v___x_1797_ = l_Lean_Expr_app___override(v___x_1795_, v___x_1796_);
v___x_1798_ = l_Lean_Expr_app___override(v___x_1792_, v___x_1797_);
v___x_1799_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14);
v___x_1800_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2);
v___x_1801_ = l_Lean_Expr_app___override(v___x_1800_, v_fst_1781_);
v___x_1802_ = l_Lean_Expr_app___override(v___x_1801_, v_fst_1782_);
v___x_1803_ = l_Lean_Expr_app___override(v___x_1799_, v___x_1802_);
v___x_1804_ = l_Lean_Expr_app___override(v___x_1803_, v___x_1798_);
v___x_1805_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__17);
v___x_1806_ = l_Lean_Expr_app___override(v___x_1805_, v_fst_1781_);
v___x_1807_ = l_Lean_Expr_app___override(v___x_1806_, v_fst_1782_);
v___x_1808_ = l_Lean_Expr_app___override(v___x_1807_, v_a_1786_);
v___x_1809_ = l_Lean_Expr_app___override(v___x_1804_, v___x_1808_);
v___x_1810_ = l_Lean_Expr_app___override(v___x_1809_, v___x_1787_);
v_p_1641_ = v___x_1810_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1811_; lean_object* v___x_1813_; uint8_t v_isShared_1814_; uint8_t v_isSharedCheck_1818_; 
lean_dec(v_fst_1782_);
lean_dec(v_fst_1781_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1811_ = lean_ctor_get(v___x_1785_, 0);
v_isSharedCheck_1818_ = !lean_is_exclusive(v___x_1785_);
if (v_isSharedCheck_1818_ == 0)
{
v___x_1813_ = v___x_1785_;
v_isShared_1814_ = v_isSharedCheck_1818_;
goto v_resetjp_1812_;
}
else
{
lean_inc(v_a_1811_);
lean_dec(v___x_1785_);
v___x_1813_ = lean_box(0);
v_isShared_1814_ = v_isSharedCheck_1818_;
goto v_resetjp_1812_;
}
v_resetjp_1812_:
{
lean_object* v___x_1816_; 
if (v_isShared_1814_ == 0)
{
v___x_1816_ = v___x_1813_;
goto v_reusejp_1815_;
}
else
{
lean_object* v_reuseFailAlloc_1817_; 
v_reuseFailAlloc_1817_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1817_, 0, v_a_1811_);
v___x_1816_ = v_reuseFailAlloc_1817_;
goto v_reusejp_1815_;
}
v_reusejp_1815_:
{
return v___x_1816_;
}
}
}
}
}
else
{
lean_object* v_a_1819_; lean_object* v___x_1821_; uint8_t v_isShared_1822_; uint8_t v_isSharedCheck_1826_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1819_ = lean_ctor_get(v___x_1722_, 0);
v_isSharedCheck_1826_ = !lean_is_exclusive(v___x_1722_);
if (v_isSharedCheck_1826_ == 0)
{
v___x_1821_ = v___x_1722_;
v_isShared_1822_ = v_isSharedCheck_1826_;
goto v_resetjp_1820_;
}
else
{
lean_inc(v_a_1819_);
lean_dec(v___x_1722_);
v___x_1821_ = lean_box(0);
v_isShared_1822_ = v_isSharedCheck_1826_;
goto v_resetjp_1820_;
}
v_resetjp_1820_:
{
lean_object* v___x_1824_; 
if (v_isShared_1822_ == 0)
{
v___x_1824_ = v___x_1821_;
goto v_reusejp_1823_;
}
else
{
lean_object* v_reuseFailAlloc_1825_; 
v_reuseFailAlloc_1825_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1825_, 0, v_a_1819_);
v___x_1824_ = v_reuseFailAlloc_1825_;
goto v_reusejp_1823_;
}
v_reusejp_1823_:
{
return v___x_1824_;
}
}
}
}
else
{
lean_object* v_fst_1827_; lean_object* v_fst_1828_; lean_object* v___x_1829_; lean_object* v___x_1830_; lean_object* v___x_1831_; 
lean_dec_ref(v_a_1649_);
v_fst_1827_ = lean_ctor_get(v_a_1715_, 0);
lean_inc(v_fst_1827_);
lean_dec(v_a_1715_);
v_fst_1828_ = lean_ctor_get(v_snd_1716_, 0);
lean_inc_n(v_fst_1828_, 2);
lean_dec(v_snd_1716_);
v___x_1829_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1830_ = l_Lean_Expr_app___override(v___x_1829_, v_fst_1828_);
v___x_1831_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1830_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1831_) == 0)
{
lean_object* v_a_1832_; lean_object* v___x_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; lean_object* v___x_1839_; lean_object* v___x_1840_; lean_object* v___x_1841_; lean_object* v___x_1842_; lean_object* v___x_1843_; lean_object* v___x_1844_; lean_object* v___x_1845_; lean_object* v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; 
v_a_1832_ = lean_ctor_get(v___x_1831_, 0);
lean_inc(v_a_1832_);
lean_dec_ref_known(v___x_1831_, 1);
lean_inc(v_a_1639_);
v___x_1833_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1834_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2);
v___x_1835_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
lean_inc_n(v_fst_1827_, 2);
v___x_1836_ = l_Lean_Expr_app___override(v___x_1835_, v_fst_1827_);
v___x_1837_ = l_Lean_Expr_app___override(v___x_1834_, v___x_1836_);
lean_inc_n(v_fst_1828_, 2);
v___x_1838_ = l_Lean_Expr_app___override(v___x_1837_, v_fst_1828_);
v___x_1839_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14);
v___x_1840_ = l_Lean_Expr_app___override(v___x_1834_, v_fst_1827_);
v___x_1841_ = l_Lean_Expr_app___override(v___x_1840_, v_fst_1828_);
v___x_1842_ = l_Lean_Expr_app___override(v___x_1835_, v___x_1841_);
v___x_1843_ = l_Lean_Expr_app___override(v___x_1839_, v___x_1842_);
v___x_1844_ = l_Lean_Expr_app___override(v___x_1843_, v___x_1838_);
v___x_1845_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__20);
v___x_1846_ = l_Lean_Expr_app___override(v___x_1845_, v_fst_1828_);
v___x_1847_ = l_Lean_Expr_app___override(v___x_1846_, v_fst_1827_);
v___x_1848_ = l_Lean_Expr_app___override(v___x_1847_, v_a_1832_);
v___x_1849_ = l_Lean_Expr_app___override(v___x_1844_, v___x_1848_);
v___x_1850_ = l_Lean_Expr_app___override(v___x_1849_, v___x_1833_);
v_p_1641_ = v___x_1850_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1851_; lean_object* v___x_1853_; uint8_t v_isShared_1854_; uint8_t v_isSharedCheck_1858_; 
lean_dec(v_fst_1828_);
lean_dec(v_fst_1827_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1851_ = lean_ctor_get(v___x_1831_, 0);
v_isSharedCheck_1858_ = !lean_is_exclusive(v___x_1831_);
if (v_isSharedCheck_1858_ == 0)
{
v___x_1853_ = v___x_1831_;
v_isShared_1854_ = v_isSharedCheck_1858_;
goto v_resetjp_1852_;
}
else
{
lean_inc(v_a_1851_);
lean_dec(v___x_1831_);
v___x_1853_ = lean_box(0);
v_isShared_1854_ = v_isSharedCheck_1858_;
goto v_resetjp_1852_;
}
v_resetjp_1852_:
{
lean_object* v___x_1856_; 
if (v_isShared_1854_ == 0)
{
v___x_1856_ = v___x_1853_;
goto v_reusejp_1855_;
}
else
{
lean_object* v_reuseFailAlloc_1857_; 
v_reuseFailAlloc_1857_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1857_, 0, v_a_1851_);
v___x_1856_ = v_reuseFailAlloc_1857_;
goto v_reusejp_1855_;
}
v_reusejp_1855_:
{
return v___x_1856_;
}
}
}
}
}
else
{
lean_object* v_a_1859_; lean_object* v___x_1861_; uint8_t v_isShared_1862_; uint8_t v_isSharedCheck_1866_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1859_ = lean_ctor_get(v___x_1714_, 0);
v_isSharedCheck_1866_ = !lean_is_exclusive(v___x_1714_);
if (v_isSharedCheck_1866_ == 0)
{
v___x_1861_ = v___x_1714_;
v_isShared_1862_ = v_isSharedCheck_1866_;
goto v_resetjp_1860_;
}
else
{
lean_inc(v_a_1859_);
lean_dec(v___x_1714_);
v___x_1861_ = lean_box(0);
v_isShared_1862_ = v_isSharedCheck_1866_;
goto v_resetjp_1860_;
}
v_resetjp_1860_:
{
lean_object* v___x_1864_; 
if (v_isShared_1862_ == 0)
{
v___x_1864_ = v___x_1861_;
goto v_reusejp_1863_;
}
else
{
lean_object* v_reuseFailAlloc_1865_; 
v_reuseFailAlloc_1865_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1865_, 0, v_a_1859_);
v___x_1864_ = v_reuseFailAlloc_1865_;
goto v_reusejp_1863_;
}
v_reusejp_1863_:
{
return v___x_1864_;
}
}
}
}
else
{
lean_object* v_fst_1867_; lean_object* v_fst_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; lean_object* v___x_1871_; 
lean_dec_ref(v_a_1649_);
v_fst_1867_ = lean_ctor_get(v_a_1707_, 0);
lean_inc_n(v_fst_1867_, 2);
lean_dec(v_a_1707_);
v_fst_1868_ = lean_ctor_get(v_snd_1708_, 0);
lean_inc(v_fst_1868_);
lean_dec(v_snd_1708_);
v___x_1869_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1870_ = l_Lean_Expr_app___override(v___x_1869_, v_fst_1867_);
v___x_1871_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1870_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1871_) == 0)
{
lean_object* v_a_1872_; lean_object* v___x_1873_; lean_object* v___x_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; lean_object* v___x_1879_; uint8_t v___x_1880_; lean_object* v___x_1881_; lean_object* v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; lean_object* v___x_1889_; lean_object* v___x_1890_; 
v_a_1872_ = lean_ctor_get(v___x_1871_, 0);
lean_inc(v_a_1872_);
lean_dec_ref_known(v___x_1871_, 1);
lean_inc(v_a_1639_);
v___x_1873_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1874_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2);
lean_inc_n(v_fst_1867_, 2);
v___x_1875_ = l_Lean_Expr_app___override(v___x_1874_, v_fst_1867_);
v___x_1876_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
lean_inc_n(v_fst_1868_, 2);
v___x_1877_ = l_Lean_Expr_app___override(v___x_1876_, v_fst_1868_);
v___x_1878_ = l_Lean_Expr_app___override(v___x_1875_, v___x_1877_);
v___x_1879_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14);
v___x_1880_ = 0;
v___x_1881_ = l_Lean_Expr_forallE___override(v___x_1654_, v_fst_1867_, v_fst_1868_, v___x_1880_);
v___x_1882_ = l_Lean_Expr_app___override(v___x_1876_, v___x_1881_);
v___x_1883_ = l_Lean_Expr_app___override(v___x_1879_, v___x_1882_);
v___x_1884_ = l_Lean_Expr_app___override(v___x_1883_, v___x_1878_);
v___x_1885_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__23);
v___x_1886_ = l_Lean_Expr_app___override(v___x_1885_, v_fst_1867_);
v___x_1887_ = l_Lean_Expr_app___override(v___x_1886_, v_fst_1868_);
v___x_1888_ = l_Lean_Expr_app___override(v___x_1887_, v_a_1872_);
v___x_1889_ = l_Lean_Expr_app___override(v___x_1884_, v___x_1888_);
v___x_1890_ = l_Lean_Expr_app___override(v___x_1889_, v___x_1873_);
v_p_1641_ = v___x_1890_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1891_; lean_object* v___x_1893_; uint8_t v_isShared_1894_; uint8_t v_isSharedCheck_1898_; 
lean_dec(v_fst_1868_);
lean_dec(v_fst_1867_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1891_ = lean_ctor_get(v___x_1871_, 0);
v_isSharedCheck_1898_ = !lean_is_exclusive(v___x_1871_);
if (v_isSharedCheck_1898_ == 0)
{
v___x_1893_ = v___x_1871_;
v_isShared_1894_ = v_isSharedCheck_1898_;
goto v_resetjp_1892_;
}
else
{
lean_inc(v_a_1891_);
lean_dec(v___x_1871_);
v___x_1893_ = lean_box(0);
v_isShared_1894_ = v_isSharedCheck_1898_;
goto v_resetjp_1892_;
}
v_resetjp_1892_:
{
lean_object* v___x_1896_; 
if (v_isShared_1894_ == 0)
{
v___x_1896_ = v___x_1893_;
goto v_reusejp_1895_;
}
else
{
lean_object* v_reuseFailAlloc_1897_; 
v_reuseFailAlloc_1897_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1897_, 0, v_a_1891_);
v___x_1896_ = v_reuseFailAlloc_1897_;
goto v_reusejp_1895_;
}
v_reusejp_1895_:
{
return v___x_1896_;
}
}
}
}
}
else
{
lean_object* v_a_1899_; lean_object* v___x_1901_; uint8_t v_isShared_1902_; uint8_t v_isSharedCheck_1906_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1899_ = lean_ctor_get(v___x_1706_, 0);
v_isSharedCheck_1906_ = !lean_is_exclusive(v___x_1706_);
if (v_isSharedCheck_1906_ == 0)
{
v___x_1901_ = v___x_1706_;
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
else
{
lean_inc(v_a_1899_);
lean_dec(v___x_1706_);
v___x_1901_ = lean_box(0);
v_isShared_1902_ = v_isSharedCheck_1906_;
goto v_resetjp_1900_;
}
v_resetjp_1900_:
{
lean_object* v___x_1904_; 
if (v_isShared_1902_ == 0)
{
v___x_1904_ = v___x_1901_;
goto v_reusejp_1903_;
}
else
{
lean_object* v_reuseFailAlloc_1905_; 
v_reuseFailAlloc_1905_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1905_, 0, v_a_1899_);
v___x_1904_ = v_reuseFailAlloc_1905_;
goto v_reusejp_1903_;
}
v_reusejp_1903_:
{
return v___x_1904_;
}
}
}
}
else
{
lean_object* v_fst_1907_; lean_object* v___x_1908_; lean_object* v___x_1909_; lean_object* v___x_1910_; 
lean_dec_ref(v_a_1649_);
v_fst_1907_ = lean_ctor_get(v_a_1700_, 0);
lean_inc_n(v_fst_1907_, 2);
lean_dec(v_a_1700_);
v___x_1908_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1909_ = l_Lean_Expr_app___override(v___x_1908_, v_fst_1907_);
v___x_1910_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1909_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1910_) == 0)
{
lean_object* v_a_1911_; lean_object* v___x_1912_; lean_object* v___x_1913_; lean_object* v___x_1914_; lean_object* v___x_1915_; lean_object* v___x_1916_; 
v_a_1911_ = lean_ctor_get(v___x_1910_, 0);
lean_inc(v_a_1911_);
lean_dec_ref_known(v___x_1910_, 1);
lean_inc(v_a_1639_);
v___x_1912_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1913_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26);
v___x_1914_ = l_Lean_Expr_app___override(v___x_1913_, v_fst_1907_);
v___x_1915_ = l_Lean_Expr_app___override(v___x_1914_, v_a_1911_);
v___x_1916_ = l_Lean_Expr_app___override(v___x_1915_, v___x_1912_);
v_p_1641_ = v___x_1916_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1917_; lean_object* v___x_1919_; uint8_t v_isShared_1920_; uint8_t v_isSharedCheck_1924_; 
lean_dec(v_fst_1907_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1917_ = lean_ctor_get(v___x_1910_, 0);
v_isSharedCheck_1924_ = !lean_is_exclusive(v___x_1910_);
if (v_isSharedCheck_1924_ == 0)
{
v___x_1919_ = v___x_1910_;
v_isShared_1920_ = v_isSharedCheck_1924_;
goto v_resetjp_1918_;
}
else
{
lean_inc(v_a_1917_);
lean_dec(v___x_1910_);
v___x_1919_ = lean_box(0);
v_isShared_1920_ = v_isSharedCheck_1924_;
goto v_resetjp_1918_;
}
v_resetjp_1918_:
{
lean_object* v___x_1922_; 
if (v_isShared_1920_ == 0)
{
v___x_1922_ = v___x_1919_;
goto v_reusejp_1921_;
}
else
{
lean_object* v_reuseFailAlloc_1923_; 
v_reuseFailAlloc_1923_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1923_, 0, v_a_1917_);
v___x_1922_ = v_reuseFailAlloc_1923_;
goto v_reusejp_1921_;
}
v_reusejp_1921_:
{
return v___x_1922_;
}
}
}
}
}
else
{
lean_object* v_a_1925_; lean_object* v___x_1927_; uint8_t v_isShared_1928_; uint8_t v_isSharedCheck_1932_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1925_ = lean_ctor_get(v___x_1699_, 0);
v_isSharedCheck_1932_ = !lean_is_exclusive(v___x_1699_);
if (v_isSharedCheck_1932_ == 0)
{
v___x_1927_ = v___x_1699_;
v_isShared_1928_ = v_isSharedCheck_1932_;
goto v_resetjp_1926_;
}
else
{
lean_inc(v_a_1925_);
lean_dec(v___x_1699_);
v___x_1927_ = lean_box(0);
v_isShared_1928_ = v_isSharedCheck_1932_;
goto v_resetjp_1926_;
}
v_resetjp_1926_:
{
lean_object* v___x_1930_; 
if (v_isShared_1928_ == 0)
{
v___x_1930_ = v___x_1927_;
goto v_reusejp_1929_;
}
else
{
lean_object* v_reuseFailAlloc_1931_; 
v_reuseFailAlloc_1931_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1931_, 0, v_a_1925_);
v___x_1930_ = v_reuseFailAlloc_1931_;
goto v_reusejp_1929_;
}
v_reusejp_1929_:
{
return v___x_1930_;
}
}
}
}
else
{
lean_object* v_fst_1933_; lean_object* v_fst_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; lean_object* v___x_1937_; lean_object* v___x_1938_; lean_object* v___x_1939_; lean_object* v___x_1940_; 
lean_dec_ref(v_a_1649_);
v_fst_1933_ = lean_ctor_get(v_a_1692_, 0);
lean_inc(v_fst_1933_);
lean_dec(v_a_1692_);
v_fst_1934_ = lean_ctor_get(v_snd_1693_, 0);
lean_inc(v_fst_1934_);
lean_dec(v_snd_1693_);
v___x_1935_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1936_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30);
v___x_1937_ = l_Lean_Expr_app___override(v___x_1936_, v_fst_1933_);
v___x_1938_ = l_Lean_Expr_app___override(v___x_1937_, v_fst_1934_);
lean_inc_ref(v___x_1938_);
v___x_1939_ = l_Lean_Expr_app___override(v___x_1935_, v___x_1938_);
v___x_1940_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1939_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1940_) == 0)
{
lean_object* v_a_1941_; lean_object* v___x_1942_; lean_object* v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; 
v_a_1941_ = lean_ctor_get(v___x_1940_, 0);
lean_inc(v_a_1941_);
lean_dec_ref_known(v___x_1940_, 1);
lean_inc(v_a_1639_);
v___x_1942_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1943_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__26);
v___x_1944_ = l_Lean_Expr_app___override(v___x_1943_, v___x_1938_);
v___x_1945_ = l_Lean_Expr_app___override(v___x_1944_, v_a_1941_);
v___x_1946_ = l_Lean_Expr_app___override(v___x_1945_, v___x_1942_);
v_p_1641_ = v___x_1946_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_1947_; lean_object* v___x_1949_; uint8_t v_isShared_1950_; uint8_t v_isSharedCheck_1954_; 
lean_dec_ref(v___x_1938_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1947_ = lean_ctor_get(v___x_1940_, 0);
v_isSharedCheck_1954_ = !lean_is_exclusive(v___x_1940_);
if (v_isSharedCheck_1954_ == 0)
{
v___x_1949_ = v___x_1940_;
v_isShared_1950_ = v_isSharedCheck_1954_;
goto v_resetjp_1948_;
}
else
{
lean_inc(v_a_1947_);
lean_dec(v___x_1940_);
v___x_1949_ = lean_box(0);
v_isShared_1950_ = v_isSharedCheck_1954_;
goto v_resetjp_1948_;
}
v_resetjp_1948_:
{
lean_object* v___x_1952_; 
if (v_isShared_1950_ == 0)
{
v___x_1952_ = v___x_1949_;
goto v_reusejp_1951_;
}
else
{
lean_object* v_reuseFailAlloc_1953_; 
v_reuseFailAlloc_1953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1953_, 0, v_a_1947_);
v___x_1952_ = v_reuseFailAlloc_1953_;
goto v_reusejp_1951_;
}
v_reusejp_1951_:
{
return v___x_1952_;
}
}
}
}
}
else
{
lean_object* v_a_1955_; lean_object* v___x_1957_; uint8_t v_isShared_1958_; uint8_t v_isSharedCheck_1962_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1955_ = lean_ctor_get(v___x_1691_, 0);
v_isSharedCheck_1962_ = !lean_is_exclusive(v___x_1691_);
if (v_isSharedCheck_1962_ == 0)
{
v___x_1957_ = v___x_1691_;
v_isShared_1958_ = v_isSharedCheck_1962_;
goto v_resetjp_1956_;
}
else
{
lean_inc(v_a_1955_);
lean_dec(v___x_1691_);
v___x_1957_ = lean_box(0);
v_isShared_1958_ = v_isSharedCheck_1962_;
goto v_resetjp_1956_;
}
v_resetjp_1956_:
{
lean_object* v___x_1960_; 
if (v_isShared_1958_ == 0)
{
v___x_1960_ = v___x_1957_;
goto v_reusejp_1959_;
}
else
{
lean_object* v_reuseFailAlloc_1961_; 
v_reuseFailAlloc_1961_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1961_, 0, v_a_1955_);
v___x_1960_ = v_reuseFailAlloc_1961_;
goto v_reusejp_1959_;
}
v_reusejp_1959_:
{
return v___x_1960_;
}
}
}
}
else
{
lean_object* v_fst_1963_; lean_object* v_fst_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; lean_object* v___x_1967_; lean_object* v___x_1968_; lean_object* v___x_1969_; lean_object* v___x_1970_; lean_object* v___x_1971_; lean_object* v___x_1972_; lean_object* v___x_1973_; lean_object* v___x_1974_; lean_object* v___x_1975_; lean_object* v___x_1976_; lean_object* v___x_1977_; lean_object* v___x_1978_; lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1981_; lean_object* v___x_1982_; lean_object* v___x_1983_; 
lean_dec_ref(v_a_1649_);
v_fst_1963_ = lean_ctor_get(v_a_1684_, 0);
lean_inc_n(v_fst_1963_, 3);
lean_dec(v_a_1684_);
v_fst_1964_ = lean_ctor_get(v_snd_1685_, 0);
lean_inc_n(v_fst_1964_, 3);
lean_dec(v_snd_1685_);
lean_inc(v_a_1639_);
v___x_1965_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1966_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2);
v___x_1967_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
v___x_1968_ = l_Lean_Expr_app___override(v___x_1967_, v_fst_1963_);
v___x_1969_ = l_Lean_Expr_app___override(v___x_1966_, v___x_1968_);
v___x_1970_ = l_Lean_Expr_app___override(v___x_1967_, v_fst_1964_);
v___x_1971_ = l_Lean_Expr_app___override(v___x_1969_, v___x_1970_);
v___x_1972_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14);
v___x_1973_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2);
v___x_1974_ = l_Lean_Expr_app___override(v___x_1973_, v_fst_1963_);
v___x_1975_ = l_Lean_Expr_app___override(v___x_1974_, v_fst_1964_);
v___x_1976_ = l_Lean_Expr_app___override(v___x_1967_, v___x_1975_);
v___x_1977_ = l_Lean_Expr_app___override(v___x_1972_, v___x_1976_);
v___x_1978_ = l_Lean_Expr_app___override(v___x_1977_, v___x_1971_);
v___x_1979_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__33);
v___x_1980_ = l_Lean_Expr_app___override(v___x_1979_, v_fst_1963_);
v___x_1981_ = l_Lean_Expr_app___override(v___x_1980_, v_fst_1964_);
v___x_1982_ = l_Lean_Expr_app___override(v___x_1978_, v___x_1981_);
v___x_1983_ = l_Lean_Expr_app___override(v___x_1982_, v___x_1965_);
v_p_1641_ = v___x_1983_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
}
else
{
lean_object* v_a_1984_; lean_object* v___x_1986_; uint8_t v_isShared_1987_; uint8_t v_isSharedCheck_1991_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_1984_ = lean_ctor_get(v___x_1683_, 0);
v_isSharedCheck_1991_ = !lean_is_exclusive(v___x_1683_);
if (v_isSharedCheck_1991_ == 0)
{
v___x_1986_ = v___x_1683_;
v_isShared_1987_ = v_isSharedCheck_1991_;
goto v_resetjp_1985_;
}
else
{
lean_inc(v_a_1984_);
lean_dec(v___x_1683_);
v___x_1986_ = lean_box(0);
v_isShared_1987_ = v_isSharedCheck_1991_;
goto v_resetjp_1985_;
}
v_resetjp_1985_:
{
lean_object* v___x_1989_; 
if (v_isShared_1987_ == 0)
{
v___x_1989_ = v___x_1986_;
goto v_reusejp_1988_;
}
else
{
lean_object* v_reuseFailAlloc_1990_; 
v_reuseFailAlloc_1990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1990_, 0, v_a_1984_);
v___x_1989_ = v_reuseFailAlloc_1990_;
goto v_reusejp_1988_;
}
v_reusejp_1988_:
{
return v___x_1989_;
}
}
}
}
else
{
lean_object* v_fst_1992_; lean_object* v_fst_1993_; lean_object* v___x_1994_; lean_object* v___x_1995_; lean_object* v___x_1996_; 
lean_dec_ref(v_a_1649_);
v_fst_1992_ = lean_ctor_get(v_a_1676_, 0);
lean_inc(v_fst_1992_);
lean_dec(v_a_1676_);
v_fst_1993_ = lean_ctor_get(v_snd_1677_, 0);
lean_inc_n(v_fst_1993_, 2);
lean_dec(v_snd_1677_);
v___x_1994_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__6);
v___x_1995_ = l_Lean_Expr_app___override(v___x_1994_, v_fst_1993_);
v___x_1996_ = lp_Qq_Qq_synthInstanceQ___redArg(v___x_1995_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
if (lean_obj_tag(v___x_1996_) == 0)
{
lean_object* v_a_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; lean_object* v___x_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; lean_object* v___x_2005_; lean_object* v___x_2006_; lean_object* v___x_2007_; lean_object* v___x_2008_; lean_object* v___x_2009_; lean_object* v___x_2010_; lean_object* v___x_2011_; lean_object* v___x_2012_; lean_object* v___x_2013_; lean_object* v___x_2014_; lean_object* v___x_2015_; lean_object* v___x_2016_; lean_object* v___x_2017_; 
v_a_1997_ = lean_ctor_get(v___x_1996_, 0);
lean_inc(v_a_1997_);
lean_dec_ref_known(v___x_1996_, 1);
lean_inc(v_a_1639_);
v___x_1998_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_1999_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__2);
v___x_2000_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__1___closed__2);
lean_inc_n(v_fst_1992_, 2);
v___x_2001_ = l_Lean_Expr_app___override(v___x_2000_, v_fst_1992_);
v___x_2002_ = l_Lean_Expr_app___override(v___x_1999_, v___x_2001_);
lean_inc_n(v_fst_1993_, 2);
v___x_2003_ = l_Lean_Expr_app___override(v___x_2000_, v_fst_1993_);
v___x_2004_ = l_Lean_Expr_app___override(v___x_2002_, v___x_2003_);
v___x_2005_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__14);
v___x_2006_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__2);
v___x_2007_ = l_Lean_Expr_app___override(v___x_2006_, v_fst_1992_);
v___x_2008_ = l_Lean_Expr_app___override(v___x_2007_, v_fst_1993_);
v___x_2009_ = l_Lean_Expr_app___override(v___x_2000_, v___x_2008_);
v___x_2010_ = l_Lean_Expr_app___override(v___x_2005_, v___x_2009_);
v___x_2011_ = l_Lean_Expr_app___override(v___x_2010_, v___x_2004_);
v___x_2012_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__36);
v___x_2013_ = l_Lean_Expr_app___override(v___x_2012_, v_fst_1993_);
v___x_2014_ = l_Lean_Expr_app___override(v___x_2013_, v_fst_1992_);
v___x_2015_ = l_Lean_Expr_app___override(v___x_2014_, v_a_1997_);
v___x_2016_ = l_Lean_Expr_app___override(v___x_2011_, v___x_2015_);
v___x_2017_ = l_Lean_Expr_app___override(v___x_2016_, v___x_1998_);
v_p_1641_ = v___x_2017_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
else
{
lean_object* v_a_2018_; lean_object* v___x_2020_; uint8_t v_isShared_2021_; uint8_t v_isSharedCheck_2025_; 
lean_dec(v_fst_1993_);
lean_dec(v_fst_1992_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2018_ = lean_ctor_get(v___x_1996_, 0);
v_isSharedCheck_2025_ = !lean_is_exclusive(v___x_1996_);
if (v_isSharedCheck_2025_ == 0)
{
v___x_2020_ = v___x_1996_;
v_isShared_2021_ = v_isSharedCheck_2025_;
goto v_resetjp_2019_;
}
else
{
lean_inc(v_a_2018_);
lean_dec(v___x_1996_);
v___x_2020_ = lean_box(0);
v_isShared_2021_ = v_isSharedCheck_2025_;
goto v_resetjp_2019_;
}
v_resetjp_2019_:
{
lean_object* v___x_2023_; 
if (v_isShared_2021_ == 0)
{
v___x_2023_ = v___x_2020_;
goto v_reusejp_2022_;
}
else
{
lean_object* v_reuseFailAlloc_2024_; 
v_reuseFailAlloc_2024_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2024_, 0, v_a_2018_);
v___x_2023_ = v_reuseFailAlloc_2024_;
goto v_reusejp_2022_;
}
v_reusejp_2022_:
{
return v___x_2023_;
}
}
}
}
}
else
{
lean_object* v_a_2026_; lean_object* v___x_2028_; uint8_t v_isShared_2029_; uint8_t v_isSharedCheck_2033_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2026_ = lean_ctor_get(v___x_1675_, 0);
v_isSharedCheck_2033_ = !lean_is_exclusive(v___x_1675_);
if (v_isSharedCheck_2033_ == 0)
{
v___x_2028_ = v___x_1675_;
v_isShared_2029_ = v_isSharedCheck_2033_;
goto v_resetjp_2027_;
}
else
{
lean_inc(v_a_2026_);
lean_dec(v___x_1675_);
v___x_2028_ = lean_box(0);
v_isShared_2029_ = v_isSharedCheck_2033_;
goto v_resetjp_2027_;
}
v_resetjp_2027_:
{
lean_object* v___x_2031_; 
if (v_isShared_2029_ == 0)
{
v___x_2031_ = v___x_2028_;
goto v_reusejp_2030_;
}
else
{
lean_object* v_reuseFailAlloc_2032_; 
v_reuseFailAlloc_2032_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2032_, 0, v_a_2026_);
v___x_2031_ = v_reuseFailAlloc_2032_;
goto v_reusejp_2030_;
}
v_reusejp_2030_:
{
return v___x_2031_;
}
}
}
}
else
{
lean_object* v_fst_2034_; lean_object* v_fst_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; lean_object* v___x_2039_; lean_object* v___x_2040_; 
lean_dec_ref(v_a_1649_);
v_fst_2034_ = lean_ctor_get(v_a_1668_, 0);
lean_inc(v_fst_2034_);
lean_dec(v_a_1668_);
v_fst_2035_ = lean_ctor_get(v_snd_1669_, 0);
lean_inc(v_fst_2035_);
lean_dec(v_snd_1669_);
lean_inc(v_a_1639_);
v___x_2036_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_2037_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__39);
v___x_2038_ = l_Lean_Expr_app___override(v___x_2037_, v_fst_2034_);
v___x_2039_ = l_Lean_Expr_app___override(v___x_2038_, v_fst_2035_);
v___x_2040_ = l_Lean_Expr_app___override(v___x_2039_, v___x_2036_);
v_p_1641_ = v___x_2040_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
}
else
{
lean_object* v_a_2041_; lean_object* v___x_2043_; uint8_t v_isShared_2044_; uint8_t v_isSharedCheck_2048_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2041_ = lean_ctor_get(v___x_1667_, 0);
v_isSharedCheck_2048_ = !lean_is_exclusive(v___x_1667_);
if (v_isSharedCheck_2048_ == 0)
{
v___x_2043_ = v___x_1667_;
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
else
{
lean_inc(v_a_2041_);
lean_dec(v___x_1667_);
v___x_2043_ = lean_box(0);
v_isShared_2044_ = v_isSharedCheck_2048_;
goto v_resetjp_2042_;
}
v_resetjp_2042_:
{
lean_object* v___x_2046_; 
if (v_isShared_2044_ == 0)
{
v___x_2046_ = v___x_2043_;
goto v_reusejp_2045_;
}
else
{
lean_object* v_reuseFailAlloc_2047_; 
v_reuseFailAlloc_2047_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2047_, 0, v_a_2041_);
v___x_2046_ = v_reuseFailAlloc_2047_;
goto v_reusejp_2045_;
}
v_reusejp_2045_:
{
return v___x_2046_;
}
}
}
}
else
{
lean_object* v_fst_2049_; lean_object* v_fst_2050_; lean_object* v___x_2051_; lean_object* v___x_2052_; lean_object* v___x_2053_; lean_object* v___x_2054_; lean_object* v___x_2055_; lean_object* v___x_2056_; lean_object* v___x_2057_; lean_object* v___x_2058_; lean_object* v___x_2059_; lean_object* v___x_2060_; lean_object* v___x_2061_; lean_object* v___x_2062_; lean_object* v___x_2063_; lean_object* v___x_2064_; lean_object* v___x_2065_; 
lean_dec_ref(v_a_1649_);
v_fst_2049_ = lean_ctor_get(v_a_1660_, 0);
lean_inc_n(v_fst_2049_, 3);
lean_dec(v_a_1660_);
v_fst_2050_ = lean_ctor_get(v_snd_1661_, 0);
lean_inc_n(v_fst_2050_, 3);
lean_dec(v_snd_1661_);
lean_inc(v_a_1639_);
v___x_2051_ = l_Lean_LocalDecl_toExpr(v_a_1639_);
v___x_2052_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__2);
v___x_2053_ = l_Lean_Expr_app___override(v___x_2052_, v_fst_2049_);
v___x_2054_ = l_Lean_Expr_app___override(v___x_2053_, v_fst_2050_);
v___x_2055_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__42);
v___x_2056_ = l_Lean_Expr_app___override(v___x_2055_, v___x_2054_);
v___x_2057_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__30);
v___x_2058_ = l_Lean_Expr_app___override(v___x_2057_, v_fst_2049_);
v___x_2059_ = l_Lean_Expr_app___override(v___x_2058_, v_fst_2050_);
v___x_2060_ = l_Lean_Expr_app___override(v___x_2056_, v___x_2059_);
v___x_2061_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__45);
v___x_2062_ = l_Lean_Expr_app___override(v___x_2061_, v_fst_2049_);
v___x_2063_ = l_Lean_Expr_app___override(v___x_2062_, v_fst_2050_);
v___x_2064_ = l_Lean_Expr_app___override(v___x_2060_, v___x_2063_);
v___x_2065_ = l_Lean_Expr_app___override(v___x_2064_, v___x_2051_);
v_p_1641_ = v___x_2065_;
v___y_1642_ = v___y_1632_;
v___y_1643_ = v___y_1633_;
v___y_1644_ = v___y_1634_;
v___y_1645_ = v___y_1635_;
goto v___jp_1640_;
}
}
else
{
lean_object* v_a_2066_; lean_object* v___x_2068_; uint8_t v_isShared_2069_; uint8_t v_isSharedCheck_2073_; 
lean_dec_ref(v_a_1649_);
lean_dec(v_a_1639_);
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2066_ = lean_ctor_get(v___x_1659_, 0);
v_isSharedCheck_2073_ = !lean_is_exclusive(v___x_1659_);
if (v_isSharedCheck_2073_ == 0)
{
v___x_2068_ = v___x_1659_;
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
else
{
lean_inc(v_a_2066_);
lean_dec(v___x_1659_);
v___x_2068_ = lean_box(0);
v_isShared_2069_ = v_isSharedCheck_2073_;
goto v_resetjp_2067_;
}
v_resetjp_2067_:
{
lean_object* v___x_2071_; 
if (v_isShared_2069_ == 0)
{
v___x_2071_ = v___x_2068_;
goto v_reusejp_2070_;
}
else
{
lean_object* v_reuseFailAlloc_2072_; 
v_reuseFailAlloc_2072_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2072_, 0, v_a_2066_);
v___x_2071_ = v_reuseFailAlloc_2072_;
goto v_reusejp_2070_;
}
v_reusejp_2070_:
{
return v___x_2071_;
}
}
}
}
}
else
{
lean_object* v_a_2096_; lean_object* v___x_2098_; uint8_t v_isShared_2099_; uint8_t v_isSharedCheck_2103_; 
lean_dec(v_fvarId_1637_);
lean_dec(v_g_1631_);
v_a_2096_ = lean_ctor_get(v___x_1638_, 0);
v_isSharedCheck_2103_ = !lean_is_exclusive(v___x_1638_);
if (v_isSharedCheck_2103_ == 0)
{
v___x_2098_ = v___x_1638_;
v_isShared_2099_ = v_isSharedCheck_2103_;
goto v_resetjp_2097_;
}
else
{
lean_inc(v_a_2096_);
lean_dec(v___x_1638_);
v___x_2098_ = lean_box(0);
v_isShared_2099_ = v_isSharedCheck_2103_;
goto v_resetjp_2097_;
}
v_resetjp_2097_:
{
lean_object* v___x_2101_; 
if (v_isShared_2099_ == 0)
{
v___x_2101_ = v___x_2098_;
goto v_reusejp_2100_;
}
else
{
lean_object* v_reuseFailAlloc_2102_; 
v_reuseFailAlloc_2102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2102_, 0, v_a_2096_);
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
lean_object* v___x_2104_; lean_object* v___x_2105_; lean_object* v___x_2106_; lean_object* v___x_2107_; 
lean_dec(v_g_1631_);
v___x_2104_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__49);
v___x_2105_ = l_Lean_MessageData_ofExpr(v_hypFVar_1630_);
v___x_2106_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2106_, 0, v___x_2104_);
lean_ctor_set(v___x_2106_, 1, v___x_2105_);
v___x_2107_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v___x_2106_, v___y_1632_, v___y_1633_, v___y_1634_, v___y_1635_);
return v___x_2107_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___boxed(lean_object* v_hypFVar_2108_, lean_object* v_g_2109_, lean_object* v___y_2110_, lean_object* v___y_2111_, lean_object* v___y_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_){
_start:
{
lean_object* v_res_2115_; 
v_res_2115_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12(v_hypFVar_2108_, v_g_2109_, v___y_2110_, v___y_2111_, v___y_2112_, v___y_2113_);
lean_dec(v___y_2113_);
lean_dec_ref(v___y_2112_);
lean_dec(v___y_2111_);
lean_dec_ref(v___y_2110_);
return v_res_2115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt(lean_object* v_hypFVar_2116_, lean_object* v_g_2117_, lean_object* v_a_2118_, lean_object* v_a_2119_, lean_object* v_a_2120_, lean_object* v_a_2121_){
_start:
{
lean_object* v___y_2123_; lean_object* v___x_2124_; 
lean_inc(v_g_2117_);
v___y_2123_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___boxed), 7, 2);
lean_closure_set(v___y_2123_, 0, v_hypFVar_2116_);
lean_closure_set(v___y_2123_, 1, v_g_2117_);
v___x_2124_ = lp_mathlib_Lean_MVarId_withContext___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__4___redArg(v_g_2117_, v___y_2123_, v_a_2118_, v_a_2119_, v_a_2120_, v_a_2121_);
return v___x_2124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___boxed(lean_object* v_hypFVar_2125_, lean_object* v_g_2126_, lean_object* v_a_2127_, lean_object* v_a_2128_, lean_object* v_a_2129_, lean_object* v_a_2130_, lean_object* v_a_2131_){
_start:
{
lean_object* v_res_2132_; 
v_res_2132_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt(v_hypFVar_2125_, v_g_2126_, v_a_2127_, v_a_2128_, v_a_2129_, v_a_2130_);
lean_dec(v_a_2130_);
lean_dec_ref(v_a_2129_);
lean_dec(v_a_2128_);
lean_dec_ref(v_a_2127_);
return v_res_2132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3(lean_object* v_00_u03b1_2133_, lean_object* v_msg_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_){
_start:
{
lean_object* v___x_2140_; 
v___x_2140_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v_msg_2134_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_);
return v___x_2140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___boxed(lean_object* v_00_u03b1_2141_, lean_object* v_msg_2142_, lean_object* v___y_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_){
_start:
{
lean_object* v_res_2148_; 
v_res_2148_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3(v_00_u03b1_2141_, v_msg_2142_, v___y_2143_, v___y_2144_, v___y_2145_, v___y_2146_);
lean_dec(v___y_2146_);
lean_dec_ref(v___y_2145_);
lean_dec(v___y_2144_);
lean_dec_ref(v___y_2143_);
return v_res_2148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Tauto_distribNotAt_spec__0(lean_object* v_a_2149_, lean_object* v_a_2150_, lean_object* v_a_2151_){
_start:
{
if (lean_obj_tag(v_a_2150_) == 0)
{
lean_object* v___x_2152_; 
lean_dec_ref(v_a_2149_);
v___x_2152_ = l_List_reverse___redArg(v_a_2151_);
return v___x_2152_;
}
else
{
lean_object* v_head_2153_; lean_object* v_tail_2154_; lean_object* v___x_2156_; uint8_t v_isShared_2157_; uint8_t v_isSharedCheck_2164_; 
v_head_2153_ = lean_ctor_get(v_a_2150_, 0);
v_tail_2154_ = lean_ctor_get(v_a_2150_, 1);
v_isSharedCheck_2164_ = !lean_is_exclusive(v_a_2150_);
if (v_isSharedCheck_2164_ == 0)
{
v___x_2156_ = v_a_2150_;
v_isShared_2157_ = v_isSharedCheck_2164_;
goto v_resetjp_2155_;
}
else
{
lean_inc(v_tail_2154_);
lean_inc(v_head_2153_);
lean_dec(v_a_2150_);
v___x_2156_ = lean_box(0);
v_isShared_2157_ = v_isSharedCheck_2164_;
goto v_resetjp_2155_;
}
v_resetjp_2155_:
{
lean_object* v_subst_2158_; lean_object* v___x_2159_; lean_object* v___x_2161_; 
v_subst_2158_ = lean_ctor_get(v_a_2149_, 2);
lean_inc(v_subst_2158_);
v___x_2159_ = l_Lean_Meta_FVarSubst_apply(v_subst_2158_, v_head_2153_);
lean_dec(v_head_2153_);
if (v_isShared_2157_ == 0)
{
lean_ctor_set(v___x_2156_, 1, v_a_2151_);
lean_ctor_set(v___x_2156_, 0, v___x_2159_);
v___x_2161_ = v___x_2156_;
goto v_reusejp_2160_;
}
else
{
lean_object* v_reuseFailAlloc_2163_; 
v_reuseFailAlloc_2163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2163_, 0, v___x_2159_);
lean_ctor_set(v_reuseFailAlloc_2163_, 1, v_a_2151_);
v___x_2161_ = v_reuseFailAlloc_2163_;
goto v_reusejp_2160_;
}
v_reusejp_2160_:
{
v_a_2150_ = v_tail_2154_;
v_a_2151_ = v___x_2161_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt(lean_object* v_nIters_2165_, lean_object* v_state_2166_, lean_object* v_a_2167_, lean_object* v_a_2168_, lean_object* v_a_2169_, lean_object* v_a_2170_){
_start:
{
lean_object* v___y_2173_; uint8_t v___y_2174_; lean_object* v___y_2177_; lean_object* v_a_2178_; lean_object* v_zero_2181_; uint8_t v_isZero_2182_; 
v_zero_2181_ = lean_unsigned_to_nat(0u);
v_isZero_2182_ = lean_nat_dec_eq(v_nIters_2165_, v_zero_2181_);
if (v_isZero_2182_ == 1)
{
lean_object* v___x_2183_; 
v___x_2183_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2183_, 0, v_state_2166_);
return v___x_2183_;
}
else
{
lean_object* v_fvars_2184_; 
v_fvars_2184_ = lean_ctor_get(v_state_2166_, 0);
lean_inc(v_fvars_2184_);
if (lean_obj_tag(v_fvars_2184_) == 0)
{
lean_object* v___x_2185_; 
v___x_2185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2185_, 0, v_state_2166_);
return v___x_2185_;
}
else
{
lean_object* v_currentGoal_2186_; lean_object* v_head_2187_; lean_object* v_tail_2188_; lean_object* v___x_2190_; uint8_t v_isShared_2191_; uint8_t v_isSharedCheck_2215_; 
v_currentGoal_2186_ = lean_ctor_get(v_state_2166_, 1);
v_head_2187_ = lean_ctor_get(v_fvars_2184_, 0);
v_tail_2188_ = lean_ctor_get(v_fvars_2184_, 1);
v_isSharedCheck_2215_ = !lean_is_exclusive(v_fvars_2184_);
if (v_isSharedCheck_2215_ == 0)
{
v___x_2190_ = v_fvars_2184_;
v_isShared_2191_ = v_isSharedCheck_2215_;
goto v_resetjp_2189_;
}
else
{
lean_inc(v_tail_2188_);
lean_inc(v_head_2187_);
lean_dec(v_fvars_2184_);
v___x_2190_ = lean_box(0);
v_isShared_2191_ = v_isSharedCheck_2215_;
goto v_resetjp_2189_;
}
v_resetjp_2189_:
{
lean_object* v___x_2192_; 
lean_inc(v_currentGoal_2186_);
v___x_2192_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt(v_head_2187_, v_currentGoal_2186_, v_a_2167_, v_a_2168_, v_a_2169_, v_a_2170_);
if (lean_obj_tag(v___x_2192_) == 0)
{
lean_object* v_a_2193_; lean_object* v_fvarId_2194_; lean_object* v_mvarId_2195_; lean_object* v_one_2196_; lean_object* v_n_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2202_; 
v_a_2193_ = lean_ctor_get(v___x_2192_, 0);
lean_inc(v_a_2193_);
lean_dec_ref_known(v___x_2192_, 1);
v_fvarId_2194_ = lean_ctor_get(v_a_2193_, 0);
v_mvarId_2195_ = lean_ctor_get(v_a_2193_, 1);
lean_inc(v_mvarId_2195_);
v_one_2196_ = lean_unsigned_to_nat(1u);
v_n_2197_ = lean_nat_sub(v_nIters_2165_, v_one_2196_);
lean_inc(v_fvarId_2194_);
v___x_2198_ = l_Lean_mkFVar(v_fvarId_2194_);
v___x_2199_ = lean_box(0);
v___x_2200_ = lp_mathlib_List_mapTR_loop___at___00Mathlib_Tactic_Tauto_distribNotAt_spec__0(v_a_2193_, v_tail_2188_, v___x_2199_);
if (v_isShared_2191_ == 0)
{
lean_ctor_set(v___x_2190_, 1, v___x_2200_);
lean_ctor_set(v___x_2190_, 0, v___x_2198_);
v___x_2202_ = v___x_2190_;
goto v_reusejp_2201_;
}
else
{
lean_object* v_reuseFailAlloc_2206_; 
v_reuseFailAlloc_2206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2206_, 0, v___x_2198_);
lean_ctor_set(v_reuseFailAlloc_2206_, 1, v___x_2200_);
v___x_2202_ = v_reuseFailAlloc_2206_;
goto v_reusejp_2201_;
}
v_reusejp_2201_:
{
lean_object* v___x_2203_; lean_object* v___x_2204_; 
v___x_2203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2203_, 0, v___x_2202_);
lean_ctor_set(v___x_2203_, 1, v_mvarId_2195_);
v___x_2204_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt(v_n_2197_, v___x_2203_, v_a_2167_, v_a_2168_, v_a_2169_, v_a_2170_);
lean_dec(v_n_2197_);
if (lean_obj_tag(v___x_2204_) == 0)
{
lean_dec_ref(v_state_2166_);
return v___x_2204_;
}
else
{
lean_object* v_a_2205_; 
v_a_2205_ = lean_ctor_get(v___x_2204_, 0);
lean_inc(v_a_2205_);
v___y_2177_ = v___x_2204_;
v_a_2178_ = v_a_2205_;
goto v___jp_2176_;
}
}
}
else
{
lean_object* v_a_2207_; lean_object* v___x_2209_; uint8_t v_isShared_2210_; uint8_t v_isSharedCheck_2214_; 
lean_del_object(v___x_2190_);
lean_dec(v_tail_2188_);
v_a_2207_ = lean_ctor_get(v___x_2192_, 0);
v_isSharedCheck_2214_ = !lean_is_exclusive(v___x_2192_);
if (v_isSharedCheck_2214_ == 0)
{
v___x_2209_ = v___x_2192_;
v_isShared_2210_ = v_isSharedCheck_2214_;
goto v_resetjp_2208_;
}
else
{
lean_inc(v_a_2207_);
lean_dec(v___x_2192_);
v___x_2209_ = lean_box(0);
v_isShared_2210_ = v_isSharedCheck_2214_;
goto v_resetjp_2208_;
}
v_resetjp_2208_:
{
lean_object* v___x_2212_; 
lean_inc(v_a_2207_);
if (v_isShared_2210_ == 0)
{
v___x_2212_ = v___x_2209_;
goto v_reusejp_2211_;
}
else
{
lean_object* v_reuseFailAlloc_2213_; 
v_reuseFailAlloc_2213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2213_, 0, v_a_2207_);
v___x_2212_ = v_reuseFailAlloc_2213_;
goto v_reusejp_2211_;
}
v_reusejp_2211_:
{
v___y_2177_ = v___x_2212_;
v_a_2178_ = v_a_2207_;
goto v___jp_2176_;
}
}
}
}
}
}
v___jp_2172_:
{
if (v___y_2174_ == 0)
{
lean_object* v___x_2175_; 
lean_dec_ref(v___y_2173_);
v___x_2175_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2175_, 0, v_state_2166_);
return v___x_2175_;
}
else
{
lean_dec_ref(v_state_2166_);
return v___y_2173_;
}
}
v___jp_2176_:
{
uint8_t v___x_2179_; 
v___x_2179_ = l_Lean_Exception_isInterrupt(v_a_2178_);
if (v___x_2179_ == 0)
{
uint8_t v___x_2180_; 
v___x_2180_ = l_Lean_Exception_isRuntime(v_a_2178_);
v___y_2173_ = v___y_2177_;
v___y_2174_ = v___x_2180_;
goto v___jp_2172_;
}
else
{
lean_dec_ref(v_a_2178_);
v___y_2173_ = v___y_2177_;
v___y_2174_ = v___x_2179_;
goto v___jp_2172_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt___boxed(lean_object* v_nIters_2216_, lean_object* v_state_2217_, lean_object* v_a_2218_, lean_object* v_a_2219_, lean_object* v_a_2220_, lean_object* v_a_2221_, lean_object* v_a_2222_){
_start:
{
lean_object* v_res_2223_; 
v_res_2223_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt(v_nIters_2216_, v_state_2217_, v_a_2218_, v_a_2219_, v_a_2220_, v_a_2221_);
lean_dec(v_a_2221_);
lean_dec_ref(v_a_2220_);
lean_dec(v_a_2219_);
lean_dec_ref(v_a_2218_);
lean_dec(v_nIters_2216_);
return v_res_2223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux(lean_object* v_fvars_2224_, lean_object* v_g_2225_, lean_object* v_a_2226_, lean_object* v_a_2227_, lean_object* v_a_2228_, lean_object* v_a_2229_){
_start:
{
if (lean_obj_tag(v_fvars_2224_) == 0)
{
lean_object* v___x_2231_; 
v___x_2231_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2231_, 0, v_g_2225_);
return v___x_2231_;
}
else
{
lean_object* v___x_2232_; lean_object* v___x_2233_; lean_object* v___x_2234_; 
v___x_2232_ = lean_unsigned_to_nat(3u);
v___x_2233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2233_, 0, v_fvars_2224_);
lean_ctor_set(v___x_2233_, 1, v_g_2225_);
v___x_2234_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotAt(v___x_2232_, v___x_2233_, v_a_2226_, v_a_2227_, v_a_2228_, v_a_2229_);
if (lean_obj_tag(v___x_2234_) == 0)
{
lean_object* v_a_2235_; lean_object* v_fvars_2236_; lean_object* v_currentGoal_2237_; lean_object* v___x_2238_; 
v_a_2235_ = lean_ctor_get(v___x_2234_, 0);
lean_inc(v_a_2235_);
lean_dec_ref_known(v___x_2234_, 1);
v_fvars_2236_ = lean_ctor_get(v_a_2235_, 0);
lean_inc(v_fvars_2236_);
v_currentGoal_2237_ = lean_ctor_get(v_a_2235_, 1);
lean_inc(v_currentGoal_2237_);
lean_dec(v_a_2235_);
v___x_2238_ = l_List_tail_x21___redArg(v_fvars_2236_);
lean_dec(v_fvars_2236_);
v_fvars_2224_ = v___x_2238_;
v_g_2225_ = v_currentGoal_2237_;
goto _start;
}
else
{
lean_object* v_a_2240_; lean_object* v___x_2242_; uint8_t v_isShared_2243_; uint8_t v_isSharedCheck_2247_; 
v_a_2240_ = lean_ctor_get(v___x_2234_, 0);
v_isSharedCheck_2247_ = !lean_is_exclusive(v___x_2234_);
if (v_isSharedCheck_2247_ == 0)
{
v___x_2242_ = v___x_2234_;
v_isShared_2243_ = v_isSharedCheck_2247_;
goto v_resetjp_2241_;
}
else
{
lean_inc(v_a_2240_);
lean_dec(v___x_2234_);
v___x_2242_ = lean_box(0);
v_isShared_2243_ = v_isSharedCheck_2247_;
goto v_resetjp_2241_;
}
v_resetjp_2241_:
{
lean_object* v___x_2245_; 
if (v_isShared_2243_ == 0)
{
v___x_2245_ = v___x_2242_;
goto v_reusejp_2244_;
}
else
{
lean_object* v_reuseFailAlloc_2246_; 
v_reuseFailAlloc_2246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2246_, 0, v_a_2240_);
v___x_2245_ = v_reuseFailAlloc_2246_;
goto v_reusejp_2244_;
}
v_reusejp_2244_:
{
return v___x_2245_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux___boxed(lean_object* v_fvars_2248_, lean_object* v_g_2249_, lean_object* v_a_2250_, lean_object* v_a_2251_, lean_object* v_a_2252_, lean_object* v_a_2253_, lean_object* v_a_2254_){
_start:
{
lean_object* v_res_2255_; 
v_res_2255_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux(v_fvars_2248_, v_g_2249_, v_a_2250_, v_a_2251_, v_a_2252_, v_a_2253_);
lean_dec(v_a_2253_);
lean_dec_ref(v_a_2252_);
lean_dec(v_a_2251_);
lean_dec_ref(v_a_2250_);
return v_res_2255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg(lean_object* v_as_2256_, size_t v_sz_2257_, size_t v_i_2258_, lean_object* v_b_2259_){
_start:
{
uint8_t v___x_2261_; 
v___x_2261_ = lean_usize_dec_lt(v_i_2258_, v_sz_2257_);
if (v___x_2261_ == 0)
{
lean_object* v___x_2262_; 
v___x_2262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2262_, 0, v_b_2259_);
return v___x_2262_;
}
else
{
lean_object* v_snd_2263_; lean_object* v___x_2265_; uint8_t v_isShared_2266_; uint8_t v_isSharedCheck_2282_; 
v_snd_2263_ = lean_ctor_get(v_b_2259_, 1);
v_isSharedCheck_2282_ = !lean_is_exclusive(v_b_2259_);
if (v_isSharedCheck_2282_ == 0)
{
lean_object* v_unused_2283_; 
v_unused_2283_ = lean_ctor_get(v_b_2259_, 0);
lean_dec(v_unused_2283_);
v___x_2265_ = v_b_2259_;
v_isShared_2266_ = v_isSharedCheck_2282_;
goto v_resetjp_2264_;
}
else
{
lean_inc(v_snd_2263_);
lean_dec(v_b_2259_);
v___x_2265_ = lean_box(0);
v_isShared_2266_ = v_isSharedCheck_2282_;
goto v_resetjp_2264_;
}
v_resetjp_2264_:
{
lean_object* v___x_2267_; lean_object* v_a_2269_; lean_object* v_a_2276_; 
v___x_2267_ = lean_box(0);
v_a_2276_ = lean_array_uget_borrowed(v_as_2256_, v_i_2258_);
if (lean_obj_tag(v_a_2276_) == 0)
{
v_a_2269_ = v_snd_2263_;
goto v___jp_2268_;
}
else
{
lean_object* v_val_2277_; uint8_t v___x_2278_; 
v_val_2277_ = lean_ctor_get(v_a_2276_, 0);
v___x_2278_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2277_);
if (v___x_2278_ == 0)
{
lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; 
v___x_2279_ = l_Lean_LocalDecl_fvarId(v_val_2277_);
v___x_2280_ = l_Lean_mkFVar(v___x_2279_);
v___x_2281_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2281_, 0, v___x_2280_);
lean_ctor_set(v___x_2281_, 1, v_snd_2263_);
v_a_2269_ = v___x_2281_;
goto v___jp_2268_;
}
else
{
v_a_2269_ = v_snd_2263_;
goto v___jp_2268_;
}
}
v___jp_2268_:
{
lean_object* v___x_2271_; 
if (v_isShared_2266_ == 0)
{
lean_ctor_set(v___x_2265_, 1, v_a_2269_);
lean_ctor_set(v___x_2265_, 0, v___x_2267_);
v___x_2271_ = v___x_2265_;
goto v_reusejp_2270_;
}
else
{
lean_object* v_reuseFailAlloc_2275_; 
v_reuseFailAlloc_2275_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2275_, 0, v___x_2267_);
lean_ctor_set(v_reuseFailAlloc_2275_, 1, v_a_2269_);
v___x_2271_ = v_reuseFailAlloc_2275_;
goto v_reusejp_2270_;
}
v_reusejp_2270_:
{
size_t v___x_2272_; size_t v___x_2273_; 
v___x_2272_ = ((size_t)1ULL);
v___x_2273_ = lean_usize_add(v_i_2258_, v___x_2272_);
v_i_2258_ = v___x_2273_;
v_b_2259_ = v___x_2271_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_as_2284_, lean_object* v_sz_2285_, lean_object* v_i_2286_, lean_object* v_b_2287_, lean_object* v___y_2288_){
_start:
{
size_t v_sz_boxed_2289_; size_t v_i_boxed_2290_; lean_object* v_res_2291_; 
v_sz_boxed_2289_ = lean_unbox_usize(v_sz_2285_);
lean_dec(v_sz_2285_);
v_i_boxed_2290_ = lean_unbox_usize(v_i_2286_);
lean_dec(v_i_2286_);
v_res_2291_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg(v_as_2284_, v_sz_boxed_2289_, v_i_boxed_2290_, v_b_2287_);
lean_dec_ref(v_as_2284_);
return v_res_2291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1(lean_object* v_as_2292_, size_t v_sz_2293_, size_t v_i_2294_, lean_object* v_b_2295_, lean_object* v___y_2296_, lean_object* v___y_2297_, lean_object* v___y_2298_, lean_object* v___y_2299_, lean_object* v___y_2300_, lean_object* v___y_2301_, lean_object* v___y_2302_, lean_object* v___y_2303_){
_start:
{
uint8_t v___x_2305_; 
v___x_2305_ = lean_usize_dec_lt(v_i_2294_, v_sz_2293_);
if (v___x_2305_ == 0)
{
lean_object* v___x_2306_; 
v___x_2306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2306_, 0, v_b_2295_);
return v___x_2306_;
}
else
{
lean_object* v_snd_2307_; lean_object* v___x_2309_; uint8_t v_isShared_2310_; uint8_t v_isSharedCheck_2326_; 
v_snd_2307_ = lean_ctor_get(v_b_2295_, 1);
v_isSharedCheck_2326_ = !lean_is_exclusive(v_b_2295_);
if (v_isSharedCheck_2326_ == 0)
{
lean_object* v_unused_2327_; 
v_unused_2327_ = lean_ctor_get(v_b_2295_, 0);
lean_dec(v_unused_2327_);
v___x_2309_ = v_b_2295_;
v_isShared_2310_ = v_isSharedCheck_2326_;
goto v_resetjp_2308_;
}
else
{
lean_inc(v_snd_2307_);
lean_dec(v_b_2295_);
v___x_2309_ = lean_box(0);
v_isShared_2310_ = v_isSharedCheck_2326_;
goto v_resetjp_2308_;
}
v_resetjp_2308_:
{
lean_object* v___x_2311_; lean_object* v_a_2313_; lean_object* v_a_2320_; 
v___x_2311_ = lean_box(0);
v_a_2320_ = lean_array_uget_borrowed(v_as_2292_, v_i_2294_);
if (lean_obj_tag(v_a_2320_) == 0)
{
v_a_2313_ = v_snd_2307_;
goto v___jp_2312_;
}
else
{
lean_object* v_val_2321_; uint8_t v___x_2322_; 
v_val_2321_ = lean_ctor_get(v_a_2320_, 0);
v___x_2322_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2321_);
if (v___x_2322_ == 0)
{
lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; 
v___x_2323_ = l_Lean_LocalDecl_fvarId(v_val_2321_);
v___x_2324_ = l_Lean_mkFVar(v___x_2323_);
v___x_2325_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2325_, 0, v___x_2324_);
lean_ctor_set(v___x_2325_, 1, v_snd_2307_);
v_a_2313_ = v___x_2325_;
goto v___jp_2312_;
}
else
{
v_a_2313_ = v_snd_2307_;
goto v___jp_2312_;
}
}
v___jp_2312_:
{
lean_object* v___x_2315_; 
if (v_isShared_2310_ == 0)
{
lean_ctor_set(v___x_2309_, 1, v_a_2313_);
lean_ctor_set(v___x_2309_, 0, v___x_2311_);
v___x_2315_ = v___x_2309_;
goto v_reusejp_2314_;
}
else
{
lean_object* v_reuseFailAlloc_2319_; 
v_reuseFailAlloc_2319_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2319_, 0, v___x_2311_);
lean_ctor_set(v_reuseFailAlloc_2319_, 1, v_a_2313_);
v___x_2315_ = v_reuseFailAlloc_2319_;
goto v_reusejp_2314_;
}
v_reusejp_2314_:
{
size_t v___x_2316_; size_t v___x_2317_; lean_object* v___x_2318_; 
v___x_2316_ = ((size_t)1ULL);
v___x_2317_ = lean_usize_add(v_i_2294_, v___x_2316_);
v___x_2318_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg(v_as_2292_, v_sz_2293_, v___x_2317_, v___x_2315_);
return v___x_2318_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1___boxed(lean_object* v_as_2328_, lean_object* v_sz_2329_, lean_object* v_i_2330_, lean_object* v_b_2331_, lean_object* v___y_2332_, lean_object* v___y_2333_, lean_object* v___y_2334_, lean_object* v___y_2335_, lean_object* v___y_2336_, lean_object* v___y_2337_, lean_object* v___y_2338_, lean_object* v___y_2339_, lean_object* v___y_2340_){
_start:
{
size_t v_sz_boxed_2341_; size_t v_i_boxed_2342_; lean_object* v_res_2343_; 
v_sz_boxed_2341_ = lean_unbox_usize(v_sz_2329_);
lean_dec(v_sz_2329_);
v_i_boxed_2342_ = lean_unbox_usize(v_i_2330_);
lean_dec(v_i_2330_);
v_res_2343_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1(v_as_2328_, v_sz_boxed_2341_, v_i_boxed_2342_, v_b_2331_, v___y_2332_, v___y_2333_, v___y_2334_, v___y_2335_, v___y_2336_, v___y_2337_, v___y_2338_, v___y_2339_);
lean_dec(v___y_2339_);
lean_dec_ref(v___y_2338_);
lean_dec(v___y_2337_);
lean_dec_ref(v___y_2336_);
lean_dec(v___y_2335_);
lean_dec_ref(v___y_2334_);
lean_dec(v___y_2333_);
lean_dec_ref(v___y_2332_);
lean_dec_ref(v_as_2328_);
return v_res_2343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg(lean_object* v_as_2344_, size_t v_sz_2345_, size_t v_i_2346_, lean_object* v_b_2347_){
_start:
{
uint8_t v___x_2349_; 
v___x_2349_ = lean_usize_dec_lt(v_i_2346_, v_sz_2345_);
if (v___x_2349_ == 0)
{
lean_object* v___x_2350_; 
v___x_2350_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2350_, 0, v_b_2347_);
return v___x_2350_;
}
else
{
lean_object* v_snd_2351_; lean_object* v___x_2353_; uint8_t v_isShared_2354_; uint8_t v_isSharedCheck_2370_; 
v_snd_2351_ = lean_ctor_get(v_b_2347_, 1);
v_isSharedCheck_2370_ = !lean_is_exclusive(v_b_2347_);
if (v_isSharedCheck_2370_ == 0)
{
lean_object* v_unused_2371_; 
v_unused_2371_ = lean_ctor_get(v_b_2347_, 0);
lean_dec(v_unused_2371_);
v___x_2353_ = v_b_2347_;
v_isShared_2354_ = v_isSharedCheck_2370_;
goto v_resetjp_2352_;
}
else
{
lean_inc(v_snd_2351_);
lean_dec(v_b_2347_);
v___x_2353_ = lean_box(0);
v_isShared_2354_ = v_isSharedCheck_2370_;
goto v_resetjp_2352_;
}
v_resetjp_2352_:
{
lean_object* v___x_2355_; lean_object* v_a_2357_; lean_object* v_a_2364_; 
v___x_2355_ = lean_box(0);
v_a_2364_ = lean_array_uget_borrowed(v_as_2344_, v_i_2346_);
if (lean_obj_tag(v_a_2364_) == 0)
{
v_a_2357_ = v_snd_2351_;
goto v___jp_2356_;
}
else
{
lean_object* v_val_2365_; uint8_t v___x_2366_; 
v_val_2365_ = lean_ctor_get(v_a_2364_, 0);
v___x_2366_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2365_);
if (v___x_2366_ == 0)
{
lean_object* v___x_2367_; lean_object* v___x_2368_; lean_object* v___x_2369_; 
v___x_2367_ = l_Lean_LocalDecl_fvarId(v_val_2365_);
v___x_2368_ = l_Lean_mkFVar(v___x_2367_);
v___x_2369_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2369_, 0, v___x_2368_);
lean_ctor_set(v___x_2369_, 1, v_snd_2351_);
v_a_2357_ = v___x_2369_;
goto v___jp_2356_;
}
else
{
v_a_2357_ = v_snd_2351_;
goto v___jp_2356_;
}
}
v___jp_2356_:
{
lean_object* v___x_2359_; 
if (v_isShared_2354_ == 0)
{
lean_ctor_set(v___x_2353_, 1, v_a_2357_);
lean_ctor_set(v___x_2353_, 0, v___x_2355_);
v___x_2359_ = v___x_2353_;
goto v_reusejp_2358_;
}
else
{
lean_object* v_reuseFailAlloc_2363_; 
v_reuseFailAlloc_2363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2363_, 0, v___x_2355_);
lean_ctor_set(v_reuseFailAlloc_2363_, 1, v_a_2357_);
v___x_2359_ = v_reuseFailAlloc_2363_;
goto v_reusejp_2358_;
}
v_reusejp_2358_:
{
size_t v___x_2360_; size_t v___x_2361_; 
v___x_2360_ = ((size_t)1ULL);
v___x_2361_ = lean_usize_add(v_i_2346_, v___x_2360_);
v_i_2346_ = v___x_2361_;
v_b_2347_ = v___x_2359_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg___boxed(lean_object* v_as_2372_, lean_object* v_sz_2373_, lean_object* v_i_2374_, lean_object* v_b_2375_, lean_object* v___y_2376_){
_start:
{
size_t v_sz_boxed_2377_; size_t v_i_boxed_2378_; lean_object* v_res_2379_; 
v_sz_boxed_2377_ = lean_unbox_usize(v_sz_2373_);
lean_dec(v_sz_2373_);
v_i_boxed_2378_ = lean_unbox_usize(v_i_2374_);
lean_dec(v_i_2374_);
v_res_2379_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg(v_as_2372_, v_sz_boxed_2377_, v_i_boxed_2378_, v_b_2375_);
lean_dec_ref(v_as_2372_);
return v_res_2379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2(lean_object* v_as_2380_, size_t v_sz_2381_, size_t v_i_2382_, lean_object* v_b_2383_, lean_object* v___y_2384_, lean_object* v___y_2385_, lean_object* v___y_2386_, lean_object* v___y_2387_, lean_object* v___y_2388_, lean_object* v___y_2389_, lean_object* v___y_2390_, lean_object* v___y_2391_){
_start:
{
uint8_t v___x_2393_; 
v___x_2393_ = lean_usize_dec_lt(v_i_2382_, v_sz_2381_);
if (v___x_2393_ == 0)
{
lean_object* v___x_2394_; 
v___x_2394_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2394_, 0, v_b_2383_);
return v___x_2394_;
}
else
{
lean_object* v_snd_2395_; lean_object* v___x_2397_; uint8_t v_isShared_2398_; uint8_t v_isSharedCheck_2414_; 
v_snd_2395_ = lean_ctor_get(v_b_2383_, 1);
v_isSharedCheck_2414_ = !lean_is_exclusive(v_b_2383_);
if (v_isSharedCheck_2414_ == 0)
{
lean_object* v_unused_2415_; 
v_unused_2415_ = lean_ctor_get(v_b_2383_, 0);
lean_dec(v_unused_2415_);
v___x_2397_ = v_b_2383_;
v_isShared_2398_ = v_isSharedCheck_2414_;
goto v_resetjp_2396_;
}
else
{
lean_inc(v_snd_2395_);
lean_dec(v_b_2383_);
v___x_2397_ = lean_box(0);
v_isShared_2398_ = v_isSharedCheck_2414_;
goto v_resetjp_2396_;
}
v_resetjp_2396_:
{
lean_object* v___x_2399_; lean_object* v_a_2401_; lean_object* v_a_2408_; 
v___x_2399_ = lean_box(0);
v_a_2408_ = lean_array_uget_borrowed(v_as_2380_, v_i_2382_);
if (lean_obj_tag(v_a_2408_) == 0)
{
v_a_2401_ = v_snd_2395_;
goto v___jp_2400_;
}
else
{
lean_object* v_val_2409_; uint8_t v___x_2410_; 
v_val_2409_ = lean_ctor_get(v_a_2408_, 0);
v___x_2410_ = l_Lean_LocalDecl_isImplementationDetail(v_val_2409_);
if (v___x_2410_ == 0)
{
lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; 
v___x_2411_ = l_Lean_LocalDecl_fvarId(v_val_2409_);
v___x_2412_ = l_Lean_mkFVar(v___x_2411_);
v___x_2413_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2413_, 0, v___x_2412_);
lean_ctor_set(v___x_2413_, 1, v_snd_2395_);
v_a_2401_ = v___x_2413_;
goto v___jp_2400_;
}
else
{
v_a_2401_ = v_snd_2395_;
goto v___jp_2400_;
}
}
v___jp_2400_:
{
lean_object* v___x_2403_; 
if (v_isShared_2398_ == 0)
{
lean_ctor_set(v___x_2397_, 1, v_a_2401_);
lean_ctor_set(v___x_2397_, 0, v___x_2399_);
v___x_2403_ = v___x_2397_;
goto v_reusejp_2402_;
}
else
{
lean_object* v_reuseFailAlloc_2407_; 
v_reuseFailAlloc_2407_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2407_, 0, v___x_2399_);
lean_ctor_set(v_reuseFailAlloc_2407_, 1, v_a_2401_);
v___x_2403_ = v_reuseFailAlloc_2407_;
goto v_reusejp_2402_;
}
v_reusejp_2402_:
{
size_t v___x_2404_; size_t v___x_2405_; lean_object* v___x_2406_; 
v___x_2404_ = ((size_t)1ULL);
v___x_2405_ = lean_usize_add(v_i_2382_, v___x_2404_);
v___x_2406_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg(v_as_2380_, v_sz_2381_, v___x_2405_, v___x_2403_);
return v___x_2406_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2___boxed(lean_object* v_as_2416_, lean_object* v_sz_2417_, lean_object* v_i_2418_, lean_object* v_b_2419_, lean_object* v___y_2420_, lean_object* v___y_2421_, lean_object* v___y_2422_, lean_object* v___y_2423_, lean_object* v___y_2424_, lean_object* v___y_2425_, lean_object* v___y_2426_, lean_object* v___y_2427_, lean_object* v___y_2428_){
_start:
{
size_t v_sz_boxed_2429_; size_t v_i_boxed_2430_; lean_object* v_res_2431_; 
v_sz_boxed_2429_ = lean_unbox_usize(v_sz_2417_);
lean_dec(v_sz_2417_);
v_i_boxed_2430_ = lean_unbox_usize(v_i_2418_);
lean_dec(v_i_2418_);
v_res_2431_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2(v_as_2416_, v_sz_boxed_2429_, v_i_boxed_2430_, v_b_2419_, v___y_2420_, v___y_2421_, v___y_2422_, v___y_2423_, v___y_2424_, v___y_2425_, v___y_2426_, v___y_2427_);
lean_dec(v___y_2427_);
lean_dec_ref(v___y_2426_);
lean_dec(v___y_2425_);
lean_dec_ref(v___y_2424_);
lean_dec(v___y_2423_);
lean_dec_ref(v___y_2422_);
lean_dec(v___y_2421_);
lean_dec_ref(v___y_2420_);
lean_dec_ref(v_as_2416_);
return v_res_2431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0(lean_object* v_init_2432_, lean_object* v_n_2433_, lean_object* v_b_2434_, lean_object* v___y_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_, lean_object* v___y_2439_, lean_object* v___y_2440_, lean_object* v___y_2441_, lean_object* v___y_2442_){
_start:
{
if (lean_obj_tag(v_n_2433_) == 0)
{
lean_object* v_cs_2444_; lean_object* v___x_2445_; lean_object* v___x_2446_; size_t v_sz_2447_; size_t v___x_2448_; lean_object* v___x_2449_; 
v_cs_2444_ = lean_ctor_get(v_n_2433_, 0);
v___x_2445_ = lean_box(0);
v___x_2446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2446_, 0, v___x_2445_);
lean_ctor_set(v___x_2446_, 1, v_b_2434_);
v_sz_2447_ = lean_array_size(v_cs_2444_);
v___x_2448_ = ((size_t)0ULL);
v___x_2449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1(v_init_2432_, v_cs_2444_, v_sz_2447_, v___x_2448_, v___x_2446_, v___y_2435_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_, v___y_2440_, v___y_2441_, v___y_2442_);
if (lean_obj_tag(v___x_2449_) == 0)
{
lean_object* v_a_2450_; lean_object* v___x_2452_; uint8_t v_isShared_2453_; uint8_t v_isSharedCheck_2464_; 
v_a_2450_ = lean_ctor_get(v___x_2449_, 0);
v_isSharedCheck_2464_ = !lean_is_exclusive(v___x_2449_);
if (v_isSharedCheck_2464_ == 0)
{
v___x_2452_ = v___x_2449_;
v_isShared_2453_ = v_isSharedCheck_2464_;
goto v_resetjp_2451_;
}
else
{
lean_inc(v_a_2450_);
lean_dec(v___x_2449_);
v___x_2452_ = lean_box(0);
v_isShared_2453_ = v_isSharedCheck_2464_;
goto v_resetjp_2451_;
}
v_resetjp_2451_:
{
lean_object* v_fst_2454_; 
v_fst_2454_ = lean_ctor_get(v_a_2450_, 0);
if (lean_obj_tag(v_fst_2454_) == 0)
{
lean_object* v_snd_2455_; lean_object* v___x_2456_; lean_object* v___x_2458_; 
v_snd_2455_ = lean_ctor_get(v_a_2450_, 1);
lean_inc(v_snd_2455_);
lean_dec(v_a_2450_);
v___x_2456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2456_, 0, v_snd_2455_);
if (v_isShared_2453_ == 0)
{
lean_ctor_set(v___x_2452_, 0, v___x_2456_);
v___x_2458_ = v___x_2452_;
goto v_reusejp_2457_;
}
else
{
lean_object* v_reuseFailAlloc_2459_; 
v_reuseFailAlloc_2459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2459_, 0, v___x_2456_);
v___x_2458_ = v_reuseFailAlloc_2459_;
goto v_reusejp_2457_;
}
v_reusejp_2457_:
{
return v___x_2458_;
}
}
else
{
lean_object* v_val_2460_; lean_object* v___x_2462_; 
lean_inc_ref(v_fst_2454_);
lean_dec(v_a_2450_);
v_val_2460_ = lean_ctor_get(v_fst_2454_, 0);
lean_inc(v_val_2460_);
lean_dec_ref_known(v_fst_2454_, 1);
if (v_isShared_2453_ == 0)
{
lean_ctor_set(v___x_2452_, 0, v_val_2460_);
v___x_2462_ = v___x_2452_;
goto v_reusejp_2461_;
}
else
{
lean_object* v_reuseFailAlloc_2463_; 
v_reuseFailAlloc_2463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2463_, 0, v_val_2460_);
v___x_2462_ = v_reuseFailAlloc_2463_;
goto v_reusejp_2461_;
}
v_reusejp_2461_:
{
return v___x_2462_;
}
}
}
}
else
{
lean_object* v_a_2465_; lean_object* v___x_2467_; uint8_t v_isShared_2468_; uint8_t v_isSharedCheck_2472_; 
v_a_2465_ = lean_ctor_get(v___x_2449_, 0);
v_isSharedCheck_2472_ = !lean_is_exclusive(v___x_2449_);
if (v_isSharedCheck_2472_ == 0)
{
v___x_2467_ = v___x_2449_;
v_isShared_2468_ = v_isSharedCheck_2472_;
goto v_resetjp_2466_;
}
else
{
lean_inc(v_a_2465_);
lean_dec(v___x_2449_);
v___x_2467_ = lean_box(0);
v_isShared_2468_ = v_isSharedCheck_2472_;
goto v_resetjp_2466_;
}
v_resetjp_2466_:
{
lean_object* v___x_2470_; 
if (v_isShared_2468_ == 0)
{
v___x_2470_ = v___x_2467_;
goto v_reusejp_2469_;
}
else
{
lean_object* v_reuseFailAlloc_2471_; 
v_reuseFailAlloc_2471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2471_, 0, v_a_2465_);
v___x_2470_ = v_reuseFailAlloc_2471_;
goto v_reusejp_2469_;
}
v_reusejp_2469_:
{
return v___x_2470_;
}
}
}
}
else
{
lean_object* v_vs_2473_; lean_object* v___x_2474_; lean_object* v___x_2475_; size_t v_sz_2476_; size_t v___x_2477_; lean_object* v___x_2478_; 
v_vs_2473_ = lean_ctor_get(v_n_2433_, 0);
v___x_2474_ = lean_box(0);
v___x_2475_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2475_, 0, v___x_2474_);
lean_ctor_set(v___x_2475_, 1, v_b_2434_);
v_sz_2476_ = lean_array_size(v_vs_2473_);
v___x_2477_ = ((size_t)0ULL);
v___x_2478_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2(v_vs_2473_, v_sz_2476_, v___x_2477_, v___x_2475_, v___y_2435_, v___y_2436_, v___y_2437_, v___y_2438_, v___y_2439_, v___y_2440_, v___y_2441_, v___y_2442_);
if (lean_obj_tag(v___x_2478_) == 0)
{
lean_object* v_a_2479_; lean_object* v___x_2481_; uint8_t v_isShared_2482_; uint8_t v_isSharedCheck_2493_; 
v_a_2479_ = lean_ctor_get(v___x_2478_, 0);
v_isSharedCheck_2493_ = !lean_is_exclusive(v___x_2478_);
if (v_isSharedCheck_2493_ == 0)
{
v___x_2481_ = v___x_2478_;
v_isShared_2482_ = v_isSharedCheck_2493_;
goto v_resetjp_2480_;
}
else
{
lean_inc(v_a_2479_);
lean_dec(v___x_2478_);
v___x_2481_ = lean_box(0);
v_isShared_2482_ = v_isSharedCheck_2493_;
goto v_resetjp_2480_;
}
v_resetjp_2480_:
{
lean_object* v_fst_2483_; 
v_fst_2483_ = lean_ctor_get(v_a_2479_, 0);
if (lean_obj_tag(v_fst_2483_) == 0)
{
lean_object* v_snd_2484_; lean_object* v___x_2485_; lean_object* v___x_2487_; 
v_snd_2484_ = lean_ctor_get(v_a_2479_, 1);
lean_inc(v_snd_2484_);
lean_dec(v_a_2479_);
v___x_2485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2485_, 0, v_snd_2484_);
if (v_isShared_2482_ == 0)
{
lean_ctor_set(v___x_2481_, 0, v___x_2485_);
v___x_2487_ = v___x_2481_;
goto v_reusejp_2486_;
}
else
{
lean_object* v_reuseFailAlloc_2488_; 
v_reuseFailAlloc_2488_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2488_, 0, v___x_2485_);
v___x_2487_ = v_reuseFailAlloc_2488_;
goto v_reusejp_2486_;
}
v_reusejp_2486_:
{
return v___x_2487_;
}
}
else
{
lean_object* v_val_2489_; lean_object* v___x_2491_; 
lean_inc_ref(v_fst_2483_);
lean_dec(v_a_2479_);
v_val_2489_ = lean_ctor_get(v_fst_2483_, 0);
lean_inc(v_val_2489_);
lean_dec_ref_known(v_fst_2483_, 1);
if (v_isShared_2482_ == 0)
{
lean_ctor_set(v___x_2481_, 0, v_val_2489_);
v___x_2491_ = v___x_2481_;
goto v_reusejp_2490_;
}
else
{
lean_object* v_reuseFailAlloc_2492_; 
v_reuseFailAlloc_2492_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2492_, 0, v_val_2489_);
v___x_2491_ = v_reuseFailAlloc_2492_;
goto v_reusejp_2490_;
}
v_reusejp_2490_:
{
return v___x_2491_;
}
}
}
}
else
{
lean_object* v_a_2494_; lean_object* v___x_2496_; uint8_t v_isShared_2497_; uint8_t v_isSharedCheck_2501_; 
v_a_2494_ = lean_ctor_get(v___x_2478_, 0);
v_isSharedCheck_2501_ = !lean_is_exclusive(v___x_2478_);
if (v_isSharedCheck_2501_ == 0)
{
v___x_2496_ = v___x_2478_;
v_isShared_2497_ = v_isSharedCheck_2501_;
goto v_resetjp_2495_;
}
else
{
lean_inc(v_a_2494_);
lean_dec(v___x_2478_);
v___x_2496_ = lean_box(0);
v_isShared_2497_ = v_isSharedCheck_2501_;
goto v_resetjp_2495_;
}
v_resetjp_2495_:
{
lean_object* v___x_2499_; 
if (v_isShared_2497_ == 0)
{
v___x_2499_ = v___x_2496_;
goto v_reusejp_2498_;
}
else
{
lean_object* v_reuseFailAlloc_2500_; 
v_reuseFailAlloc_2500_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2500_, 0, v_a_2494_);
v___x_2499_ = v_reuseFailAlloc_2500_;
goto v_reusejp_2498_;
}
v_reusejp_2498_:
{
return v___x_2499_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1(lean_object* v_init_2502_, lean_object* v_as_2503_, size_t v_sz_2504_, size_t v_i_2505_, lean_object* v_b_2506_, lean_object* v___y_2507_, lean_object* v___y_2508_, lean_object* v___y_2509_, lean_object* v___y_2510_, lean_object* v___y_2511_, lean_object* v___y_2512_, lean_object* v___y_2513_, lean_object* v___y_2514_){
_start:
{
uint8_t v___x_2516_; 
v___x_2516_ = lean_usize_dec_lt(v_i_2505_, v_sz_2504_);
if (v___x_2516_ == 0)
{
lean_object* v___x_2517_; 
v___x_2517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2517_, 0, v_b_2506_);
return v___x_2517_;
}
else
{
lean_object* v_snd_2518_; lean_object* v___x_2520_; uint8_t v_isShared_2521_; uint8_t v_isSharedCheck_2552_; 
v_snd_2518_ = lean_ctor_get(v_b_2506_, 1);
v_isSharedCheck_2552_ = !lean_is_exclusive(v_b_2506_);
if (v_isSharedCheck_2552_ == 0)
{
lean_object* v_unused_2553_; 
v_unused_2553_ = lean_ctor_get(v_b_2506_, 0);
lean_dec(v_unused_2553_);
v___x_2520_ = v_b_2506_;
v_isShared_2521_ = v_isSharedCheck_2552_;
goto v_resetjp_2519_;
}
else
{
lean_inc(v_snd_2518_);
lean_dec(v_b_2506_);
v___x_2520_ = lean_box(0);
v_isShared_2521_ = v_isSharedCheck_2552_;
goto v_resetjp_2519_;
}
v_resetjp_2519_:
{
lean_object* v_a_2522_; lean_object* v___x_2523_; 
v_a_2522_ = lean_array_uget_borrowed(v_as_2503_, v_i_2505_);
lean_inc(v_snd_2518_);
v___x_2523_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0(v_init_2502_, v_a_2522_, v_snd_2518_, v___y_2507_, v___y_2508_, v___y_2509_, v___y_2510_, v___y_2511_, v___y_2512_, v___y_2513_, v___y_2514_);
if (lean_obj_tag(v___x_2523_) == 0)
{
lean_object* v_a_2524_; lean_object* v___x_2526_; uint8_t v_isShared_2527_; uint8_t v_isSharedCheck_2543_; 
v_a_2524_ = lean_ctor_get(v___x_2523_, 0);
v_isSharedCheck_2543_ = !lean_is_exclusive(v___x_2523_);
if (v_isSharedCheck_2543_ == 0)
{
v___x_2526_ = v___x_2523_;
v_isShared_2527_ = v_isSharedCheck_2543_;
goto v_resetjp_2525_;
}
else
{
lean_inc(v_a_2524_);
lean_dec(v___x_2523_);
v___x_2526_ = lean_box(0);
v_isShared_2527_ = v_isSharedCheck_2543_;
goto v_resetjp_2525_;
}
v_resetjp_2525_:
{
if (lean_obj_tag(v_a_2524_) == 0)
{
lean_object* v___x_2528_; lean_object* v___x_2530_; 
v___x_2528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2528_, 0, v_a_2524_);
if (v_isShared_2521_ == 0)
{
lean_ctor_set(v___x_2520_, 0, v___x_2528_);
v___x_2530_ = v___x_2520_;
goto v_reusejp_2529_;
}
else
{
lean_object* v_reuseFailAlloc_2534_; 
v_reuseFailAlloc_2534_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2534_, 0, v___x_2528_);
lean_ctor_set(v_reuseFailAlloc_2534_, 1, v_snd_2518_);
v___x_2530_ = v_reuseFailAlloc_2534_;
goto v_reusejp_2529_;
}
v_reusejp_2529_:
{
lean_object* v___x_2532_; 
if (v_isShared_2527_ == 0)
{
lean_ctor_set(v___x_2526_, 0, v___x_2530_);
v___x_2532_ = v___x_2526_;
goto v_reusejp_2531_;
}
else
{
lean_object* v_reuseFailAlloc_2533_; 
v_reuseFailAlloc_2533_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2533_, 0, v___x_2530_);
v___x_2532_ = v_reuseFailAlloc_2533_;
goto v_reusejp_2531_;
}
v_reusejp_2531_:
{
return v___x_2532_;
}
}
}
else
{
lean_object* v_a_2535_; lean_object* v___x_2536_; lean_object* v___x_2538_; 
lean_del_object(v___x_2526_);
lean_dec(v_snd_2518_);
v_a_2535_ = lean_ctor_get(v_a_2524_, 0);
lean_inc(v_a_2535_);
lean_dec_ref_known(v_a_2524_, 1);
v___x_2536_ = lean_box(0);
if (v_isShared_2521_ == 0)
{
lean_ctor_set(v___x_2520_, 1, v_a_2535_);
lean_ctor_set(v___x_2520_, 0, v___x_2536_);
v___x_2538_ = v___x_2520_;
goto v_reusejp_2537_;
}
else
{
lean_object* v_reuseFailAlloc_2542_; 
v_reuseFailAlloc_2542_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2542_, 0, v___x_2536_);
lean_ctor_set(v_reuseFailAlloc_2542_, 1, v_a_2535_);
v___x_2538_ = v_reuseFailAlloc_2542_;
goto v_reusejp_2537_;
}
v_reusejp_2537_:
{
size_t v___x_2539_; size_t v___x_2540_; 
v___x_2539_ = ((size_t)1ULL);
v___x_2540_ = lean_usize_add(v_i_2505_, v___x_2539_);
v_i_2505_ = v___x_2540_;
v_b_2506_ = v___x_2538_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_2544_; lean_object* v___x_2546_; uint8_t v_isShared_2547_; uint8_t v_isSharedCheck_2551_; 
lean_del_object(v___x_2520_);
lean_dec(v_snd_2518_);
v_a_2544_ = lean_ctor_get(v___x_2523_, 0);
v_isSharedCheck_2551_ = !lean_is_exclusive(v___x_2523_);
if (v_isSharedCheck_2551_ == 0)
{
v___x_2546_ = v___x_2523_;
v_isShared_2547_ = v_isSharedCheck_2551_;
goto v_resetjp_2545_;
}
else
{
lean_inc(v_a_2544_);
lean_dec(v___x_2523_);
v___x_2546_ = lean_box(0);
v_isShared_2547_ = v_isSharedCheck_2551_;
goto v_resetjp_2545_;
}
v_resetjp_2545_:
{
lean_object* v___x_2549_; 
if (v_isShared_2547_ == 0)
{
v___x_2549_ = v___x_2546_;
goto v_reusejp_2548_;
}
else
{
lean_object* v_reuseFailAlloc_2550_; 
v_reuseFailAlloc_2550_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2550_, 0, v_a_2544_);
v___x_2549_ = v_reuseFailAlloc_2550_;
goto v_reusejp_2548_;
}
v_reusejp_2548_:
{
return v___x_2549_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1___boxed(lean_object* v_init_2554_, lean_object* v_as_2555_, lean_object* v_sz_2556_, lean_object* v_i_2557_, lean_object* v_b_2558_, lean_object* v___y_2559_, lean_object* v___y_2560_, lean_object* v___y_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_, lean_object* v___y_2565_, lean_object* v___y_2566_, lean_object* v___y_2567_){
_start:
{
size_t v_sz_boxed_2568_; size_t v_i_boxed_2569_; lean_object* v_res_2570_; 
v_sz_boxed_2568_ = lean_unbox_usize(v_sz_2556_);
lean_dec(v_sz_2556_);
v_i_boxed_2569_ = lean_unbox_usize(v_i_2557_);
lean_dec(v_i_2557_);
v_res_2570_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__1(v_init_2554_, v_as_2555_, v_sz_boxed_2568_, v_i_boxed_2569_, v_b_2558_, v___y_2559_, v___y_2560_, v___y_2561_, v___y_2562_, v___y_2563_, v___y_2564_, v___y_2565_, v___y_2566_);
lean_dec(v___y_2566_);
lean_dec_ref(v___y_2565_);
lean_dec(v___y_2564_);
lean_dec_ref(v___y_2563_);
lean_dec(v___y_2562_);
lean_dec_ref(v___y_2561_);
lean_dec(v___y_2560_);
lean_dec_ref(v___y_2559_);
lean_dec_ref(v_as_2555_);
lean_dec(v_init_2554_);
return v_res_2570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0___boxed(lean_object* v_init_2571_, lean_object* v_n_2572_, lean_object* v_b_2573_, lean_object* v___y_2574_, lean_object* v___y_2575_, lean_object* v___y_2576_, lean_object* v___y_2577_, lean_object* v___y_2578_, lean_object* v___y_2579_, lean_object* v___y_2580_, lean_object* v___y_2581_, lean_object* v___y_2582_){
_start:
{
lean_object* v_res_2583_; 
v_res_2583_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0(v_init_2571_, v_n_2572_, v_b_2573_, v___y_2574_, v___y_2575_, v___y_2576_, v___y_2577_, v___y_2578_, v___y_2579_, v___y_2580_, v___y_2581_);
lean_dec(v___y_2581_);
lean_dec_ref(v___y_2580_);
lean_dec(v___y_2579_);
lean_dec_ref(v___y_2578_);
lean_dec(v___y_2577_);
lean_dec_ref(v___y_2576_);
lean_dec(v___y_2575_);
lean_dec_ref(v___y_2574_);
lean_dec_ref(v_n_2572_);
lean_dec(v_init_2571_);
return v_res_2583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0(lean_object* v_t_2584_, lean_object* v_init_2585_, lean_object* v___y_2586_, lean_object* v___y_2587_, lean_object* v___y_2588_, lean_object* v___y_2589_, lean_object* v___y_2590_, lean_object* v___y_2591_, lean_object* v___y_2592_, lean_object* v___y_2593_){
_start:
{
lean_object* v_root_2595_; lean_object* v_tail_2596_; lean_object* v___x_2597_; 
v_root_2595_ = lean_ctor_get(v_t_2584_, 0);
v_tail_2596_ = lean_ctor_get(v_t_2584_, 1);
lean_inc(v_init_2585_);
v___x_2597_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0(v_init_2585_, v_root_2595_, v_init_2585_, v___y_2586_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_);
lean_dec(v_init_2585_);
if (lean_obj_tag(v___x_2597_) == 0)
{
lean_object* v_a_2598_; lean_object* v___x_2600_; uint8_t v_isShared_2601_; uint8_t v_isSharedCheck_2634_; 
v_a_2598_ = lean_ctor_get(v___x_2597_, 0);
v_isSharedCheck_2634_ = !lean_is_exclusive(v___x_2597_);
if (v_isSharedCheck_2634_ == 0)
{
v___x_2600_ = v___x_2597_;
v_isShared_2601_ = v_isSharedCheck_2634_;
goto v_resetjp_2599_;
}
else
{
lean_inc(v_a_2598_);
lean_dec(v___x_2597_);
v___x_2600_ = lean_box(0);
v_isShared_2601_ = v_isSharedCheck_2634_;
goto v_resetjp_2599_;
}
v_resetjp_2599_:
{
if (lean_obj_tag(v_a_2598_) == 0)
{
lean_object* v_a_2602_; lean_object* v___x_2604_; 
v_a_2602_ = lean_ctor_get(v_a_2598_, 0);
lean_inc(v_a_2602_);
lean_dec_ref_known(v_a_2598_, 1);
if (v_isShared_2601_ == 0)
{
lean_ctor_set(v___x_2600_, 0, v_a_2602_);
v___x_2604_ = v___x_2600_;
goto v_reusejp_2603_;
}
else
{
lean_object* v_reuseFailAlloc_2605_; 
v_reuseFailAlloc_2605_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2605_, 0, v_a_2602_);
v___x_2604_ = v_reuseFailAlloc_2605_;
goto v_reusejp_2603_;
}
v_reusejp_2603_:
{
return v___x_2604_;
}
}
else
{
lean_object* v_a_2606_; lean_object* v___x_2607_; lean_object* v___x_2608_; size_t v_sz_2609_; size_t v___x_2610_; lean_object* v___x_2611_; 
lean_del_object(v___x_2600_);
v_a_2606_ = lean_ctor_get(v_a_2598_, 0);
lean_inc(v_a_2606_);
lean_dec_ref_known(v_a_2598_, 1);
v___x_2607_ = lean_box(0);
v___x_2608_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2608_, 0, v___x_2607_);
lean_ctor_set(v___x_2608_, 1, v_a_2606_);
v_sz_2609_ = lean_array_size(v_tail_2596_);
v___x_2610_ = ((size_t)0ULL);
v___x_2611_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1(v_tail_2596_, v_sz_2609_, v___x_2610_, v___x_2608_, v___y_2586_, v___y_2587_, v___y_2588_, v___y_2589_, v___y_2590_, v___y_2591_, v___y_2592_, v___y_2593_);
if (lean_obj_tag(v___x_2611_) == 0)
{
lean_object* v_a_2612_; lean_object* v___x_2614_; uint8_t v_isShared_2615_; uint8_t v_isSharedCheck_2625_; 
v_a_2612_ = lean_ctor_get(v___x_2611_, 0);
v_isSharedCheck_2625_ = !lean_is_exclusive(v___x_2611_);
if (v_isSharedCheck_2625_ == 0)
{
v___x_2614_ = v___x_2611_;
v_isShared_2615_ = v_isSharedCheck_2625_;
goto v_resetjp_2613_;
}
else
{
lean_inc(v_a_2612_);
lean_dec(v___x_2611_);
v___x_2614_ = lean_box(0);
v_isShared_2615_ = v_isSharedCheck_2625_;
goto v_resetjp_2613_;
}
v_resetjp_2613_:
{
lean_object* v_fst_2616_; 
v_fst_2616_ = lean_ctor_get(v_a_2612_, 0);
if (lean_obj_tag(v_fst_2616_) == 0)
{
lean_object* v_snd_2617_; lean_object* v___x_2619_; 
v_snd_2617_ = lean_ctor_get(v_a_2612_, 1);
lean_inc(v_snd_2617_);
lean_dec(v_a_2612_);
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 0, v_snd_2617_);
v___x_2619_ = v___x_2614_;
goto v_reusejp_2618_;
}
else
{
lean_object* v_reuseFailAlloc_2620_; 
v_reuseFailAlloc_2620_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2620_, 0, v_snd_2617_);
v___x_2619_ = v_reuseFailAlloc_2620_;
goto v_reusejp_2618_;
}
v_reusejp_2618_:
{
return v___x_2619_;
}
}
else
{
lean_object* v_val_2621_; lean_object* v___x_2623_; 
lean_inc_ref(v_fst_2616_);
lean_dec(v_a_2612_);
v_val_2621_ = lean_ctor_get(v_fst_2616_, 0);
lean_inc(v_val_2621_);
lean_dec_ref_known(v_fst_2616_, 1);
if (v_isShared_2615_ == 0)
{
lean_ctor_set(v___x_2614_, 0, v_val_2621_);
v___x_2623_ = v___x_2614_;
goto v_reusejp_2622_;
}
else
{
lean_object* v_reuseFailAlloc_2624_; 
v_reuseFailAlloc_2624_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2624_, 0, v_val_2621_);
v___x_2623_ = v_reuseFailAlloc_2624_;
goto v_reusejp_2622_;
}
v_reusejp_2622_:
{
return v___x_2623_;
}
}
}
}
else
{
lean_object* v_a_2626_; lean_object* v___x_2628_; uint8_t v_isShared_2629_; uint8_t v_isSharedCheck_2633_; 
v_a_2626_ = lean_ctor_get(v___x_2611_, 0);
v_isSharedCheck_2633_ = !lean_is_exclusive(v___x_2611_);
if (v_isSharedCheck_2633_ == 0)
{
v___x_2628_ = v___x_2611_;
v_isShared_2629_ = v_isSharedCheck_2633_;
goto v_resetjp_2627_;
}
else
{
lean_inc(v_a_2626_);
lean_dec(v___x_2611_);
v___x_2628_ = lean_box(0);
v_isShared_2629_ = v_isSharedCheck_2633_;
goto v_resetjp_2627_;
}
v_resetjp_2627_:
{
lean_object* v___x_2631_; 
if (v_isShared_2629_ == 0)
{
v___x_2631_ = v___x_2628_;
goto v_reusejp_2630_;
}
else
{
lean_object* v_reuseFailAlloc_2632_; 
v_reuseFailAlloc_2632_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2632_, 0, v_a_2626_);
v___x_2631_ = v_reuseFailAlloc_2632_;
goto v_reusejp_2630_;
}
v_reusejp_2630_:
{
return v___x_2631_;
}
}
}
}
}
}
else
{
lean_object* v_a_2635_; lean_object* v___x_2637_; uint8_t v_isShared_2638_; uint8_t v_isSharedCheck_2642_; 
v_a_2635_ = lean_ctor_get(v___x_2597_, 0);
v_isSharedCheck_2642_ = !lean_is_exclusive(v___x_2597_);
if (v_isSharedCheck_2642_ == 0)
{
v___x_2637_ = v___x_2597_;
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
else
{
lean_inc(v_a_2635_);
lean_dec(v___x_2597_);
v___x_2637_ = lean_box(0);
v_isShared_2638_ = v_isSharedCheck_2642_;
goto v_resetjp_2636_;
}
v_resetjp_2636_:
{
lean_object* v___x_2640_; 
if (v_isShared_2638_ == 0)
{
v___x_2640_ = v___x_2637_;
goto v_reusejp_2639_;
}
else
{
lean_object* v_reuseFailAlloc_2641_; 
v_reuseFailAlloc_2641_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2641_, 0, v_a_2635_);
v___x_2640_ = v_reuseFailAlloc_2641_;
goto v_reusejp_2639_;
}
v_reusejp_2639_:
{
return v___x_2640_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0___boxed(lean_object* v_t_2643_, lean_object* v_init_2644_, lean_object* v___y_2645_, lean_object* v___y_2646_, lean_object* v___y_2647_, lean_object* v___y_2648_, lean_object* v___y_2649_, lean_object* v___y_2650_, lean_object* v___y_2651_, lean_object* v___y_2652_, lean_object* v___y_2653_){
_start:
{
lean_object* v_res_2654_; 
v_res_2654_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0(v_t_2643_, v_init_2644_, v___y_2645_, v___y_2646_, v___y_2647_, v___y_2648_, v___y_2649_, v___y_2650_, v___y_2651_, v___y_2652_);
lean_dec(v___y_2652_);
lean_dec_ref(v___y_2651_);
lean_dec(v___y_2650_);
lean_dec_ref(v___y_2649_);
lean_dec(v___y_2648_);
lean_dec_ref(v___y_2647_);
lean_dec(v___y_2646_);
lean_dec_ref(v___y_2645_);
lean_dec_ref(v_t_2643_);
return v_res_2654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0(lean_object* v_fvars_2655_, lean_object* v___y_2656_, lean_object* v___y_2657_, lean_object* v___y_2658_, lean_object* v___y_2659_, lean_object* v___y_2660_, lean_object* v___y_2661_, lean_object* v___y_2662_, lean_object* v___y_2663_){
_start:
{
lean_object* v_lctx_2665_; lean_object* v_decls_2666_; lean_object* v___x_2667_; 
v_lctx_2665_ = lean_ctor_get(v___y_2660_, 2);
v_decls_2666_ = lean_ctor_get(v_lctx_2665_, 1);
v___x_2667_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0(v_decls_2666_, v_fvars_2655_, v___y_2656_, v___y_2657_, v___y_2658_, v___y_2659_, v___y_2660_, v___y_2661_, v___y_2662_, v___y_2663_);
if (lean_obj_tag(v___x_2667_) == 0)
{
lean_object* v_a_2668_; lean_object* v___x_2669_; lean_object* v___x_2670_; 
v_a_2668_ = lean_ctor_get(v___x_2667_, 0);
lean_inc(v_a_2668_);
lean_dec_ref_known(v___x_2667_, 1);
v___x_2669_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotAux___boxed), 7, 1);
lean_closure_set(v___x_2669_, 0, v_a_2668_);
v___x_2670_ = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(v___x_2669_, v___y_2656_, v___y_2657_, v___y_2658_, v___y_2659_, v___y_2660_, v___y_2661_, v___y_2662_, v___y_2663_);
return v___x_2670_;
}
else
{
lean_object* v_a_2671_; lean_object* v___x_2673_; uint8_t v_isShared_2674_; uint8_t v_isSharedCheck_2678_; 
v_a_2671_ = lean_ctor_get(v___x_2667_, 0);
v_isSharedCheck_2678_ = !lean_is_exclusive(v___x_2667_);
if (v_isSharedCheck_2678_ == 0)
{
v___x_2673_ = v___x_2667_;
v_isShared_2674_ = v_isSharedCheck_2678_;
goto v_resetjp_2672_;
}
else
{
lean_inc(v_a_2671_);
lean_dec(v___x_2667_);
v___x_2673_ = lean_box(0);
v_isShared_2674_ = v_isSharedCheck_2678_;
goto v_resetjp_2672_;
}
v_resetjp_2672_:
{
lean_object* v___x_2676_; 
if (v_isShared_2674_ == 0)
{
v___x_2676_ = v___x_2673_;
goto v_reusejp_2675_;
}
else
{
lean_object* v_reuseFailAlloc_2677_; 
v_reuseFailAlloc_2677_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2677_, 0, v_a_2671_);
v___x_2676_ = v_reuseFailAlloc_2677_;
goto v_reusejp_2675_;
}
v_reusejp_2675_:
{
return v___x_2676_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0___boxed(lean_object* v_fvars_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_, lean_object* v___y_2683_, lean_object* v___y_2684_, lean_object* v___y_2685_, lean_object* v___y_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_){
_start:
{
lean_object* v_res_2689_; 
v_res_2689_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNot___lam__0(v_fvars_2679_, v___y_2680_, v___y_2681_, v___y_2682_, v___y_2683_, v___y_2684_, v___y_2685_, v___y_2686_, v___y_2687_);
lean_dec(v___y_2687_);
lean_dec_ref(v___y_2686_);
lean_dec(v___y_2685_);
lean_dec_ref(v___y_2684_);
lean_dec(v___y_2683_);
lean_dec_ref(v___y_2682_);
lean_dec(v___y_2681_);
lean_dec_ref(v___y_2680_);
return v_res_2689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot(lean_object* v_a_2692_, lean_object* v_a_2693_, lean_object* v_a_2694_, lean_object* v_a_2695_, lean_object* v_a_2696_, lean_object* v_a_2697_, lean_object* v_a_2698_, lean_object* v_a_2699_){
_start:
{
lean_object* v___f_2701_; lean_object* v___x_2702_; 
v___f_2701_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNot___closed__0));
v___x_2702_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_2701_, v_a_2692_, v_a_2693_, v_a_2694_, v_a_2695_, v_a_2696_, v_a_2697_, v_a_2698_, v_a_2699_);
return v___x_2702_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_distribNot___boxed(lean_object* v_a_2703_, lean_object* v_a_2704_, lean_object* v_a_2705_, lean_object* v_a_2706_, lean_object* v_a_2707_, lean_object* v_a_2708_, lean_object* v_a_2709_, lean_object* v_a_2710_, lean_object* v_a_2711_){
_start:
{
lean_object* v_res_2712_; 
v_res_2712_ = lp_mathlib_Mathlib_Tactic_Tauto_distribNot(v_a_2703_, v_a_2704_, v_a_2705_, v_a_2706_, v_a_2707_, v_a_2708_, v_a_2709_, v_a_2710_);
lean_dec(v_a_2710_);
lean_dec_ref(v_a_2709_);
lean_dec(v_a_2708_);
lean_dec_ref(v_a_2707_);
lean_dec(v_a_2706_);
lean_dec_ref(v_a_2705_);
lean_dec(v_a_2704_);
lean_dec_ref(v_a_2703_);
return v_res_2712_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4(lean_object* v_as_2713_, size_t v_sz_2714_, size_t v_i_2715_, lean_object* v_b_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_, lean_object* v___y_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_){
_start:
{
lean_object* v___x_2726_; 
v___x_2726_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___redArg(v_as_2713_, v_sz_2714_, v_i_2715_, v_b_2716_);
return v___x_2726_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4___boxed(lean_object* v_as_2727_, lean_object* v_sz_2728_, lean_object* v_i_2729_, lean_object* v_b_2730_, lean_object* v___y_2731_, lean_object* v___y_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_, lean_object* v___y_2735_, lean_object* v___y_2736_, lean_object* v___y_2737_, lean_object* v___y_2738_, lean_object* v___y_2739_){
_start:
{
size_t v_sz_boxed_2740_; size_t v_i_boxed_2741_; lean_object* v_res_2742_; 
v_sz_boxed_2740_ = lean_unbox_usize(v_sz_2728_);
lean_dec(v_sz_2728_);
v_i_boxed_2741_ = lean_unbox_usize(v_i_2729_);
lean_dec(v_i_2729_);
v_res_2742_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__1_spec__4(v_as_2727_, v_sz_boxed_2740_, v_i_boxed_2741_, v_b_2730_, v___y_2731_, v___y_2732_, v___y_2733_, v___y_2734_, v___y_2735_, v___y_2736_, v___y_2737_, v___y_2738_);
lean_dec(v___y_2738_);
lean_dec_ref(v___y_2737_);
lean_dec(v___y_2736_);
lean_dec_ref(v___y_2735_);
lean_dec(v___y_2734_);
lean_dec_ref(v___y_2733_);
lean_dec(v___y_2732_);
lean_dec_ref(v___y_2731_);
lean_dec_ref(v_as_2727_);
return v_res_2742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3(lean_object* v_as_2743_, size_t v_sz_2744_, size_t v_i_2745_, lean_object* v_b_2746_, lean_object* v___y_2747_, lean_object* v___y_2748_, lean_object* v___y_2749_, lean_object* v___y_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_){
_start:
{
lean_object* v___x_2756_; 
v___x_2756_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___redArg(v_as_2743_, v_sz_2744_, v_i_2745_, v_b_2746_);
return v___x_2756_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_as_2757_, lean_object* v_sz_2758_, lean_object* v_i_2759_, lean_object* v_b_2760_, lean_object* v___y_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_, lean_object* v___y_2767_, lean_object* v___y_2768_, lean_object* v___y_2769_){
_start:
{
size_t v_sz_boxed_2770_; size_t v_i_boxed_2771_; lean_object* v_res_2772_; 
v_sz_boxed_2770_ = lean_unbox_usize(v_sz_2758_);
lean_dec(v_sz_2758_);
v_i_boxed_2771_ = lean_unbox_usize(v_i_2759_);
lean_dec(v_i_2759_);
v_res_2772_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic_Tauto_distribNot_spec__0_spec__0_spec__2_spec__3(v_as_2757_, v_sz_boxed_2770_, v_i_boxed_2771_, v_b_2760_, v___y_2761_, v___y_2762_, v___y_2763_, v___y_2764_, v___y_2765_, v___y_2766_, v___y_2767_, v___y_2768_);
lean_dec(v___y_2768_);
lean_dec_ref(v___y_2767_);
lean_dec(v___y_2766_);
lean_dec_ref(v___y_2765_);
lean_dec(v___y_2764_);
lean_dec_ref(v___y_2763_);
lean_dec(v___y_2762_);
lean_dec_ref(v___y_2761_);
lean_dec_ref(v_as_2757_);
return v_res_2772_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2773_; lean_object* v___x_2774_; lean_object* v___x_2775_; 
v___x_2773_ = lean_box(0);
v___x_2774_ = l_Lean_Elab_ConfigEval_unsupportedExprExceptionId;
v___x_2775_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2775_, 0, v___x_2774_);
lean_ctor_set(v___x_2775_, 1, v___x_2773_);
return v___x_2775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg(){
_start:
{
lean_object* v___x_2777_; lean_object* v___x_2778_; 
v___x_2777_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___closed__0);
v___x_2778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2778_, 0, v___x_2777_);
return v___x_2778_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg___boxed(lean_object* v___y_2779_){
_start:
{
lean_object* v_res_2780_; 
v_res_2780_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg();
return v_res_2780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0(lean_object* v_00_u03b1_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_){
_start:
{
lean_object* v___x_2787_; 
v___x_2787_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_2787_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___boxed(lean_object* v_00_u03b1_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_){
_start:
{
lean_object* v_res_2794_; 
v_res_2794_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0(v_00_u03b1_2788_, v___y_2789_, v___y_2790_, v___y_2791_, v___y_2792_);
lean_dec(v___y_2792_);
lean_dec_ref(v___y_2791_);
lean_dec(v___y_2790_);
lean_dec_ref(v___y_2789_);
return v_res_2794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0(lean_object* v_ctor_2796_, lean_object* v_args_2797_, lean_object* v___y_2798_, lean_object* v___y_2799_, lean_object* v___y_2800_, lean_object* v___y_2801_){
_start:
{
lean_object* v___x_2806_; uint8_t v___x_2807_; 
v___x_2806_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___closed__0));
v___x_2807_ = lean_string_dec_eq(v_ctor_2796_, v___x_2806_);
if (v___x_2807_ == 0)
{
lean_object* v___x_2808_; 
v___x_2808_ = lp_mathlib_Lean_Elab_ConfigEval_throwUnsupportedExpr___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr_spec__0___redArg();
return v___x_2808_;
}
else
{
lean_object* v___x_2809_; lean_object* v___x_2810_; uint8_t v___x_2811_; 
v___x_2809_ = lean_array_get_size(v_args_2797_);
v___x_2810_ = lean_unsigned_to_nat(0u);
v___x_2811_ = lean_nat_dec_eq(v___x_2809_, v___x_2810_);
if (v___x_2811_ == 0)
{
lean_object* v___x_2812_; lean_object* v___x_2813_; lean_object* v_a_2814_; lean_object* v___x_2816_; uint8_t v_isShared_2817_; uint8_t v_isSharedCheck_2821_; 
v___x_2812_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__47);
v___x_2813_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3___redArg(v___x_2812_, v___y_2798_, v___y_2799_, v___y_2800_, v___y_2801_);
v_a_2814_ = lean_ctor_get(v___x_2813_, 0);
v_isSharedCheck_2821_ = !lean_is_exclusive(v___x_2813_);
if (v_isSharedCheck_2821_ == 0)
{
v___x_2816_ = v___x_2813_;
v_isShared_2817_ = v_isSharedCheck_2821_;
goto v_resetjp_2815_;
}
else
{
lean_inc(v_a_2814_);
lean_dec(v___x_2813_);
v___x_2816_ = lean_box(0);
v_isShared_2817_ = v_isSharedCheck_2821_;
goto v_resetjp_2815_;
}
v_resetjp_2815_:
{
lean_object* v___x_2819_; 
if (v_isShared_2817_ == 0)
{
v___x_2819_ = v___x_2816_;
goto v_reusejp_2818_;
}
else
{
lean_object* v_reuseFailAlloc_2820_; 
v_reuseFailAlloc_2820_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2820_, 0, v_a_2814_);
v___x_2819_ = v_reuseFailAlloc_2820_;
goto v_reusejp_2818_;
}
v_reusejp_2818_:
{
return v___x_2819_;
}
}
}
else
{
goto v___jp_2803_;
}
}
v___jp_2803_:
{
lean_object* v___x_2804_; lean_object* v___x_2805_; 
v___x_2804_ = lean_box(0);
v___x_2805_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2805_, 0, v___x_2804_);
return v___x_2805_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0___boxed(lean_object* v_ctor_2822_, lean_object* v_args_2823_, lean_object* v___y_2824_, lean_object* v___y_2825_, lean_object* v___y_2826_, lean_object* v___y_2827_, lean_object* v___y_2828_){
_start:
{
lean_object* v_res_2829_; 
v_res_2829_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___lam__0(v_ctor_2822_, v_args_2823_, v___y_2824_, v___y_2825_, v___y_2826_, v___y_2827_);
lean_dec(v___y_2827_);
lean_dec_ref(v___y_2826_);
lean_dec(v___y_2825_);
lean_dec_ref(v___y_2824_);
lean_dec_ref(v_args_2823_);
lean_dec_ref(v_ctor_2822_);
return v_res_2829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr(lean_object* v_a_2837_, lean_object* v_a_2838_, lean_object* v_a_2839_, lean_object* v_a_2840_, lean_object* v_a_2841_){
_start:
{
lean_object* v___f_2843_; lean_object* v___x_2844_; lean_object* v___x_2845_; 
v___f_2843_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__0));
v___x_2844_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2));
v___x_2845_ = l_Lean_Elab_ConfigEval_EvalExpr_withSimpleEvalExpr___redArg(v___x_2844_, v___f_2843_, v_a_2837_, v_a_2838_, v_a_2839_, v_a_2840_, v_a_2841_);
return v___x_2845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___boxed(lean_object* v_a_2846_, lean_object* v_a_2847_, lean_object* v_a_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_){
_start:
{
lean_object* v_res_2852_; 
v_res_2852_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr(v_a_2846_, v_a_2847_, v_a_2848_, v_a_2849_, v_a_2850_);
lean_dec(v_a_2850_);
lean_dec_ref(v_a_2849_);
lean_dec(v_a_2848_);
lean_dec_ref(v_a_2847_);
return v_res_2852_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1(void){
_start:
{
lean_object* v___x_2854_; lean_object* v___x_2855_; lean_object* v___x_2856_; 
v___x_2854_ = lean_box(0);
v___x_2855_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2));
v___x_2856_ = l_Lean_Expr_const___override(v___x_2855_, v___x_2854_);
return v___x_2856_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2(void){
_start:
{
lean_object* v___x_2857_; lean_object* v___x_2858_; 
v___x_2857_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1);
v___x_2858_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2858_, 0, v___x_2857_);
return v___x_2858_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3(void){
_start:
{
lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; 
v___x_2859_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2);
v___x_2860_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__0));
v___x_2861_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2861_, 0, v___x_2860_);
lean_ctor_set(v___x_2861_, 1, v___x_2859_);
return v___x_2861_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig(void){
_start:
{
lean_object* v___x_2862_; 
v___x_2862_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__3);
return v___x_2862_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0(void){
_start:
{
lean_object* v___x_2863_; lean_object* v___x_2864_; lean_object* v___x_2865_; 
v___x_2863_ = lean_box(0);
v___x_2864_ = l_Lean_Elab_abortTermExceptionId;
v___x_2865_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2865_, 0, v___x_2864_);
lean_ctor_set(v___x_2865_, 1, v___x_2863_);
return v___x_2865_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg(){
_start:
{
lean_object* v___x_2867_; lean_object* v___x_2868_; 
v___x_2867_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0, &lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___closed__0);
v___x_2868_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2868_, 0, v___x_2867_);
return v___x_2868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg___boxed(lean_object* v___y_2869_){
_start:
{
lean_object* v_res_2870_; 
v_res_2870_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
return v_res_2870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg(lean_object* v_e_2871_, lean_object* v___y_2872_){
_start:
{
uint8_t v___x_2874_; 
v___x_2874_ = l_Lean_Expr_hasMVar(v_e_2871_);
if (v___x_2874_ == 0)
{
lean_object* v___x_2875_; 
v___x_2875_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2875_, 0, v_e_2871_);
return v___x_2875_;
}
else
{
lean_object* v___x_2876_; lean_object* v_mctx_2877_; lean_object* v___x_2878_; lean_object* v_fst_2879_; lean_object* v_snd_2880_; lean_object* v___x_2881_; lean_object* v_cache_2882_; lean_object* v_zetaDeltaFVarIds_2883_; lean_object* v_postponed_2884_; lean_object* v_diag_2885_; lean_object* v___x_2887_; uint8_t v_isShared_2888_; uint8_t v_isSharedCheck_2894_; 
v___x_2876_ = lean_st_ref_get(v___y_2872_);
v_mctx_2877_ = lean_ctor_get(v___x_2876_, 0);
lean_inc_ref(v_mctx_2877_);
lean_dec(v___x_2876_);
v___x_2878_ = l_Lean_instantiateMVarsCore(v_mctx_2877_, v_e_2871_);
v_fst_2879_ = lean_ctor_get(v___x_2878_, 0);
lean_inc(v_fst_2879_);
v_snd_2880_ = lean_ctor_get(v___x_2878_, 1);
lean_inc(v_snd_2880_);
lean_dec_ref(v___x_2878_);
v___x_2881_ = lean_st_ref_take(v___y_2872_);
v_cache_2882_ = lean_ctor_get(v___x_2881_, 1);
v_zetaDeltaFVarIds_2883_ = lean_ctor_get(v___x_2881_, 2);
v_postponed_2884_ = lean_ctor_get(v___x_2881_, 3);
v_diag_2885_ = lean_ctor_get(v___x_2881_, 4);
v_isSharedCheck_2894_ = !lean_is_exclusive(v___x_2881_);
if (v_isSharedCheck_2894_ == 0)
{
lean_object* v_unused_2895_; 
v_unused_2895_ = lean_ctor_get(v___x_2881_, 0);
lean_dec(v_unused_2895_);
v___x_2887_ = v___x_2881_;
v_isShared_2888_ = v_isSharedCheck_2894_;
goto v_resetjp_2886_;
}
else
{
lean_inc(v_diag_2885_);
lean_inc(v_postponed_2884_);
lean_inc(v_zetaDeltaFVarIds_2883_);
lean_inc(v_cache_2882_);
lean_dec(v___x_2881_);
v___x_2887_ = lean_box(0);
v_isShared_2888_ = v_isSharedCheck_2894_;
goto v_resetjp_2886_;
}
v_resetjp_2886_:
{
lean_object* v___x_2890_; 
if (v_isShared_2888_ == 0)
{
lean_ctor_set(v___x_2887_, 0, v_snd_2880_);
v___x_2890_ = v___x_2887_;
goto v_reusejp_2889_;
}
else
{
lean_object* v_reuseFailAlloc_2893_; 
v_reuseFailAlloc_2893_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_2893_, 0, v_snd_2880_);
lean_ctor_set(v_reuseFailAlloc_2893_, 1, v_cache_2882_);
lean_ctor_set(v_reuseFailAlloc_2893_, 2, v_zetaDeltaFVarIds_2883_);
lean_ctor_set(v_reuseFailAlloc_2893_, 3, v_postponed_2884_);
lean_ctor_set(v_reuseFailAlloc_2893_, 4, v_diag_2885_);
v___x_2890_ = v_reuseFailAlloc_2893_;
goto v_reusejp_2889_;
}
v_reusejp_2889_:
{
lean_object* v___x_2891_; lean_object* v___x_2892_; 
v___x_2891_ = lean_st_ref_set(v___y_2872_, v___x_2890_);
v___x_2892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2892_, 0, v_fst_2879_);
return v___x_2892_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg___boxed(lean_object* v_e_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_){
_start:
{
lean_object* v_res_2899_; 
v_res_2899_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_2896_, v___y_2897_);
lean_dec(v___y_2897_);
return v_res_2899_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0(void){
_start:
{
lean_object* v___x_2900_; lean_object* v___x_2901_; 
v___x_2900_ = lean_box(1);
v___x_2901_ = l_Lean_MessageData_ofFormat(v___x_2900_);
return v___x_2901_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3(void){
_start:
{
lean_object* v___x_2905_; lean_object* v___x_2906_; 
v___x_2905_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__2));
v___x_2906_ = l_Lean_MessageData_ofFormat(v___x_2905_);
return v___x_2906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(lean_object* v_x_2907_, lean_object* v_x_2908_){
_start:
{
if (lean_obj_tag(v_x_2908_) == 0)
{
return v_x_2907_;
}
else
{
lean_object* v_head_2909_; lean_object* v_tail_2910_; lean_object* v___x_2912_; uint8_t v_isShared_2913_; uint8_t v_isSharedCheck_2932_; 
v_head_2909_ = lean_ctor_get(v_x_2908_, 0);
v_tail_2910_ = lean_ctor_get(v_x_2908_, 1);
v_isSharedCheck_2932_ = !lean_is_exclusive(v_x_2908_);
if (v_isSharedCheck_2932_ == 0)
{
v___x_2912_ = v_x_2908_;
v_isShared_2913_ = v_isSharedCheck_2932_;
goto v_resetjp_2911_;
}
else
{
lean_inc(v_tail_2910_);
lean_inc(v_head_2909_);
lean_dec(v_x_2908_);
v___x_2912_ = lean_box(0);
v_isShared_2913_ = v_isSharedCheck_2932_;
goto v_resetjp_2911_;
}
v_resetjp_2911_:
{
lean_object* v_before_2914_; lean_object* v___x_2916_; uint8_t v_isShared_2917_; uint8_t v_isSharedCheck_2930_; 
v_before_2914_ = lean_ctor_get(v_head_2909_, 0);
v_isSharedCheck_2930_ = !lean_is_exclusive(v_head_2909_);
if (v_isSharedCheck_2930_ == 0)
{
lean_object* v_unused_2931_; 
v_unused_2931_ = lean_ctor_get(v_head_2909_, 1);
lean_dec(v_unused_2931_);
v___x_2916_ = v_head_2909_;
v_isShared_2917_ = v_isSharedCheck_2930_;
goto v_resetjp_2915_;
}
else
{
lean_inc(v_before_2914_);
lean_dec(v_head_2909_);
v___x_2916_ = lean_box(0);
v_isShared_2917_ = v_isSharedCheck_2930_;
goto v_resetjp_2915_;
}
v_resetjp_2915_:
{
lean_object* v___x_2918_; lean_object* v___x_2920_; 
v___x_2918_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_2917_ == 0)
{
lean_ctor_set_tag(v___x_2916_, 7);
lean_ctor_set(v___x_2916_, 1, v___x_2918_);
lean_ctor_set(v___x_2916_, 0, v_x_2907_);
v___x_2920_ = v___x_2916_;
goto v_reusejp_2919_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v_x_2907_);
lean_ctor_set(v_reuseFailAlloc_2929_, 1, v___x_2918_);
v___x_2920_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2919_;
}
v_reusejp_2919_:
{
lean_object* v___x_2921_; lean_object* v___x_2923_; 
v___x_2921_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__3);
if (v_isShared_2913_ == 0)
{
lean_ctor_set_tag(v___x_2912_, 7);
lean_ctor_set(v___x_2912_, 1, v___x_2921_);
lean_ctor_set(v___x_2912_, 0, v___x_2920_);
v___x_2923_ = v___x_2912_;
goto v_reusejp_2922_;
}
else
{
lean_object* v_reuseFailAlloc_2928_; 
v_reuseFailAlloc_2928_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2928_, 0, v___x_2920_);
lean_ctor_set(v_reuseFailAlloc_2928_, 1, v___x_2921_);
v___x_2923_ = v_reuseFailAlloc_2928_;
goto v_reusejp_2922_;
}
v_reusejp_2922_:
{
lean_object* v___x_2924_; lean_object* v___x_2925_; lean_object* v___x_2926_; 
v___x_2924_ = l_Lean_MessageData_ofSyntax(v_before_2914_);
v___x_2925_ = l_Lean_indentD(v___x_2924_);
v___x_2926_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2926_, 0, v___x_2923_);
lean_ctor_set(v___x_2926_, 1, v___x_2925_);
v_x_2907_ = v___x_2926_;
v_x_2908_ = v_tail_2910_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(lean_object* v_opts_2933_, lean_object* v_opt_2934_){
_start:
{
lean_object* v_name_2935_; lean_object* v_defValue_2936_; lean_object* v_map_2937_; lean_object* v___x_2938_; 
v_name_2935_ = lean_ctor_get(v_opt_2934_, 0);
v_defValue_2936_ = lean_ctor_get(v_opt_2934_, 1);
v_map_2937_ = lean_ctor_get(v_opts_2933_, 0);
v___x_2938_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2937_, v_name_2935_);
if (lean_obj_tag(v___x_2938_) == 0)
{
uint8_t v___x_2939_; 
v___x_2939_ = lean_unbox(v_defValue_2936_);
return v___x_2939_;
}
else
{
lean_object* v_val_2940_; 
v_val_2940_ = lean_ctor_get(v___x_2938_, 0);
lean_inc(v_val_2940_);
lean_dec_ref_known(v___x_2938_, 1);
if (lean_obj_tag(v_val_2940_) == 1)
{
uint8_t v_v_2941_; 
v_v_2941_ = lean_ctor_get_uint8(v_val_2940_, 0);
lean_dec_ref_known(v_val_2940_, 0);
return v_v_2941_;
}
else
{
uint8_t v___x_2942_; 
lean_dec(v_val_2940_);
v___x_2942_ = lean_unbox(v_defValue_2936_);
return v___x_2942_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4___boxed(lean_object* v_opts_2943_, lean_object* v_opt_2944_){
_start:
{
uint8_t v_res_2945_; lean_object* v_r_2946_; 
v_res_2945_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_opts_2943_, v_opt_2944_);
lean_dec_ref(v_opt_2944_);
lean_dec_ref(v_opts_2943_);
v_r_2946_ = lean_box(v_res_2945_);
return v_r_2946_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2(void){
_start:
{
lean_object* v___x_2950_; lean_object* v___x_2951_; 
v___x_2950_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__1));
v___x_2951_ = l_Lean_MessageData_ofFormat(v___x_2950_);
return v___x_2951_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(lean_object* v_msgData_2952_, lean_object* v_macroStack_2953_, lean_object* v___y_2954_){
_start:
{
lean_object* v_options_2956_; lean_object* v___x_2957_; uint8_t v___x_2958_; 
v_options_2956_ = lean_ctor_get(v___y_2954_, 2);
v___x_2957_ = l_Lean_Elab_pp_macroStack;
v___x_2958_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__4(v_options_2956_, v___x_2957_);
if (v___x_2958_ == 0)
{
lean_object* v___x_2959_; 
lean_dec(v_macroStack_2953_);
v___x_2959_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2959_, 0, v_msgData_2952_);
return v___x_2959_;
}
else
{
if (lean_obj_tag(v_macroStack_2953_) == 0)
{
lean_object* v___x_2960_; 
v___x_2960_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2960_, 0, v_msgData_2952_);
return v___x_2960_;
}
else
{
lean_object* v_head_2961_; lean_object* v_after_2962_; lean_object* v___x_2964_; uint8_t v_isShared_2965_; uint8_t v_isSharedCheck_2977_; 
v_head_2961_ = lean_ctor_get(v_macroStack_2953_, 0);
lean_inc(v_head_2961_);
v_after_2962_ = lean_ctor_get(v_head_2961_, 1);
v_isSharedCheck_2977_ = !lean_is_exclusive(v_head_2961_);
if (v_isSharedCheck_2977_ == 0)
{
lean_object* v_unused_2978_; 
v_unused_2978_ = lean_ctor_get(v_head_2961_, 0);
lean_dec(v_unused_2978_);
v___x_2964_ = v_head_2961_;
v_isShared_2965_ = v_isSharedCheck_2977_;
goto v_resetjp_2963_;
}
else
{
lean_inc(v_after_2962_);
lean_dec(v_head_2961_);
v___x_2964_ = lean_box(0);
v_isShared_2965_ = v_isSharedCheck_2977_;
goto v_resetjp_2963_;
}
v_resetjp_2963_:
{
lean_object* v___x_2966_; lean_object* v___x_2968_; 
v___x_2966_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5___closed__0);
if (v_isShared_2965_ == 0)
{
lean_ctor_set_tag(v___x_2964_, 7);
lean_ctor_set(v___x_2964_, 1, v___x_2966_);
lean_ctor_set(v___x_2964_, 0, v_msgData_2952_);
v___x_2968_ = v___x_2964_;
goto v_reusejp_2967_;
}
else
{
lean_object* v_reuseFailAlloc_2976_; 
v_reuseFailAlloc_2976_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2976_, 0, v_msgData_2952_);
lean_ctor_set(v_reuseFailAlloc_2976_, 1, v___x_2966_);
v___x_2968_ = v_reuseFailAlloc_2976_;
goto v_reusejp_2967_;
}
v_reusejp_2967_:
{
lean_object* v___x_2969_; lean_object* v___x_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v_msgData_2973_; lean_object* v___x_2974_; lean_object* v___x_2975_; 
v___x_2969_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___closed__2);
v___x_2970_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2970_, 0, v___x_2968_);
lean_ctor_set(v___x_2970_, 1, v___x_2969_);
v___x_2971_ = l_Lean_MessageData_ofSyntax(v_after_2962_);
v___x_2972_ = l_Lean_indentD(v___x_2971_);
v_msgData_2973_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2973_, 0, v___x_2970_);
lean_ctor_set(v_msgData_2973_, 1, v___x_2972_);
v___x_2974_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2_spec__5(v_msgData_2973_, v_macroStack_2953_);
v___x_2975_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2975_, 0, v___x_2974_);
return v___x_2975_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg___boxed(lean_object* v_msgData_2979_, lean_object* v_macroStack_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_){
_start:
{
lean_object* v_res_2983_; 
v_res_2983_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_2979_, v_macroStack_2980_, v___y_2981_);
lean_dec_ref(v___y_2981_);
return v_res_2983_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(lean_object* v_msg_2984_, lean_object* v___y_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_){
_start:
{
lean_object* v_ref_2992_; lean_object* v___x_2993_; lean_object* v_a_2994_; lean_object* v_macroStack_2995_; lean_object* v___x_2996_; lean_object* v___x_2997_; lean_object* v_a_2998_; lean_object* v___x_3000_; uint8_t v_isShared_3001_; uint8_t v_isSharedCheck_3006_; 
v_ref_2992_ = lean_ctor_get(v___y_2989_, 5);
v___x_2993_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(v_msg_2984_, v___y_2987_, v___y_2988_, v___y_2989_, v___y_2990_);
v_a_2994_ = lean_ctor_get(v___x_2993_, 0);
lean_inc(v_a_2994_);
lean_dec_ref(v___x_2993_);
v_macroStack_2995_ = lean_ctor_get(v___y_2985_, 1);
v___x_2996_ = l_Lean_Elab_getBetterRef(v_ref_2992_, v_macroStack_2995_);
lean_inc(v_macroStack_2995_);
v___x_2997_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_a_2994_, v_macroStack_2995_, v___y_2989_);
v_a_2998_ = lean_ctor_get(v___x_2997_, 0);
v_isSharedCheck_3006_ = !lean_is_exclusive(v___x_2997_);
if (v_isSharedCheck_3006_ == 0)
{
v___x_3000_ = v___x_2997_;
v_isShared_3001_ = v_isSharedCheck_3006_;
goto v_resetjp_2999_;
}
else
{
lean_inc(v_a_2998_);
lean_dec(v___x_2997_);
v___x_3000_ = lean_box(0);
v_isShared_3001_ = v_isSharedCheck_3006_;
goto v_resetjp_2999_;
}
v_resetjp_2999_:
{
lean_object* v___x_3002_; lean_object* v___x_3004_; 
v___x_3002_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3002_, 0, v___x_2996_);
lean_ctor_set(v___x_3002_, 1, v_a_2998_);
if (v_isShared_3001_ == 0)
{
lean_ctor_set_tag(v___x_3000_, 1);
lean_ctor_set(v___x_3000_, 0, v___x_3002_);
v___x_3004_ = v___x_3000_;
goto v_reusejp_3003_;
}
else
{
lean_object* v_reuseFailAlloc_3005_; 
v_reuseFailAlloc_3005_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3005_, 0, v___x_3002_);
v___x_3004_ = v_reuseFailAlloc_3005_;
goto v_reusejp_3003_;
}
v_reusejp_3003_:
{
return v___x_3004_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg___boxed(lean_object* v_msg_3007_, lean_object* v___y_3008_, lean_object* v___y_3009_, lean_object* v___y_3010_, lean_object* v___y_3011_, lean_object* v___y_3012_, lean_object* v___y_3013_, lean_object* v___y_3014_){
_start:
{
lean_object* v_res_3015_; 
v_res_3015_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_3007_, v___y_3008_, v___y_3009_, v___y_3010_, v___y_3011_, v___y_3012_, v___y_3013_);
lean_dec(v___y_3013_);
lean_dec_ref(v___y_3012_);
lean_dec(v___y_3011_);
lean_dec_ref(v___y_3010_);
lean_dec(v___y_3009_);
lean_dec_ref(v___y_3008_);
return v_res_3015_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1(void){
_start:
{
lean_object* v___x_3017_; lean_object* v___x_3018_; 
v___x_3017_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__0));
v___x_3018_ = l_Lean_stringToMessageData(v___x_3017_);
return v___x_3018_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2(void){
_start:
{
lean_object* v___x_3019_; lean_object* v___x_3020_; 
v___x_3019_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__1);
v___x_3020_ = l_Lean_MessageData_ofExpr(v___x_3019_);
return v___x_3020_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3(void){
_start:
{
lean_object* v___x_3021_; lean_object* v___x_3022_; lean_object* v___x_3023_; 
v___x_3021_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__2);
v___x_3022_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__1);
v___x_3023_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3023_, 0, v___x_3022_);
lean_ctor_set(v___x_3023_, 1, v___x_3021_);
return v___x_3023_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5(void){
_start:
{
lean_object* v___x_3025_; lean_object* v___x_3026_; 
v___x_3025_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__4));
v___x_3026_ = l_Lean_stringToMessageData(v___x_3025_);
return v___x_3026_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6(void){
_start:
{
lean_object* v___x_3027_; lean_object* v___x_3028_; lean_object* v___x_3029_; 
v___x_3027_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__5);
v___x_3028_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__3);
v___x_3029_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3029_, 0, v___x_3028_);
lean_ctor_set(v___x_3029_, 1, v___x_3027_);
return v___x_3029_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8(void){
_start:
{
lean_object* v___x_3031_; lean_object* v___x_3032_; 
v___x_3031_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__7));
v___x_3032_ = l_Lean_stringToMessageData(v___x_3031_);
return v___x_3032_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10(void){
_start:
{
lean_object* v___x_3034_; lean_object* v___x_3035_; 
v___x_3034_ = ((lean_object*)(lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__9));
v___x_3035_ = l_Lean_stringToMessageData(v___x_3034_);
return v___x_3035_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0(lean_object* v_stx_3036_, lean_object* v_a_3037_, lean_object* v_a_3038_, lean_object* v_a_3039_, lean_object* v_a_3040_, lean_object* v_a_3041_, lean_object* v_a_3042_){
_start:
{
lean_object* v_ty_x3f_3044_; uint8_t v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; lean_object* v_fileName_3050_; lean_object* v_fileMap_3051_; lean_object* v_options_3052_; lean_object* v_currRecDepth_3053_; lean_object* v_maxRecDepth_3054_; lean_object* v_ref_3055_; lean_object* v_currNamespace_3056_; lean_object* v_openDecls_3057_; lean_object* v_initHeartbeats_3058_; lean_object* v_maxHeartbeats_3059_; lean_object* v_quotContext_3060_; lean_object* v_currMacroScope_3061_; uint8_t v_diag_3062_; lean_object* v_cancelTk_x3f_3063_; uint8_t v_suppressElabErrors_3064_; lean_object* v_inheritedTraceOptions_3065_; uint8_t v___x_3066_; lean_object* v_ref_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; 
v_ty_x3f_3044_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig___closed__2);
v___x_3045_ = 1;
v___x_3046_ = lean_box(0);
v___x_3047_ = lean_box(v___x_3045_);
v___x_3048_ = lean_box(v___x_3045_);
lean_inc(v_stx_3036_);
v___x_3049_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_elabTermEnsuringType___boxed), 12, 5);
lean_closure_set(v___x_3049_, 0, v_stx_3036_);
lean_closure_set(v___x_3049_, 1, v_ty_x3f_3044_);
lean_closure_set(v___x_3049_, 2, v___x_3047_);
lean_closure_set(v___x_3049_, 3, v___x_3048_);
lean_closure_set(v___x_3049_, 4, v___x_3046_);
v_fileName_3050_ = lean_ctor_get(v_a_3041_, 0);
v_fileMap_3051_ = lean_ctor_get(v_a_3041_, 1);
v_options_3052_ = lean_ctor_get(v_a_3041_, 2);
v_currRecDepth_3053_ = lean_ctor_get(v_a_3041_, 3);
v_maxRecDepth_3054_ = lean_ctor_get(v_a_3041_, 4);
v_ref_3055_ = lean_ctor_get(v_a_3041_, 5);
v_currNamespace_3056_ = lean_ctor_get(v_a_3041_, 6);
v_openDecls_3057_ = lean_ctor_get(v_a_3041_, 7);
v_initHeartbeats_3058_ = lean_ctor_get(v_a_3041_, 8);
v_maxHeartbeats_3059_ = lean_ctor_get(v_a_3041_, 9);
v_quotContext_3060_ = lean_ctor_get(v_a_3041_, 10);
v_currMacroScope_3061_ = lean_ctor_get(v_a_3041_, 11);
v_diag_3062_ = lean_ctor_get_uint8(v_a_3041_, sizeof(void*)*14);
v_cancelTk_x3f_3063_ = lean_ctor_get(v_a_3041_, 12);
v_suppressElabErrors_3064_ = lean_ctor_get_uint8(v_a_3041_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3065_ = lean_ctor_get(v_a_3041_, 13);
v___x_3066_ = 1;
v_ref_3067_ = l_Lean_replaceRef(v_stx_3036_, v_ref_3055_);
lean_dec(v_stx_3036_);
lean_inc_ref(v_inheritedTraceOptions_3065_);
lean_inc(v_cancelTk_x3f_3063_);
lean_inc(v_currMacroScope_3061_);
lean_inc(v_quotContext_3060_);
lean_inc(v_maxHeartbeats_3059_);
lean_inc(v_initHeartbeats_3058_);
lean_inc(v_openDecls_3057_);
lean_inc(v_currNamespace_3056_);
lean_inc(v_maxRecDepth_3054_);
lean_inc(v_currRecDepth_3053_);
lean_inc_ref(v_options_3052_);
lean_inc_ref(v_fileMap_3051_);
lean_inc_ref(v_fileName_3050_);
v___x_3068_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3068_, 0, v_fileName_3050_);
lean_ctor_set(v___x_3068_, 1, v_fileMap_3051_);
lean_ctor_set(v___x_3068_, 2, v_options_3052_);
lean_ctor_set(v___x_3068_, 3, v_currRecDepth_3053_);
lean_ctor_set(v___x_3068_, 4, v_maxRecDepth_3054_);
lean_ctor_set(v___x_3068_, 5, v_ref_3067_);
lean_ctor_set(v___x_3068_, 6, v_currNamespace_3056_);
lean_ctor_set(v___x_3068_, 7, v_openDecls_3057_);
lean_ctor_set(v___x_3068_, 8, v_initHeartbeats_3058_);
lean_ctor_set(v___x_3068_, 9, v_maxHeartbeats_3059_);
lean_ctor_set(v___x_3068_, 10, v_quotContext_3060_);
lean_ctor_set(v___x_3068_, 11, v_currMacroScope_3061_);
lean_ctor_set(v___x_3068_, 12, v_cancelTk_x3f_3063_);
lean_ctor_set(v___x_3068_, 13, v_inheritedTraceOptions_3065_);
lean_ctor_set_uint8(v___x_3068_, sizeof(void*)*14, v_diag_3062_);
lean_ctor_set_uint8(v___x_3068_, sizeof(void*)*14 + 1, v_suppressElabErrors_3064_);
v___x_3069_ = l___private_Lean_Elab_SyntheticMVars_0__Lean_Elab_Term_withSynthesizeImp(lean_box(0), v___x_3049_, v___x_3066_, v_a_3037_, v_a_3038_, v_a_3039_, v_a_3040_, v___x_3068_, v_a_3042_);
if (lean_obj_tag(v___x_3069_) == 0)
{
lean_object* v_a_3070_; lean_object* v___x_3071_; lean_object* v_a_3072_; lean_object* v___y_3074_; lean_object* v___y_3075_; lean_object* v___y_3076_; lean_object* v___y_3077_; lean_object* v___y_3078_; lean_object* v___y_3079_; lean_object* v___y_3080_; lean_object* v___y_3081_; lean_object* v___y_3082_; uint8_t v___y_3083_; lean_object* v___y_3100_; lean_object* v___y_3101_; lean_object* v___y_3102_; lean_object* v___y_3103_; lean_object* v___y_3104_; lean_object* v___y_3105_; lean_object* v___y_3112_; lean_object* v___y_3113_; lean_object* v___y_3114_; lean_object* v___y_3115_; lean_object* v___y_3116_; lean_object* v___y_3117_; lean_object* v___y_3149_; lean_object* v___y_3150_; lean_object* v___y_3151_; lean_object* v___y_3152_; lean_object* v___y_3153_; lean_object* v___y_3154_; uint8_t v___x_3167_; 
v_a_3070_ = lean_ctor_get(v___x_3069_, 0);
lean_inc(v_a_3070_);
lean_dec_ref_known(v___x_3069_, 1);
v___x_3071_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_a_3070_, v_a_3040_);
v_a_3072_ = lean_ctor_get(v___x_3071_, 0);
lean_inc(v_a_3072_);
lean_dec_ref(v___x_3071_);
v___x_3167_ = l_Lean_Expr_hasSorry(v_a_3072_);
if (v___x_3167_ == 0)
{
v___y_3112_ = v_a_3037_;
v___y_3113_ = v_a_3038_;
v___y_3114_ = v_a_3039_;
v___y_3115_ = v_a_3040_;
v___y_3116_ = v___x_3068_;
v___y_3117_ = v_a_3042_;
goto v___jp_3111_;
}
else
{
uint8_t v___x_3168_; 
v___x_3168_ = l_Lean_Expr_hasSyntheticSorry(v_a_3072_);
if (v___x_3168_ == 0)
{
v___y_3149_ = v_a_3037_;
v___y_3150_ = v_a_3038_;
v___y_3151_ = v_a_3039_;
v___y_3152_ = v_a_3040_;
v___y_3153_ = v___x_3068_;
v___y_3154_ = v_a_3042_;
goto v___jp_3148_;
}
else
{
lean_object* v___x_3169_; lean_object* v_a_3170_; lean_object* v___x_3172_; uint8_t v_isShared_3173_; uint8_t v_isSharedCheck_3177_; 
lean_dec(v_a_3072_);
lean_dec_ref_known(v___x_3068_, 14);
v___x_3169_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_3170_ = lean_ctor_get(v___x_3169_, 0);
v_isSharedCheck_3177_ = !lean_is_exclusive(v___x_3169_);
if (v_isSharedCheck_3177_ == 0)
{
v___x_3172_ = v___x_3169_;
v_isShared_3173_ = v_isSharedCheck_3177_;
goto v_resetjp_3171_;
}
else
{
lean_inc(v_a_3170_);
lean_dec(v___x_3169_);
v___x_3172_ = lean_box(0);
v_isShared_3173_ = v_isSharedCheck_3177_;
goto v_resetjp_3171_;
}
v_resetjp_3171_:
{
lean_object* v___x_3175_; 
if (v_isShared_3173_ == 0)
{
v___x_3175_ = v___x_3172_;
goto v_reusejp_3174_;
}
else
{
lean_object* v_reuseFailAlloc_3176_; 
v_reuseFailAlloc_3176_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3176_, 0, v_a_3170_);
v___x_3175_ = v_reuseFailAlloc_3176_;
goto v_reusejp_3174_;
}
v_reusejp_3174_:
{
return v___x_3175_;
}
}
}
}
v___jp_3073_:
{
if (v___y_3083_ == 0)
{
if (lean_obj_tag(v___y_3081_) == 0)
{
lean_dec_ref_known(v___y_3081_, 2);
lean_dec_ref(v___y_3077_);
lean_dec(v_a_3072_);
return v___y_3074_;
}
else
{
lean_object* v_id_3084_; lean_object* v___x_3086_; uint8_t v_isShared_3087_; uint8_t v_isSharedCheck_3097_; 
v_id_3084_ = lean_ctor_get(v___y_3081_, 0);
v_isSharedCheck_3097_ = !lean_is_exclusive(v___y_3081_);
if (v_isSharedCheck_3097_ == 0)
{
lean_object* v_unused_3098_; 
v_unused_3098_ = lean_ctor_get(v___y_3081_, 1);
lean_dec(v_unused_3098_);
v___x_3086_ = v___y_3081_;
v_isShared_3087_ = v_isSharedCheck_3097_;
goto v_resetjp_3085_;
}
else
{
lean_inc(v_id_3084_);
lean_dec(v___y_3081_);
v___x_3086_ = lean_box(0);
v_isShared_3087_ = v_isSharedCheck_3097_;
goto v_resetjp_3085_;
}
v_resetjp_3085_:
{
uint8_t v___x_3088_; 
v___x_3088_ = l_Lean_instBEqInternalExceptionId_beq(v___y_3076_, v_id_3084_);
lean_dec(v_id_3084_);
if (v___x_3088_ == 0)
{
lean_del_object(v___x_3086_);
lean_dec_ref(v___y_3077_);
lean_dec(v_a_3072_);
return v___y_3074_;
}
else
{
lean_object* v___x_3089_; lean_object* v___x_3090_; lean_object* v___x_3091_; lean_object* v___x_3093_; 
lean_dec_ref(v___y_3074_);
v___x_3089_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__6);
v___x_3090_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__8);
v___x_3091_ = l_Lean_indentExpr(v_a_3072_);
if (v_isShared_3087_ == 0)
{
lean_ctor_set_tag(v___x_3086_, 7);
lean_ctor_set(v___x_3086_, 1, v___x_3091_);
lean_ctor_set(v___x_3086_, 0, v___x_3090_);
v___x_3093_ = v___x_3086_;
goto v_reusejp_3092_;
}
else
{
lean_object* v_reuseFailAlloc_3096_; 
v_reuseFailAlloc_3096_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3096_, 0, v___x_3090_);
lean_ctor_set(v_reuseFailAlloc_3096_, 1, v___x_3091_);
v___x_3093_ = v_reuseFailAlloc_3096_;
goto v_reusejp_3092_;
}
v_reusejp_3092_:
{
lean_object* v___x_3094_; lean_object* v___x_3095_; 
v___x_3094_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3094_, 0, v___x_3093_);
lean_ctor_set(v___x_3094_, 1, v___x_3089_);
v___x_3095_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3094_, v___y_3082_, v___y_3079_, v___y_3080_, v___y_3075_, v___y_3077_, v___y_3078_);
lean_dec_ref(v___y_3077_);
return v___x_3095_;
}
}
}
}
}
else
{
lean_dec_ref(v___y_3081_);
lean_dec_ref(v___y_3077_);
lean_dec(v_a_3072_);
return v___y_3074_;
}
}
v___jp_3099_:
{
lean_object* v___x_3106_; 
lean_inc(v_a_3072_);
v___x_3106_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr(v_a_3072_, v___y_3102_, v___y_3103_, v___y_3104_, v___y_3105_);
if (lean_obj_tag(v___x_3106_) == 0)
{
lean_dec_ref(v___y_3104_);
lean_dec(v_a_3072_);
return v___x_3106_;
}
else
{
lean_object* v_a_3107_; lean_object* v___x_3108_; uint8_t v___x_3109_; 
v_a_3107_ = lean_ctor_get(v___x_3106_, 0);
lean_inc(v_a_3107_);
v___x_3108_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_3109_ = l_Lean_Exception_isInterrupt(v_a_3107_);
if (v___x_3109_ == 0)
{
uint8_t v___x_3110_; 
lean_inc(v_a_3107_);
v___x_3110_ = l_Lean_Exception_isRuntime(v_a_3107_);
v___y_3074_ = v___x_3106_;
v___y_3075_ = v___y_3103_;
v___y_3076_ = v___x_3108_;
v___y_3077_ = v___y_3104_;
v___y_3078_ = v___y_3105_;
v___y_3079_ = v___y_3101_;
v___y_3080_ = v___y_3102_;
v___y_3081_ = v_a_3107_;
v___y_3082_ = v___y_3100_;
v___y_3083_ = v___x_3110_;
goto v___jp_3073_;
}
else
{
v___y_3074_ = v___x_3106_;
v___y_3075_ = v___y_3103_;
v___y_3076_ = v___x_3108_;
v___y_3077_ = v___y_3104_;
v___y_3078_ = v___y_3105_;
v___y_3079_ = v___y_3101_;
v___y_3080_ = v___y_3102_;
v___y_3081_ = v_a_3107_;
v___y_3082_ = v___y_3100_;
v___y_3083_ = v___x_3109_;
goto v___jp_3073_;
}
}
}
v___jp_3111_:
{
lean_object* v___x_3118_; 
lean_inc(v_a_3072_);
v___x_3118_ = l_Lean_Meta_getMVars(v_a_3072_, v___y_3114_, v___y_3115_, v___y_3116_, v___y_3117_);
if (lean_obj_tag(v___x_3118_) == 0)
{
lean_object* v_a_3119_; lean_object* v___x_3120_; 
v_a_3119_ = lean_ctor_get(v___x_3118_, 0);
lean_inc(v_a_3119_);
lean_dec_ref_known(v___x_3118_, 1);
v___x_3120_ = l_Lean_Elab_Term_logUnassignedUsingErrorInfos(v_a_3119_, v___x_3046_, v___y_3112_, v___y_3113_, v___y_3114_, v___y_3115_, v___y_3116_, v___y_3117_);
lean_dec(v_a_3119_);
if (lean_obj_tag(v___x_3120_) == 0)
{
lean_object* v_a_3121_; uint8_t v___x_3122_; 
v_a_3121_ = lean_ctor_get(v___x_3120_, 0);
lean_inc(v_a_3121_);
lean_dec_ref_known(v___x_3120_, 1);
v___x_3122_ = lean_unbox(v_a_3121_);
lean_dec(v_a_3121_);
if (v___x_3122_ == 0)
{
v___y_3100_ = v___y_3112_;
v___y_3101_ = v___y_3113_;
v___y_3102_ = v___y_3114_;
v___y_3103_ = v___y_3115_;
v___y_3104_ = v___y_3116_;
v___y_3105_ = v___y_3117_;
goto v___jp_3099_;
}
else
{
lean_object* v___x_3123_; lean_object* v_a_3124_; lean_object* v___x_3126_; uint8_t v_isShared_3127_; uint8_t v_isSharedCheck_3131_; 
lean_dec_ref(v___y_3116_);
lean_dec(v_a_3072_);
v___x_3123_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
v_a_3124_ = lean_ctor_get(v___x_3123_, 0);
v_isSharedCheck_3131_ = !lean_is_exclusive(v___x_3123_);
if (v_isSharedCheck_3131_ == 0)
{
v___x_3126_ = v___x_3123_;
v_isShared_3127_ = v_isSharedCheck_3131_;
goto v_resetjp_3125_;
}
else
{
lean_inc(v_a_3124_);
lean_dec(v___x_3123_);
v___x_3126_ = lean_box(0);
v_isShared_3127_ = v_isSharedCheck_3131_;
goto v_resetjp_3125_;
}
v_resetjp_3125_:
{
lean_object* v___x_3129_; 
if (v_isShared_3127_ == 0)
{
v___x_3129_ = v___x_3126_;
goto v_reusejp_3128_;
}
else
{
lean_object* v_reuseFailAlloc_3130_; 
v_reuseFailAlloc_3130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3130_, 0, v_a_3124_);
v___x_3129_ = v_reuseFailAlloc_3130_;
goto v_reusejp_3128_;
}
v_reusejp_3128_:
{
return v___x_3129_;
}
}
}
}
else
{
lean_object* v_a_3132_; lean_object* v___x_3134_; uint8_t v_isShared_3135_; uint8_t v_isSharedCheck_3139_; 
lean_dec_ref(v___y_3116_);
lean_dec(v_a_3072_);
v_a_3132_ = lean_ctor_get(v___x_3120_, 0);
v_isSharedCheck_3139_ = !lean_is_exclusive(v___x_3120_);
if (v_isSharedCheck_3139_ == 0)
{
v___x_3134_ = v___x_3120_;
v_isShared_3135_ = v_isSharedCheck_3139_;
goto v_resetjp_3133_;
}
else
{
lean_inc(v_a_3132_);
lean_dec(v___x_3120_);
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
}
else
{
lean_object* v_a_3140_; lean_object* v___x_3142_; uint8_t v_isShared_3143_; uint8_t v_isSharedCheck_3147_; 
lean_dec_ref(v___y_3116_);
lean_dec(v_a_3072_);
v_a_3140_ = lean_ctor_get(v___x_3118_, 0);
v_isSharedCheck_3147_ = !lean_is_exclusive(v___x_3118_);
if (v_isSharedCheck_3147_ == 0)
{
v___x_3142_ = v___x_3118_;
v_isShared_3143_ = v_isSharedCheck_3147_;
goto v_resetjp_3141_;
}
else
{
lean_inc(v_a_3140_);
lean_dec(v___x_3118_);
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
v___jp_3148_:
{
lean_object* v___x_3155_; lean_object* v___x_3156_; lean_object* v___x_3157_; lean_object* v___x_3158_; lean_object* v_a_3159_; lean_object* v___x_3161_; uint8_t v_isShared_3162_; uint8_t v_isSharedCheck_3166_; 
v___x_3155_ = lean_obj_once(&lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10, &lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10_once, _init_lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___closed__10);
v___x_3156_ = l_Lean_indentExpr(v_a_3072_);
v___x_3157_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3157_, 0, v___x_3155_);
lean_ctor_set(v___x_3157_, 1, v___x_3156_);
v___x_3158_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v___x_3157_, v___y_3149_, v___y_3150_, v___y_3151_, v___y_3152_, v___y_3153_, v___y_3154_);
lean_dec_ref(v___y_3153_);
v_a_3159_ = lean_ctor_get(v___x_3158_, 0);
v_isSharedCheck_3166_ = !lean_is_exclusive(v___x_3158_);
if (v_isSharedCheck_3166_ == 0)
{
v___x_3161_ = v___x_3158_;
v_isShared_3162_ = v_isSharedCheck_3166_;
goto v_resetjp_3160_;
}
else
{
lean_inc(v_a_3159_);
lean_dec(v___x_3158_);
v___x_3161_ = lean_box(0);
v_isShared_3162_ = v_isSharedCheck_3166_;
goto v_resetjp_3160_;
}
v_resetjp_3160_:
{
lean_object* v___x_3164_; 
if (v_isShared_3162_ == 0)
{
v___x_3164_ = v___x_3161_;
goto v_reusejp_3163_;
}
else
{
lean_object* v_reuseFailAlloc_3165_; 
v_reuseFailAlloc_3165_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3165_, 0, v_a_3159_);
v___x_3164_ = v_reuseFailAlloc_3165_;
goto v_reusejp_3163_;
}
v_reusejp_3163_:
{
return v___x_3164_;
}
}
}
}
else
{
lean_object* v_a_3178_; lean_object* v___x_3180_; uint8_t v_isShared_3181_; uint8_t v_isSharedCheck_3185_; 
lean_dec_ref_known(v___x_3068_, 14);
v_a_3178_ = lean_ctor_get(v___x_3069_, 0);
v_isSharedCheck_3185_ = !lean_is_exclusive(v___x_3069_);
if (v_isSharedCheck_3185_ == 0)
{
v___x_3180_ = v___x_3069_;
v_isShared_3181_ = v_isSharedCheck_3185_;
goto v_resetjp_3179_;
}
else
{
lean_inc(v_a_3178_);
lean_dec(v___x_3069_);
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
v_reuseFailAlloc_3184_ = lean_alloc_ctor(1, 1, 0);
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
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0___boxed(lean_object* v_stx_3186_, lean_object* v_a_3187_, lean_object* v_a_3188_, lean_object* v_a_3189_, lean_object* v_a_3190_, lean_object* v_a_3191_, lean_object* v_a_3192_, lean_object* v_a_3193_){
_start:
{
lean_object* v_res_3194_; 
v_res_3194_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0(v_stx_3186_, v_a_3187_, v_a_3188_, v_a_3189_, v_a_3190_, v_a_3191_, v_a_3192_);
lean_dec(v_a_3192_);
lean_dec_ref(v_a_3191_);
lean_dec(v_a_3190_);
lean_dec_ref(v_a_3189_);
lean_dec(v_a_3188_);
lean_dec_ref(v_a_3187_);
return v_res_3194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0(lean_object* v_config_3198_, lean_object* v_item_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_){
_start:
{
lean_object* v___x_3207_; lean_object* v___x_3208_; 
v___x_3207_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2));
v___x_3208_ = l_Lean_Elab_ConfigEval_ConfigItem_addCompletionInfo(v_item_3199_, v___x_3207_, v___y_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_, v___y_3205_);
if (lean_obj_tag(v___x_3208_) == 0)
{
lean_object* v_item_3210_; lean_object* v___y_3211_; lean_object* v___y_3212_; lean_object* v___y_3213_; lean_object* v___y_3214_; lean_object* v___y_3215_; lean_object* v___y_3216_; uint8_t v___x_3219_; 
lean_dec_ref_known(v___x_3208_, 1);
v___x_3219_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v_item_3199_);
if (v___x_3219_ == 0)
{
lean_object* v___x_3220_; lean_object* v___x_3221_; lean_object* v___x_3222_; uint8_t v___x_3223_; 
v___x_3220_ = l_Lean_Elab_ConfigEval_ConfigItem_getRootStr(v_item_3199_);
lean_inc_ref(v_item_3199_);
v___x_3221_ = l_Lean_Elab_ConfigEval_ConfigItem_shift(v_item_3199_);
v___x_3222_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__1));
v___x_3223_ = lean_string_dec_eq(v___x_3220_, v___x_3222_);
lean_dec_ref(v___x_3220_);
if (v___x_3223_ == 0)
{
lean_dec_ref(v_item_3199_);
v_item_3210_ = v___x_3221_;
v___y_3211_ = v___y_3200_;
v___y_3212_ = v___y_3201_;
v___y_3213_ = v___y_3202_;
v___y_3214_ = v___y_3203_;
v___y_3215_ = v___y_3204_;
v___y_3216_ = v___y_3205_;
goto v___jp_3209_;
}
else
{
uint8_t v___x_3224_; 
v___x_3224_ = l_Lean_Elab_ConfigEval_ConfigItem_isAnonymous(v___x_3221_);
if (v___x_3224_ == 0)
{
lean_dec_ref(v_item_3199_);
v_item_3210_ = v___x_3221_;
v___y_3211_ = v___y_3200_;
v___y_3212_ = v___y_3201_;
v___y_3213_ = v___y_3202_;
v___y_3214_ = v___y_3203_;
v___y_3215_ = v___y_3204_;
v___y_3216_ = v___y_3205_;
goto v___jp_3209_;
}
else
{
lean_object* v_value_3225_; lean_object* v___x_3226_; 
lean_dec_ref(v___x_3221_);
v_value_3225_ = lean_ctor_get(v_item_3199_, 2);
lean_inc(v_value_3225_);
lean_dec_ref(v_item_3199_);
v___x_3226_ = lp_mathlib_Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0(v_value_3225_, v___y_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_, v___y_3205_);
return v___x_3226_;
}
}
}
else
{
v_item_3210_ = v_item_3199_;
v___y_3211_ = v___y_3200_;
v___y_3212_ = v___y_3201_;
v___y_3213_ = v___y_3202_;
v___y_3214_ = v___y_3203_;
v___y_3215_ = v___y_3204_;
v___y_3216_ = v___y_3205_;
goto v___jp_3209_;
}
v___jp_3209_:
{
lean_object* v___x_3217_; lean_object* v___x_3218_; 
v___x_3217_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___closed__0));
v___x_3218_ = l_Lean_Elab_ConfigEval_ConfigItem_throwInvalidOption___redArg(v_item_3210_, v___x_3217_, v___y_3211_, v___y_3212_, v___y_3213_, v___y_3214_, v___y_3215_, v___y_3216_);
return v___x_3218_;
}
}
else
{
lean_object* v_a_3227_; lean_object* v___x_3229_; uint8_t v_isShared_3230_; uint8_t v_isSharedCheck_3234_; 
lean_dec_ref(v_item_3199_);
v_a_3227_ = lean_ctor_get(v___x_3208_, 0);
v_isSharedCheck_3234_ = !lean_is_exclusive(v___x_3208_);
if (v_isSharedCheck_3234_ == 0)
{
v___x_3229_ = v___x_3208_;
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
else
{
lean_inc(v_a_3227_);
lean_dec(v___x_3208_);
v___x_3229_ = lean_box(0);
v_isShared_3230_ = v_isSharedCheck_3234_;
goto v_resetjp_3228_;
}
v_resetjp_3228_:
{
lean_object* v___x_3232_; 
if (v_isShared_3230_ == 0)
{
v___x_3232_ = v___x_3229_;
goto v_reusejp_3231_;
}
else
{
lean_object* v_reuseFailAlloc_3233_; 
v_reuseFailAlloc_3233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3233_, 0, v_a_3227_);
v___x_3232_ = v_reuseFailAlloc_3233_;
goto v_reusejp_3231_;
}
v_reusejp_3231_:
{
return v___x_3232_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0___boxed(lean_object* v_config_3235_, lean_object* v_item_3236_, lean_object* v___y_3237_, lean_object* v___y_3238_, lean_object* v___y_3239_, lean_object* v___y_3240_, lean_object* v___y_3241_, lean_object* v___y_3242_, lean_object* v___y_3243_){
_start:
{
lean_object* v_res_3244_; 
v_res_3244_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___lam__0(v_config_3235_, v_item_3236_, v___y_3237_, v___y_3238_, v___y_3239_, v___y_3240_, v___y_3241_, v___y_3242_);
lean_dec(v___y_3242_);
lean_dec_ref(v___y_3241_);
lean_dec(v___y_3240_);
lean_dec_ref(v___y_3239_);
lean_dec(v___y_3238_);
lean_dec_ref(v___y_3237_);
return v_res_3244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0(lean_object* v_e_3247_, lean_object* v___y_3248_, lean_object* v___y_3249_, lean_object* v___y_3250_, lean_object* v___y_3251_, lean_object* v___y_3252_, lean_object* v___y_3253_){
_start:
{
lean_object* v___x_3255_; 
v___x_3255_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___redArg(v_e_3247_, v___y_3251_);
return v___x_3255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0___boxed(lean_object* v_e_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_, lean_object* v___y_3263_){
_start:
{
lean_object* v_res_3264_; 
v_res_3264_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__0(v_e_3256_, v___y_3257_, v___y_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_);
lean_dec(v___y_3262_);
lean_dec_ref(v___y_3261_);
lean_dec(v___y_3260_);
lean_dec_ref(v___y_3259_);
lean_dec(v___y_3258_);
lean_dec_ref(v___y_3257_);
return v_res_3264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2(lean_object* v_00_u03b1_3265_, lean_object* v___y_3266_, lean_object* v___y_3267_, lean_object* v___y_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_){
_start:
{
lean_object* v___x_3273_; 
v___x_3273_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___redArg();
return v___x_3273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2___boxed(lean_object* v_00_u03b1_3274_, lean_object* v___y_3275_, lean_object* v___y_3276_, lean_object* v___y_3277_, lean_object* v___y_3278_, lean_object* v___y_3279_, lean_object* v___y_3280_, lean_object* v___y_3281_){
_start:
{
lean_object* v_res_3282_; 
v_res_3282_ = lp_mathlib_Lean_Elab_throwAbortTerm___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__2(v_00_u03b1_3274_, v___y_3275_, v___y_3276_, v___y_3277_, v___y_3278_, v___y_3279_, v___y_3280_);
lean_dec(v___y_3280_);
lean_dec_ref(v___y_3279_);
lean_dec(v___y_3278_);
lean_dec_ref(v___y_3277_);
lean_dec(v___y_3276_);
lean_dec_ref(v___y_3275_);
return v_res_3282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1(lean_object* v_00_u03b1_3283_, lean_object* v_msg_3284_, lean_object* v___y_3285_, lean_object* v___y_3286_, lean_object* v___y_3287_, lean_object* v___y_3288_, lean_object* v___y_3289_, lean_object* v___y_3290_){
_start:
{
lean_object* v___x_3292_; 
v___x_3292_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___redArg(v_msg_3284_, v___y_3285_, v___y_3286_, v___y_3287_, v___y_3288_, v___y_3289_, v___y_3290_);
return v___x_3292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1___boxed(lean_object* v_00_u03b1_3293_, lean_object* v_msg_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_, lean_object* v___y_3301_){
_start:
{
lean_object* v_res_3302_; 
v_res_3302_ = lp_mathlib_Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1(v_00_u03b1_3293_, v_msg_3294_, v___y_3295_, v___y_3296_, v___y_3297_, v___y_3298_, v___y_3299_, v___y_3300_);
lean_dec(v___y_3300_);
lean_dec_ref(v___y_3299_);
lean_dec(v___y_3298_);
lean_dec_ref(v___y_3297_);
lean_dec(v___y_3296_);
lean_dec_ref(v___y_3295_);
return v_res_3302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(lean_object* v_msgData_3303_, lean_object* v_macroStack_3304_, lean_object* v___y_3305_, lean_object* v___y_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_){
_start:
{
lean_object* v___x_3312_; 
v___x_3312_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___redArg(v_msgData_3303_, v_macroStack_3304_, v___y_3309_);
return v___x_3312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2___boxed(lean_object* v_msgData_3313_, lean_object* v_macroStack_3314_, lean_object* v___y_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_, lean_object* v___y_3319_, lean_object* v___y_3320_, lean_object* v___y_3321_){
_start:
{
lean_object* v_res_3322_; 
v_res_3322_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00Lean_Elab_ConfigEval_evalExprWithElab___at___00__private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem_spec__0_spec__1_spec__2(v_msgData_3313_, v_macroStack_3314_, v___y_3315_, v___y_3316_, v___y_3317_, v___y_3318_, v___y_3319_, v___y_3320_);
lean_dec(v___y_3320_);
lean_dec_ref(v___y_3319_);
lean_dec(v___y_3318_);
lean_dec_ref(v___y_3317_);
lean_dec(v___y_3316_);
lean_dec_ref(v___y_3315_);
return v_res_3322_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_3323_; lean_object* v___x_3324_; lean_object* v___x_3325_; 
v___x_3323_ = lean_box(0);
v___x_3324_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig_evalExpr___closed__2));
v___x_3325_ = l_Lean_mkConst(v___x_3324_, v___x_3323_);
return v___x_3325_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1(void){
_start:
{
lean_object* v___x_3326_; lean_object* v___x_3327_; 
v___x_3326_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0, &lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__0);
v___x_3327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3327_, 0, v___x_3326_);
return v___x_3327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0(lean_object* v_cfg_3328_, lean_object* v_cfgItem_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_){
_start:
{
lean_object* v___x_3337_; lean_object* v___x_3338_; 
v___x_3337_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1, &lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___closed__1);
v___x_3338_ = l_Lean_Elab_ConfigEval_EvalConfigItem_defaultOnErr___redArg(v_cfg_3328_, v_cfgItem_3329_, v___x_3337_, v___y_3330_, v___y_3331_, v___y_3332_, v___y_3333_, v___y_3334_, v___y_3335_);
return v___x_3338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0___boxed(lean_object* v_cfg_3339_, lean_object* v_cfgItem_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_, lean_object* v___y_3345_, lean_object* v___y_3346_, lean_object* v___y_3347_){
_start:
{
lean_object* v_res_3348_; 
v_res_3348_ = lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___lam__0(v_cfg_3339_, v_cfgItem_3340_, v___y_3341_, v___y_3342_, v___y_3343_, v___y_3344_, v___y_3345_, v___y_3346_);
lean_dec(v___y_3346_);
lean_dec_ref(v___y_3345_);
lean_dec(v___y_3344_);
lean_dec_ref(v___y_3343_);
lean_dec(v___y_3342_);
lean_dec_ref(v___y_3341_);
lean_dec(v_cfgItem_3340_);
return v_res_3348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg(lean_object* v_cfg_3350_, lean_object* v_init_3351_, uint8_t v_logExceptions_3352_, lean_object* v_a_3353_, lean_object* v_a_3354_, lean_object* v_a_3355_){
_start:
{
lean_object* v_onErr_3357_; lean_object* v_eval_3358_; 
v_onErr_3357_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___closed__0));
v_eval_3358_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_elabConfig_evalConfigItem___closed__0));
if (v_logExceptions_3352_ == 0)
{
lean_object* v___x_3359_; 
v___x_3359_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3358_, v_init_3351_, v_cfg_3350_, v_onErr_3357_, v_logExceptions_3352_, v_a_3354_, v_a_3355_);
return v___x_3359_;
}
else
{
uint8_t v_recover_3360_; lean_object* v___x_3361_; 
v_recover_3360_ = lean_ctor_get_uint8(v_a_3353_, sizeof(void*)*1);
v___x_3361_ = l_Lean_Elab_ConfigEval_EvalConfigItem_setConfig_x27___redArg(v_eval_3358_, v_init_3351_, v_cfg_3350_, v_onErr_3357_, v_recover_3360_, v_a_3354_, v_a_3355_);
return v___x_3361_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg___boxed(lean_object* v_cfg_3362_, lean_object* v_init_3363_, lean_object* v_logExceptions_3364_, lean_object* v_a_3365_, lean_object* v_a_3366_, lean_object* v_a_3367_, lean_object* v_a_3368_){
_start:
{
uint8_t v_logExceptions_boxed_3369_; lean_object* v_res_3370_; 
v_logExceptions_boxed_3369_ = lean_unbox(v_logExceptions_3364_);
v_res_3370_ = lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg(v_cfg_3362_, v_init_3363_, v_logExceptions_boxed_3369_, v_a_3365_, v_a_3366_, v_a_3367_);
lean_dec(v_a_3367_);
lean_dec_ref(v_a_3366_);
lean_dec_ref(v_a_3365_);
return v_res_3370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig(lean_object* v_cfg_3371_, lean_object* v_init_3372_, uint8_t v_logExceptions_3373_, lean_object* v_a_3374_, lean_object* v_a_3375_, lean_object* v_a_3376_, lean_object* v_a_3377_, lean_object* v_a_3378_, lean_object* v_a_3379_, lean_object* v_a_3380_, lean_object* v_a_3381_){
_start:
{
lean_object* v___x_3383_; 
v___x_3383_ = lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg(v_cfg_3371_, v_init_3372_, v_logExceptions_3373_, v_a_3374_, v_a_3380_, v_a_3381_);
return v___x_3383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___boxed(lean_object* v_cfg_3384_, lean_object* v_init_3385_, lean_object* v_logExceptions_3386_, lean_object* v_a_3387_, lean_object* v_a_3388_, lean_object* v_a_3389_, lean_object* v_a_3390_, lean_object* v_a_3391_, lean_object* v_a_3392_, lean_object* v_a_3393_, lean_object* v_a_3394_, lean_object* v_a_3395_){
_start:
{
uint8_t v_logExceptions_boxed_3396_; lean_object* v_res_3397_; 
v_logExceptions_boxed_3396_ = lean_unbox(v_logExceptions_3386_);
v_res_3397_ = lp_mathlib_Mathlib_Tactic_Tauto_elabConfig(v_cfg_3384_, v_init_3385_, v_logExceptions_boxed_3396_, v_a_3387_, v_a_3388_, v_a_3389_, v_a_3390_, v_a_3391_, v_a_3392_, v_a_3393_, v_a_3394_);
lean_dec(v_a_3394_);
lean_dec_ref(v_a_3393_);
lean_dec(v_a_3392_);
lean_dec_ref(v_a_3391_);
lean_dec(v_a_3390_);
lean_dec_ref(v_a_3389_);
lean_dec(v_a_3388_);
lean_dec_ref(v_a_3387_);
return v_res_3397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0(lean_object* v___x_3398_, uint8_t v___x_3399_, lean_object* v___x_3400_, lean_object* v___x_3401_, lean_object* v_e_3402_, uint8_t v___x_3403_, lean_object* v___y_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_){
_start:
{
lean_object* v___x_3409_; 
lean_inc(v___x_3400_);
lean_inc(v___x_3398_);
v___x_3409_ = l_Lean_Meta_mkFreshExprMVar(v___x_3398_, v___x_3399_, v___x_3400_, v___y_3404_, v___y_3405_, v___y_3406_, v___y_3407_);
if (lean_obj_tag(v___x_3409_) == 0)
{
lean_object* v_a_3410_; lean_object* v___x_3411_; 
v_a_3410_ = lean_ctor_get(v___x_3409_, 0);
lean_inc(v_a_3410_);
lean_dec_ref_known(v___x_3409_, 1);
v___x_3411_ = l_Lean_Meta_mkFreshExprMVar(v___x_3398_, v___x_3399_, v___x_3400_, v___y_3404_, v___y_3405_, v___y_3406_, v___y_3407_);
if (lean_obj_tag(v___x_3411_) == 0)
{
lean_object* v_a_3412_; lean_object* v_keyedConfig_3413_; uint8_t v_trackZetaDelta_3414_; lean_object* v_zetaDeltaSet_3415_; lean_object* v_lctx_3416_; lean_object* v_localInstances_3417_; lean_object* v_defEqCtx_x3f_3418_; lean_object* v_synthPendingDepth_3419_; lean_object* v_customCanUnfoldPredicate_x3f_3420_; uint8_t v_univApprox_3421_; uint8_t v_inTypeClassResolution_3422_; uint8_t v_cacheInferType_3423_; lean_object* v___x_3425_; uint8_t v_isShared_3426_; uint8_t v_isSharedCheck_3470_; 
v_a_3412_ = lean_ctor_get(v___x_3411_, 0);
lean_inc(v_a_3412_);
lean_dec_ref_known(v___x_3411_, 1);
v_keyedConfig_3413_ = lean_ctor_get(v___y_3404_, 0);
v_trackZetaDelta_3414_ = lean_ctor_get_uint8(v___y_3404_, sizeof(void*)*7);
v_zetaDeltaSet_3415_ = lean_ctor_get(v___y_3404_, 1);
v_lctx_3416_ = lean_ctor_get(v___y_3404_, 2);
v_localInstances_3417_ = lean_ctor_get(v___y_3404_, 3);
v_defEqCtx_x3f_3418_ = lean_ctor_get(v___y_3404_, 4);
v_synthPendingDepth_3419_ = lean_ctor_get(v___y_3404_, 5);
v_customCanUnfoldPredicate_x3f_3420_ = lean_ctor_get(v___y_3404_, 6);
v_univApprox_3421_ = lean_ctor_get_uint8(v___y_3404_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3422_ = lean_ctor_get_uint8(v___y_3404_, sizeof(void*)*7 + 2);
v_cacheInferType_3423_ = lean_ctor_get_uint8(v___y_3404_, sizeof(void*)*7 + 3);
v_isSharedCheck_3470_ = !lean_is_exclusive(v___y_3404_);
if (v_isSharedCheck_3470_ == 0)
{
v___x_3425_ = v___y_3404_;
v_isShared_3426_ = v_isSharedCheck_3470_;
goto v_resetjp_3424_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3420_);
lean_inc(v_synthPendingDepth_3419_);
lean_inc(v_defEqCtx_x3f_3418_);
lean_inc(v_localInstances_3417_);
lean_inc(v_lctx_3416_);
lean_inc(v_zetaDeltaSet_3415_);
lean_inc(v_keyedConfig_3413_);
lean_dec(v___y_3404_);
v___x_3425_ = lean_box(0);
v_isShared_3426_ = v_isSharedCheck_3470_;
goto v_resetjp_3424_;
}
v_resetjp_3424_:
{
lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; uint8_t v___x_3431_; lean_object* v___x_3432_; lean_object* v___x_3434_; 
v___x_3427_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__3___closed__1));
v___x_3428_ = l_Lean_Expr_const___override(v___x_3427_, v___x_3401_);
lean_inc(v_a_3410_);
v___x_3429_ = l_Lean_Expr_app___override(v___x_3428_, v_a_3410_);
lean_inc(v_a_3412_);
v___x_3430_ = l_Lean_Expr_app___override(v___x_3429_, v_a_3412_);
v___x_3431_ = 2;
v___x_3432_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3431_, v_keyedConfig_3413_);
if (v_isShared_3426_ == 0)
{
lean_ctor_set(v___x_3425_, 0, v___x_3432_);
v___x_3434_ = v___x_3425_;
goto v_reusejp_3433_;
}
else
{
lean_object* v_reuseFailAlloc_3469_; 
v_reuseFailAlloc_3469_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3469_, 0, v___x_3432_);
lean_ctor_set(v_reuseFailAlloc_3469_, 1, v_zetaDeltaSet_3415_);
lean_ctor_set(v_reuseFailAlloc_3469_, 2, v_lctx_3416_);
lean_ctor_set(v_reuseFailAlloc_3469_, 3, v_localInstances_3417_);
lean_ctor_set(v_reuseFailAlloc_3469_, 4, v_defEqCtx_x3f_3418_);
lean_ctor_set(v_reuseFailAlloc_3469_, 5, v_synthPendingDepth_3419_);
lean_ctor_set(v_reuseFailAlloc_3469_, 6, v_customCanUnfoldPredicate_x3f_3420_);
lean_ctor_set_uint8(v_reuseFailAlloc_3469_, sizeof(void*)*7, v_trackZetaDelta_3414_);
lean_ctor_set_uint8(v_reuseFailAlloc_3469_, sizeof(void*)*7 + 1, v_univApprox_3421_);
lean_ctor_set_uint8(v_reuseFailAlloc_3469_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3422_);
lean_ctor_set_uint8(v_reuseFailAlloc_3469_, sizeof(void*)*7 + 3, v_cacheInferType_3423_);
v___x_3434_ = v_reuseFailAlloc_3469_;
goto v_reusejp_3433_;
}
v_reusejp_3433_:
{
lean_object* v___x_3435_; 
v___x_3435_ = l_Lean_Meta_isExprDefEq(v___x_3430_, v_e_3402_, v___x_3434_, v___y_3405_, v___y_3406_, v___y_3407_);
lean_dec_ref(v___x_3434_);
if (lean_obj_tag(v___x_3435_) == 0)
{
lean_object* v_a_3436_; lean_object* v___x_3438_; uint8_t v_isShared_3439_; uint8_t v_isSharedCheck_3460_; 
v_a_3436_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3460_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3460_ == 0)
{
v___x_3438_ = v___x_3435_;
v_isShared_3439_ = v_isSharedCheck_3460_;
goto v_resetjp_3437_;
}
else
{
lean_inc(v_a_3436_);
lean_dec(v___x_3435_);
v___x_3438_ = lean_box(0);
v_isShared_3439_ = v_isSharedCheck_3460_;
goto v_resetjp_3437_;
}
v_resetjp_3437_:
{
uint8_t v___x_3440_; 
v___x_3440_ = lean_unbox(v_a_3436_);
if (v___x_3440_ == 0)
{
lean_object* v___x_3441_; lean_object* v___x_3442_; lean_object* v___x_3443_; lean_object* v___x_3445_; 
lean_dec(v_a_3436_);
v___x_3441_ = lean_box(v___x_3403_);
v___x_3442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3442_, 0, v_a_3412_);
lean_ctor_set(v___x_3442_, 1, v___x_3441_);
v___x_3443_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3443_, 0, v_a_3410_);
lean_ctor_set(v___x_3443_, 1, v___x_3442_);
if (v_isShared_3439_ == 0)
{
lean_ctor_set(v___x_3438_, 0, v___x_3443_);
v___x_3445_ = v___x_3438_;
goto v_reusejp_3444_;
}
else
{
lean_object* v_reuseFailAlloc_3446_; 
v_reuseFailAlloc_3446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3446_, 0, v___x_3443_);
v___x_3445_ = v_reuseFailAlloc_3446_;
goto v_reusejp_3444_;
}
v_reusejp_3444_:
{
return v___x_3445_;
}
}
else
{
lean_object* v___x_3447_; lean_object* v_a_3448_; lean_object* v___x_3449_; lean_object* v_a_3450_; lean_object* v___x_3452_; uint8_t v_isShared_3453_; uint8_t v_isSharedCheck_3459_; 
lean_del_object(v___x_3438_);
v___x_3447_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3410_, v___y_3405_);
v_a_3448_ = lean_ctor_get(v___x_3447_, 0);
lean_inc(v_a_3448_);
lean_dec_ref(v___x_3447_);
v___x_3449_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3412_, v___y_3405_);
v_a_3450_ = lean_ctor_get(v___x_3449_, 0);
v_isSharedCheck_3459_ = !lean_is_exclusive(v___x_3449_);
if (v_isSharedCheck_3459_ == 0)
{
v___x_3452_ = v___x_3449_;
v_isShared_3453_ = v_isSharedCheck_3459_;
goto v_resetjp_3451_;
}
else
{
lean_inc(v_a_3450_);
lean_dec(v___x_3449_);
v___x_3452_ = lean_box(0);
v_isShared_3453_ = v_isSharedCheck_3459_;
goto v_resetjp_3451_;
}
v_resetjp_3451_:
{
lean_object* v___x_3454_; lean_object* v___x_3455_; lean_object* v___x_3457_; 
v___x_3454_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3454_, 0, v_a_3450_);
lean_ctor_set(v___x_3454_, 1, v_a_3436_);
v___x_3455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3455_, 0, v_a_3448_);
lean_ctor_set(v___x_3455_, 1, v___x_3454_);
if (v_isShared_3453_ == 0)
{
lean_ctor_set(v___x_3452_, 0, v___x_3455_);
v___x_3457_ = v___x_3452_;
goto v_reusejp_3456_;
}
else
{
lean_object* v_reuseFailAlloc_3458_; 
v_reuseFailAlloc_3458_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3458_, 0, v___x_3455_);
v___x_3457_ = v_reuseFailAlloc_3458_;
goto v_reusejp_3456_;
}
v_reusejp_3456_:
{
return v___x_3457_;
}
}
}
}
}
else
{
lean_object* v_a_3461_; lean_object* v___x_3463_; uint8_t v_isShared_3464_; uint8_t v_isSharedCheck_3468_; 
lean_dec(v_a_3412_);
lean_dec(v_a_3410_);
v_a_3461_ = lean_ctor_get(v___x_3435_, 0);
v_isSharedCheck_3468_ = !lean_is_exclusive(v___x_3435_);
if (v_isSharedCheck_3468_ == 0)
{
v___x_3463_ = v___x_3435_;
v_isShared_3464_ = v_isSharedCheck_3468_;
goto v_resetjp_3462_;
}
else
{
lean_inc(v_a_3461_);
lean_dec(v___x_3435_);
v___x_3463_ = lean_box(0);
v_isShared_3464_ = v_isSharedCheck_3468_;
goto v_resetjp_3462_;
}
v_resetjp_3462_:
{
lean_object* v___x_3466_; 
if (v_isShared_3464_ == 0)
{
v___x_3466_ = v___x_3463_;
goto v_reusejp_3465_;
}
else
{
lean_object* v_reuseFailAlloc_3467_; 
v_reuseFailAlloc_3467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3467_, 0, v_a_3461_);
v___x_3466_ = v_reuseFailAlloc_3467_;
goto v_reusejp_3465_;
}
v_reusejp_3465_:
{
return v___x_3466_;
}
}
}
}
}
}
else
{
lean_object* v_a_3471_; lean_object* v___x_3473_; uint8_t v_isShared_3474_; uint8_t v_isSharedCheck_3478_; 
lean_dec(v_a_3410_);
lean_dec_ref(v___y_3404_);
lean_dec_ref(v_e_3402_);
lean_dec(v___x_3401_);
v_a_3471_ = lean_ctor_get(v___x_3411_, 0);
v_isSharedCheck_3478_ = !lean_is_exclusive(v___x_3411_);
if (v_isSharedCheck_3478_ == 0)
{
v___x_3473_ = v___x_3411_;
v_isShared_3474_ = v_isSharedCheck_3478_;
goto v_resetjp_3472_;
}
else
{
lean_inc(v_a_3471_);
lean_dec(v___x_3411_);
v___x_3473_ = lean_box(0);
v_isShared_3474_ = v_isSharedCheck_3478_;
goto v_resetjp_3472_;
}
v_resetjp_3472_:
{
lean_object* v___x_3476_; 
if (v_isShared_3474_ == 0)
{
v___x_3476_ = v___x_3473_;
goto v_reusejp_3475_;
}
else
{
lean_object* v_reuseFailAlloc_3477_; 
v_reuseFailAlloc_3477_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3477_, 0, v_a_3471_);
v___x_3476_ = v_reuseFailAlloc_3477_;
goto v_reusejp_3475_;
}
v_reusejp_3475_:
{
return v___x_3476_;
}
}
}
}
else
{
lean_object* v_a_3479_; lean_object* v___x_3481_; uint8_t v_isShared_3482_; uint8_t v_isSharedCheck_3486_; 
lean_dec_ref(v___y_3404_);
lean_dec_ref(v_e_3402_);
lean_dec(v___x_3401_);
lean_dec(v___x_3400_);
lean_dec(v___x_3398_);
v_a_3479_ = lean_ctor_get(v___x_3409_, 0);
v_isSharedCheck_3486_ = !lean_is_exclusive(v___x_3409_);
if (v_isSharedCheck_3486_ == 0)
{
v___x_3481_ = v___x_3409_;
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
else
{
lean_inc(v_a_3479_);
lean_dec(v___x_3409_);
v___x_3481_ = lean_box(0);
v_isShared_3482_ = v_isSharedCheck_3486_;
goto v_resetjp_3480_;
}
v_resetjp_3480_:
{
lean_object* v___x_3484_; 
if (v_isShared_3482_ == 0)
{
v___x_3484_ = v___x_3481_;
goto v_reusejp_3483_;
}
else
{
lean_object* v_reuseFailAlloc_3485_; 
v_reuseFailAlloc_3485_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3485_, 0, v_a_3479_);
v___x_3484_ = v_reuseFailAlloc_3485_;
goto v_reusejp_3483_;
}
v_reusejp_3483_:
{
return v___x_3484_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0___boxed(lean_object* v___x_3487_, lean_object* v___x_3488_, lean_object* v___x_3489_, lean_object* v___x_3490_, lean_object* v_e_3491_, lean_object* v___x_3492_, lean_object* v___y_3493_, lean_object* v___y_3494_, lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_){
_start:
{
uint8_t v___x_3280__boxed_3498_; uint8_t v___x_3283__boxed_3499_; lean_object* v_res_3500_; 
v___x_3280__boxed_3498_ = lean_unbox(v___x_3488_);
v___x_3283__boxed_3499_ = lean_unbox(v___x_3492_);
v_res_3500_ = lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0(v___x_3487_, v___x_3280__boxed_3498_, v___x_3489_, v___x_3490_, v_e_3491_, v___x_3283__boxed_3499_, v___y_3493_, v___y_3494_, v___y_3495_, v___y_3496_);
lean_dec(v___y_3496_);
lean_dec_ref(v___y_3495_);
lean_dec(v___y_3494_);
return v_res_3500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1(lean_object* v___x_3501_, uint8_t v___x_3502_, lean_object* v___x_3503_, lean_object* v___x_3504_, lean_object* v_e_3505_, uint8_t v___x_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_){
_start:
{
lean_object* v___x_3512_; 
lean_inc(v___x_3503_);
lean_inc(v___x_3501_);
v___x_3512_ = l_Lean_Meta_mkFreshExprMVar(v___x_3501_, v___x_3502_, v___x_3503_, v___y_3507_, v___y_3508_, v___y_3509_, v___y_3510_);
if (lean_obj_tag(v___x_3512_) == 0)
{
lean_object* v_a_3513_; lean_object* v___x_3514_; 
v_a_3513_ = lean_ctor_get(v___x_3512_, 0);
lean_inc(v_a_3513_);
lean_dec_ref_known(v___x_3512_, 1);
v___x_3514_ = l_Lean_Meta_mkFreshExprMVar(v___x_3501_, v___x_3502_, v___x_3503_, v___y_3507_, v___y_3508_, v___y_3509_, v___y_3510_);
if (lean_obj_tag(v___x_3514_) == 0)
{
lean_object* v_a_3515_; lean_object* v_keyedConfig_3516_; uint8_t v_trackZetaDelta_3517_; lean_object* v_zetaDeltaSet_3518_; lean_object* v_lctx_3519_; lean_object* v_localInstances_3520_; lean_object* v_defEqCtx_x3f_3521_; lean_object* v_synthPendingDepth_3522_; lean_object* v_customCanUnfoldPredicate_x3f_3523_; uint8_t v_univApprox_3524_; uint8_t v_inTypeClassResolution_3525_; uint8_t v_cacheInferType_3526_; lean_object* v___x_3528_; uint8_t v_isShared_3529_; uint8_t v_isSharedCheck_3573_; 
v_a_3515_ = lean_ctor_get(v___x_3514_, 0);
lean_inc(v_a_3515_);
lean_dec_ref_known(v___x_3514_, 1);
v_keyedConfig_3516_ = lean_ctor_get(v___y_3507_, 0);
v_trackZetaDelta_3517_ = lean_ctor_get_uint8(v___y_3507_, sizeof(void*)*7);
v_zetaDeltaSet_3518_ = lean_ctor_get(v___y_3507_, 1);
v_lctx_3519_ = lean_ctor_get(v___y_3507_, 2);
v_localInstances_3520_ = lean_ctor_get(v___y_3507_, 3);
v_defEqCtx_x3f_3521_ = lean_ctor_get(v___y_3507_, 4);
v_synthPendingDepth_3522_ = lean_ctor_get(v___y_3507_, 5);
v_customCanUnfoldPredicate_x3f_3523_ = lean_ctor_get(v___y_3507_, 6);
v_univApprox_3524_ = lean_ctor_get_uint8(v___y_3507_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3525_ = lean_ctor_get_uint8(v___y_3507_, sizeof(void*)*7 + 2);
v_cacheInferType_3526_ = lean_ctor_get_uint8(v___y_3507_, sizeof(void*)*7 + 3);
v_isSharedCheck_3573_ = !lean_is_exclusive(v___y_3507_);
if (v_isSharedCheck_3573_ == 0)
{
v___x_3528_ = v___y_3507_;
v_isShared_3529_ = v_isSharedCheck_3573_;
goto v_resetjp_3527_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3523_);
lean_inc(v_synthPendingDepth_3522_);
lean_inc(v_defEqCtx_x3f_3521_);
lean_inc(v_localInstances_3520_);
lean_inc(v_lctx_3519_);
lean_inc(v_zetaDeltaSet_3518_);
lean_inc(v_keyedConfig_3516_);
lean_dec(v___y_3507_);
v___x_3528_ = lean_box(0);
v_isShared_3529_ = v_isSharedCheck_3573_;
goto v_resetjp_3527_;
}
v_resetjp_3527_:
{
lean_object* v___x_3530_; lean_object* v___x_3531_; lean_object* v___x_3532_; lean_object* v___x_3533_; uint8_t v___x_3534_; lean_object* v___x_3535_; lean_object* v___x_3537_; 
v___x_3530_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__8___closed__1));
v___x_3531_ = l_Lean_Expr_const___override(v___x_3530_, v___x_3504_);
lean_inc(v_a_3513_);
v___x_3532_ = l_Lean_Expr_app___override(v___x_3531_, v_a_3513_);
lean_inc(v_a_3515_);
v___x_3533_ = l_Lean_Expr_app___override(v___x_3532_, v_a_3515_);
v___x_3534_ = 2;
v___x_3535_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3534_, v_keyedConfig_3516_);
if (v_isShared_3529_ == 0)
{
lean_ctor_set(v___x_3528_, 0, v___x_3535_);
v___x_3537_ = v___x_3528_;
goto v_reusejp_3536_;
}
else
{
lean_object* v_reuseFailAlloc_3572_; 
v_reuseFailAlloc_3572_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3572_, 0, v___x_3535_);
lean_ctor_set(v_reuseFailAlloc_3572_, 1, v_zetaDeltaSet_3518_);
lean_ctor_set(v_reuseFailAlloc_3572_, 2, v_lctx_3519_);
lean_ctor_set(v_reuseFailAlloc_3572_, 3, v_localInstances_3520_);
lean_ctor_set(v_reuseFailAlloc_3572_, 4, v_defEqCtx_x3f_3521_);
lean_ctor_set(v_reuseFailAlloc_3572_, 5, v_synthPendingDepth_3522_);
lean_ctor_set(v_reuseFailAlloc_3572_, 6, v_customCanUnfoldPredicate_x3f_3523_);
lean_ctor_set_uint8(v_reuseFailAlloc_3572_, sizeof(void*)*7, v_trackZetaDelta_3517_);
lean_ctor_set_uint8(v_reuseFailAlloc_3572_, sizeof(void*)*7 + 1, v_univApprox_3524_);
lean_ctor_set_uint8(v_reuseFailAlloc_3572_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3525_);
lean_ctor_set_uint8(v_reuseFailAlloc_3572_, sizeof(void*)*7 + 3, v_cacheInferType_3526_);
v___x_3537_ = v_reuseFailAlloc_3572_;
goto v_reusejp_3536_;
}
v_reusejp_3536_:
{
lean_object* v___x_3538_; 
v___x_3538_ = l_Lean_Meta_isExprDefEq(v___x_3533_, v_e_3505_, v___x_3537_, v___y_3508_, v___y_3509_, v___y_3510_);
lean_dec_ref(v___x_3537_);
if (lean_obj_tag(v___x_3538_) == 0)
{
lean_object* v_a_3539_; lean_object* v___x_3541_; uint8_t v_isShared_3542_; uint8_t v_isSharedCheck_3563_; 
v_a_3539_ = lean_ctor_get(v___x_3538_, 0);
v_isSharedCheck_3563_ = !lean_is_exclusive(v___x_3538_);
if (v_isSharedCheck_3563_ == 0)
{
v___x_3541_ = v___x_3538_;
v_isShared_3542_ = v_isSharedCheck_3563_;
goto v_resetjp_3540_;
}
else
{
lean_inc(v_a_3539_);
lean_dec(v___x_3538_);
v___x_3541_ = lean_box(0);
v_isShared_3542_ = v_isSharedCheck_3563_;
goto v_resetjp_3540_;
}
v_resetjp_3540_:
{
uint8_t v___x_3543_; 
v___x_3543_ = lean_unbox(v_a_3539_);
if (v___x_3543_ == 0)
{
lean_object* v___x_3544_; lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3548_; 
lean_dec(v_a_3539_);
v___x_3544_ = lean_box(v___x_3506_);
v___x_3545_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3545_, 0, v_a_3515_);
lean_ctor_set(v___x_3545_, 1, v___x_3544_);
v___x_3546_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3546_, 0, v_a_3513_);
lean_ctor_set(v___x_3546_, 1, v___x_3545_);
if (v_isShared_3542_ == 0)
{
lean_ctor_set(v___x_3541_, 0, v___x_3546_);
v___x_3548_ = v___x_3541_;
goto v_reusejp_3547_;
}
else
{
lean_object* v_reuseFailAlloc_3549_; 
v_reuseFailAlloc_3549_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3549_, 0, v___x_3546_);
v___x_3548_ = v_reuseFailAlloc_3549_;
goto v_reusejp_3547_;
}
v_reusejp_3547_:
{
return v___x_3548_;
}
}
else
{
lean_object* v___x_3550_; lean_object* v_a_3551_; lean_object* v___x_3552_; lean_object* v_a_3553_; lean_object* v___x_3555_; uint8_t v_isShared_3556_; uint8_t v_isSharedCheck_3562_; 
lean_del_object(v___x_3541_);
v___x_3550_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3513_, v___y_3508_);
v_a_3551_ = lean_ctor_get(v___x_3550_, 0);
lean_inc(v_a_3551_);
lean_dec_ref(v___x_3550_);
v___x_3552_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3515_, v___y_3508_);
v_a_3553_ = lean_ctor_get(v___x_3552_, 0);
v_isSharedCheck_3562_ = !lean_is_exclusive(v___x_3552_);
if (v_isSharedCheck_3562_ == 0)
{
v___x_3555_ = v___x_3552_;
v_isShared_3556_ = v_isSharedCheck_3562_;
goto v_resetjp_3554_;
}
else
{
lean_inc(v_a_3553_);
lean_dec(v___x_3552_);
v___x_3555_ = lean_box(0);
v_isShared_3556_ = v_isSharedCheck_3562_;
goto v_resetjp_3554_;
}
v_resetjp_3554_:
{
lean_object* v___x_3557_; lean_object* v___x_3558_; lean_object* v___x_3560_; 
v___x_3557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3557_, 0, v_a_3553_);
lean_ctor_set(v___x_3557_, 1, v_a_3539_);
v___x_3558_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3558_, 0, v_a_3551_);
lean_ctor_set(v___x_3558_, 1, v___x_3557_);
if (v_isShared_3556_ == 0)
{
lean_ctor_set(v___x_3555_, 0, v___x_3558_);
v___x_3560_ = v___x_3555_;
goto v_reusejp_3559_;
}
else
{
lean_object* v_reuseFailAlloc_3561_; 
v_reuseFailAlloc_3561_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3561_, 0, v___x_3558_);
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
else
{
lean_object* v_a_3564_; lean_object* v___x_3566_; uint8_t v_isShared_3567_; uint8_t v_isSharedCheck_3571_; 
lean_dec(v_a_3515_);
lean_dec(v_a_3513_);
v_a_3564_ = lean_ctor_get(v___x_3538_, 0);
v_isSharedCheck_3571_ = !lean_is_exclusive(v___x_3538_);
if (v_isSharedCheck_3571_ == 0)
{
v___x_3566_ = v___x_3538_;
v_isShared_3567_ = v_isSharedCheck_3571_;
goto v_resetjp_3565_;
}
else
{
lean_inc(v_a_3564_);
lean_dec(v___x_3538_);
v___x_3566_ = lean_box(0);
v_isShared_3567_ = v_isSharedCheck_3571_;
goto v_resetjp_3565_;
}
v_resetjp_3565_:
{
lean_object* v___x_3569_; 
if (v_isShared_3567_ == 0)
{
v___x_3569_ = v___x_3566_;
goto v_reusejp_3568_;
}
else
{
lean_object* v_reuseFailAlloc_3570_; 
v_reuseFailAlloc_3570_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3570_, 0, v_a_3564_);
v___x_3569_ = v_reuseFailAlloc_3570_;
goto v_reusejp_3568_;
}
v_reusejp_3568_:
{
return v___x_3569_;
}
}
}
}
}
}
else
{
lean_object* v_a_3574_; lean_object* v___x_3576_; uint8_t v_isShared_3577_; uint8_t v_isSharedCheck_3581_; 
lean_dec(v_a_3513_);
lean_dec_ref(v___y_3507_);
lean_dec_ref(v_e_3505_);
lean_dec(v___x_3504_);
v_a_3574_ = lean_ctor_get(v___x_3514_, 0);
v_isSharedCheck_3581_ = !lean_is_exclusive(v___x_3514_);
if (v_isSharedCheck_3581_ == 0)
{
v___x_3576_ = v___x_3514_;
v_isShared_3577_ = v_isSharedCheck_3581_;
goto v_resetjp_3575_;
}
else
{
lean_inc(v_a_3574_);
lean_dec(v___x_3514_);
v___x_3576_ = lean_box(0);
v_isShared_3577_ = v_isSharedCheck_3581_;
goto v_resetjp_3575_;
}
v_resetjp_3575_:
{
lean_object* v___x_3579_; 
if (v_isShared_3577_ == 0)
{
v___x_3579_ = v___x_3576_;
goto v_reusejp_3578_;
}
else
{
lean_object* v_reuseFailAlloc_3580_; 
v_reuseFailAlloc_3580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3580_, 0, v_a_3574_);
v___x_3579_ = v_reuseFailAlloc_3580_;
goto v_reusejp_3578_;
}
v_reusejp_3578_:
{
return v___x_3579_;
}
}
}
}
else
{
lean_object* v_a_3582_; lean_object* v___x_3584_; uint8_t v_isShared_3585_; uint8_t v_isSharedCheck_3589_; 
lean_dec_ref(v___y_3507_);
lean_dec_ref(v_e_3505_);
lean_dec(v___x_3504_);
lean_dec(v___x_3503_);
lean_dec(v___x_3501_);
v_a_3582_ = lean_ctor_get(v___x_3512_, 0);
v_isSharedCheck_3589_ = !lean_is_exclusive(v___x_3512_);
if (v_isSharedCheck_3589_ == 0)
{
v___x_3584_ = v___x_3512_;
v_isShared_3585_ = v_isSharedCheck_3589_;
goto v_resetjp_3583_;
}
else
{
lean_inc(v_a_3582_);
lean_dec(v___x_3512_);
v___x_3584_ = lean_box(0);
v_isShared_3585_ = v_isSharedCheck_3589_;
goto v_resetjp_3583_;
}
v_resetjp_3583_:
{
lean_object* v___x_3587_; 
if (v_isShared_3585_ == 0)
{
v___x_3587_ = v___x_3584_;
goto v_reusejp_3586_;
}
else
{
lean_object* v_reuseFailAlloc_3588_; 
v_reuseFailAlloc_3588_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3588_, 0, v_a_3582_);
v___x_3587_ = v_reuseFailAlloc_3588_;
goto v_reusejp_3586_;
}
v_reusejp_3586_:
{
return v___x_3587_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1___boxed(lean_object* v___x_3590_, lean_object* v___x_3591_, lean_object* v___x_3592_, lean_object* v___x_3593_, lean_object* v_e_3594_, lean_object* v___x_3595_, lean_object* v___y_3596_, lean_object* v___y_3597_, lean_object* v___y_3598_, lean_object* v___y_3599_, lean_object* v___y_3600_){
_start:
{
uint8_t v___x_3450__boxed_3601_; uint8_t v___x_3453__boxed_3602_; lean_object* v_res_3603_; 
v___x_3450__boxed_3601_ = lean_unbox(v___x_3591_);
v___x_3453__boxed_3602_ = lean_unbox(v___x_3595_);
v_res_3603_ = lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1(v___x_3590_, v___x_3450__boxed_3601_, v___x_3592_, v___x_3593_, v_e_3594_, v___x_3453__boxed_3602_, v___y_3596_, v___y_3597_, v___y_3598_, v___y_3599_);
lean_dec(v___y_3599_);
lean_dec_ref(v___y_3598_);
lean_dec(v___y_3597_);
return v_res_3603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2(uint8_t v___x_3604_, lean_object* v___x_3605_, lean_object* v_e_3606_, uint8_t v___x_3607_, lean_object* v___y_3608_, lean_object* v___y_3609_, lean_object* v___y_3610_, lean_object* v___y_3611_){
_start:
{
lean_object* v_keyedConfig_3613_; uint8_t v_trackZetaDelta_3614_; lean_object* v_zetaDeltaSet_3615_; lean_object* v_lctx_3616_; lean_object* v_localInstances_3617_; lean_object* v_defEqCtx_x3f_3618_; lean_object* v_synthPendingDepth_3619_; lean_object* v_customCanUnfoldPredicate_x3f_3620_; uint8_t v_univApprox_3621_; uint8_t v_inTypeClassResolution_3622_; uint8_t v_cacheInferType_3623_; lean_object* v___x_3625_; uint8_t v_isShared_3626_; uint8_t v_isSharedCheck_3643_; 
v_keyedConfig_3613_ = lean_ctor_get(v___y_3608_, 0);
v_trackZetaDelta_3614_ = lean_ctor_get_uint8(v___y_3608_, sizeof(void*)*7);
v_zetaDeltaSet_3615_ = lean_ctor_get(v___y_3608_, 1);
v_lctx_3616_ = lean_ctor_get(v___y_3608_, 2);
v_localInstances_3617_ = lean_ctor_get(v___y_3608_, 3);
v_defEqCtx_x3f_3618_ = lean_ctor_get(v___y_3608_, 4);
v_synthPendingDepth_3619_ = lean_ctor_get(v___y_3608_, 5);
v_customCanUnfoldPredicate_x3f_3620_ = lean_ctor_get(v___y_3608_, 6);
v_univApprox_3621_ = lean_ctor_get_uint8(v___y_3608_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3622_ = lean_ctor_get_uint8(v___y_3608_, sizeof(void*)*7 + 2);
v_cacheInferType_3623_ = lean_ctor_get_uint8(v___y_3608_, sizeof(void*)*7 + 3);
v_isSharedCheck_3643_ = !lean_is_exclusive(v___y_3608_);
if (v_isSharedCheck_3643_ == 0)
{
v___x_3625_ = v___y_3608_;
v_isShared_3626_ = v_isSharedCheck_3643_;
goto v_resetjp_3624_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3620_);
lean_inc(v_synthPendingDepth_3619_);
lean_inc(v_defEqCtx_x3f_3618_);
lean_inc(v_localInstances_3617_);
lean_inc(v_lctx_3616_);
lean_inc(v_zetaDeltaSet_3615_);
lean_inc(v_keyedConfig_3613_);
lean_dec(v___y_3608_);
v___x_3625_ = lean_box(0);
v_isShared_3626_ = v_isSharedCheck_3643_;
goto v_resetjp_3624_;
}
v_resetjp_3624_:
{
lean_object* v___x_3627_; lean_object* v___x_3629_; 
v___x_3627_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3604_, v_keyedConfig_3613_);
if (v_isShared_3626_ == 0)
{
lean_ctor_set(v___x_3625_, 0, v___x_3627_);
v___x_3629_ = v___x_3625_;
goto v_reusejp_3628_;
}
else
{
lean_object* v_reuseFailAlloc_3642_; 
v_reuseFailAlloc_3642_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3642_, 0, v___x_3627_);
lean_ctor_set(v_reuseFailAlloc_3642_, 1, v_zetaDeltaSet_3615_);
lean_ctor_set(v_reuseFailAlloc_3642_, 2, v_lctx_3616_);
lean_ctor_set(v_reuseFailAlloc_3642_, 3, v_localInstances_3617_);
lean_ctor_set(v_reuseFailAlloc_3642_, 4, v_defEqCtx_x3f_3618_);
lean_ctor_set(v_reuseFailAlloc_3642_, 5, v_synthPendingDepth_3619_);
lean_ctor_set(v_reuseFailAlloc_3642_, 6, v_customCanUnfoldPredicate_x3f_3620_);
lean_ctor_set_uint8(v_reuseFailAlloc_3642_, sizeof(void*)*7, v_trackZetaDelta_3614_);
lean_ctor_set_uint8(v_reuseFailAlloc_3642_, sizeof(void*)*7 + 1, v_univApprox_3621_);
lean_ctor_set_uint8(v_reuseFailAlloc_3642_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3622_);
lean_ctor_set_uint8(v_reuseFailAlloc_3642_, sizeof(void*)*7 + 3, v_cacheInferType_3623_);
v___x_3629_ = v_reuseFailAlloc_3642_;
goto v_reusejp_3628_;
}
v_reusejp_3628_:
{
lean_object* v___x_3630_; 
v___x_3630_ = l_Lean_Meta_isExprDefEq(v___x_3605_, v_e_3606_, v___x_3629_, v___y_3609_, v___y_3610_, v___y_3611_);
lean_dec_ref(v___x_3629_);
if (lean_obj_tag(v___x_3630_) == 0)
{
lean_object* v_a_3631_; uint8_t v___x_3632_; 
v_a_3631_ = lean_ctor_get(v___x_3630_, 0);
lean_inc(v_a_3631_);
v___x_3632_ = lean_unbox(v_a_3631_);
lean_dec(v_a_3631_);
if (v___x_3632_ == 0)
{
lean_object* v___x_3634_; uint8_t v_isShared_3635_; uint8_t v_isSharedCheck_3640_; 
v_isSharedCheck_3640_ = !lean_is_exclusive(v___x_3630_);
if (v_isSharedCheck_3640_ == 0)
{
lean_object* v_unused_3641_; 
v_unused_3641_ = lean_ctor_get(v___x_3630_, 0);
lean_dec(v_unused_3641_);
v___x_3634_ = v___x_3630_;
v_isShared_3635_ = v_isSharedCheck_3640_;
goto v_resetjp_3633_;
}
else
{
lean_dec(v___x_3630_);
v___x_3634_ = lean_box(0);
v_isShared_3635_ = v_isSharedCheck_3640_;
goto v_resetjp_3633_;
}
v_resetjp_3633_:
{
lean_object* v___x_3636_; lean_object* v___x_3638_; 
v___x_3636_ = lean_box(v___x_3607_);
if (v_isShared_3635_ == 0)
{
lean_ctor_set(v___x_3634_, 0, v___x_3636_);
v___x_3638_ = v___x_3634_;
goto v_reusejp_3637_;
}
else
{
lean_object* v_reuseFailAlloc_3639_; 
v_reuseFailAlloc_3639_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3639_, 0, v___x_3636_);
v___x_3638_ = v_reuseFailAlloc_3639_;
goto v_reusejp_3637_;
}
v_reusejp_3637_:
{
return v___x_3638_;
}
}
}
else
{
return v___x_3630_;
}
}
else
{
return v___x_3630_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2___boxed(lean_object* v___x_3644_, lean_object* v___x_3645_, lean_object* v_e_3646_, lean_object* v___x_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_, lean_object* v___y_3651_, lean_object* v___y_3652_){
_start:
{
uint8_t v___x_3616__boxed_3653_; uint8_t v___x_3618__boxed_3654_; lean_object* v_res_3655_; 
v___x_3616__boxed_3653_ = lean_unbox(v___x_3644_);
v___x_3618__boxed_3654_ = lean_unbox(v___x_3647_);
v_res_3655_ = lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2(v___x_3616__boxed_3653_, v___x_3645_, v_e_3646_, v___x_3618__boxed_3654_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_);
lean_dec(v___y_3651_);
lean_dec_ref(v___y_3650_);
lean_dec(v___y_3649_);
return v_res_3655_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2(void){
_start:
{
lean_object* v___x_3659_; lean_object* v___x_3660_; lean_object* v___x_3661_; 
v___x_3659_ = lean_box(0);
v___x_3660_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__1));
v___x_3661_ = l_Lean_Expr_const___override(v___x_3660_, v___x_3659_);
return v___x_3661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher(lean_object* v_e_3662_, lean_object* v_a_3663_, lean_object* v_a_3664_, lean_object* v_a_3665_, lean_object* v_a_3666_){
_start:
{
uint8_t v___x_3668_; lean_object* v___x_3669_; lean_object* v___x_3670_; uint8_t v___x_3671_; lean_object* v___x_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; lean_object* v___f_3675_; lean_object* v___x_3676_; 
v___x_3668_ = 0;
v___x_3669_ = lean_box(0);
v___x_3670_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1);
v___x_3671_ = 0;
v___x_3672_ = lean_box(0);
v___x_3673_ = lean_box(v___x_3671_);
v___x_3674_ = lean_box(v___x_3668_);
lean_inc_ref(v_e_3662_);
v___f_3675_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0___boxed), 11, 6);
lean_closure_set(v___f_3675_, 0, v___x_3670_);
lean_closure_set(v___f_3675_, 1, v___x_3673_);
lean_closure_set(v___f_3675_, 2, v___x_3672_);
lean_closure_set(v___f_3675_, 3, v___x_3669_);
lean_closure_set(v___f_3675_, 4, v_e_3662_);
lean_closure_set(v___f_3675_, 5, v___x_3674_);
v___x_3676_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_3675_, v___x_3668_, v_a_3663_, v_a_3664_, v_a_3665_, v_a_3666_);
if (lean_obj_tag(v___x_3676_) == 0)
{
lean_object* v_a_3677_; lean_object* v___x_3679_; uint8_t v_isShared_3680_; uint8_t v_isSharedCheck_3727_; 
v_a_3677_ = lean_ctor_get(v___x_3676_, 0);
v_isSharedCheck_3727_ = !lean_is_exclusive(v___x_3676_);
if (v_isSharedCheck_3727_ == 0)
{
v___x_3679_ = v___x_3676_;
v_isShared_3680_ = v_isSharedCheck_3727_;
goto v_resetjp_3678_;
}
else
{
lean_inc(v_a_3677_);
lean_dec(v___x_3676_);
v___x_3679_ = lean_box(0);
v_isShared_3680_ = v_isSharedCheck_3727_;
goto v_resetjp_3678_;
}
v_resetjp_3678_:
{
lean_object* v_snd_3681_; lean_object* v_snd_3682_; uint8_t v___x_3683_; 
v_snd_3681_ = lean_ctor_get(v_a_3677_, 1);
lean_inc(v_snd_3681_);
lean_dec(v_a_3677_);
v_snd_3682_ = lean_ctor_get(v_snd_3681_, 1);
lean_inc(v_snd_3682_);
lean_dec(v_snd_3681_);
v___x_3683_ = lean_unbox(v_snd_3682_);
if (v___x_3683_ == 0)
{
lean_object* v___x_3684_; lean_object* v___x_3685_; lean_object* v___f_3686_; lean_object* v___x_3687_; 
lean_dec(v_snd_3682_);
lean_del_object(v___x_3679_);
v___x_3684_ = lean_box(v___x_3671_);
v___x_3685_ = lean_box(v___x_3668_);
lean_inc_ref(v_e_3662_);
v___f_3686_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1___boxed), 11, 6);
lean_closure_set(v___f_3686_, 0, v___x_3670_);
lean_closure_set(v___f_3686_, 1, v___x_3684_);
lean_closure_set(v___f_3686_, 2, v___x_3672_);
lean_closure_set(v___f_3686_, 3, v___x_3669_);
lean_closure_set(v___f_3686_, 4, v_e_3662_);
lean_closure_set(v___f_3686_, 5, v___x_3685_);
v___x_3687_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_3686_, v___x_3668_, v_a_3663_, v_a_3664_, v_a_3665_, v_a_3666_);
if (lean_obj_tag(v___x_3687_) == 0)
{
lean_object* v_a_3688_; lean_object* v___x_3690_; uint8_t v_isShared_3691_; uint8_t v_isSharedCheck_3715_; 
v_a_3688_ = lean_ctor_get(v___x_3687_, 0);
v_isSharedCheck_3715_ = !lean_is_exclusive(v___x_3687_);
if (v_isSharedCheck_3715_ == 0)
{
v___x_3690_ = v___x_3687_;
v_isShared_3691_ = v_isSharedCheck_3715_;
goto v_resetjp_3689_;
}
else
{
lean_inc(v_a_3688_);
lean_dec(v___x_3687_);
v___x_3690_ = lean_box(0);
v_isShared_3691_ = v_isSharedCheck_3715_;
goto v_resetjp_3689_;
}
v_resetjp_3689_:
{
lean_object* v_snd_3692_; lean_object* v_snd_3693_; uint8_t v___x_3694_; 
v_snd_3692_ = lean_ctor_get(v_a_3688_, 1);
lean_inc(v_snd_3692_);
lean_dec(v_a_3688_);
v_snd_3693_ = lean_ctor_get(v_snd_3692_, 1);
lean_inc(v_snd_3693_);
lean_dec(v_snd_3692_);
v___x_3694_ = lean_unbox(v_snd_3693_);
if (v___x_3694_ == 0)
{
lean_object* v___x_3695_; uint8_t v___x_3696_; lean_object* v___x_3697_; lean_object* v___x_3698_; lean_object* v___f_3699_; lean_object* v___x_3700_; 
lean_dec(v_snd_3693_);
lean_del_object(v___x_3690_);
v___x_3695_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2);
v___x_3696_ = 2;
v___x_3697_ = lean_box(v___x_3696_);
v___x_3698_ = lean_box(v___x_3668_);
v___f_3699_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2___boxed), 9, 4);
lean_closure_set(v___f_3699_, 0, v___x_3697_);
lean_closure_set(v___f_3699_, 1, v___x_3695_);
lean_closure_set(v___f_3699_, 2, v_e_3662_);
lean_closure_set(v___f_3699_, 3, v___x_3698_);
v___x_3700_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_3699_, v___x_3668_, v_a_3663_, v_a_3664_, v_a_3665_, v_a_3666_);
if (lean_obj_tag(v___x_3700_) == 0)
{
lean_object* v_a_3701_; uint8_t v___x_3702_; 
v_a_3701_ = lean_ctor_get(v___x_3700_, 0);
lean_inc(v_a_3701_);
v___x_3702_ = lean_unbox(v_a_3701_);
lean_dec(v_a_3701_);
if (v___x_3702_ == 0)
{
lean_object* v___x_3704_; uint8_t v_isShared_3705_; uint8_t v_isSharedCheck_3710_; 
v_isSharedCheck_3710_ = !lean_is_exclusive(v___x_3700_);
if (v_isSharedCheck_3710_ == 0)
{
lean_object* v_unused_3711_; 
v_unused_3711_ = lean_ctor_get(v___x_3700_, 0);
lean_dec(v_unused_3711_);
v___x_3704_ = v___x_3700_;
v_isShared_3705_ = v_isSharedCheck_3710_;
goto v_resetjp_3703_;
}
else
{
lean_dec(v___x_3700_);
v___x_3704_ = lean_box(0);
v_isShared_3705_ = v_isSharedCheck_3710_;
goto v_resetjp_3703_;
}
v_resetjp_3703_:
{
lean_object* v___x_3706_; lean_object* v___x_3708_; 
v___x_3706_ = lean_box(v___x_3668_);
if (v_isShared_3705_ == 0)
{
lean_ctor_set(v___x_3704_, 0, v___x_3706_);
v___x_3708_ = v___x_3704_;
goto v_reusejp_3707_;
}
else
{
lean_object* v_reuseFailAlloc_3709_; 
v_reuseFailAlloc_3709_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3709_, 0, v___x_3706_);
v___x_3708_ = v_reuseFailAlloc_3709_;
goto v_reusejp_3707_;
}
v_reusejp_3707_:
{
return v___x_3708_;
}
}
}
else
{
return v___x_3700_;
}
}
else
{
return v___x_3700_;
}
}
else
{
lean_object* v___x_3713_; 
lean_dec_ref(v_e_3662_);
if (v_isShared_3691_ == 0)
{
lean_ctor_set(v___x_3690_, 0, v_snd_3693_);
v___x_3713_ = v___x_3690_;
goto v_reusejp_3712_;
}
else
{
lean_object* v_reuseFailAlloc_3714_; 
v_reuseFailAlloc_3714_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3714_, 0, v_snd_3693_);
v___x_3713_ = v_reuseFailAlloc_3714_;
goto v_reusejp_3712_;
}
v_reusejp_3712_:
{
return v___x_3713_;
}
}
}
}
else
{
lean_object* v_a_3716_; lean_object* v___x_3718_; uint8_t v_isShared_3719_; uint8_t v_isSharedCheck_3723_; 
lean_dec_ref(v_e_3662_);
v_a_3716_ = lean_ctor_get(v___x_3687_, 0);
v_isSharedCheck_3723_ = !lean_is_exclusive(v___x_3687_);
if (v_isSharedCheck_3723_ == 0)
{
v___x_3718_ = v___x_3687_;
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
else
{
lean_inc(v_a_3716_);
lean_dec(v___x_3687_);
v___x_3718_ = lean_box(0);
v_isShared_3719_ = v_isSharedCheck_3723_;
goto v_resetjp_3717_;
}
v_resetjp_3717_:
{
lean_object* v___x_3721_; 
if (v_isShared_3719_ == 0)
{
v___x_3721_ = v___x_3718_;
goto v_reusejp_3720_;
}
else
{
lean_object* v_reuseFailAlloc_3722_; 
v_reuseFailAlloc_3722_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3722_, 0, v_a_3716_);
v___x_3721_ = v_reuseFailAlloc_3722_;
goto v_reusejp_3720_;
}
v_reusejp_3720_:
{
return v___x_3721_;
}
}
}
}
else
{
lean_object* v___x_3725_; 
lean_dec_ref(v_e_3662_);
if (v_isShared_3680_ == 0)
{
lean_ctor_set(v___x_3679_, 0, v_snd_3682_);
v___x_3725_ = v___x_3679_;
goto v_reusejp_3724_;
}
else
{
lean_object* v_reuseFailAlloc_3726_; 
v_reuseFailAlloc_3726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3726_, 0, v_snd_3682_);
v___x_3725_ = v_reuseFailAlloc_3726_;
goto v_reusejp_3724_;
}
v_reusejp_3724_:
{
return v___x_3725_;
}
}
}
}
else
{
lean_object* v_a_3728_; lean_object* v___x_3730_; uint8_t v_isShared_3731_; uint8_t v_isSharedCheck_3735_; 
lean_dec_ref(v_e_3662_);
v_a_3728_ = lean_ctor_get(v___x_3676_, 0);
v_isSharedCheck_3735_ = !lean_is_exclusive(v___x_3676_);
if (v_isSharedCheck_3735_ == 0)
{
v___x_3730_ = v___x_3676_;
v_isShared_3731_ = v_isSharedCheck_3735_;
goto v_resetjp_3729_;
}
else
{
lean_inc(v_a_3728_);
lean_dec(v___x_3676_);
v___x_3730_ = lean_box(0);
v_isShared_3731_ = v_isSharedCheck_3735_;
goto v_resetjp_3729_;
}
v_resetjp_3729_:
{
lean_object* v___x_3733_; 
if (v_isShared_3731_ == 0)
{
v___x_3733_ = v___x_3730_;
goto v_reusejp_3732_;
}
else
{
lean_object* v_reuseFailAlloc_3734_; 
v_reuseFailAlloc_3734_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3734_, 0, v_a_3728_);
v___x_3733_ = v_reuseFailAlloc_3734_;
goto v_reusejp_3732_;
}
v_reusejp_3732_:
{
return v___x_3733_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___boxed(lean_object* v_e_3736_, lean_object* v_a_3737_, lean_object* v_a_3738_, lean_object* v_a_3739_, lean_object* v_a_3740_, lean_object* v_a_3741_){
_start:
{
lean_object* v_res_3742_; 
v_res_3742_ = lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher(v_e_3736_, v_a_3737_, v_a_3738_, v_a_3739_, v_a_3740_);
lean_dec(v_a_3740_);
lean_dec_ref(v_a_3739_);
lean_dec(v_a_3738_);
lean_dec_ref(v_a_3737_);
return v_res_3742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg(lean_object* v_l_3743_, lean_object* v___y_3744_){
_start:
{
lean_object* v___x_3746_; lean_object* v_mctx_3747_; lean_object* v___x_3748_; lean_object* v_fst_3749_; lean_object* v_snd_3750_; lean_object* v___x_3751_; lean_object* v_cache_3752_; lean_object* v_zetaDeltaFVarIds_3753_; lean_object* v_postponed_3754_; lean_object* v_diag_3755_; lean_object* v___x_3757_; uint8_t v_isShared_3758_; uint8_t v_isSharedCheck_3764_; 
v___x_3746_ = lean_st_ref_get(v___y_3744_);
v_mctx_3747_ = lean_ctor_get(v___x_3746_, 0);
lean_inc_ref(v_mctx_3747_);
lean_dec(v___x_3746_);
v___x_3748_ = lean_instantiate_level_mvars(v_mctx_3747_, v_l_3743_);
v_fst_3749_ = lean_ctor_get(v___x_3748_, 0);
lean_inc(v_fst_3749_);
v_snd_3750_ = lean_ctor_get(v___x_3748_, 1);
lean_inc(v_snd_3750_);
lean_dec_ref(v___x_3748_);
v___x_3751_ = lean_st_ref_take(v___y_3744_);
v_cache_3752_ = lean_ctor_get(v___x_3751_, 1);
v_zetaDeltaFVarIds_3753_ = lean_ctor_get(v___x_3751_, 2);
v_postponed_3754_ = lean_ctor_get(v___x_3751_, 3);
v_diag_3755_ = lean_ctor_get(v___x_3751_, 4);
v_isSharedCheck_3764_ = !lean_is_exclusive(v___x_3751_);
if (v_isSharedCheck_3764_ == 0)
{
lean_object* v_unused_3765_; 
v_unused_3765_ = lean_ctor_get(v___x_3751_, 0);
lean_dec(v_unused_3765_);
v___x_3757_ = v___x_3751_;
v_isShared_3758_ = v_isSharedCheck_3764_;
goto v_resetjp_3756_;
}
else
{
lean_inc(v_diag_3755_);
lean_inc(v_postponed_3754_);
lean_inc(v_zetaDeltaFVarIds_3753_);
lean_inc(v_cache_3752_);
lean_dec(v___x_3751_);
v___x_3757_ = lean_box(0);
v_isShared_3758_ = v_isSharedCheck_3764_;
goto v_resetjp_3756_;
}
v_resetjp_3756_:
{
lean_object* v___x_3760_; 
if (v_isShared_3758_ == 0)
{
lean_ctor_set(v___x_3757_, 0, v_fst_3749_);
v___x_3760_ = v___x_3757_;
goto v_reusejp_3759_;
}
else
{
lean_object* v_reuseFailAlloc_3763_; 
v_reuseFailAlloc_3763_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3763_, 0, v_fst_3749_);
lean_ctor_set(v_reuseFailAlloc_3763_, 1, v_cache_3752_);
lean_ctor_set(v_reuseFailAlloc_3763_, 2, v_zetaDeltaFVarIds_3753_);
lean_ctor_set(v_reuseFailAlloc_3763_, 3, v_postponed_3754_);
lean_ctor_set(v_reuseFailAlloc_3763_, 4, v_diag_3755_);
v___x_3760_ = v_reuseFailAlloc_3763_;
goto v_reusejp_3759_;
}
v_reusejp_3759_:
{
lean_object* v___x_3761_; lean_object* v___x_3762_; 
v___x_3761_ = lean_st_ref_set(v___y_3744_, v___x_3760_);
v___x_3762_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3762_, 0, v_snd_3750_);
return v___x_3762_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg___boxed(lean_object* v_l_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_){
_start:
{
lean_object* v_res_3769_; 
v_res_3769_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg(v_l_3766_, v___y_3767_);
lean_dec(v___y_3767_);
return v_res_3769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0(lean_object* v_l_3770_, lean_object* v___y_3771_, lean_object* v___y_3772_, lean_object* v___y_3773_, lean_object* v___y_3774_){
_start:
{
lean_object* v___x_3776_; 
v___x_3776_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg(v_l_3770_, v___y_3772_);
return v___x_3776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___boxed(lean_object* v_l_3777_, lean_object* v___y_3778_, lean_object* v___y_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_, lean_object* v___y_3782_){
_start:
{
lean_object* v_res_3783_; 
v_res_3783_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0(v_l_3777_, v___y_3778_, v___y_3779_, v___y_3780_, v___y_3781_);
lean_dec(v___y_3781_);
lean_dec_ref(v___y_3780_);
lean_dec(v___y_3779_);
lean_dec_ref(v___y_3778_);
return v_res_3783_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1(lean_object* v___x_3784_, uint8_t v___x_3785_, lean_object* v___x_3786_, lean_object* v___x_3787_, lean_object* v_e_3788_, uint8_t v___x_3789_, lean_object* v___y_3790_, lean_object* v___y_3791_, lean_object* v___y_3792_, lean_object* v___y_3793_){
_start:
{
lean_object* v___x_3795_; 
lean_inc(v___x_3786_);
lean_inc(v___x_3784_);
v___x_3795_ = l_Lean_Meta_mkFreshExprMVar(v___x_3784_, v___x_3785_, v___x_3786_, v___y_3790_, v___y_3791_, v___y_3792_, v___y_3793_);
if (lean_obj_tag(v___x_3795_) == 0)
{
lean_object* v_a_3796_; lean_object* v___x_3797_; 
v_a_3796_ = lean_ctor_get(v___x_3795_, 0);
lean_inc(v_a_3796_);
lean_dec_ref_known(v___x_3795_, 1);
v___x_3797_ = l_Lean_Meta_mkFreshExprMVar(v___x_3784_, v___x_3785_, v___x_3786_, v___y_3790_, v___y_3791_, v___y_3792_, v___y_3793_);
if (lean_obj_tag(v___x_3797_) == 0)
{
lean_object* v_a_3798_; lean_object* v_keyedConfig_3799_; uint8_t v_trackZetaDelta_3800_; lean_object* v_zetaDeltaSet_3801_; lean_object* v_lctx_3802_; lean_object* v_localInstances_3803_; lean_object* v_defEqCtx_x3f_3804_; lean_object* v_synthPendingDepth_3805_; lean_object* v_customCanUnfoldPredicate_x3f_3806_; uint8_t v_univApprox_3807_; uint8_t v_inTypeClassResolution_3808_; uint8_t v_cacheInferType_3809_; lean_object* v___x_3811_; uint8_t v_isShared_3812_; uint8_t v_isSharedCheck_3856_; 
v_a_3798_ = lean_ctor_get(v___x_3797_, 0);
lean_inc(v_a_3798_);
lean_dec_ref_known(v___x_3797_, 1);
v_keyedConfig_3799_ = lean_ctor_get(v___y_3790_, 0);
v_trackZetaDelta_3800_ = lean_ctor_get_uint8(v___y_3790_, sizeof(void*)*7);
v_zetaDeltaSet_3801_ = lean_ctor_get(v___y_3790_, 1);
v_lctx_3802_ = lean_ctor_get(v___y_3790_, 2);
v_localInstances_3803_ = lean_ctor_get(v___y_3790_, 3);
v_defEqCtx_x3f_3804_ = lean_ctor_get(v___y_3790_, 4);
v_synthPendingDepth_3805_ = lean_ctor_get(v___y_3790_, 5);
v_customCanUnfoldPredicate_x3f_3806_ = lean_ctor_get(v___y_3790_, 6);
v_univApprox_3807_ = lean_ctor_get_uint8(v___y_3790_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3808_ = lean_ctor_get_uint8(v___y_3790_, sizeof(void*)*7 + 2);
v_cacheInferType_3809_ = lean_ctor_get_uint8(v___y_3790_, sizeof(void*)*7 + 3);
v_isSharedCheck_3856_ = !lean_is_exclusive(v___y_3790_);
if (v_isSharedCheck_3856_ == 0)
{
v___x_3811_ = v___y_3790_;
v_isShared_3812_ = v_isSharedCheck_3856_;
goto v_resetjp_3810_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3806_);
lean_inc(v_synthPendingDepth_3805_);
lean_inc(v_defEqCtx_x3f_3804_);
lean_inc(v_localInstances_3803_);
lean_inc(v_lctx_3802_);
lean_inc(v_zetaDeltaSet_3801_);
lean_inc(v_keyedConfig_3799_);
lean_dec(v___y_3790_);
v___x_3811_ = lean_box(0);
v_isShared_3812_ = v_isSharedCheck_3856_;
goto v_resetjp_3810_;
}
v_resetjp_3810_:
{
lean_object* v___x_3813_; lean_object* v___x_3814_; lean_object* v___x_3815_; lean_object* v___x_3816_; uint8_t v___x_3817_; lean_object* v___x_3818_; lean_object* v___x_3820_; 
v___x_3813_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__4___closed__1));
v___x_3814_ = l_Lean_Expr_const___override(v___x_3813_, v___x_3787_);
lean_inc(v_a_3796_);
v___x_3815_ = l_Lean_Expr_app___override(v___x_3814_, v_a_3796_);
lean_inc(v_a_3798_);
v___x_3816_ = l_Lean_Expr_app___override(v___x_3815_, v_a_3798_);
v___x_3817_ = 2;
v___x_3818_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3817_, v_keyedConfig_3799_);
if (v_isShared_3812_ == 0)
{
lean_ctor_set(v___x_3811_, 0, v___x_3818_);
v___x_3820_ = v___x_3811_;
goto v_reusejp_3819_;
}
else
{
lean_object* v_reuseFailAlloc_3855_; 
v_reuseFailAlloc_3855_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3855_, 0, v___x_3818_);
lean_ctor_set(v_reuseFailAlloc_3855_, 1, v_zetaDeltaSet_3801_);
lean_ctor_set(v_reuseFailAlloc_3855_, 2, v_lctx_3802_);
lean_ctor_set(v_reuseFailAlloc_3855_, 3, v_localInstances_3803_);
lean_ctor_set(v_reuseFailAlloc_3855_, 4, v_defEqCtx_x3f_3804_);
lean_ctor_set(v_reuseFailAlloc_3855_, 5, v_synthPendingDepth_3805_);
lean_ctor_set(v_reuseFailAlloc_3855_, 6, v_customCanUnfoldPredicate_x3f_3806_);
lean_ctor_set_uint8(v_reuseFailAlloc_3855_, sizeof(void*)*7, v_trackZetaDelta_3800_);
lean_ctor_set_uint8(v_reuseFailAlloc_3855_, sizeof(void*)*7 + 1, v_univApprox_3807_);
lean_ctor_set_uint8(v_reuseFailAlloc_3855_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3808_);
lean_ctor_set_uint8(v_reuseFailAlloc_3855_, sizeof(void*)*7 + 3, v_cacheInferType_3809_);
v___x_3820_ = v_reuseFailAlloc_3855_;
goto v_reusejp_3819_;
}
v_reusejp_3819_:
{
lean_object* v___x_3821_; 
v___x_3821_ = l_Lean_Meta_isExprDefEq(v___x_3816_, v_e_3788_, v___x_3820_, v___y_3791_, v___y_3792_, v___y_3793_);
lean_dec_ref(v___x_3820_);
if (lean_obj_tag(v___x_3821_) == 0)
{
lean_object* v_a_3822_; lean_object* v___x_3824_; uint8_t v_isShared_3825_; uint8_t v_isSharedCheck_3846_; 
v_a_3822_ = lean_ctor_get(v___x_3821_, 0);
v_isSharedCheck_3846_ = !lean_is_exclusive(v___x_3821_);
if (v_isSharedCheck_3846_ == 0)
{
v___x_3824_ = v___x_3821_;
v_isShared_3825_ = v_isSharedCheck_3846_;
goto v_resetjp_3823_;
}
else
{
lean_inc(v_a_3822_);
lean_dec(v___x_3821_);
v___x_3824_ = lean_box(0);
v_isShared_3825_ = v_isSharedCheck_3846_;
goto v_resetjp_3823_;
}
v_resetjp_3823_:
{
uint8_t v___x_3826_; 
v___x_3826_ = lean_unbox(v_a_3822_);
if (v___x_3826_ == 0)
{
lean_object* v___x_3827_; lean_object* v___x_3828_; lean_object* v___x_3829_; lean_object* v___x_3831_; 
lean_dec(v_a_3822_);
v___x_3827_ = lean_box(v___x_3789_);
v___x_3828_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3828_, 0, v_a_3798_);
lean_ctor_set(v___x_3828_, 1, v___x_3827_);
v___x_3829_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3829_, 0, v_a_3796_);
lean_ctor_set(v___x_3829_, 1, v___x_3828_);
if (v_isShared_3825_ == 0)
{
lean_ctor_set(v___x_3824_, 0, v___x_3829_);
v___x_3831_ = v___x_3824_;
goto v_reusejp_3830_;
}
else
{
lean_object* v_reuseFailAlloc_3832_; 
v_reuseFailAlloc_3832_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3832_, 0, v___x_3829_);
v___x_3831_ = v_reuseFailAlloc_3832_;
goto v_reusejp_3830_;
}
v_reusejp_3830_:
{
return v___x_3831_;
}
}
else
{
lean_object* v___x_3833_; lean_object* v_a_3834_; lean_object* v___x_3835_; lean_object* v_a_3836_; lean_object* v___x_3838_; uint8_t v_isShared_3839_; uint8_t v_isSharedCheck_3845_; 
lean_del_object(v___x_3824_);
v___x_3833_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3796_, v___y_3791_);
v_a_3834_ = lean_ctor_get(v___x_3833_, 0);
lean_inc(v_a_3834_);
lean_dec_ref(v___x_3833_);
v___x_3835_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3798_, v___y_3791_);
v_a_3836_ = lean_ctor_get(v___x_3835_, 0);
v_isSharedCheck_3845_ = !lean_is_exclusive(v___x_3835_);
if (v_isSharedCheck_3845_ == 0)
{
v___x_3838_ = v___x_3835_;
v_isShared_3839_ = v_isSharedCheck_3845_;
goto v_resetjp_3837_;
}
else
{
lean_inc(v_a_3836_);
lean_dec(v___x_3835_);
v___x_3838_ = lean_box(0);
v_isShared_3839_ = v_isSharedCheck_3845_;
goto v_resetjp_3837_;
}
v_resetjp_3837_:
{
lean_object* v___x_3840_; lean_object* v___x_3841_; lean_object* v___x_3843_; 
v___x_3840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3840_, 0, v_a_3836_);
lean_ctor_set(v___x_3840_, 1, v_a_3822_);
v___x_3841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3841_, 0, v_a_3834_);
lean_ctor_set(v___x_3841_, 1, v___x_3840_);
if (v_isShared_3839_ == 0)
{
lean_ctor_set(v___x_3838_, 0, v___x_3841_);
v___x_3843_ = v___x_3838_;
goto v_reusejp_3842_;
}
else
{
lean_object* v_reuseFailAlloc_3844_; 
v_reuseFailAlloc_3844_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3844_, 0, v___x_3841_);
v___x_3843_ = v_reuseFailAlloc_3844_;
goto v_reusejp_3842_;
}
v_reusejp_3842_:
{
return v___x_3843_;
}
}
}
}
}
else
{
lean_object* v_a_3847_; lean_object* v___x_3849_; uint8_t v_isShared_3850_; uint8_t v_isSharedCheck_3854_; 
lean_dec(v_a_3798_);
lean_dec(v_a_3796_);
v_a_3847_ = lean_ctor_get(v___x_3821_, 0);
v_isSharedCheck_3854_ = !lean_is_exclusive(v___x_3821_);
if (v_isSharedCheck_3854_ == 0)
{
v___x_3849_ = v___x_3821_;
v_isShared_3850_ = v_isSharedCheck_3854_;
goto v_resetjp_3848_;
}
else
{
lean_inc(v_a_3847_);
lean_dec(v___x_3821_);
v___x_3849_ = lean_box(0);
v_isShared_3850_ = v_isSharedCheck_3854_;
goto v_resetjp_3848_;
}
v_resetjp_3848_:
{
lean_object* v___x_3852_; 
if (v_isShared_3850_ == 0)
{
v___x_3852_ = v___x_3849_;
goto v_reusejp_3851_;
}
else
{
lean_object* v_reuseFailAlloc_3853_; 
v_reuseFailAlloc_3853_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3853_, 0, v_a_3847_);
v___x_3852_ = v_reuseFailAlloc_3853_;
goto v_reusejp_3851_;
}
v_reusejp_3851_:
{
return v___x_3852_;
}
}
}
}
}
}
else
{
lean_object* v_a_3857_; lean_object* v___x_3859_; uint8_t v_isShared_3860_; uint8_t v_isSharedCheck_3864_; 
lean_dec(v_a_3796_);
lean_dec_ref(v___y_3790_);
lean_dec_ref(v_e_3788_);
lean_dec(v___x_3787_);
v_a_3857_ = lean_ctor_get(v___x_3797_, 0);
v_isSharedCheck_3864_ = !lean_is_exclusive(v___x_3797_);
if (v_isSharedCheck_3864_ == 0)
{
v___x_3859_ = v___x_3797_;
v_isShared_3860_ = v_isSharedCheck_3864_;
goto v_resetjp_3858_;
}
else
{
lean_inc(v_a_3857_);
lean_dec(v___x_3797_);
v___x_3859_ = lean_box(0);
v_isShared_3860_ = v_isSharedCheck_3864_;
goto v_resetjp_3858_;
}
v_resetjp_3858_:
{
lean_object* v___x_3862_; 
if (v_isShared_3860_ == 0)
{
v___x_3862_ = v___x_3859_;
goto v_reusejp_3861_;
}
else
{
lean_object* v_reuseFailAlloc_3863_; 
v_reuseFailAlloc_3863_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3863_, 0, v_a_3857_);
v___x_3862_ = v_reuseFailAlloc_3863_;
goto v_reusejp_3861_;
}
v_reusejp_3861_:
{
return v___x_3862_;
}
}
}
}
else
{
lean_object* v_a_3865_; lean_object* v___x_3867_; uint8_t v_isShared_3868_; uint8_t v_isSharedCheck_3872_; 
lean_dec_ref(v___y_3790_);
lean_dec_ref(v_e_3788_);
lean_dec(v___x_3787_);
lean_dec(v___x_3786_);
lean_dec(v___x_3784_);
v_a_3865_ = lean_ctor_get(v___x_3795_, 0);
v_isSharedCheck_3872_ = !lean_is_exclusive(v___x_3795_);
if (v_isSharedCheck_3872_ == 0)
{
v___x_3867_ = v___x_3795_;
v_isShared_3868_ = v_isSharedCheck_3872_;
goto v_resetjp_3866_;
}
else
{
lean_inc(v_a_3865_);
lean_dec(v___x_3795_);
v___x_3867_ = lean_box(0);
v_isShared_3868_ = v_isSharedCheck_3872_;
goto v_resetjp_3866_;
}
v_resetjp_3866_:
{
lean_object* v___x_3870_; 
if (v_isShared_3868_ == 0)
{
v___x_3870_ = v___x_3867_;
goto v_reusejp_3869_;
}
else
{
lean_object* v_reuseFailAlloc_3871_; 
v_reuseFailAlloc_3871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3871_, 0, v_a_3865_);
v___x_3870_ = v_reuseFailAlloc_3871_;
goto v_reusejp_3869_;
}
v_reusejp_3869_:
{
return v___x_3870_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1___boxed(lean_object* v___x_3873_, lean_object* v___x_3874_, lean_object* v___x_3875_, lean_object* v___x_3876_, lean_object* v_e_3877_, lean_object* v___x_3878_, lean_object* v___y_3879_, lean_object* v___y_3880_, lean_object* v___y_3881_, lean_object* v___y_3882_, lean_object* v___y_3883_){
_start:
{
uint8_t v___x_5396__boxed_3884_; uint8_t v___x_5399__boxed_3885_; lean_object* v_res_3886_; 
v___x_5396__boxed_3884_ = lean_unbox(v___x_3874_);
v___x_5399__boxed_3885_ = lean_unbox(v___x_3878_);
v_res_3886_ = lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1(v___x_3873_, v___x_5396__boxed_3884_, v___x_3875_, v___x_3876_, v_e_3877_, v___x_5399__boxed_3885_, v___y_3879_, v___y_3880_, v___y_3881_, v___y_3882_);
lean_dec(v___y_3882_);
lean_dec_ref(v___y_3881_);
lean_dec(v___y_3880_);
return v_res_3886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0(lean_object* v___x_3890_, lean_object* v_e_3891_, uint8_t v___x_3892_, lean_object* v___y_3893_, lean_object* v___y_3894_, lean_object* v___y_3895_, lean_object* v___y_3896_){
_start:
{
lean_object* v___x_3898_; 
v___x_3898_ = l_Lean_Meta_mkFreshLevelMVar(v___y_3893_, v___y_3894_, v___y_3895_, v___y_3896_);
if (lean_obj_tag(v___x_3898_) == 0)
{
lean_object* v_a_3899_; lean_object* v___x_3900_; lean_object* v___x_3901_; uint8_t v___x_3902_; lean_object* v___x_3903_; lean_object* v___x_3904_; 
v_a_3899_ = lean_ctor_get(v___x_3898_, 0);
lean_inc_n(v_a_3899_, 2);
lean_dec_ref_known(v___x_3898_, 1);
v___x_3900_ = l_Lean_Expr_sort___override(v_a_3899_);
v___x_3901_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3901_, 0, v___x_3900_);
v___x_3902_ = 0;
v___x_3903_ = lean_box(0);
v___x_3904_ = l_Lean_Meta_mkFreshExprMVar(v___x_3901_, v___x_3902_, v___x_3903_, v___y_3893_, v___y_3894_, v___y_3895_, v___y_3896_);
if (lean_obj_tag(v___x_3904_) == 0)
{
lean_object* v_a_3905_; lean_object* v___x_3906_; uint8_t v___x_3907_; lean_object* v___x_3908_; lean_object* v___x_3909_; lean_object* v___x_3910_; 
v_a_3905_ = lean_ctor_get(v___x_3904_, 0);
lean_inc_n(v_a_3905_, 2);
lean_dec_ref_known(v___x_3904_, 1);
v___x_3906_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__0);
v___x_3907_ = 0;
v___x_3908_ = l_Lean_Expr_forallE___override(v___x_3903_, v_a_3905_, v___x_3906_, v___x_3907_);
v___x_3909_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3909_, 0, v___x_3908_);
v___x_3910_ = l_Lean_Meta_mkFreshExprMVar(v___x_3909_, v___x_3902_, v___x_3903_, v___y_3893_, v___y_3894_, v___y_3895_, v___y_3896_);
if (lean_obj_tag(v___x_3910_) == 0)
{
lean_object* v_a_3911_; lean_object* v_keyedConfig_3912_; uint8_t v_trackZetaDelta_3913_; lean_object* v_zetaDeltaSet_3914_; lean_object* v_lctx_3915_; lean_object* v_localInstances_3916_; lean_object* v_defEqCtx_x3f_3917_; lean_object* v_synthPendingDepth_3918_; lean_object* v_customCanUnfoldPredicate_x3f_3919_; uint8_t v_univApprox_3920_; uint8_t v_inTypeClassResolution_3921_; uint8_t v_cacheInferType_3922_; lean_object* v___x_3924_; uint8_t v_isShared_3925_; uint8_t v_isSharedCheck_3974_; 
v_a_3911_ = lean_ctor_get(v___x_3910_, 0);
lean_inc(v_a_3911_);
lean_dec_ref_known(v___x_3910_, 1);
v_keyedConfig_3912_ = lean_ctor_get(v___y_3893_, 0);
v_trackZetaDelta_3913_ = lean_ctor_get_uint8(v___y_3893_, sizeof(void*)*7);
v_zetaDeltaSet_3914_ = lean_ctor_get(v___y_3893_, 1);
v_lctx_3915_ = lean_ctor_get(v___y_3893_, 2);
v_localInstances_3916_ = lean_ctor_get(v___y_3893_, 3);
v_defEqCtx_x3f_3917_ = lean_ctor_get(v___y_3893_, 4);
v_synthPendingDepth_3918_ = lean_ctor_get(v___y_3893_, 5);
v_customCanUnfoldPredicate_x3f_3919_ = lean_ctor_get(v___y_3893_, 6);
v_univApprox_3920_ = lean_ctor_get_uint8(v___y_3893_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3921_ = lean_ctor_get_uint8(v___y_3893_, sizeof(void*)*7 + 2);
v_cacheInferType_3922_ = lean_ctor_get_uint8(v___y_3893_, sizeof(void*)*7 + 3);
v_isSharedCheck_3974_ = !lean_is_exclusive(v___y_3893_);
if (v_isSharedCheck_3974_ == 0)
{
v___x_3924_ = v___y_3893_;
v_isShared_3925_ = v_isSharedCheck_3974_;
goto v_resetjp_3923_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_3919_);
lean_inc(v_synthPendingDepth_3918_);
lean_inc(v_defEqCtx_x3f_3917_);
lean_inc(v_localInstances_3916_);
lean_inc(v_lctx_3915_);
lean_inc(v_zetaDeltaSet_3914_);
lean_inc(v_keyedConfig_3912_);
lean_dec(v___y_3893_);
v___x_3924_ = lean_box(0);
v_isShared_3925_ = v_isSharedCheck_3974_;
goto v_resetjp_3923_;
}
v_resetjp_3923_:
{
lean_object* v___x_3926_; lean_object* v___x_3927_; lean_object* v___x_3928_; lean_object* v___x_3929_; lean_object* v___x_3930_; uint8_t v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3934_; 
v___x_3926_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___closed__1));
lean_inc(v_a_3899_);
v___x_3927_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_3927_, 0, v_a_3899_);
lean_ctor_set(v___x_3927_, 1, v___x_3890_);
v___x_3928_ = l_Lean_Expr_const___override(v___x_3926_, v___x_3927_);
lean_inc(v_a_3905_);
v___x_3929_ = l_Lean_Expr_app___override(v___x_3928_, v_a_3905_);
lean_inc(v_a_3911_);
v___x_3930_ = l_Lean_Expr_app___override(v___x_3929_, v_a_3911_);
v___x_3931_ = 2;
v___x_3932_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3931_, v_keyedConfig_3912_);
if (v_isShared_3925_ == 0)
{
lean_ctor_set(v___x_3924_, 0, v___x_3932_);
v___x_3934_ = v___x_3924_;
goto v_reusejp_3933_;
}
else
{
lean_object* v_reuseFailAlloc_3973_; 
v_reuseFailAlloc_3973_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_3973_, 0, v___x_3932_);
lean_ctor_set(v_reuseFailAlloc_3973_, 1, v_zetaDeltaSet_3914_);
lean_ctor_set(v_reuseFailAlloc_3973_, 2, v_lctx_3915_);
lean_ctor_set(v_reuseFailAlloc_3973_, 3, v_localInstances_3916_);
lean_ctor_set(v_reuseFailAlloc_3973_, 4, v_defEqCtx_x3f_3917_);
lean_ctor_set(v_reuseFailAlloc_3973_, 5, v_synthPendingDepth_3918_);
lean_ctor_set(v_reuseFailAlloc_3973_, 6, v_customCanUnfoldPredicate_x3f_3919_);
lean_ctor_set_uint8(v_reuseFailAlloc_3973_, sizeof(void*)*7, v_trackZetaDelta_3913_);
lean_ctor_set_uint8(v_reuseFailAlloc_3973_, sizeof(void*)*7 + 1, v_univApprox_3920_);
lean_ctor_set_uint8(v_reuseFailAlloc_3973_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3921_);
lean_ctor_set_uint8(v_reuseFailAlloc_3973_, sizeof(void*)*7 + 3, v_cacheInferType_3922_);
v___x_3934_ = v_reuseFailAlloc_3973_;
goto v_reusejp_3933_;
}
v_reusejp_3933_:
{
lean_object* v___x_3935_; 
v___x_3935_ = l_Lean_Meta_isExprDefEq(v___x_3930_, v_e_3891_, v___x_3934_, v___y_3894_, v___y_3895_, v___y_3896_);
lean_dec_ref(v___x_3934_);
if (lean_obj_tag(v___x_3935_) == 0)
{
lean_object* v_a_3936_; lean_object* v___x_3938_; uint8_t v_isShared_3939_; uint8_t v_isSharedCheck_3964_; 
v_a_3936_ = lean_ctor_get(v___x_3935_, 0);
v_isSharedCheck_3964_ = !lean_is_exclusive(v___x_3935_);
if (v_isSharedCheck_3964_ == 0)
{
v___x_3938_ = v___x_3935_;
v_isShared_3939_ = v_isSharedCheck_3964_;
goto v_resetjp_3937_;
}
else
{
lean_inc(v_a_3936_);
lean_dec(v___x_3935_);
v___x_3938_ = lean_box(0);
v_isShared_3939_ = v_isSharedCheck_3964_;
goto v_resetjp_3937_;
}
v_resetjp_3937_:
{
uint8_t v___x_3940_; 
v___x_3940_ = lean_unbox(v_a_3936_);
if (v___x_3940_ == 0)
{
lean_object* v___x_3941_; lean_object* v___x_3942_; lean_object* v___x_3943_; lean_object* v___x_3944_; lean_object* v___x_3946_; 
lean_dec(v_a_3936_);
v___x_3941_ = lean_box(v___x_3892_);
v___x_3942_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3942_, 0, v_a_3911_);
lean_ctor_set(v___x_3942_, 1, v___x_3941_);
v___x_3943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3943_, 0, v_a_3905_);
lean_ctor_set(v___x_3943_, 1, v___x_3942_);
v___x_3944_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3944_, 0, v_a_3899_);
lean_ctor_set(v___x_3944_, 1, v___x_3943_);
if (v_isShared_3939_ == 0)
{
lean_ctor_set(v___x_3938_, 0, v___x_3944_);
v___x_3946_ = v___x_3938_;
goto v_reusejp_3945_;
}
else
{
lean_object* v_reuseFailAlloc_3947_; 
v_reuseFailAlloc_3947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3947_, 0, v___x_3944_);
v___x_3946_ = v_reuseFailAlloc_3947_;
goto v_reusejp_3945_;
}
v_reusejp_3945_:
{
return v___x_3946_;
}
}
else
{
lean_object* v___x_3948_; lean_object* v_a_3949_; lean_object* v___x_3950_; lean_object* v_a_3951_; lean_object* v___x_3952_; lean_object* v_a_3953_; lean_object* v___x_3955_; uint8_t v_isShared_3956_; uint8_t v_isSharedCheck_3963_; 
lean_del_object(v___x_3938_);
v___x_3948_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Mathlib_Tactic_Tauto_casesMatcher_spec__0___redArg(v_a_3899_, v___y_3894_);
v_a_3949_ = lean_ctor_get(v___x_3948_, 0);
lean_inc(v_a_3949_);
lean_dec_ref(v___x_3948_);
v___x_3950_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3905_, v___y_3894_);
v_a_3951_ = lean_ctor_get(v___x_3950_, 0);
lean_inc(v_a_3951_);
lean_dec_ref(v___x_3950_);
v___x_3952_ = lp_mathlib_Lean_instantiateMVars___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__1___redArg(v_a_3911_, v___y_3894_);
v_a_3953_ = lean_ctor_get(v___x_3952_, 0);
v_isSharedCheck_3963_ = !lean_is_exclusive(v___x_3952_);
if (v_isSharedCheck_3963_ == 0)
{
v___x_3955_ = v___x_3952_;
v_isShared_3956_ = v_isSharedCheck_3963_;
goto v_resetjp_3954_;
}
else
{
lean_inc(v_a_3953_);
lean_dec(v___x_3952_);
v___x_3955_ = lean_box(0);
v_isShared_3956_ = v_isSharedCheck_3963_;
goto v_resetjp_3954_;
}
v_resetjp_3954_:
{
lean_object* v___x_3957_; lean_object* v___x_3958_; lean_object* v___x_3959_; lean_object* v___x_3961_; 
v___x_3957_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3957_, 0, v_a_3953_);
lean_ctor_set(v___x_3957_, 1, v_a_3936_);
v___x_3958_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3958_, 0, v_a_3951_);
lean_ctor_set(v___x_3958_, 1, v___x_3957_);
v___x_3959_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3959_, 0, v_a_3949_);
lean_ctor_set(v___x_3959_, 1, v___x_3958_);
if (v_isShared_3956_ == 0)
{
lean_ctor_set(v___x_3955_, 0, v___x_3959_);
v___x_3961_ = v___x_3955_;
goto v_reusejp_3960_;
}
else
{
lean_object* v_reuseFailAlloc_3962_; 
v_reuseFailAlloc_3962_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3962_, 0, v___x_3959_);
v___x_3961_ = v_reuseFailAlloc_3962_;
goto v_reusejp_3960_;
}
v_reusejp_3960_:
{
return v___x_3961_;
}
}
}
}
}
else
{
lean_object* v_a_3965_; lean_object* v___x_3967_; uint8_t v_isShared_3968_; uint8_t v_isSharedCheck_3972_; 
lean_dec(v_a_3911_);
lean_dec(v_a_3905_);
lean_dec(v_a_3899_);
v_a_3965_ = lean_ctor_get(v___x_3935_, 0);
v_isSharedCheck_3972_ = !lean_is_exclusive(v___x_3935_);
if (v_isSharedCheck_3972_ == 0)
{
v___x_3967_ = v___x_3935_;
v_isShared_3968_ = v_isSharedCheck_3972_;
goto v_resetjp_3966_;
}
else
{
lean_inc(v_a_3965_);
lean_dec(v___x_3935_);
v___x_3967_ = lean_box(0);
v_isShared_3968_ = v_isSharedCheck_3972_;
goto v_resetjp_3966_;
}
v_resetjp_3966_:
{
lean_object* v___x_3970_; 
if (v_isShared_3968_ == 0)
{
v___x_3970_ = v___x_3967_;
goto v_reusejp_3969_;
}
else
{
lean_object* v_reuseFailAlloc_3971_; 
v_reuseFailAlloc_3971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3971_, 0, v_a_3965_);
v___x_3970_ = v_reuseFailAlloc_3971_;
goto v_reusejp_3969_;
}
v_reusejp_3969_:
{
return v___x_3970_;
}
}
}
}
}
}
else
{
lean_object* v_a_3975_; lean_object* v___x_3977_; uint8_t v_isShared_3978_; uint8_t v_isSharedCheck_3982_; 
lean_dec(v_a_3905_);
lean_dec(v_a_3899_);
lean_dec_ref(v___y_3893_);
lean_dec_ref(v_e_3891_);
lean_dec(v___x_3890_);
v_a_3975_ = lean_ctor_get(v___x_3910_, 0);
v_isSharedCheck_3982_ = !lean_is_exclusive(v___x_3910_);
if (v_isSharedCheck_3982_ == 0)
{
v___x_3977_ = v___x_3910_;
v_isShared_3978_ = v_isSharedCheck_3982_;
goto v_resetjp_3976_;
}
else
{
lean_inc(v_a_3975_);
lean_dec(v___x_3910_);
v___x_3977_ = lean_box(0);
v_isShared_3978_ = v_isSharedCheck_3982_;
goto v_resetjp_3976_;
}
v_resetjp_3976_:
{
lean_object* v___x_3980_; 
if (v_isShared_3978_ == 0)
{
v___x_3980_ = v___x_3977_;
goto v_reusejp_3979_;
}
else
{
lean_object* v_reuseFailAlloc_3981_; 
v_reuseFailAlloc_3981_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3981_, 0, v_a_3975_);
v___x_3980_ = v_reuseFailAlloc_3981_;
goto v_reusejp_3979_;
}
v_reusejp_3979_:
{
return v___x_3980_;
}
}
}
}
else
{
lean_object* v_a_3983_; lean_object* v___x_3985_; uint8_t v_isShared_3986_; uint8_t v_isSharedCheck_3990_; 
lean_dec(v_a_3899_);
lean_dec_ref(v___y_3893_);
lean_dec_ref(v_e_3891_);
lean_dec(v___x_3890_);
v_a_3983_ = lean_ctor_get(v___x_3904_, 0);
v_isSharedCheck_3990_ = !lean_is_exclusive(v___x_3904_);
if (v_isSharedCheck_3990_ == 0)
{
v___x_3985_ = v___x_3904_;
v_isShared_3986_ = v_isSharedCheck_3990_;
goto v_resetjp_3984_;
}
else
{
lean_inc(v_a_3983_);
lean_dec(v___x_3904_);
v___x_3985_ = lean_box(0);
v_isShared_3986_ = v_isSharedCheck_3990_;
goto v_resetjp_3984_;
}
v_resetjp_3984_:
{
lean_object* v___x_3988_; 
if (v_isShared_3986_ == 0)
{
v___x_3988_ = v___x_3985_;
goto v_reusejp_3987_;
}
else
{
lean_object* v_reuseFailAlloc_3989_; 
v_reuseFailAlloc_3989_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3989_, 0, v_a_3983_);
v___x_3988_ = v_reuseFailAlloc_3989_;
goto v_reusejp_3987_;
}
v_reusejp_3987_:
{
return v___x_3988_;
}
}
}
}
else
{
lean_object* v_a_3991_; lean_object* v___x_3993_; uint8_t v_isShared_3994_; uint8_t v_isSharedCheck_3998_; 
lean_dec_ref(v___y_3893_);
lean_dec_ref(v_e_3891_);
lean_dec(v___x_3890_);
v_a_3991_ = lean_ctor_get(v___x_3898_, 0);
v_isSharedCheck_3998_ = !lean_is_exclusive(v___x_3898_);
if (v_isSharedCheck_3998_ == 0)
{
v___x_3993_ = v___x_3898_;
v_isShared_3994_ = v_isSharedCheck_3998_;
goto v_resetjp_3992_;
}
else
{
lean_inc(v_a_3991_);
lean_dec(v___x_3898_);
v___x_3993_ = lean_box(0);
v_isShared_3994_ = v_isSharedCheck_3998_;
goto v_resetjp_3992_;
}
v_resetjp_3992_:
{
lean_object* v___x_3996_; 
if (v_isShared_3994_ == 0)
{
v___x_3996_ = v___x_3993_;
goto v_reusejp_3995_;
}
else
{
lean_object* v_reuseFailAlloc_3997_; 
v_reuseFailAlloc_3997_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3997_, 0, v_a_3991_);
v___x_3996_ = v_reuseFailAlloc_3997_;
goto v_reusejp_3995_;
}
v_reusejp_3995_:
{
return v___x_3996_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___boxed(lean_object* v___x_3999_, lean_object* v_e_4000_, lean_object* v___x_4001_, lean_object* v___y_4002_, lean_object* v___y_4003_, lean_object* v___y_4004_, lean_object* v___y_4005_, lean_object* v___y_4006_){
_start:
{
uint8_t v___x_5571__boxed_4007_; lean_object* v_res_4008_; 
v___x_5571__boxed_4007_ = lean_unbox(v___x_4001_);
v_res_4008_ = lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0(v___x_3999_, v_e_4000_, v___x_5571__boxed_4007_, v___y_4002_, v___y_4003_, v___y_4004_, v___y_4005_);
lean_dec(v___y_4005_);
lean_dec_ref(v___y_4004_);
lean_dec(v___y_4003_);
return v_res_4008_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher(lean_object* v_e_4009_, lean_object* v_a_4010_, lean_object* v_a_4011_, lean_object* v_a_4012_, lean_object* v_a_4013_){
_start:
{
uint8_t v___x_4015_; lean_object* v___x_4016_; lean_object* v___x_4017_; uint8_t v___x_4018_; lean_object* v___x_4019_; lean_object* v___x_4020_; lean_object* v___x_4021_; lean_object* v___f_4022_; lean_object* v___x_4023_; 
v___x_4015_ = 0;
v___x_4016_ = lean_box(0);
v___x_4017_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1);
v___x_4018_ = 0;
v___x_4019_ = lean_box(0);
v___x_4020_ = lean_box(v___x_4018_);
v___x_4021_ = lean_box(v___x_4015_);
lean_inc_ref(v_e_4009_);
v___f_4022_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0___boxed), 11, 6);
lean_closure_set(v___f_4022_, 0, v___x_4017_);
lean_closure_set(v___f_4022_, 1, v___x_4020_);
lean_closure_set(v___f_4022_, 2, v___x_4019_);
lean_closure_set(v___f_4022_, 3, v___x_4016_);
lean_closure_set(v___f_4022_, 4, v_e_4009_);
lean_closure_set(v___f_4022_, 5, v___x_4021_);
v___x_4023_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4022_, v___x_4015_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
if (lean_obj_tag(v___x_4023_) == 0)
{
lean_object* v_a_4024_; lean_object* v___x_4026_; uint8_t v_isShared_4027_; uint8_t v_isSharedCheck_4097_; 
v_a_4024_ = lean_ctor_get(v___x_4023_, 0);
v_isSharedCheck_4097_ = !lean_is_exclusive(v___x_4023_);
if (v_isSharedCheck_4097_ == 0)
{
v___x_4026_ = v___x_4023_;
v_isShared_4027_ = v_isSharedCheck_4097_;
goto v_resetjp_4025_;
}
else
{
lean_inc(v_a_4024_);
lean_dec(v___x_4023_);
v___x_4026_ = lean_box(0);
v_isShared_4027_ = v_isSharedCheck_4097_;
goto v_resetjp_4025_;
}
v_resetjp_4025_:
{
lean_object* v_snd_4028_; lean_object* v_snd_4029_; uint8_t v___x_4030_; 
v_snd_4028_ = lean_ctor_get(v_a_4024_, 1);
lean_inc(v_snd_4028_);
lean_dec(v_a_4024_);
v_snd_4029_ = lean_ctor_get(v_snd_4028_, 1);
lean_inc(v_snd_4029_);
lean_dec(v_snd_4028_);
v___x_4030_ = lean_unbox(v_snd_4029_);
if (v___x_4030_ == 0)
{
lean_object* v___x_4031_; lean_object* v___x_4032_; lean_object* v___f_4033_; lean_object* v___x_4034_; 
lean_dec(v_snd_4029_);
lean_del_object(v___x_4026_);
v___x_4031_ = lean_box(v___x_4018_);
v___x_4032_ = lean_box(v___x_4015_);
lean_inc_ref(v_e_4009_);
v___f_4033_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__1___boxed), 11, 6);
lean_closure_set(v___f_4033_, 0, v___x_4017_);
lean_closure_set(v___f_4033_, 1, v___x_4031_);
lean_closure_set(v___f_4033_, 2, v___x_4019_);
lean_closure_set(v___f_4033_, 3, v___x_4016_);
lean_closure_set(v___f_4033_, 4, v_e_4009_);
lean_closure_set(v___f_4033_, 5, v___x_4032_);
v___x_4034_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4033_, v___x_4015_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
if (lean_obj_tag(v___x_4034_) == 0)
{
lean_object* v_a_4035_; lean_object* v___x_4037_; uint8_t v_isShared_4038_; uint8_t v_isSharedCheck_4085_; 
v_a_4035_ = lean_ctor_get(v___x_4034_, 0);
v_isSharedCheck_4085_ = !lean_is_exclusive(v___x_4034_);
if (v_isSharedCheck_4085_ == 0)
{
v___x_4037_ = v___x_4034_;
v_isShared_4038_ = v_isSharedCheck_4085_;
goto v_resetjp_4036_;
}
else
{
lean_inc(v_a_4035_);
lean_dec(v___x_4034_);
v___x_4037_ = lean_box(0);
v_isShared_4038_ = v_isSharedCheck_4085_;
goto v_resetjp_4036_;
}
v_resetjp_4036_:
{
lean_object* v_snd_4039_; lean_object* v_snd_4040_; uint8_t v___x_4041_; 
v_snd_4039_ = lean_ctor_get(v_a_4035_, 1);
lean_inc(v_snd_4039_);
lean_dec(v_a_4035_);
v_snd_4040_ = lean_ctor_get(v_snd_4039_, 1);
lean_inc(v_snd_4040_);
lean_dec(v_snd_4039_);
v___x_4041_ = lean_unbox(v_snd_4040_);
if (v___x_4041_ == 0)
{
lean_object* v___x_4042_; lean_object* v___f_4043_; lean_object* v___x_4044_; 
lean_dec(v_snd_4040_);
lean_del_object(v___x_4037_);
v___x_4042_ = lean_box(v___x_4015_);
lean_inc_ref(v_e_4009_);
v___f_4043_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___boxed), 8, 3);
lean_closure_set(v___f_4043_, 0, v___x_4016_);
lean_closure_set(v___f_4043_, 1, v_e_4009_);
lean_closure_set(v___f_4043_, 2, v___x_4042_);
v___x_4044_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4043_, v___x_4015_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
if (lean_obj_tag(v___x_4044_) == 0)
{
lean_object* v_a_4045_; lean_object* v___x_4047_; uint8_t v_isShared_4048_; uint8_t v_isSharedCheck_4073_; 
v_a_4045_ = lean_ctor_get(v___x_4044_, 0);
v_isSharedCheck_4073_ = !lean_is_exclusive(v___x_4044_);
if (v_isSharedCheck_4073_ == 0)
{
v___x_4047_ = v___x_4044_;
v_isShared_4048_ = v_isSharedCheck_4073_;
goto v_resetjp_4046_;
}
else
{
lean_inc(v_a_4045_);
lean_dec(v___x_4044_);
v___x_4047_ = lean_box(0);
v_isShared_4048_ = v_isSharedCheck_4073_;
goto v_resetjp_4046_;
}
v_resetjp_4046_:
{
lean_object* v_snd_4049_; lean_object* v_snd_4050_; lean_object* v_snd_4051_; uint8_t v___x_4052_; 
v_snd_4049_ = lean_ctor_get(v_a_4045_, 1);
lean_inc(v_snd_4049_);
lean_dec(v_a_4045_);
v_snd_4050_ = lean_ctor_get(v_snd_4049_, 1);
lean_inc(v_snd_4050_);
lean_dec(v_snd_4049_);
v_snd_4051_ = lean_ctor_get(v_snd_4050_, 1);
lean_inc(v_snd_4051_);
lean_dec(v_snd_4050_);
v___x_4052_ = lean_unbox(v_snd_4051_);
if (v___x_4052_ == 0)
{
lean_object* v___x_4053_; uint8_t v___x_4054_; lean_object* v___x_4055_; lean_object* v___x_4056_; lean_object* v___f_4057_; lean_object* v___x_4058_; 
lean_dec(v_snd_4051_);
lean_del_object(v___x_4047_);
v___x_4053_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__10___closed__2);
v___x_4054_ = 2;
v___x_4055_ = lean_box(v___x_4054_);
v___x_4056_ = lean_box(v___x_4015_);
v___f_4057_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2___boxed), 9, 4);
lean_closure_set(v___f_4057_, 0, v___x_4055_);
lean_closure_set(v___f_4057_, 1, v___x_4053_);
lean_closure_set(v___f_4057_, 2, v_e_4009_);
lean_closure_set(v___f_4057_, 3, v___x_4056_);
v___x_4058_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4057_, v___x_4015_, v_a_4010_, v_a_4011_, v_a_4012_, v_a_4013_);
if (lean_obj_tag(v___x_4058_) == 0)
{
lean_object* v_a_4059_; uint8_t v___x_4060_; 
v_a_4059_ = lean_ctor_get(v___x_4058_, 0);
lean_inc(v_a_4059_);
v___x_4060_ = lean_unbox(v_a_4059_);
lean_dec(v_a_4059_);
if (v___x_4060_ == 0)
{
lean_object* v___x_4062_; uint8_t v_isShared_4063_; uint8_t v_isSharedCheck_4068_; 
v_isSharedCheck_4068_ = !lean_is_exclusive(v___x_4058_);
if (v_isSharedCheck_4068_ == 0)
{
lean_object* v_unused_4069_; 
v_unused_4069_ = lean_ctor_get(v___x_4058_, 0);
lean_dec(v_unused_4069_);
v___x_4062_ = v___x_4058_;
v_isShared_4063_ = v_isSharedCheck_4068_;
goto v_resetjp_4061_;
}
else
{
lean_dec(v___x_4058_);
v___x_4062_ = lean_box(0);
v_isShared_4063_ = v_isSharedCheck_4068_;
goto v_resetjp_4061_;
}
v_resetjp_4061_:
{
lean_object* v___x_4064_; lean_object* v___x_4066_; 
v___x_4064_ = lean_box(v___x_4015_);
if (v_isShared_4063_ == 0)
{
lean_ctor_set(v___x_4062_, 0, v___x_4064_);
v___x_4066_ = v___x_4062_;
goto v_reusejp_4065_;
}
else
{
lean_object* v_reuseFailAlloc_4067_; 
v_reuseFailAlloc_4067_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4067_, 0, v___x_4064_);
v___x_4066_ = v_reuseFailAlloc_4067_;
goto v_reusejp_4065_;
}
v_reusejp_4065_:
{
return v___x_4066_;
}
}
}
else
{
return v___x_4058_;
}
}
else
{
return v___x_4058_;
}
}
else
{
lean_object* v___x_4071_; 
lean_dec_ref(v_e_4009_);
if (v_isShared_4048_ == 0)
{
lean_ctor_set(v___x_4047_, 0, v_snd_4051_);
v___x_4071_ = v___x_4047_;
goto v_reusejp_4070_;
}
else
{
lean_object* v_reuseFailAlloc_4072_; 
v_reuseFailAlloc_4072_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4072_, 0, v_snd_4051_);
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
else
{
lean_object* v_a_4074_; lean_object* v___x_4076_; uint8_t v_isShared_4077_; uint8_t v_isSharedCheck_4081_; 
lean_dec_ref(v_e_4009_);
v_a_4074_ = lean_ctor_get(v___x_4044_, 0);
v_isSharedCheck_4081_ = !lean_is_exclusive(v___x_4044_);
if (v_isSharedCheck_4081_ == 0)
{
v___x_4076_ = v___x_4044_;
v_isShared_4077_ = v_isSharedCheck_4081_;
goto v_resetjp_4075_;
}
else
{
lean_inc(v_a_4074_);
lean_dec(v___x_4044_);
v___x_4076_ = lean_box(0);
v_isShared_4077_ = v_isSharedCheck_4081_;
goto v_resetjp_4075_;
}
v_resetjp_4075_:
{
lean_object* v___x_4079_; 
if (v_isShared_4077_ == 0)
{
v___x_4079_ = v___x_4076_;
goto v_reusejp_4078_;
}
else
{
lean_object* v_reuseFailAlloc_4080_; 
v_reuseFailAlloc_4080_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4080_, 0, v_a_4074_);
v___x_4079_ = v_reuseFailAlloc_4080_;
goto v_reusejp_4078_;
}
v_reusejp_4078_:
{
return v___x_4079_;
}
}
}
}
else
{
lean_object* v___x_4083_; 
lean_dec_ref(v_e_4009_);
if (v_isShared_4038_ == 0)
{
lean_ctor_set(v___x_4037_, 0, v_snd_4040_);
v___x_4083_ = v___x_4037_;
goto v_reusejp_4082_;
}
else
{
lean_object* v_reuseFailAlloc_4084_; 
v_reuseFailAlloc_4084_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4084_, 0, v_snd_4040_);
v___x_4083_ = v_reuseFailAlloc_4084_;
goto v_reusejp_4082_;
}
v_reusejp_4082_:
{
return v___x_4083_;
}
}
}
}
else
{
lean_object* v_a_4086_; lean_object* v___x_4088_; uint8_t v_isShared_4089_; uint8_t v_isSharedCheck_4093_; 
lean_dec_ref(v_e_4009_);
v_a_4086_ = lean_ctor_get(v___x_4034_, 0);
v_isSharedCheck_4093_ = !lean_is_exclusive(v___x_4034_);
if (v_isSharedCheck_4093_ == 0)
{
v___x_4088_ = v___x_4034_;
v_isShared_4089_ = v_isSharedCheck_4093_;
goto v_resetjp_4087_;
}
else
{
lean_inc(v_a_4086_);
lean_dec(v___x_4034_);
v___x_4088_ = lean_box(0);
v_isShared_4089_ = v_isSharedCheck_4093_;
goto v_resetjp_4087_;
}
v_resetjp_4087_:
{
lean_object* v___x_4091_; 
if (v_isShared_4089_ == 0)
{
v___x_4091_ = v___x_4088_;
goto v_reusejp_4090_;
}
else
{
lean_object* v_reuseFailAlloc_4092_; 
v_reuseFailAlloc_4092_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4092_, 0, v_a_4086_);
v___x_4091_ = v_reuseFailAlloc_4092_;
goto v_reusejp_4090_;
}
v_reusejp_4090_:
{
return v___x_4091_;
}
}
}
}
else
{
lean_object* v___x_4095_; 
lean_dec_ref(v_e_4009_);
if (v_isShared_4027_ == 0)
{
lean_ctor_set(v___x_4026_, 0, v_snd_4029_);
v___x_4095_ = v___x_4026_;
goto v_reusejp_4094_;
}
else
{
lean_object* v_reuseFailAlloc_4096_; 
v_reuseFailAlloc_4096_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4096_, 0, v_snd_4029_);
v___x_4095_ = v_reuseFailAlloc_4096_;
goto v_reusejp_4094_;
}
v_reusejp_4094_:
{
return v___x_4095_;
}
}
}
}
else
{
lean_object* v_a_4098_; lean_object* v___x_4100_; uint8_t v_isShared_4101_; uint8_t v_isSharedCheck_4105_; 
lean_dec_ref(v_e_4009_);
v_a_4098_ = lean_ctor_get(v___x_4023_, 0);
v_isSharedCheck_4105_ = !lean_is_exclusive(v___x_4023_);
if (v_isSharedCheck_4105_ == 0)
{
v___x_4100_ = v___x_4023_;
v_isShared_4101_ = v_isSharedCheck_4105_;
goto v_resetjp_4099_;
}
else
{
lean_inc(v_a_4098_);
lean_dec(v___x_4023_);
v___x_4100_ = lean_box(0);
v_isShared_4101_ = v_isSharedCheck_4105_;
goto v_resetjp_4099_;
}
v_resetjp_4099_:
{
lean_object* v___x_4103_; 
if (v_isShared_4101_ == 0)
{
v___x_4103_ = v___x_4100_;
goto v_reusejp_4102_;
}
else
{
lean_object* v_reuseFailAlloc_4104_; 
v_reuseFailAlloc_4104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4104_, 0, v_a_4098_);
v___x_4103_ = v_reuseFailAlloc_4104_;
goto v_reusejp_4102_;
}
v_reusejp_4102_:
{
return v___x_4103_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___boxed(lean_object* v_e_4106_, lean_object* v_a_4107_, lean_object* v_a_4108_, lean_object* v_a_4109_, lean_object* v_a_4110_, lean_object* v_a_4111_){
_start:
{
lean_object* v_res_4112_; 
v_res_4112_ = lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher(v_e_4106_, v_a_4107_, v_a_4108_, v_a_4109_, v_a_4110_);
lean_dec(v_a_4110_);
lean_dec_ref(v_a_4109_);
lean_dec(v_a_4108_);
lean_dec_ref(v_a_4107_);
return v_res_4112_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6(void){
_start:
{
lean_object* v___x_4148_; lean_object* v___x_4149_; 
v___x_4148_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__5));
v___x_4149_ = l_String_toRawSubstring_x27(v___x_4148_);
return v___x_4149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1(lean_object* v_x_4167_, lean_object* v_a_4168_, lean_object* v_a_4169_){
_start:
{
lean_object* v___x_4170_; lean_object* v___x_4171_; uint8_t v___x_4172_; 
v___x_4170_ = lean_unsigned_to_nat(0u);
v___x_4171_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1));
lean_inc(v_x_4167_);
v___x_4172_ = l_Lean_Syntax_isOfKind(v_x_4167_, v___x_4171_);
if (v___x_4172_ == 0)
{
lean_object* v___x_4173_; lean_object* v___x_4174_; 
lean_dec(v_x_4167_);
v___x_4173_ = lean_box(1);
v___x_4174_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4174_, 0, v___x_4173_);
lean_ctor_set(v___x_4174_, 1, v_a_4169_);
return v___x_4174_;
}
else
{
lean_object* v_quotContext_4175_; lean_object* v_currMacroScope_4176_; lean_object* v_ref_4177_; lean_object* v___x_4178_; lean_object* v___x_4179_; lean_object* v___x_4180_; uint8_t v___x_4181_; lean_object* v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4187_; lean_object* v___x_4188_; lean_object* v___x_4189_; lean_object* v___x_4190_; lean_object* v___x_4191_; lean_object* v___x_4192_; 
v_quotContext_4175_ = lean_ctor_get(v_a_4168_, 1);
v_currMacroScope_4176_ = lean_ctor_get(v_a_4168_, 2);
v_ref_4177_ = lean_ctor_get(v_a_4168_, 5);
v___x_4178_ = l_Lean_Syntax_getArg(v_x_4167_, v___x_4170_);
v___x_4179_ = lean_unsigned_to_nat(2u);
v___x_4180_ = l_Lean_Syntax_getArg(v_x_4167_, v___x_4179_);
lean_dec(v_x_4167_);
v___x_4181_ = 0;
v___x_4182_ = l_Lean_SourceInfo_fromRef(v_ref_4177_, v___x_4181_);
v___x_4183_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4));
v___x_4184_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6, &lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6_once, _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__6);
v___x_4185_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__7));
lean_inc(v_currMacroScope_4176_);
lean_inc(v_quotContext_4175_);
v___x_4186_ = l_Lean_addMacroScope(v_quotContext_4175_, v___x_4185_, v_currMacroScope_4176_);
v___x_4187_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__11));
lean_inc_n(v___x_4182_, 2);
v___x_4188_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4188_, 0, v___x_4182_);
lean_ctor_set(v___x_4188_, 1, v___x_4184_);
lean_ctor_set(v___x_4188_, 2, v___x_4186_);
lean_ctor_set(v___x_4188_, 3, v___x_4187_);
v___x_4189_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13));
v___x_4190_ = l_Lean_Syntax_node2(v___x_4182_, v___x_4189_, v___x_4178_, v___x_4180_);
v___x_4191_ = l_Lean_Syntax_node2(v___x_4182_, v___x_4183_, v___x_4188_, v___x_4190_);
v___x_4192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4192_, 0, v___x_4191_);
lean_ctor_set(v___x_4192_, 1, v_a_4169_);
return v___x_4192_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___boxed(lean_object* v_x_4193_, lean_object* v_a_4194_, lean_object* v_a_4195_){
_start:
{
lean_object* v_res_4196_; 
v_res_4196_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1(v_x_4193_, v_a_4194_, v_a_4195_);
lean_dec_ref(v_a_4194_);
return v_res_4196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1(lean_object* v_x_4200_, lean_object* v_a_4201_, lean_object* v_a_4202_){
_start:
{
lean_object* v___x_4203_; uint8_t v___x_4204_; 
v___x_4203_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4));
lean_inc(v_x_4200_);
v___x_4204_ = l_Lean_Syntax_isOfKind(v_x_4200_, v___x_4203_);
if (v___x_4204_ == 0)
{
lean_object* v___x_4205_; lean_object* v___x_4206_; 
lean_dec(v_x_4200_);
v___x_4205_ = lean_box(0);
v___x_4206_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4206_, 0, v___x_4205_);
lean_ctor_set(v___x_4206_, 1, v_a_4202_);
return v___x_4206_;
}
else
{
lean_object* v___x_4207_; lean_object* v___x_4208_; lean_object* v___x_4209_; uint8_t v___x_4210_; 
v___x_4207_ = lean_unsigned_to_nat(0u);
v___x_4208_ = l_Lean_Syntax_getArg(v_x_4200_, v___x_4207_);
v___x_4209_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___closed__1));
lean_inc(v___x_4208_);
v___x_4210_ = l_Lean_Syntax_isOfKind(v___x_4208_, v___x_4209_);
if (v___x_4210_ == 0)
{
lean_object* v___x_4211_; lean_object* v___x_4212_; 
lean_dec(v___x_4208_);
lean_dec(v_x_4200_);
v___x_4211_ = lean_box(0);
v___x_4212_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4212_, 0, v___x_4211_);
lean_ctor_set(v___x_4212_, 1, v_a_4202_);
return v___x_4212_;
}
else
{
lean_object* v___x_4213_; lean_object* v___x_4214_; lean_object* v___x_4215_; uint8_t v___x_4216_; 
v___x_4213_ = lean_unsigned_to_nat(1u);
v___x_4214_ = l_Lean_Syntax_getArg(v_x_4200_, v___x_4213_);
lean_dec(v_x_4200_);
v___x_4215_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_4214_);
v___x_4216_ = l_Lean_Syntax_matchesNull(v___x_4214_, v___x_4215_);
if (v___x_4216_ == 0)
{
lean_object* v___x_4217_; lean_object* v___x_4218_; 
lean_dec(v___x_4214_);
lean_dec(v___x_4208_);
v___x_4217_ = lean_box(0);
v___x_4218_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_4218_, 0, v___x_4217_);
lean_ctor_set(v___x_4218_, 1, v_a_4202_);
return v___x_4218_;
}
else
{
lean_object* v___x_4219_; lean_object* v___x_4220_; lean_object* v_ref_4221_; uint8_t v___x_4222_; lean_object* v___x_4223_; lean_object* v___x_4224_; lean_object* v___x_4225_; lean_object* v___x_4226_; lean_object* v___x_4227_; lean_object* v___x_4228_; 
v___x_4219_ = l_Lean_Syntax_getArg(v___x_4214_, v___x_4207_);
v___x_4220_ = l_Lean_Syntax_getArg(v___x_4214_, v___x_4213_);
lean_dec(v___x_4214_);
v_ref_4221_ = l_Lean_replaceRef(v___x_4208_, v_a_4201_);
lean_dec(v___x_4208_);
v___x_4222_ = 0;
v___x_4223_ = l_Lean_SourceInfo_fromRef(v_ref_4221_, v___x_4222_);
lean_dec(v_ref_4221_);
v___x_4224_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__1));
v___x_4225_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__4));
lean_inc(v___x_4223_);
v___x_4226_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4226_, 0, v___x_4223_);
lean_ctor_set(v___x_4226_, 1, v___x_4225_);
v___x_4227_ = l_Lean_Syntax_node3(v___x_4223_, v___x_4224_, v___x_4219_, v___x_4226_, v___x_4220_);
v___x_4228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4228_, 0, v___x_4227_);
lean_ctor_set(v___x_4228_, 1, v_a_4202_);
return v___x_4228_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1___boxed(lean_object* v_x_4229_, lean_object* v_a_4230_, lean_object* v_a_4231_){
_start:
{
lean_object* v_res_4232_; 
v_res_4232_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______unexpand__Lean__Elab__Tactic__andThenOnSubgoals__1(v_x_4229_, v_a_4230_, v_a_4231_);
lean_dec(v_a_4230_);
return v_res_4232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0(lean_object* v___y_4233_, lean_object* v___y_4234_, lean_object* v___y_4235_, lean_object* v___y_4236_, lean_object* v___y_4237_, lean_object* v___y_4238_, lean_object* v___y_4239_, lean_object* v___y_4240_){
_start:
{
lean_object* v_ref_4242_; uint8_t v___x_4243_; lean_object* v___x_4244_; lean_object* v___x_4245_; 
v_ref_4242_ = lean_ctor_get(v___y_4239_, 5);
v___x_4243_ = 0;
v___x_4244_ = l_Lean_SourceInfo_fromRef(v_ref_4242_, v___x_4243_);
v___x_4245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4245_, 0, v___x_4244_);
return v___x_4245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0___boxed(lean_object* v___y_4246_, lean_object* v___y_4247_, lean_object* v___y_4248_, lean_object* v___y_4249_, lean_object* v___y_4250_, lean_object* v___y_4251_, lean_object* v___y_4252_, lean_object* v___y_4253_, lean_object* v___y_4254_){
_start:
{
lean_object* v_res_4255_; 
v_res_4255_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0(v___y_4246_, v___y_4247_, v___y_4248_, v___y_4249_, v___y_4250_, v___y_4251_, v___y_4252_, v___y_4253_);
lean_dec(v___y_4253_);
lean_dec_ref(v___y_4252_);
lean_dec(v___y_4251_);
lean_dec_ref(v___y_4250_);
lean_dec(v___y_4249_);
lean_dec_ref(v___y_4248_);
lean_dec(v___y_4247_);
lean_dec_ref(v___y_4246_);
return v_res_4255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1(lean_object* v_____do__lift_4256_, lean_object* v___y_4257_, lean_object* v___y_4258_, lean_object* v___y_4259_, lean_object* v___y_4260_, lean_object* v___y_4261_, lean_object* v___y_4262_, lean_object* v___y_4263_, lean_object* v___y_4264_){
_start:
{
lean_object* v___x_4266_; lean_object* v___x_4267_; 
v___x_4266_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_4266_, 0, v_____do__lift_4256_);
v___x_4267_ = l_Lean_Elab_Tactic_tryTactic___redArg(v___x_4266_, v___y_4257_, v___y_4258_, v___y_4259_, v___y_4260_, v___y_4261_, v___y_4262_, v___y_4263_, v___y_4264_);
if (lean_obj_tag(v___x_4267_) == 0)
{
lean_object* v___x_4269_; uint8_t v_isShared_4270_; uint8_t v_isSharedCheck_4275_; 
v_isSharedCheck_4275_ = !lean_is_exclusive(v___x_4267_);
if (v_isSharedCheck_4275_ == 0)
{
lean_object* v_unused_4276_; 
v_unused_4276_ = lean_ctor_get(v___x_4267_, 0);
lean_dec(v_unused_4276_);
v___x_4269_ = v___x_4267_;
v_isShared_4270_ = v_isSharedCheck_4275_;
goto v_resetjp_4268_;
}
else
{
lean_dec(v___x_4267_);
v___x_4269_ = lean_box(0);
v_isShared_4270_ = v_isSharedCheck_4275_;
goto v_resetjp_4268_;
}
v_resetjp_4268_:
{
lean_object* v___x_4271_; lean_object* v___x_4273_; 
v___x_4271_ = lean_box(0);
if (v_isShared_4270_ == 0)
{
lean_ctor_set(v___x_4269_, 0, v___x_4271_);
v___x_4273_ = v___x_4269_;
goto v_reusejp_4272_;
}
else
{
lean_object* v_reuseFailAlloc_4274_; 
v_reuseFailAlloc_4274_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4274_, 0, v___x_4271_);
v___x_4273_ = v_reuseFailAlloc_4274_;
goto v_reusejp_4272_;
}
v_reusejp_4272_:
{
return v___x_4273_;
}
}
}
else
{
lean_object* v_a_4277_; lean_object* v___x_4279_; uint8_t v_isShared_4280_; uint8_t v_isSharedCheck_4284_; 
v_a_4277_ = lean_ctor_get(v___x_4267_, 0);
v_isSharedCheck_4284_ = !lean_is_exclusive(v___x_4267_);
if (v_isSharedCheck_4284_ == 0)
{
v___x_4279_ = v___x_4267_;
v_isShared_4280_ = v_isSharedCheck_4284_;
goto v_resetjp_4278_;
}
else
{
lean_inc(v_a_4277_);
lean_dec(v___x_4267_);
v___x_4279_ = lean_box(0);
v_isShared_4280_ = v_isSharedCheck_4284_;
goto v_resetjp_4278_;
}
v_resetjp_4278_:
{
lean_object* v___x_4282_; 
if (v_isShared_4280_ == 0)
{
v___x_4282_ = v___x_4279_;
goto v_reusejp_4281_;
}
else
{
lean_object* v_reuseFailAlloc_4283_; 
v_reuseFailAlloc_4283_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4283_, 0, v_a_4277_);
v___x_4282_ = v_reuseFailAlloc_4283_;
goto v_reusejp_4281_;
}
v_reusejp_4281_:
{
return v___x_4282_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1___boxed(lean_object* v_____do__lift_4285_, lean_object* v___y_4286_, lean_object* v___y_4287_, lean_object* v___y_4288_, lean_object* v___y_4289_, lean_object* v___y_4290_, lean_object* v___y_4291_, lean_object* v___y_4292_, lean_object* v___y_4293_, lean_object* v___y_4294_){
_start:
{
lean_object* v_res_4295_; 
v_res_4295_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__1(v_____do__lift_4285_, v___y_4286_, v___y_4287_, v___y_4288_, v___y_4289_, v___y_4290_, v___y_4291_, v___y_4292_, v___y_4293_);
lean_dec(v___y_4293_);
lean_dec_ref(v___y_4292_);
lean_dec(v___y_4291_);
lean_dec_ref(v___y_4290_);
lean_dec(v___y_4289_);
lean_dec_ref(v___y_4288_);
lean_dec(v___y_4287_);
lean_dec_ref(v___y_4286_);
return v_res_4295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2(lean_object* v___y_4302_, lean_object* v___y_4303_, lean_object* v___y_4304_, lean_object* v___y_4305_, lean_object* v___y_4306_, lean_object* v___y_4307_, lean_object* v___y_4308_, lean_object* v___y_4309_){
_start:
{
lean_object* v_ref_4311_; uint8_t v___x_4312_; lean_object* v___x_4313_; lean_object* v___x_4314_; lean_object* v___x_4315_; lean_object* v___x_4316_; lean_object* v___x_4317_; lean_object* v___x_4318_; lean_object* v___x_4319_; 
v_ref_4311_ = lean_ctor_get(v___y_4308_, 5);
v___x_4312_ = 0;
v___x_4313_ = l_Lean_SourceInfo_fromRef(v_ref_4311_, v___x_4312_);
v___x_4314_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0));
v___x_4315_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1));
lean_inc(v___x_4313_);
v___x_4316_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4316_, 0, v___x_4313_);
lean_ctor_set(v___x_4316_, 1, v___x_4314_);
v___x_4317_ = l_Lean_Syntax_node1(v___x_4313_, v___x_4315_, v___x_4316_);
v___x_4318_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_4318_, 0, v___x_4317_);
v___x_4319_ = l_Lean_Elab_Tactic_tryTactic___redArg(v___x_4318_, v___y_4302_, v___y_4303_, v___y_4304_, v___y_4305_, v___y_4306_, v___y_4307_, v___y_4308_, v___y_4309_);
if (lean_obj_tag(v___x_4319_) == 0)
{
lean_object* v___x_4321_; uint8_t v_isShared_4322_; uint8_t v_isSharedCheck_4327_; 
v_isSharedCheck_4327_ = !lean_is_exclusive(v___x_4319_);
if (v_isSharedCheck_4327_ == 0)
{
lean_object* v_unused_4328_; 
v_unused_4328_ = lean_ctor_get(v___x_4319_, 0);
lean_dec(v_unused_4328_);
v___x_4321_ = v___x_4319_;
v_isShared_4322_ = v_isSharedCheck_4327_;
goto v_resetjp_4320_;
}
else
{
lean_dec(v___x_4319_);
v___x_4321_ = lean_box(0);
v_isShared_4322_ = v_isSharedCheck_4327_;
goto v_resetjp_4320_;
}
v_resetjp_4320_:
{
lean_object* v___x_4323_; lean_object* v___x_4325_; 
v___x_4323_ = lean_box(0);
if (v_isShared_4322_ == 0)
{
lean_ctor_set(v___x_4321_, 0, v___x_4323_);
v___x_4325_ = v___x_4321_;
goto v_reusejp_4324_;
}
else
{
lean_object* v_reuseFailAlloc_4326_; 
v_reuseFailAlloc_4326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4326_, 0, v___x_4323_);
v___x_4325_ = v_reuseFailAlloc_4326_;
goto v_reusejp_4324_;
}
v_reusejp_4324_:
{
return v___x_4325_;
}
}
}
else
{
lean_object* v_a_4329_; lean_object* v___x_4331_; uint8_t v_isShared_4332_; uint8_t v_isSharedCheck_4336_; 
v_a_4329_ = lean_ctor_get(v___x_4319_, 0);
v_isSharedCheck_4336_ = !lean_is_exclusive(v___x_4319_);
if (v_isSharedCheck_4336_ == 0)
{
v___x_4331_ = v___x_4319_;
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
else
{
lean_inc(v_a_4329_);
lean_dec(v___x_4319_);
v___x_4331_ = lean_box(0);
v_isShared_4332_ = v_isSharedCheck_4336_;
goto v_resetjp_4330_;
}
v_resetjp_4330_:
{
lean_object* v___x_4334_; 
if (v_isShared_4332_ == 0)
{
v___x_4334_ = v___x_4331_;
goto v_reusejp_4333_;
}
else
{
lean_object* v_reuseFailAlloc_4335_; 
v_reuseFailAlloc_4335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4335_, 0, v_a_4329_);
v___x_4334_ = v_reuseFailAlloc_4335_;
goto v_reusejp_4333_;
}
v_reusejp_4333_:
{
return v___x_4334_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___boxed(lean_object* v___y_4337_, lean_object* v___y_4338_, lean_object* v___y_4339_, lean_object* v___y_4340_, lean_object* v___y_4341_, lean_object* v___y_4342_, lean_object* v___y_4343_, lean_object* v___y_4344_, lean_object* v___y_4345_){
_start:
{
lean_object* v_res_4346_; 
v_res_4346_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2(v___y_4337_, v___y_4338_, v___y_4339_, v___y_4340_, v___y_4341_, v___y_4342_, v___y_4343_, v___y_4344_);
lean_dec(v___y_4344_);
lean_dec_ref(v___y_4343_);
lean_dec(v___y_4342_);
lean_dec_ref(v___y_4341_);
lean_dec(v___y_4340_);
lean_dec_ref(v___y_4339_);
lean_dec(v___y_4338_);
lean_dec_ref(v___y_4337_);
return v_res_4346_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3(void){
_start:
{
lean_object* v___x_4354_; lean_object* v___x_4355_; 
v___x_4354_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__2));
v___x_4355_ = l_String_toRawSubstring_x27(v___x_4354_);
return v___x_4355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3(lean_object* v___f_4382_, lean_object* v___f_4383_, lean_object* v___y_4384_, lean_object* v___y_4385_, lean_object* v___y_4386_, lean_object* v___y_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_){
_start:
{
lean_object* v___x_4393_; 
lean_inc(v___y_4391_);
lean_inc_ref(v___y_4390_);
lean_inc(v___y_4389_);
lean_inc_ref(v___y_4388_);
lean_inc(v___y_4387_);
lean_inc_ref(v___y_4386_);
lean_inc(v___y_4385_);
lean_inc_ref(v___y_4384_);
v___x_4393_ = lean_apply_9(v___f_4382_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, lean_box(0));
if (lean_obj_tag(v___x_4393_) == 0)
{
lean_object* v_a_4394_; lean_object* v_quotContext_4395_; lean_object* v_currMacroScope_4396_; lean_object* v___x_4397_; lean_object* v___x_4398_; lean_object* v___x_4399_; lean_object* v___x_4400_; lean_object* v___x_4401_; lean_object* v___x_4402_; lean_object* v___x_4403_; lean_object* v___x_4404_; lean_object* v___x_4405_; lean_object* v___x_4406_; lean_object* v___x_4407_; lean_object* v___x_4408_; lean_object* v___x_4409_; lean_object* v___x_4410_; lean_object* v___x_4411_; lean_object* v___x_4412_; lean_object* v___x_4413_; lean_object* v___x_4414_; lean_object* v___x_4415_; lean_object* v___x_4416_; 
v_a_4394_ = lean_ctor_get(v___x_4393_, 0);
lean_inc_n(v_a_4394_, 8);
lean_dec_ref_known(v___x_4393_, 1);
v_quotContext_4395_ = lean_ctor_get(v___y_4390_, 10);
v_currMacroScope_4396_ = lean_ctor_get(v___y_4390_, 11);
v___x_4397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__0));
v___x_4398_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__1));
v___x_4399_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4399_, 0, v_a_4394_);
lean_ctor_set(v___x_4399_, 1, v___x_4397_);
v___x_4400_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__4));
v___x_4401_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3, &lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__3);
v___x_4402_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__6));
lean_inc(v_currMacroScope_4396_);
lean_inc(v_quotContext_4395_);
v___x_4403_ = l_Lean_addMacroScope(v_quotContext_4395_, v___x_4402_, v_currMacroScope_4396_);
v___x_4404_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__11));
v___x_4405_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_4405_, 0, v_a_4394_);
lean_ctor_set(v___x_4405_, 1, v___x_4401_);
lean_ctor_set(v___x_4405_, 2, v___x_4403_);
lean_ctor_set(v___x_4405_, 3, v___x_4404_);
v___x_4406_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13));
v___x_4407_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__13));
v___x_4408_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__14));
v___x_4409_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4409_, 0, v_a_4394_);
lean_ctor_set(v___x_4409_, 1, v___x_4408_);
v___x_4410_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___closed__15));
v___x_4411_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4411_, 0, v_a_4394_);
lean_ctor_set(v___x_4411_, 1, v___x_4410_);
v___x_4412_ = l_Lean_Syntax_node2(v_a_4394_, v___x_4407_, v___x_4409_, v___x_4411_);
v___x_4413_ = l_Lean_Syntax_node1(v_a_4394_, v___x_4406_, v___x_4412_);
v___x_4414_ = l_Lean_Syntax_node2(v_a_4394_, v___x_4400_, v___x_4405_, v___x_4413_);
v___x_4415_ = l_Lean_Syntax_node2(v_a_4394_, v___x_4398_, v___x_4399_, v___x_4414_);
v___x_4416_ = lean_apply_10(v___f_4383_, v___x_4415_, v___y_4384_, v___y_4385_, v___y_4386_, v___y_4387_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, lean_box(0));
return v___x_4416_;
}
else
{
lean_object* v_a_4417_; lean_object* v___x_4419_; uint8_t v_isShared_4420_; uint8_t v_isSharedCheck_4424_; 
lean_dec(v___y_4391_);
lean_dec_ref(v___y_4390_);
lean_dec(v___y_4389_);
lean_dec_ref(v___y_4388_);
lean_dec(v___y_4387_);
lean_dec_ref(v___y_4386_);
lean_dec(v___y_4385_);
lean_dec_ref(v___y_4384_);
lean_dec_ref(v___f_4383_);
v_a_4417_ = lean_ctor_get(v___x_4393_, 0);
v_isSharedCheck_4424_ = !lean_is_exclusive(v___x_4393_);
if (v_isSharedCheck_4424_ == 0)
{
v___x_4419_ = v___x_4393_;
v_isShared_4420_ = v_isSharedCheck_4424_;
goto v_resetjp_4418_;
}
else
{
lean_inc(v_a_4417_);
lean_dec(v___x_4393_);
v___x_4419_ = lean_box(0);
v_isShared_4420_ = v_isSharedCheck_4424_;
goto v_resetjp_4418_;
}
v_resetjp_4418_:
{
lean_object* v___x_4422_; 
if (v_isShared_4420_ == 0)
{
v___x_4422_ = v___x_4419_;
goto v_reusejp_4421_;
}
else
{
lean_object* v_reuseFailAlloc_4423_; 
v_reuseFailAlloc_4423_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4423_, 0, v_a_4417_);
v___x_4422_ = v_reuseFailAlloc_4423_;
goto v_reusejp_4421_;
}
v_reusejp_4421_:
{
return v___x_4422_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3___boxed(lean_object* v___f_4425_, lean_object* v___f_4426_, lean_object* v___y_4427_, lean_object* v___y_4428_, lean_object* v___y_4429_, lean_object* v___y_4430_, lean_object* v___y_4431_, lean_object* v___y_4432_, lean_object* v___y_4433_, lean_object* v___y_4434_, lean_object* v___y_4435_){
_start:
{
lean_object* v_res_4436_; 
v_res_4436_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__3(v___f_4425_, v___f_4426_, v___y_4427_, v___y_4428_, v___y_4429_, v___y_4430_, v___y_4431_, v___y_4432_, v___y_4433_, v___y_4434_);
return v_res_4436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4(lean_object* v___f_4443_, lean_object* v___f_4444_, lean_object* v___y_4445_, lean_object* v___y_4446_, lean_object* v___y_4447_, lean_object* v___y_4448_, lean_object* v___y_4449_, lean_object* v___y_4450_, lean_object* v___y_4451_, lean_object* v___y_4452_){
_start:
{
lean_object* v___x_4454_; 
lean_inc(v___y_4452_);
lean_inc_ref(v___y_4451_);
lean_inc(v___y_4450_);
lean_inc_ref(v___y_4449_);
lean_inc(v___y_4448_);
lean_inc_ref(v___y_4447_);
lean_inc(v___y_4446_);
lean_inc_ref(v___y_4445_);
v___x_4454_ = lean_apply_9(v___f_4443_, v___y_4445_, v___y_4446_, v___y_4447_, v___y_4448_, v___y_4449_, v___y_4450_, v___y_4451_, v___y_4452_, lean_box(0));
if (lean_obj_tag(v___x_4454_) == 0)
{
lean_object* v_a_4455_; lean_object* v___x_4456_; lean_object* v___x_4457_; lean_object* v___x_4458_; lean_object* v___x_4459_; lean_object* v___x_4460_; 
v_a_4455_ = lean_ctor_get(v___x_4454_, 0);
lean_inc_n(v_a_4455_, 2);
lean_dec_ref_known(v___x_4454_, 1);
v___x_4456_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0));
v___x_4457_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1));
v___x_4458_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4458_, 0, v_a_4455_);
lean_ctor_set(v___x_4458_, 1, v___x_4456_);
v___x_4459_ = l_Lean_Syntax_node1(v_a_4455_, v___x_4457_, v___x_4458_);
v___x_4460_ = lean_apply_10(v___f_4444_, v___x_4459_, v___y_4445_, v___y_4446_, v___y_4447_, v___y_4448_, v___y_4449_, v___y_4450_, v___y_4451_, v___y_4452_, lean_box(0));
return v___x_4460_;
}
else
{
lean_object* v_a_4461_; lean_object* v___x_4463_; uint8_t v_isShared_4464_; uint8_t v_isSharedCheck_4468_; 
lean_dec(v___y_4452_);
lean_dec_ref(v___y_4451_);
lean_dec(v___y_4450_);
lean_dec_ref(v___y_4449_);
lean_dec(v___y_4448_);
lean_dec_ref(v___y_4447_);
lean_dec(v___y_4446_);
lean_dec_ref(v___y_4445_);
lean_dec_ref(v___f_4444_);
v_a_4461_ = lean_ctor_get(v___x_4454_, 0);
v_isSharedCheck_4468_ = !lean_is_exclusive(v___x_4454_);
if (v_isSharedCheck_4468_ == 0)
{
v___x_4463_ = v___x_4454_;
v_isShared_4464_ = v_isSharedCheck_4468_;
goto v_resetjp_4462_;
}
else
{
lean_inc(v_a_4461_);
lean_dec(v___x_4454_);
v___x_4463_ = lean_box(0);
v_isShared_4464_ = v_isSharedCheck_4468_;
goto v_resetjp_4462_;
}
v_resetjp_4462_:
{
lean_object* v___x_4466_; 
if (v_isShared_4464_ == 0)
{
v___x_4466_ = v___x_4463_;
goto v_reusejp_4465_;
}
else
{
lean_object* v_reuseFailAlloc_4467_; 
v_reuseFailAlloc_4467_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4467_, 0, v_a_4461_);
v___x_4466_ = v_reuseFailAlloc_4467_;
goto v_reusejp_4465_;
}
v_reusejp_4465_:
{
return v___x_4466_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___boxed(lean_object* v___f_4469_, lean_object* v___f_4470_, lean_object* v___y_4471_, lean_object* v___y_4472_, lean_object* v___y_4473_, lean_object* v___y_4474_, lean_object* v___y_4475_, lean_object* v___y_4476_, lean_object* v___y_4477_, lean_object* v___y_4478_, lean_object* v___y_4479_){
_start:
{
lean_object* v_res_4480_; 
v_res_4480_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4(v___f_4469_, v___f_4470_, v___y_4471_, v___y_4472_, v___y_4473_, v___y_4474_, v___y_4475_, v___y_4476_, v___y_4477_, v___y_4478_);
return v_res_4480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5(lean_object* v___y_4481_, lean_object* v___y_4482_, lean_object* v___y_4483_, lean_object* v___y_4484_, lean_object* v___y_4485_, lean_object* v___y_4486_, lean_object* v___y_4487_, lean_object* v___y_4488_){
_start:
{
lean_object* v___x_4490_; 
v___x_4490_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4482_, v___y_4485_, v___y_4486_, v___y_4487_, v___y_4488_);
if (lean_obj_tag(v___x_4490_) == 0)
{
lean_object* v_a_4491_; lean_object* v___x_4492_; 
v_a_4491_ = lean_ctor_get(v___x_4490_, 0);
lean_inc(v_a_4491_);
lean_dec_ref_known(v___x_4490_, 1);
v___x_4492_ = lp_mathlib_Lean_MVarId_intros_x21(v_a_4491_, v___y_4485_, v___y_4486_, v___y_4487_, v___y_4488_);
if (lean_obj_tag(v___x_4492_) == 0)
{
lean_object* v_a_4493_; lean_object* v_snd_4494_; lean_object* v___x_4496_; uint8_t v_isShared_4497_; uint8_t v_isSharedCheck_4512_; 
v_a_4493_ = lean_ctor_get(v___x_4492_, 0);
lean_inc(v_a_4493_);
lean_dec_ref_known(v___x_4492_, 1);
v_snd_4494_ = lean_ctor_get(v_a_4493_, 1);
v_isSharedCheck_4512_ = !lean_is_exclusive(v_a_4493_);
if (v_isSharedCheck_4512_ == 0)
{
lean_object* v_unused_4513_; 
v_unused_4513_ = lean_ctor_get(v_a_4493_, 0);
lean_dec(v_unused_4513_);
v___x_4496_ = v_a_4493_;
v_isShared_4497_ = v_isSharedCheck_4512_;
goto v_resetjp_4495_;
}
else
{
lean_inc(v_snd_4494_);
lean_dec(v_a_4493_);
v___x_4496_ = lean_box(0);
v_isShared_4497_ = v_isSharedCheck_4512_;
goto v_resetjp_4495_;
}
v_resetjp_4495_:
{
lean_object* v___x_4498_; lean_object* v___x_4500_; 
v___x_4498_ = lean_box(0);
if (v_isShared_4497_ == 0)
{
lean_ctor_set_tag(v___x_4496_, 1);
lean_ctor_set(v___x_4496_, 1, v___x_4498_);
lean_ctor_set(v___x_4496_, 0, v_snd_4494_);
v___x_4500_ = v___x_4496_;
goto v_reusejp_4499_;
}
else
{
lean_object* v_reuseFailAlloc_4511_; 
v_reuseFailAlloc_4511_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4511_, 0, v_snd_4494_);
lean_ctor_set(v_reuseFailAlloc_4511_, 1, v___x_4498_);
v___x_4500_ = v_reuseFailAlloc_4511_;
goto v_reusejp_4499_;
}
v_reusejp_4499_:
{
lean_object* v___x_4501_; 
v___x_4501_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_4500_, v___y_4482_, v___y_4485_, v___y_4486_, v___y_4487_, v___y_4488_);
if (lean_obj_tag(v___x_4501_) == 0)
{
lean_object* v___x_4503_; uint8_t v_isShared_4504_; uint8_t v_isSharedCheck_4509_; 
v_isSharedCheck_4509_ = !lean_is_exclusive(v___x_4501_);
if (v_isSharedCheck_4509_ == 0)
{
lean_object* v_unused_4510_; 
v_unused_4510_ = lean_ctor_get(v___x_4501_, 0);
lean_dec(v_unused_4510_);
v___x_4503_ = v___x_4501_;
v_isShared_4504_ = v_isSharedCheck_4509_;
goto v_resetjp_4502_;
}
else
{
lean_dec(v___x_4501_);
v___x_4503_ = lean_box(0);
v_isShared_4504_ = v_isSharedCheck_4509_;
goto v_resetjp_4502_;
}
v_resetjp_4502_:
{
lean_object* v___x_4505_; lean_object* v___x_4507_; 
v___x_4505_ = lean_box(0);
if (v_isShared_4504_ == 0)
{
lean_ctor_set(v___x_4503_, 0, v___x_4505_);
v___x_4507_ = v___x_4503_;
goto v_reusejp_4506_;
}
else
{
lean_object* v_reuseFailAlloc_4508_; 
v_reuseFailAlloc_4508_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4508_, 0, v___x_4505_);
v___x_4507_ = v_reuseFailAlloc_4508_;
goto v_reusejp_4506_;
}
v_reusejp_4506_:
{
return v___x_4507_;
}
}
}
else
{
return v___x_4501_;
}
}
}
}
else
{
lean_object* v_a_4514_; lean_object* v___x_4516_; uint8_t v_isShared_4517_; uint8_t v_isSharedCheck_4521_; 
v_a_4514_ = lean_ctor_get(v___x_4492_, 0);
v_isSharedCheck_4521_ = !lean_is_exclusive(v___x_4492_);
if (v_isSharedCheck_4521_ == 0)
{
v___x_4516_ = v___x_4492_;
v_isShared_4517_ = v_isSharedCheck_4521_;
goto v_resetjp_4515_;
}
else
{
lean_inc(v_a_4514_);
lean_dec(v___x_4492_);
v___x_4516_ = lean_box(0);
v_isShared_4517_ = v_isSharedCheck_4521_;
goto v_resetjp_4515_;
}
v_resetjp_4515_:
{
lean_object* v___x_4519_; 
if (v_isShared_4517_ == 0)
{
v___x_4519_ = v___x_4516_;
goto v_reusejp_4518_;
}
else
{
lean_object* v_reuseFailAlloc_4520_; 
v_reuseFailAlloc_4520_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4520_, 0, v_a_4514_);
v___x_4519_ = v_reuseFailAlloc_4520_;
goto v_reusejp_4518_;
}
v_reusejp_4518_:
{
return v___x_4519_;
}
}
}
}
else
{
lean_object* v_a_4522_; lean_object* v___x_4524_; uint8_t v_isShared_4525_; uint8_t v_isSharedCheck_4529_; 
v_a_4522_ = lean_ctor_get(v___x_4490_, 0);
v_isSharedCheck_4529_ = !lean_is_exclusive(v___x_4490_);
if (v_isSharedCheck_4529_ == 0)
{
v___x_4524_ = v___x_4490_;
v_isShared_4525_ = v_isSharedCheck_4529_;
goto v_resetjp_4523_;
}
else
{
lean_inc(v_a_4522_);
lean_dec(v___x_4490_);
v___x_4524_ = lean_box(0);
v_isShared_4525_ = v_isSharedCheck_4529_;
goto v_resetjp_4523_;
}
v_resetjp_4523_:
{
lean_object* v___x_4527_; 
if (v_isShared_4525_ == 0)
{
v___x_4527_ = v___x_4524_;
goto v_reusejp_4526_;
}
else
{
lean_object* v_reuseFailAlloc_4528_; 
v_reuseFailAlloc_4528_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4528_, 0, v_a_4522_);
v___x_4527_ = v_reuseFailAlloc_4528_;
goto v_reusejp_4526_;
}
v_reusejp_4526_:
{
return v___x_4527_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5___boxed(lean_object* v___y_4530_, lean_object* v___y_4531_, lean_object* v___y_4532_, lean_object* v___y_4533_, lean_object* v___y_4534_, lean_object* v___y_4535_, lean_object* v___y_4536_, lean_object* v___y_4537_, lean_object* v___y_4538_){
_start:
{
lean_object* v_res_4539_; 
v_res_4539_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__5(v___y_4530_, v___y_4531_, v___y_4532_, v___y_4533_, v___y_4534_, v___y_4535_, v___y_4536_, v___y_4537_);
lean_dec(v___y_4537_);
lean_dec_ref(v___y_4536_);
lean_dec(v___y_4535_);
lean_dec_ref(v___y_4534_);
lean_dec(v___y_4533_);
lean_dec_ref(v___y_4532_);
lean_dec(v___y_4531_);
lean_dec_ref(v___y_4530_);
return v_res_4539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6(lean_object* v___f_4540_, lean_object* v___y_4541_, lean_object* v___y_4542_, lean_object* v___y_4543_, lean_object* v___y_4544_, lean_object* v___y_4545_, lean_object* v___y_4546_, lean_object* v___y_4547_, lean_object* v___y_4548_){
_start:
{
lean_object* v___x_4550_; 
v___x_4550_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_4540_, v___y_4541_, v___y_4542_, v___y_4543_, v___y_4544_, v___y_4545_, v___y_4546_, v___y_4547_, v___y_4548_);
return v___x_4550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6___boxed(lean_object* v___f_4551_, lean_object* v___y_4552_, lean_object* v___y_4553_, lean_object* v___y_4554_, lean_object* v___y_4555_, lean_object* v___y_4556_, lean_object* v___y_4557_, lean_object* v___y_4558_, lean_object* v___y_4559_, lean_object* v___y_4560_){
_start:
{
lean_object* v_res_4561_; 
v_res_4561_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__6(v___f_4551_, v___y_4552_, v___y_4553_, v___y_4554_, v___y_4555_, v___y_4556_, v___y_4557_, v___y_4558_, v___y_4559_);
lean_dec(v___y_4559_);
lean_dec_ref(v___y_4558_);
lean_dec(v___y_4557_);
lean_dec_ref(v___y_4556_);
lean_dec(v___y_4555_);
lean_dec_ref(v___y_4554_);
lean_dec(v___y_4553_);
lean_dec_ref(v___y_4552_);
return v_res_4561_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7(uint8_t v___x_4563_, uint8_t v___x_4564_, lean_object* v___y_4565_, lean_object* v___y_4566_, lean_object* v___y_4567_, lean_object* v___y_4568_, lean_object* v___y_4569_, lean_object* v___y_4570_, lean_object* v___y_4571_, lean_object* v___y_4572_){
_start:
{
lean_object* v___x_4574_; 
v___x_4574_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4566_, v___y_4569_, v___y_4570_, v___y_4571_, v___y_4572_);
if (lean_obj_tag(v___x_4574_) == 0)
{
lean_object* v_a_4575_; lean_object* v___x_4576_; lean_object* v___x_4577_; 
v_a_4575_ = lean_ctor_get(v___x_4574_, 0);
lean_inc(v_a_4575_);
lean_dec_ref_known(v___x_4574_, 1);
v___x_4576_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___closed__0));
v___x_4577_ = lp_mathlib_Mathlib_Tactic_constructorMatching(v_a_4575_, v___x_4576_, v___x_4563_, v___x_4564_, v___y_4569_, v___y_4570_, v___y_4571_, v___y_4572_);
if (lean_obj_tag(v___x_4577_) == 0)
{
lean_object* v_a_4578_; lean_object* v___x_4579_; 
v_a_4578_ = lean_ctor_get(v___x_4577_, 0);
lean_inc(v_a_4578_);
lean_dec_ref_known(v___x_4577_, 1);
v___x_4579_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_4578_, v___y_4566_, v___y_4569_, v___y_4570_, v___y_4571_, v___y_4572_);
if (lean_obj_tag(v___x_4579_) == 0)
{
lean_object* v___x_4581_; uint8_t v_isShared_4582_; uint8_t v_isSharedCheck_4587_; 
v_isSharedCheck_4587_ = !lean_is_exclusive(v___x_4579_);
if (v_isSharedCheck_4587_ == 0)
{
lean_object* v_unused_4588_; 
v_unused_4588_ = lean_ctor_get(v___x_4579_, 0);
lean_dec(v_unused_4588_);
v___x_4581_ = v___x_4579_;
v_isShared_4582_ = v_isSharedCheck_4587_;
goto v_resetjp_4580_;
}
else
{
lean_dec(v___x_4579_);
v___x_4581_ = lean_box(0);
v_isShared_4582_ = v_isSharedCheck_4587_;
goto v_resetjp_4580_;
}
v_resetjp_4580_:
{
lean_object* v___x_4583_; lean_object* v___x_4585_; 
v___x_4583_ = lean_box(0);
if (v_isShared_4582_ == 0)
{
lean_ctor_set(v___x_4581_, 0, v___x_4583_);
v___x_4585_ = v___x_4581_;
goto v_reusejp_4584_;
}
else
{
lean_object* v_reuseFailAlloc_4586_; 
v_reuseFailAlloc_4586_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4586_, 0, v___x_4583_);
v___x_4585_ = v_reuseFailAlloc_4586_;
goto v_reusejp_4584_;
}
v_reusejp_4584_:
{
return v___x_4585_;
}
}
}
else
{
return v___x_4579_;
}
}
else
{
lean_object* v_a_4589_; lean_object* v___x_4591_; uint8_t v_isShared_4592_; uint8_t v_isSharedCheck_4596_; 
v_a_4589_ = lean_ctor_get(v___x_4577_, 0);
v_isSharedCheck_4596_ = !lean_is_exclusive(v___x_4577_);
if (v_isSharedCheck_4596_ == 0)
{
v___x_4591_ = v___x_4577_;
v_isShared_4592_ = v_isSharedCheck_4596_;
goto v_resetjp_4590_;
}
else
{
lean_inc(v_a_4589_);
lean_dec(v___x_4577_);
v___x_4591_ = lean_box(0);
v_isShared_4592_ = v_isSharedCheck_4596_;
goto v_resetjp_4590_;
}
v_resetjp_4590_:
{
lean_object* v___x_4594_; 
if (v_isShared_4592_ == 0)
{
v___x_4594_ = v___x_4591_;
goto v_reusejp_4593_;
}
else
{
lean_object* v_reuseFailAlloc_4595_; 
v_reuseFailAlloc_4595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4595_, 0, v_a_4589_);
v___x_4594_ = v_reuseFailAlloc_4595_;
goto v_reusejp_4593_;
}
v_reusejp_4593_:
{
return v___x_4594_;
}
}
}
}
else
{
lean_object* v_a_4597_; lean_object* v___x_4599_; uint8_t v_isShared_4600_; uint8_t v_isSharedCheck_4604_; 
v_a_4597_ = lean_ctor_get(v___x_4574_, 0);
v_isSharedCheck_4604_ = !lean_is_exclusive(v___x_4574_);
if (v_isSharedCheck_4604_ == 0)
{
v___x_4599_ = v___x_4574_;
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
else
{
lean_inc(v_a_4597_);
lean_dec(v___x_4574_);
v___x_4599_ = lean_box(0);
v_isShared_4600_ = v_isSharedCheck_4604_;
goto v_resetjp_4598_;
}
v_resetjp_4598_:
{
lean_object* v___x_4602_; 
if (v_isShared_4600_ == 0)
{
v___x_4602_ = v___x_4599_;
goto v_reusejp_4601_;
}
else
{
lean_object* v_reuseFailAlloc_4603_; 
v_reuseFailAlloc_4603_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4603_, 0, v_a_4597_);
v___x_4602_ = v_reuseFailAlloc_4603_;
goto v_reusejp_4601_;
}
v_reusejp_4601_:
{
return v___x_4602_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7___boxed(lean_object* v___x_4605_, lean_object* v___x_4606_, lean_object* v___y_4607_, lean_object* v___y_4608_, lean_object* v___y_4609_, lean_object* v___y_4610_, lean_object* v___y_4611_, lean_object* v___y_4612_, lean_object* v___y_4613_, lean_object* v___y_4614_, lean_object* v___y_4615_){
_start:
{
uint8_t v___x_12553__boxed_4616_; uint8_t v___x_12554__boxed_4617_; lean_object* v_res_4618_; 
v___x_12553__boxed_4616_ = lean_unbox(v___x_4605_);
v___x_12554__boxed_4617_ = lean_unbox(v___x_4606_);
v_res_4618_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__7(v___x_12553__boxed_4616_, v___x_12554__boxed_4617_, v___y_4607_, v___y_4608_, v___y_4609_, v___y_4610_, v___y_4611_, v___y_4612_, v___y_4613_, v___y_4614_);
lean_dec(v___y_4614_);
lean_dec_ref(v___y_4613_);
lean_dec(v___y_4612_);
lean_dec_ref(v___y_4611_);
lean_dec(v___y_4610_);
lean_dec_ref(v___y_4609_);
lean_dec(v___y_4608_);
lean_dec_ref(v___y_4607_);
return v_res_4618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9(lean_object* v___x_4619_, uint8_t v___x_4620_, uint8_t v___x_4621_, lean_object* v___y_4622_, lean_object* v___y_4623_, lean_object* v___y_4624_, lean_object* v___y_4625_, lean_object* v___y_4626_, lean_object* v___y_4627_, lean_object* v___y_4628_, lean_object* v___y_4629_){
_start:
{
lean_object* v___x_4631_; 
v___x_4631_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_4623_, v___y_4626_, v___y_4627_, v___y_4628_, v___y_4629_);
if (lean_obj_tag(v___x_4631_) == 0)
{
lean_object* v_a_4632_; lean_object* v___x_4633_; 
v_a_4632_ = lean_ctor_get(v___x_4631_, 0);
lean_inc(v_a_4632_);
lean_dec_ref_known(v___x_4631_, 1);
v___x_4633_ = lp_mathlib_Lean_MVarId_casesMatching(v___x_4619_, v___x_4620_, v___x_4620_, v___x_4621_, v_a_4632_, v___y_4626_, v___y_4627_, v___y_4628_, v___y_4629_);
if (lean_obj_tag(v___x_4633_) == 0)
{
lean_object* v_a_4634_; lean_object* v___x_4635_; 
v_a_4634_ = lean_ctor_get(v___x_4633_, 0);
lean_inc(v_a_4634_);
lean_dec_ref_known(v___x_4633_, 1);
v___x_4635_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_4634_, v___y_4623_, v___y_4626_, v___y_4627_, v___y_4628_, v___y_4629_);
if (lean_obj_tag(v___x_4635_) == 0)
{
lean_object* v___x_4637_; uint8_t v_isShared_4638_; uint8_t v_isSharedCheck_4643_; 
v_isSharedCheck_4643_ = !lean_is_exclusive(v___x_4635_);
if (v_isSharedCheck_4643_ == 0)
{
lean_object* v_unused_4644_; 
v_unused_4644_ = lean_ctor_get(v___x_4635_, 0);
lean_dec(v_unused_4644_);
v___x_4637_ = v___x_4635_;
v_isShared_4638_ = v_isSharedCheck_4643_;
goto v_resetjp_4636_;
}
else
{
lean_dec(v___x_4635_);
v___x_4637_ = lean_box(0);
v_isShared_4638_ = v_isSharedCheck_4643_;
goto v_resetjp_4636_;
}
v_resetjp_4636_:
{
lean_object* v___x_4639_; lean_object* v___x_4641_; 
v___x_4639_ = lean_box(0);
if (v_isShared_4638_ == 0)
{
lean_ctor_set(v___x_4637_, 0, v___x_4639_);
v___x_4641_ = v___x_4637_;
goto v_reusejp_4640_;
}
else
{
lean_object* v_reuseFailAlloc_4642_; 
v_reuseFailAlloc_4642_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4642_, 0, v___x_4639_);
v___x_4641_ = v_reuseFailAlloc_4642_;
goto v_reusejp_4640_;
}
v_reusejp_4640_:
{
return v___x_4641_;
}
}
}
else
{
return v___x_4635_;
}
}
else
{
lean_object* v_a_4645_; lean_object* v___x_4647_; uint8_t v_isShared_4648_; uint8_t v_isSharedCheck_4652_; 
v_a_4645_ = lean_ctor_get(v___x_4633_, 0);
v_isSharedCheck_4652_ = !lean_is_exclusive(v___x_4633_);
if (v_isSharedCheck_4652_ == 0)
{
v___x_4647_ = v___x_4633_;
v_isShared_4648_ = v_isSharedCheck_4652_;
goto v_resetjp_4646_;
}
else
{
lean_inc(v_a_4645_);
lean_dec(v___x_4633_);
v___x_4647_ = lean_box(0);
v_isShared_4648_ = v_isSharedCheck_4652_;
goto v_resetjp_4646_;
}
v_resetjp_4646_:
{
lean_object* v___x_4650_; 
if (v_isShared_4648_ == 0)
{
v___x_4650_ = v___x_4647_;
goto v_reusejp_4649_;
}
else
{
lean_object* v_reuseFailAlloc_4651_; 
v_reuseFailAlloc_4651_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4651_, 0, v_a_4645_);
v___x_4650_ = v_reuseFailAlloc_4651_;
goto v_reusejp_4649_;
}
v_reusejp_4649_:
{
return v___x_4650_;
}
}
}
}
else
{
lean_object* v_a_4653_; lean_object* v___x_4655_; uint8_t v_isShared_4656_; uint8_t v_isSharedCheck_4660_; 
lean_dec_ref(v___x_4619_);
v_a_4653_ = lean_ctor_get(v___x_4631_, 0);
v_isSharedCheck_4660_ = !lean_is_exclusive(v___x_4631_);
if (v_isSharedCheck_4660_ == 0)
{
v___x_4655_ = v___x_4631_;
v_isShared_4656_ = v_isSharedCheck_4660_;
goto v_resetjp_4654_;
}
else
{
lean_inc(v_a_4653_);
lean_dec(v___x_4631_);
v___x_4655_ = lean_box(0);
v_isShared_4656_ = v_isSharedCheck_4660_;
goto v_resetjp_4654_;
}
v_resetjp_4654_:
{
lean_object* v___x_4658_; 
if (v_isShared_4656_ == 0)
{
v___x_4658_ = v___x_4655_;
goto v_reusejp_4657_;
}
else
{
lean_object* v_reuseFailAlloc_4659_; 
v_reuseFailAlloc_4659_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4659_, 0, v_a_4653_);
v___x_4658_ = v_reuseFailAlloc_4659_;
goto v_reusejp_4657_;
}
v_reusejp_4657_:
{
return v___x_4658_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9___boxed(lean_object* v___x_4661_, lean_object* v___x_4662_, lean_object* v___x_4663_, lean_object* v___y_4664_, lean_object* v___y_4665_, lean_object* v___y_4666_, lean_object* v___y_4667_, lean_object* v___y_4668_, lean_object* v___y_4669_, lean_object* v___y_4670_, lean_object* v___y_4671_, lean_object* v___y_4672_){
_start:
{
uint8_t v___x_12652__boxed_4673_; uint8_t v___x_12653__boxed_4674_; lean_object* v_res_4675_; 
v___x_12652__boxed_4673_ = lean_unbox(v___x_4662_);
v___x_12653__boxed_4674_ = lean_unbox(v___x_4663_);
v_res_4675_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__9(v___x_4661_, v___x_12652__boxed_4673_, v___x_12653__boxed_4674_, v___y_4664_, v___y_4665_, v___y_4666_, v___y_4667_, v___y_4668_, v___y_4669_, v___y_4670_, v___y_4671_);
lean_dec(v___y_4671_);
lean_dec_ref(v___y_4670_);
lean_dec(v___y_4669_);
lean_dec_ref(v___y_4668_);
lean_dec(v___y_4667_);
lean_dec_ref(v___y_4666_);
lean_dec(v___y_4665_);
lean_dec_ref(v___y_4664_);
return v_res_4675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg(lean_object* v_msg_4676_, lean_object* v___y_4677_, lean_object* v___y_4678_, lean_object* v___y_4679_, lean_object* v___y_4680_){
_start:
{
lean_object* v_ref_4682_; lean_object* v___x_4683_; lean_object* v_a_4684_; lean_object* v___x_4686_; uint8_t v_isShared_4687_; uint8_t v_isSharedCheck_4692_; 
v_ref_4682_ = lean_ctor_get(v___y_4679_, 5);
v___x_4683_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__3_spec__3(v_msg_4676_, v___y_4677_, v___y_4678_, v___y_4679_, v___y_4680_);
v_a_4684_ = lean_ctor_get(v___x_4683_, 0);
v_isSharedCheck_4692_ = !lean_is_exclusive(v___x_4683_);
if (v_isSharedCheck_4692_ == 0)
{
v___x_4686_ = v___x_4683_;
v_isShared_4687_ = v_isSharedCheck_4692_;
goto v_resetjp_4685_;
}
else
{
lean_inc(v_a_4684_);
lean_dec(v___x_4683_);
v___x_4686_ = lean_box(0);
v_isShared_4687_ = v_isSharedCheck_4692_;
goto v_resetjp_4685_;
}
v_resetjp_4685_:
{
lean_object* v___x_4688_; lean_object* v___x_4690_; 
lean_inc(v_ref_4682_);
v___x_4688_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4688_, 0, v_ref_4682_);
lean_ctor_set(v___x_4688_, 1, v_a_4684_);
if (v_isShared_4687_ == 0)
{
lean_ctor_set_tag(v___x_4686_, 1);
lean_ctor_set(v___x_4686_, 0, v___x_4688_);
v___x_4690_ = v___x_4686_;
goto v_reusejp_4689_;
}
else
{
lean_object* v_reuseFailAlloc_4691_; 
v_reuseFailAlloc_4691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4691_, 0, v___x_4688_);
v___x_4690_ = v_reuseFailAlloc_4691_;
goto v_reusejp_4689_;
}
v_reusejp_4689_:
{
return v___x_4690_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg___boxed(lean_object* v_msg_4693_, lean_object* v___y_4694_, lean_object* v___y_4695_, lean_object* v___y_4696_, lean_object* v___y_4697_, lean_object* v___y_4698_){
_start:
{
lean_object* v_res_4699_; 
v_res_4699_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg(v_msg_4693_, v___y_4694_, v___y_4695_, v___y_4696_, v___y_4697_);
lean_dec(v___y_4697_);
lean_dec_ref(v___y_4696_);
lean_dec(v___y_4695_);
lean_dec_ref(v___y_4694_);
return v_res_4699_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0(lean_object* v_x_4700_, lean_object* v_x_4701_){
_start:
{
if (lean_obj_tag(v_x_4700_) == 0)
{
if (lean_obj_tag(v_x_4701_) == 0)
{
uint8_t v___x_4702_; 
v___x_4702_ = 1;
return v___x_4702_;
}
else
{
uint8_t v___x_4703_; 
v___x_4703_ = 0;
return v___x_4703_;
}
}
else
{
if (lean_obj_tag(v_x_4701_) == 0)
{
uint8_t v___x_4704_; 
v___x_4704_ = 0;
return v___x_4704_;
}
else
{
lean_object* v_head_4705_; lean_object* v_tail_4706_; lean_object* v_head_4707_; lean_object* v_tail_4708_; uint8_t v___x_4709_; 
v_head_4705_ = lean_ctor_get(v_x_4700_, 0);
v_tail_4706_ = lean_ctor_get(v_x_4700_, 1);
v_head_4707_ = lean_ctor_get(v_x_4701_, 0);
v_tail_4708_ = lean_ctor_get(v_x_4701_, 1);
v___x_4709_ = l_Lean_instBEqMVarId_beq(v_head_4705_, v_head_4707_);
if (v___x_4709_ == 0)
{
return v___x_4709_;
}
else
{
v_x_4700_ = v_tail_4706_;
v_x_4701_ = v_tail_4708_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0___boxed(lean_object* v_x_4711_, lean_object* v_x_4712_){
_start:
{
uint8_t v_res_4713_; lean_object* v_r_4714_; 
v_res_4713_ = lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0(v_x_4711_, v_x_4712_);
lean_dec(v_x_4712_);
lean_dec(v_x_4711_);
v_r_4714_ = lean_box(v_res_4713_);
return v_r_4714_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6(void){
_start:
{
lean_object* v___x_4732_; lean_object* v___x_4733_; 
v___x_4732_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__5));
v___x_4733_ = l_Lean_stringToMessageData(v___x_4732_);
return v___x_4733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10(lean_object* v___f_4734_, lean_object* v___f_4735_, lean_object* v___f_4736_, lean_object* v___f_4737_, lean_object* v___y_4738_, lean_object* v___y_4739_, lean_object* v___y_4740_, lean_object* v___y_4741_, lean_object* v___y_4742_, lean_object* v___y_4743_, lean_object* v___y_4744_, lean_object* v___y_4745_){
_start:
{
lean_object* v___x_4750_; 
v___x_4750_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___y_4738_, v___y_4739_, v___y_4740_, v___y_4741_, v___y_4742_, v___y_4743_, v___y_4744_, v___y_4745_);
if (lean_obj_tag(v___x_4750_) == 0)
{
lean_object* v_a_4751_; lean_object* v___x_4752_; lean_object* v___x_4753_; lean_object* v___f_4754_; lean_object* v___f_4755_; lean_object* v___x_4756_; lean_object* v___x_4757_; lean_object* v___x_4758_; lean_object* v___x_4759_; lean_object* v___x_4760_; lean_object* v___x_4761_; lean_object* v___x_4762_; 
v_a_4751_ = lean_ctor_get(v___x_4750_, 0);
lean_inc(v_a_4751_);
lean_dec_ref_known(v___x_4750_, 1);
v___x_4752_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_distribNot___boxed), 9, 0);
lean_inc_ref(v___f_4734_);
v___x_4753_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4753_, 0, v___f_4734_);
lean_closure_set(v___x_4753_, 1, v___x_4752_);
v___f_4754_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__2));
v___f_4755_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__4));
v___x_4756_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4756_, 0, v___x_4753_);
lean_closure_set(v___x_4756_, 1, v___f_4755_);
v___x_4757_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4757_, 0, v___x_4756_);
lean_closure_set(v___x_4757_, 1, v___f_4735_);
v___x_4758_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4758_, 0, v___x_4757_);
lean_closure_set(v___x_4758_, 1, v___f_4736_);
v___x_4759_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4759_, 0, v___x_4758_);
lean_closure_set(v___x_4759_, 1, v___f_4734_);
v___x_4760_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4760_, 0, v___x_4759_);
lean_closure_set(v___x_4760_, 1, v___f_4754_);
v___x_4761_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_andThenOnSubgoals___boxed), 11, 2);
lean_closure_set(v___x_4761_, 0, v___x_4760_);
lean_closure_set(v___x_4761_, 1, v___f_4737_);
v___x_4762_ = lp_mathlib_Lean_Elab_Tactic_allGoals(v___x_4761_, v___y_4738_, v___y_4739_, v___y_4740_, v___y_4741_, v___y_4742_, v___y_4743_, v___y_4744_, v___y_4745_);
if (lean_obj_tag(v___x_4762_) == 0)
{
lean_object* v___x_4763_; 
lean_dec_ref_known(v___x_4762_, 1);
v___x_4763_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___y_4738_, v___y_4739_, v___y_4740_, v___y_4741_, v___y_4742_, v___y_4743_, v___y_4744_, v___y_4745_);
if (lean_obj_tag(v___x_4763_) == 0)
{
lean_object* v_a_4764_; uint8_t v___x_4765_; 
v_a_4764_ = lean_ctor_get(v___x_4763_, 0);
lean_inc(v_a_4764_);
lean_dec_ref_known(v___x_4763_, 1);
v___x_4765_ = lp_mathlib_List_beq___at___00Mathlib_Tactic_Tauto_tautoCore_spec__0(v_a_4751_, v_a_4764_);
lean_dec(v_a_4764_);
lean_dec(v_a_4751_);
if (v___x_4765_ == 0)
{
goto v___jp_4747_;
}
else
{
lean_object* v___x_4766_; lean_object* v___x_4767_; 
v___x_4766_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___closed__6);
v___x_4767_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg(v___x_4766_, v___y_4742_, v___y_4743_, v___y_4744_, v___y_4745_);
return v___x_4767_;
}
}
else
{
lean_object* v_a_4768_; lean_object* v___x_4770_; uint8_t v_isShared_4771_; uint8_t v_isSharedCheck_4775_; 
lean_dec(v_a_4751_);
v_a_4768_ = lean_ctor_get(v___x_4763_, 0);
v_isSharedCheck_4775_ = !lean_is_exclusive(v___x_4763_);
if (v_isSharedCheck_4775_ == 0)
{
v___x_4770_ = v___x_4763_;
v_isShared_4771_ = v_isSharedCheck_4775_;
goto v_resetjp_4769_;
}
else
{
lean_inc(v_a_4768_);
lean_dec(v___x_4763_);
v___x_4770_ = lean_box(0);
v_isShared_4771_ = v_isSharedCheck_4775_;
goto v_resetjp_4769_;
}
v_resetjp_4769_:
{
lean_object* v___x_4773_; 
if (v_isShared_4771_ == 0)
{
v___x_4773_ = v___x_4770_;
goto v_reusejp_4772_;
}
else
{
lean_object* v_reuseFailAlloc_4774_; 
v_reuseFailAlloc_4774_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4774_, 0, v_a_4768_);
v___x_4773_ = v_reuseFailAlloc_4774_;
goto v_reusejp_4772_;
}
v_reusejp_4772_:
{
return v___x_4773_;
}
}
}
}
else
{
lean_dec(v_a_4751_);
return v___x_4762_;
}
}
else
{
lean_object* v_a_4776_; lean_object* v___x_4778_; uint8_t v_isShared_4779_; uint8_t v_isSharedCheck_4783_; 
lean_dec_ref(v___f_4737_);
lean_dec_ref(v___f_4736_);
lean_dec_ref(v___f_4735_);
lean_dec_ref(v___f_4734_);
v_a_4776_ = lean_ctor_get(v___x_4750_, 0);
v_isSharedCheck_4783_ = !lean_is_exclusive(v___x_4750_);
if (v_isSharedCheck_4783_ == 0)
{
v___x_4778_ = v___x_4750_;
v_isShared_4779_ = v_isSharedCheck_4783_;
goto v_resetjp_4777_;
}
else
{
lean_inc(v_a_4776_);
lean_dec(v___x_4750_);
v___x_4778_ = lean_box(0);
v_isShared_4779_ = v_isSharedCheck_4783_;
goto v_resetjp_4777_;
}
v_resetjp_4777_:
{
lean_object* v___x_4781_; 
if (v_isShared_4779_ == 0)
{
v___x_4781_ = v___x_4778_;
goto v_reusejp_4780_;
}
else
{
lean_object* v_reuseFailAlloc_4782_; 
v_reuseFailAlloc_4782_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4782_, 0, v_a_4776_);
v___x_4781_ = v_reuseFailAlloc_4782_;
goto v_reusejp_4780_;
}
v_reusejp_4780_:
{
return v___x_4781_;
}
}
}
v___jp_4747_:
{
lean_object* v___x_4748_; lean_object* v___x_4749_; 
v___x_4748_ = lean_box(0);
v___x_4749_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4749_, 0, v___x_4748_);
return v___x_4749_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10___boxed(lean_object* v___f_4784_, lean_object* v___f_4785_, lean_object* v___f_4786_, lean_object* v___f_4787_, lean_object* v___y_4788_, lean_object* v___y_4789_, lean_object* v___y_4790_, lean_object* v___y_4791_, lean_object* v___y_4792_, lean_object* v___y_4793_, lean_object* v___y_4794_, lean_object* v___y_4795_, lean_object* v___y_4796_){
_start:
{
lean_object* v_res_4797_; 
v_res_4797_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__10(v___f_4784_, v___f_4785_, v___f_4786_, v___f_4787_, v___y_4788_, v___y_4789_, v___y_4790_, v___y_4791_, v___y_4792_, v___y_4793_, v___y_4794_, v___y_4795_);
lean_dec(v___y_4795_);
lean_dec_ref(v___y_4794_);
lean_dec(v___y_4793_);
lean_dec_ref(v___y_4792_);
lean_dec(v___y_4791_);
lean_dec_ref(v___y_4790_);
lean_dec(v___y_4789_);
lean_dec_ref(v___y_4788_);
return v_res_4797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2(lean_object* v_tac_4798_, lean_object* v___y_4799_, lean_object* v___y_4800_, lean_object* v___y_4801_, lean_object* v___y_4802_, lean_object* v___y_4803_, lean_object* v___y_4804_, lean_object* v___y_4805_, lean_object* v___y_4806_){
_start:
{
lean_object* v___x_4808_; 
v___x_4808_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_4800_, v___y_4802_, v___y_4804_, v___y_4806_);
if (lean_obj_tag(v___x_4808_) == 0)
{
lean_object* v_a_4809_; lean_object* v___y_4811_; uint8_t v___y_4812_; lean_object* v___y_4824_; lean_object* v_a_4825_; lean_object* v___x_4828_; 
v_a_4809_ = lean_ctor_get(v___x_4808_, 0);
lean_inc(v_a_4809_);
lean_dec_ref_known(v___x_4808_, 1);
lean_inc_ref(v_tac_4798_);
lean_inc(v___y_4806_);
lean_inc_ref(v___y_4805_);
lean_inc(v___y_4804_);
lean_inc_ref(v___y_4803_);
lean_inc(v___y_4802_);
lean_inc_ref(v___y_4801_);
lean_inc(v___y_4800_);
lean_inc_ref(v___y_4799_);
v___x_4828_ = lean_apply_9(v_tac_4798_, v___y_4799_, v___y_4800_, v___y_4801_, v___y_4802_, v___y_4803_, v___y_4804_, v___y_4805_, v___y_4806_, lean_box(0));
if (lean_obj_tag(v___x_4828_) == 0)
{
lean_object* v___x_4829_; 
lean_dec_ref_known(v___x_4828_, 1);
v___x_4829_ = lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2(v_tac_4798_, v___y_4799_, v___y_4800_, v___y_4801_, v___y_4802_, v___y_4803_, v___y_4804_, v___y_4805_, v___y_4806_);
if (lean_obj_tag(v___x_4829_) == 0)
{
lean_dec(v_a_4809_);
return v___x_4829_;
}
else
{
lean_object* v_a_4830_; 
v_a_4830_ = lean_ctor_get(v___x_4829_, 0);
lean_inc(v_a_4830_);
v___y_4824_ = v___x_4829_;
v_a_4825_ = v_a_4830_;
goto v___jp_4823_;
}
}
else
{
lean_object* v_a_4831_; 
lean_dec_ref(v_tac_4798_);
v_a_4831_ = lean_ctor_get(v___x_4828_, 0);
lean_inc(v_a_4831_);
v___y_4824_ = v___x_4828_;
v_a_4825_ = v_a_4831_;
goto v___jp_4823_;
}
v___jp_4810_:
{
if (v___y_4812_ == 0)
{
lean_object* v___x_4813_; 
lean_dec_ref(v___y_4811_);
v___x_4813_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_4809_, v___y_4812_, v___y_4800_, v___y_4801_, v___y_4802_, v___y_4803_, v___y_4804_, v___y_4805_, v___y_4806_);
if (lean_obj_tag(v___x_4813_) == 0)
{
lean_object* v___x_4815_; uint8_t v_isShared_4816_; uint8_t v_isSharedCheck_4821_; 
v_isSharedCheck_4821_ = !lean_is_exclusive(v___x_4813_);
if (v_isSharedCheck_4821_ == 0)
{
lean_object* v_unused_4822_; 
v_unused_4822_ = lean_ctor_get(v___x_4813_, 0);
lean_dec(v_unused_4822_);
v___x_4815_ = v___x_4813_;
v_isShared_4816_ = v_isSharedCheck_4821_;
goto v_resetjp_4814_;
}
else
{
lean_dec(v___x_4813_);
v___x_4815_ = lean_box(0);
v_isShared_4816_ = v_isSharedCheck_4821_;
goto v_resetjp_4814_;
}
v_resetjp_4814_:
{
lean_object* v___x_4817_; lean_object* v___x_4819_; 
v___x_4817_ = lean_box(0);
if (v_isShared_4816_ == 0)
{
lean_ctor_set(v___x_4815_, 0, v___x_4817_);
v___x_4819_ = v___x_4815_;
goto v_reusejp_4818_;
}
else
{
lean_object* v_reuseFailAlloc_4820_; 
v_reuseFailAlloc_4820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4820_, 0, v___x_4817_);
v___x_4819_ = v_reuseFailAlloc_4820_;
goto v_reusejp_4818_;
}
v_reusejp_4818_:
{
return v___x_4819_;
}
}
}
else
{
return v___x_4813_;
}
}
else
{
lean_dec(v_a_4809_);
return v___y_4811_;
}
}
v___jp_4823_:
{
uint8_t v___x_4826_; 
v___x_4826_ = l_Lean_Exception_isInterrupt(v_a_4825_);
if (v___x_4826_ == 0)
{
uint8_t v___x_4827_; 
v___x_4827_ = l_Lean_Exception_isRuntime(v_a_4825_);
v___y_4811_ = v___y_4824_;
v___y_4812_ = v___x_4827_;
goto v___jp_4810_;
}
else
{
lean_dec_ref(v_a_4825_);
v___y_4811_ = v___y_4824_;
v___y_4812_ = v___x_4826_;
goto v___jp_4810_;
}
}
}
else
{
lean_object* v_a_4832_; lean_object* v___x_4834_; uint8_t v_isShared_4835_; uint8_t v_isSharedCheck_4839_; 
lean_dec_ref(v_tac_4798_);
v_a_4832_ = lean_ctor_get(v___x_4808_, 0);
v_isSharedCheck_4839_ = !lean_is_exclusive(v___x_4808_);
if (v_isSharedCheck_4839_ == 0)
{
v___x_4834_ = v___x_4808_;
v_isShared_4835_ = v_isSharedCheck_4839_;
goto v_resetjp_4833_;
}
else
{
lean_inc(v_a_4832_);
lean_dec(v___x_4808_);
v___x_4834_ = lean_box(0);
v_isShared_4835_ = v_isSharedCheck_4839_;
goto v_resetjp_4833_;
}
v_resetjp_4833_:
{
lean_object* v___x_4837_; 
if (v_isShared_4835_ == 0)
{
v___x_4837_ = v___x_4834_;
goto v_reusejp_4836_;
}
else
{
lean_object* v_reuseFailAlloc_4838_; 
v_reuseFailAlloc_4838_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4838_, 0, v_a_4832_);
v___x_4837_ = v_reuseFailAlloc_4838_;
goto v_reusejp_4836_;
}
v_reusejp_4836_:
{
return v___x_4837_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2___boxed(lean_object* v_tac_4840_, lean_object* v___y_4841_, lean_object* v___y_4842_, lean_object* v___y_4843_, lean_object* v___y_4844_, lean_object* v___y_4845_, lean_object* v___y_4846_, lean_object* v___y_4847_, lean_object* v___y_4848_, lean_object* v___y_4849_){
_start:
{
lean_object* v_res_4850_; 
v_res_4850_ = lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2(v_tac_4840_, v___y_4841_, v___y_4842_, v___y_4843_, v___y_4844_, v___y_4845_, v___y_4846_, v___y_4847_, v___y_4848_);
lean_dec(v___y_4848_);
lean_dec_ref(v___y_4847_);
lean_dec(v___y_4846_);
lean_dec_ref(v___y_4845_);
lean_dec(v___y_4844_);
lean_dec_ref(v___y_4843_);
lean_dec(v___y_4842_);
lean_dec_ref(v___y_4841_);
return v_res_4850_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore(lean_object* v_a_4868_, lean_object* v_a_4869_, lean_object* v_a_4870_, lean_object* v_a_4871_, lean_object* v_a_4872_, lean_object* v_a_4873_, lean_object* v_a_4874_, lean_object* v_a_4875_){
_start:
{
lean_object* v_ref_4877_; uint8_t v___x_4878_; lean_object* v___x_4879_; lean_object* v___x_4880_; lean_object* v___x_4881_; lean_object* v___x_4882_; lean_object* v___x_4883_; lean_object* v___x_4884_; lean_object* v___x_4885_; 
v_ref_4877_ = lean_ctor_get(v_a_4874_, 5);
v___x_4878_ = 0;
v___x_4879_ = l_Lean_SourceInfo_fromRef(v_ref_4877_, v___x_4878_);
v___x_4880_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__0));
v___x_4881_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__2___closed__1));
lean_inc(v___x_4879_);
v___x_4882_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4882_, 0, v___x_4879_);
lean_ctor_set(v___x_4882_, 1, v___x_4880_);
v___x_4883_ = l_Lean_Syntax_node1(v___x_4879_, v___x_4881_, v___x_4882_);
v___x_4884_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_4884_, 0, v___x_4883_);
v___x_4885_ = l_Lean_Elab_Tactic_tryTactic___redArg(v___x_4884_, v_a_4868_, v_a_4869_, v_a_4870_, v_a_4871_, v_a_4872_, v_a_4873_, v_a_4874_, v_a_4875_);
if (lean_obj_tag(v___x_4885_) == 0)
{
lean_object* v___x_4886_; lean_object* v_a_4887_; lean_object* v___x_4888_; lean_object* v___x_4889_; lean_object* v___x_4890_; lean_object* v___x_4891_; lean_object* v___x_4892_; lean_object* v___x_4893_; 
lean_dec_ref_known(v___x_4885_, 1);
v___x_4886_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__0(v_a_4868_, v_a_4869_, v_a_4870_, v_a_4871_, v_a_4872_, v_a_4873_, v_a_4874_, v_a_4875_);
v_a_4887_ = lean_ctor_get(v___x_4886_, 0);
lean_inc_n(v_a_4887_, 2);
lean_dec_ref(v___x_4886_);
v___x_4888_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__0));
v___x_4889_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___lam__4___closed__1));
v___x_4890_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_4890_, 0, v_a_4887_);
lean_ctor_set(v___x_4890_, 1, v___x_4888_);
v___x_4891_ = l_Lean_Syntax_node1(v_a_4887_, v___x_4889_, v___x_4890_);
v___x_4892_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_4892_, 0, v___x_4891_);
v___x_4893_ = l_Lean_Elab_Tactic_tryTactic___redArg(v___x_4892_, v_a_4868_, v_a_4869_, v_a_4870_, v_a_4871_, v_a_4872_, v_a_4873_, v_a_4874_, v_a_4875_);
if (lean_obj_tag(v___x_4893_) == 0)
{
lean_object* v___f_4894_; lean_object* v___x_4895_; 
lean_dec_ref_known(v___x_4893_, 1);
v___f_4894_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___closed__7));
v___x_4895_ = lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2(v___f_4894_, v_a_4868_, v_a_4869_, v_a_4870_, v_a_4871_, v_a_4872_, v_a_4873_, v_a_4874_, v_a_4875_);
return v___x_4895_;
}
else
{
lean_object* v_a_4896_; lean_object* v___x_4898_; uint8_t v_isShared_4899_; uint8_t v_isSharedCheck_4903_; 
v_a_4896_ = lean_ctor_get(v___x_4893_, 0);
v_isSharedCheck_4903_ = !lean_is_exclusive(v___x_4893_);
if (v_isSharedCheck_4903_ == 0)
{
v___x_4898_ = v___x_4893_;
v_isShared_4899_ = v_isSharedCheck_4903_;
goto v_resetjp_4897_;
}
else
{
lean_inc(v_a_4896_);
lean_dec(v___x_4893_);
v___x_4898_ = lean_box(0);
v_isShared_4899_ = v_isSharedCheck_4903_;
goto v_resetjp_4897_;
}
v_resetjp_4897_:
{
lean_object* v___x_4901_; 
if (v_isShared_4899_ == 0)
{
v___x_4901_ = v___x_4898_;
goto v_reusejp_4900_;
}
else
{
lean_object* v_reuseFailAlloc_4902_; 
v_reuseFailAlloc_4902_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4902_, 0, v_a_4896_);
v___x_4901_ = v_reuseFailAlloc_4902_;
goto v_reusejp_4900_;
}
v_reusejp_4900_:
{
return v___x_4901_;
}
}
}
}
else
{
lean_object* v_a_4904_; lean_object* v___x_4906_; uint8_t v_isShared_4907_; uint8_t v_isSharedCheck_4911_; 
v_a_4904_ = lean_ctor_get(v___x_4885_, 0);
v_isSharedCheck_4911_ = !lean_is_exclusive(v___x_4885_);
if (v_isSharedCheck_4911_ == 0)
{
v___x_4906_ = v___x_4885_;
v_isShared_4907_ = v_isSharedCheck_4911_;
goto v_resetjp_4905_;
}
else
{
lean_inc(v_a_4904_);
lean_dec(v___x_4885_);
v___x_4906_ = lean_box(0);
v_isShared_4907_ = v_isSharedCheck_4911_;
goto v_resetjp_4905_;
}
v_resetjp_4905_:
{
lean_object* v___x_4909_; 
if (v_isShared_4907_ == 0)
{
v___x_4909_ = v___x_4906_;
goto v_reusejp_4908_;
}
else
{
lean_object* v_reuseFailAlloc_4910_; 
v_reuseFailAlloc_4910_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4910_, 0, v_a_4904_);
v___x_4909_ = v_reuseFailAlloc_4910_;
goto v_reusejp_4908_;
}
v_reusejp_4908_:
{
return v___x_4909_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautoCore___boxed(lean_object* v_a_4912_, lean_object* v_a_4913_, lean_object* v_a_4914_, lean_object* v_a_4915_, lean_object* v_a_4916_, lean_object* v_a_4917_, lean_object* v_a_4918_, lean_object* v_a_4919_, lean_object* v_a_4920_){
_start:
{
lean_object* v_res_4921_; 
v_res_4921_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore(v_a_4912_, v_a_4913_, v_a_4914_, v_a_4915_, v_a_4916_, v_a_4917_, v_a_4918_, v_a_4919_);
lean_dec(v_a_4919_);
lean_dec_ref(v_a_4918_);
lean_dec(v_a_4917_);
lean_dec_ref(v_a_4916_);
lean_dec(v_a_4915_);
lean_dec_ref(v_a_4914_);
lean_dec(v_a_4913_);
lean_dec_ref(v_a_4912_);
return v_res_4921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1(lean_object* v_00_u03b1_4922_, lean_object* v_msg_4923_, lean_object* v___y_4924_, lean_object* v___y_4925_, lean_object* v___y_4926_, lean_object* v___y_4927_, lean_object* v___y_4928_, lean_object* v___y_4929_, lean_object* v___y_4930_, lean_object* v___y_4931_){
_start:
{
lean_object* v___x_4933_; 
v___x_4933_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___redArg(v_msg_4923_, v___y_4928_, v___y_4929_, v___y_4930_, v___y_4931_);
return v___x_4933_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1___boxed(lean_object* v_00_u03b1_4934_, lean_object* v_msg_4935_, lean_object* v___y_4936_, lean_object* v___y_4937_, lean_object* v___y_4938_, lean_object* v___y_4939_, lean_object* v___y_4940_, lean_object* v___y_4941_, lean_object* v___y_4942_, lean_object* v___y_4943_, lean_object* v___y_4944_){
_start:
{
lean_object* v_res_4945_; 
v_res_4945_ = lp_mathlib_Lean_throwError___at___00Mathlib_Tactic_Tauto_tautoCore_spec__1(v_00_u03b1_4934_, v_msg_4935_, v___y_4936_, v___y_4937_, v___y_4938_, v___y_4939_, v___y_4940_, v___y_4941_, v___y_4942_, v___y_4943_);
lean_dec(v___y_4943_);
lean_dec_ref(v___y_4942_);
lean_dec(v___y_4941_);
lean_dec_ref(v___y_4940_);
lean_dec(v___y_4939_);
lean_dec_ref(v___y_4938_);
lean_dec(v___y_4937_);
lean_dec_ref(v___y_4936_);
return v_res_4945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher(lean_object* v_e_4946_, lean_object* v_a_4947_, lean_object* v_a_4948_, lean_object* v_a_4949_, lean_object* v_a_4950_){
_start:
{
uint8_t v___x_4952_; lean_object* v___x_4953_; lean_object* v___x_4954_; uint8_t v___x_4955_; lean_object* v___x_4956_; lean_object* v___x_4957_; lean_object* v___x_4958_; lean_object* v___f_4959_; lean_object* v___x_4960_; 
v___x_4952_ = 0;
v___x_4953_ = lean_box(0);
v___x_4954_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1, &lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_distribNotOnceAt___lam__12___closed__1);
v___x_4955_ = 0;
v___x_4956_ = lean_box(0);
v___x_4957_ = lean_box(v___x_4955_);
v___x_4958_ = lean_box(v___x_4952_);
lean_inc_ref(v_e_4946_);
v___f_4959_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__0___boxed), 11, 6);
lean_closure_set(v___f_4959_, 0, v___x_4954_);
lean_closure_set(v___f_4959_, 1, v___x_4957_);
lean_closure_set(v___f_4959_, 2, v___x_4956_);
lean_closure_set(v___f_4959_, 3, v___x_4953_);
lean_closure_set(v___f_4959_, 4, v_e_4946_);
lean_closure_set(v___f_4959_, 5, v___x_4958_);
v___x_4960_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4959_, v___x_4952_, v_a_4947_, v_a_4948_, v_a_4949_, v_a_4950_);
if (lean_obj_tag(v___x_4960_) == 0)
{
lean_object* v_a_4961_; lean_object* v___x_4963_; uint8_t v_isShared_4964_; uint8_t v_isSharedCheck_5034_; 
v_a_4961_ = lean_ctor_get(v___x_4960_, 0);
v_isSharedCheck_5034_ = !lean_is_exclusive(v___x_4960_);
if (v_isSharedCheck_5034_ == 0)
{
v___x_4963_ = v___x_4960_;
v_isShared_4964_ = v_isSharedCheck_5034_;
goto v_resetjp_4962_;
}
else
{
lean_inc(v_a_4961_);
lean_dec(v___x_4960_);
v___x_4963_ = lean_box(0);
v_isShared_4964_ = v_isSharedCheck_5034_;
goto v_resetjp_4962_;
}
v_resetjp_4962_:
{
lean_object* v_snd_4965_; lean_object* v_snd_4966_; uint8_t v___x_4967_; 
v_snd_4965_ = lean_ctor_get(v_a_4961_, 1);
lean_inc(v_snd_4965_);
lean_dec(v_a_4961_);
v_snd_4966_ = lean_ctor_get(v_snd_4965_, 1);
lean_inc(v_snd_4966_);
lean_dec(v_snd_4965_);
v___x_4967_ = lean_unbox(v_snd_4966_);
if (v___x_4967_ == 0)
{
lean_object* v___x_4968_; lean_object* v___x_4969_; lean_object* v___f_4970_; lean_object* v___x_4971_; 
lean_dec(v_snd_4966_);
lean_del_object(v___x_4963_);
v___x_4968_ = lean_box(v___x_4955_);
v___x_4969_ = lean_box(v___x_4952_);
lean_inc_ref(v_e_4946_);
v___f_4970_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__1___boxed), 11, 6);
lean_closure_set(v___f_4970_, 0, v___x_4954_);
lean_closure_set(v___f_4970_, 1, v___x_4968_);
lean_closure_set(v___f_4970_, 2, v___x_4956_);
lean_closure_set(v___f_4970_, 3, v___x_4953_);
lean_closure_set(v___f_4970_, 4, v_e_4946_);
lean_closure_set(v___f_4970_, 5, v___x_4969_);
v___x_4971_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4970_, v___x_4952_, v_a_4947_, v_a_4948_, v_a_4949_, v_a_4950_);
if (lean_obj_tag(v___x_4971_) == 0)
{
lean_object* v_a_4972_; lean_object* v___x_4974_; uint8_t v_isShared_4975_; uint8_t v_isSharedCheck_5022_; 
v_a_4972_ = lean_ctor_get(v___x_4971_, 0);
v_isSharedCheck_5022_ = !lean_is_exclusive(v___x_4971_);
if (v_isSharedCheck_5022_ == 0)
{
v___x_4974_ = v___x_4971_;
v_isShared_4975_ = v_isSharedCheck_5022_;
goto v_resetjp_4973_;
}
else
{
lean_inc(v_a_4972_);
lean_dec(v___x_4971_);
v___x_4974_ = lean_box(0);
v_isShared_4975_ = v_isSharedCheck_5022_;
goto v_resetjp_4973_;
}
v_resetjp_4973_:
{
lean_object* v_snd_4976_; lean_object* v_snd_4977_; uint8_t v___x_4978_; 
v_snd_4976_ = lean_ctor_get(v_a_4972_, 1);
lean_inc(v_snd_4976_);
lean_dec(v_a_4972_);
v_snd_4977_ = lean_ctor_get(v_snd_4976_, 1);
lean_inc(v_snd_4977_);
lean_dec(v_snd_4976_);
v___x_4978_ = lean_unbox(v_snd_4977_);
if (v___x_4978_ == 0)
{
lean_object* v___x_4979_; lean_object* v___f_4980_; lean_object* v___x_4981_; 
lean_dec(v_snd_4977_);
lean_del_object(v___x_4974_);
v___x_4979_ = lean_box(v___x_4952_);
lean_inc_ref(v_e_4946_);
v___f_4980_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_casesMatcher___lam__0___boxed), 8, 3);
lean_closure_set(v___f_4980_, 0, v___x_4953_);
lean_closure_set(v___f_4980_, 1, v_e_4946_);
lean_closure_set(v___f_4980_, 2, v___x_4979_);
v___x_4981_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4980_, v___x_4952_, v_a_4947_, v_a_4948_, v_a_4949_, v_a_4950_);
if (lean_obj_tag(v___x_4981_) == 0)
{
lean_object* v_a_4982_; lean_object* v___x_4984_; uint8_t v_isShared_4985_; uint8_t v_isSharedCheck_5010_; 
v_a_4982_ = lean_ctor_get(v___x_4981_, 0);
v_isSharedCheck_5010_ = !lean_is_exclusive(v___x_4981_);
if (v_isSharedCheck_5010_ == 0)
{
v___x_4984_ = v___x_4981_;
v_isShared_4985_ = v_isSharedCheck_5010_;
goto v_resetjp_4983_;
}
else
{
lean_inc(v_a_4982_);
lean_dec(v___x_4981_);
v___x_4984_ = lean_box(0);
v_isShared_4985_ = v_isSharedCheck_5010_;
goto v_resetjp_4983_;
}
v_resetjp_4983_:
{
lean_object* v_snd_4986_; lean_object* v_snd_4987_; lean_object* v_snd_4988_; uint8_t v___x_4989_; 
v_snd_4986_ = lean_ctor_get(v_a_4982_, 1);
lean_inc(v_snd_4986_);
lean_dec(v_a_4982_);
v_snd_4987_ = lean_ctor_get(v_snd_4986_, 1);
lean_inc(v_snd_4987_);
lean_dec(v_snd_4986_);
v_snd_4988_ = lean_ctor_get(v_snd_4987_, 1);
lean_inc(v_snd_4988_);
lean_dec(v_snd_4987_);
v___x_4989_ = lean_unbox(v_snd_4988_);
if (v___x_4989_ == 0)
{
lean_object* v___x_4990_; uint8_t v___x_4991_; lean_object* v___x_4992_; lean_object* v___x_4993_; lean_object* v___f_4994_; lean_object* v___x_4995_; 
lean_dec(v_snd_4988_);
lean_del_object(v___x_4984_);
v___x_4990_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___closed__2);
v___x_4991_ = 2;
v___x_4992_ = lean_box(v___x_4991_);
v___x_4993_ = lean_box(v___x_4952_);
v___f_4994_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_coreConstructorMatcher___lam__2___boxed), 9, 4);
lean_closure_set(v___f_4994_, 0, v___x_4992_);
lean_closure_set(v___f_4994_, 1, v___x_4990_);
lean_closure_set(v___f_4994_, 2, v_e_4946_);
lean_closure_set(v___f_4994_, 3, v___x_4993_);
v___x_4995_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Mathlib_Tactic_Tauto_distribNotOnceAt_spec__2___redArg(v___f_4994_, v___x_4952_, v_a_4947_, v_a_4948_, v_a_4949_, v_a_4950_);
if (lean_obj_tag(v___x_4995_) == 0)
{
lean_object* v_a_4996_; uint8_t v___x_4997_; 
v_a_4996_ = lean_ctor_get(v___x_4995_, 0);
lean_inc(v_a_4996_);
v___x_4997_ = lean_unbox(v_a_4996_);
lean_dec(v_a_4996_);
if (v___x_4997_ == 0)
{
lean_object* v___x_4999_; uint8_t v_isShared_5000_; uint8_t v_isSharedCheck_5005_; 
v_isSharedCheck_5005_ = !lean_is_exclusive(v___x_4995_);
if (v_isSharedCheck_5005_ == 0)
{
lean_object* v_unused_5006_; 
v_unused_5006_ = lean_ctor_get(v___x_4995_, 0);
lean_dec(v_unused_5006_);
v___x_4999_ = v___x_4995_;
v_isShared_5000_ = v_isSharedCheck_5005_;
goto v_resetjp_4998_;
}
else
{
lean_dec(v___x_4995_);
v___x_4999_ = lean_box(0);
v_isShared_5000_ = v_isSharedCheck_5005_;
goto v_resetjp_4998_;
}
v_resetjp_4998_:
{
lean_object* v___x_5001_; lean_object* v___x_5003_; 
v___x_5001_ = lean_box(v___x_4952_);
if (v_isShared_5000_ == 0)
{
lean_ctor_set(v___x_4999_, 0, v___x_5001_);
v___x_5003_ = v___x_4999_;
goto v_reusejp_5002_;
}
else
{
lean_object* v_reuseFailAlloc_5004_; 
v_reuseFailAlloc_5004_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5004_, 0, v___x_5001_);
v___x_5003_ = v_reuseFailAlloc_5004_;
goto v_reusejp_5002_;
}
v_reusejp_5002_:
{
return v___x_5003_;
}
}
}
else
{
return v___x_4995_;
}
}
else
{
return v___x_4995_;
}
}
else
{
lean_object* v___x_5008_; 
lean_dec_ref(v_e_4946_);
if (v_isShared_4985_ == 0)
{
lean_ctor_set(v___x_4984_, 0, v_snd_4988_);
v___x_5008_ = v___x_4984_;
goto v_reusejp_5007_;
}
else
{
lean_object* v_reuseFailAlloc_5009_; 
v_reuseFailAlloc_5009_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5009_, 0, v_snd_4988_);
v___x_5008_ = v_reuseFailAlloc_5009_;
goto v_reusejp_5007_;
}
v_reusejp_5007_:
{
return v___x_5008_;
}
}
}
}
else
{
lean_object* v_a_5011_; lean_object* v___x_5013_; uint8_t v_isShared_5014_; uint8_t v_isSharedCheck_5018_; 
lean_dec_ref(v_e_4946_);
v_a_5011_ = lean_ctor_get(v___x_4981_, 0);
v_isSharedCheck_5018_ = !lean_is_exclusive(v___x_4981_);
if (v_isSharedCheck_5018_ == 0)
{
v___x_5013_ = v___x_4981_;
v_isShared_5014_ = v_isSharedCheck_5018_;
goto v_resetjp_5012_;
}
else
{
lean_inc(v_a_5011_);
lean_dec(v___x_4981_);
v___x_5013_ = lean_box(0);
v_isShared_5014_ = v_isSharedCheck_5018_;
goto v_resetjp_5012_;
}
v_resetjp_5012_:
{
lean_object* v___x_5016_; 
if (v_isShared_5014_ == 0)
{
v___x_5016_ = v___x_5013_;
goto v_reusejp_5015_;
}
else
{
lean_object* v_reuseFailAlloc_5017_; 
v_reuseFailAlloc_5017_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5017_, 0, v_a_5011_);
v___x_5016_ = v_reuseFailAlloc_5017_;
goto v_reusejp_5015_;
}
v_reusejp_5015_:
{
return v___x_5016_;
}
}
}
}
else
{
lean_object* v___x_5020_; 
lean_dec_ref(v_e_4946_);
if (v_isShared_4975_ == 0)
{
lean_ctor_set(v___x_4974_, 0, v_snd_4977_);
v___x_5020_ = v___x_4974_;
goto v_reusejp_5019_;
}
else
{
lean_object* v_reuseFailAlloc_5021_; 
v_reuseFailAlloc_5021_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5021_, 0, v_snd_4977_);
v___x_5020_ = v_reuseFailAlloc_5021_;
goto v_reusejp_5019_;
}
v_reusejp_5019_:
{
return v___x_5020_;
}
}
}
}
else
{
lean_object* v_a_5023_; lean_object* v___x_5025_; uint8_t v_isShared_5026_; uint8_t v_isSharedCheck_5030_; 
lean_dec_ref(v_e_4946_);
v_a_5023_ = lean_ctor_get(v___x_4971_, 0);
v_isSharedCheck_5030_ = !lean_is_exclusive(v___x_4971_);
if (v_isSharedCheck_5030_ == 0)
{
v___x_5025_ = v___x_4971_;
v_isShared_5026_ = v_isSharedCheck_5030_;
goto v_resetjp_5024_;
}
else
{
lean_inc(v_a_5023_);
lean_dec(v___x_4971_);
v___x_5025_ = lean_box(0);
v_isShared_5026_ = v_isSharedCheck_5030_;
goto v_resetjp_5024_;
}
v_resetjp_5024_:
{
lean_object* v___x_5028_; 
if (v_isShared_5026_ == 0)
{
v___x_5028_ = v___x_5025_;
goto v_reusejp_5027_;
}
else
{
lean_object* v_reuseFailAlloc_5029_; 
v_reuseFailAlloc_5029_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5029_, 0, v_a_5023_);
v___x_5028_ = v_reuseFailAlloc_5029_;
goto v_reusejp_5027_;
}
v_reusejp_5027_:
{
return v___x_5028_;
}
}
}
}
else
{
lean_object* v___x_5032_; 
lean_dec_ref(v_e_4946_);
if (v_isShared_4964_ == 0)
{
lean_ctor_set(v___x_4963_, 0, v_snd_4966_);
v___x_5032_ = v___x_4963_;
goto v_reusejp_5031_;
}
else
{
lean_object* v_reuseFailAlloc_5033_; 
v_reuseFailAlloc_5033_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5033_, 0, v_snd_4966_);
v___x_5032_ = v_reuseFailAlloc_5033_;
goto v_reusejp_5031_;
}
v_reusejp_5031_:
{
return v___x_5032_;
}
}
}
}
else
{
lean_object* v_a_5035_; lean_object* v___x_5037_; uint8_t v_isShared_5038_; uint8_t v_isSharedCheck_5042_; 
lean_dec_ref(v_e_4946_);
v_a_5035_ = lean_ctor_get(v___x_4960_, 0);
v_isSharedCheck_5042_ = !lean_is_exclusive(v___x_4960_);
if (v_isSharedCheck_5042_ == 0)
{
v___x_5037_ = v___x_4960_;
v_isShared_5038_ = v_isSharedCheck_5042_;
goto v_resetjp_5036_;
}
else
{
lean_inc(v_a_5035_);
lean_dec(v___x_4960_);
v___x_5037_ = lean_box(0);
v_isShared_5038_ = v_isSharedCheck_5042_;
goto v_resetjp_5036_;
}
v_resetjp_5036_:
{
lean_object* v___x_5040_; 
if (v_isShared_5038_ == 0)
{
v___x_5040_ = v___x_5037_;
goto v_reusejp_5039_;
}
else
{
lean_object* v_reuseFailAlloc_5041_; 
v_reuseFailAlloc_5041_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5041_, 0, v_a_5035_);
v___x_5040_ = v_reuseFailAlloc_5041_;
goto v_reusejp_5039_;
}
v_reusejp_5039_:
{
return v___x_5040_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher___boxed(lean_object* v_e_5043_, lean_object* v_a_5044_, lean_object* v_a_5045_, lean_object* v_a_5046_, lean_object* v_a_5047_, lean_object* v_a_5048_){
_start:
{
lean_object* v_res_5049_; 
v_res_5049_ = lp_mathlib_Mathlib_Tactic_Tauto_finishingConstructorMatcher(v_e_5043_, v_a_5044_, v_a_5045_, v_a_5046_, v_a_5047_);
lean_dec(v_a_5047_);
lean_dec_ref(v_a_5046_);
lean_dec(v_a_5045_);
lean_dec_ref(v_a_5044_);
return v_res_5049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0(lean_object* v___y_5050_, lean_object* v___x_5051_, lean_object* v___x_5052_, lean_object* v___y_5053_, lean_object* v___x_5054_, lean_object* v_a_x3f_5055_){
_start:
{
lean_object* v___x_5057_; lean_object* v_env_5058_; lean_object* v_nextMacroScope_5059_; lean_object* v_ngen_5060_; lean_object* v_auxDeclNGen_5061_; lean_object* v_traceState_5062_; lean_object* v_messages_5063_; lean_object* v_infoState_5064_; lean_object* v_snapshotTasks_5065_; lean_object* v___x_5067_; uint8_t v_isShared_5068_; uint8_t v_isSharedCheck_5090_; 
v___x_5057_ = lean_st_ref_take(v___y_5050_);
v_env_5058_ = lean_ctor_get(v___x_5057_, 0);
v_nextMacroScope_5059_ = lean_ctor_get(v___x_5057_, 1);
v_ngen_5060_ = lean_ctor_get(v___x_5057_, 2);
v_auxDeclNGen_5061_ = lean_ctor_get(v___x_5057_, 3);
v_traceState_5062_ = lean_ctor_get(v___x_5057_, 4);
v_messages_5063_ = lean_ctor_get(v___x_5057_, 6);
v_infoState_5064_ = lean_ctor_get(v___x_5057_, 7);
v_snapshotTasks_5065_ = lean_ctor_get(v___x_5057_, 8);
v_isSharedCheck_5090_ = !lean_is_exclusive(v___x_5057_);
if (v_isSharedCheck_5090_ == 0)
{
lean_object* v_unused_5091_; 
v_unused_5091_ = lean_ctor_get(v___x_5057_, 5);
lean_dec(v_unused_5091_);
v___x_5067_ = v___x_5057_;
v_isShared_5068_ = v_isSharedCheck_5090_;
goto v_resetjp_5066_;
}
else
{
lean_inc(v_snapshotTasks_5065_);
lean_inc(v_infoState_5064_);
lean_inc(v_messages_5063_);
lean_inc(v_traceState_5062_);
lean_inc(v_auxDeclNGen_5061_);
lean_inc(v_ngen_5060_);
lean_inc(v_nextMacroScope_5059_);
lean_inc(v_env_5058_);
lean_dec(v___x_5057_);
v___x_5067_ = lean_box(0);
v_isShared_5068_ = v_isSharedCheck_5090_;
goto v_resetjp_5066_;
}
v_resetjp_5066_:
{
lean_object* v___x_5069_; lean_object* v___x_5071_; 
v___x_5069_ = l_Lean_ScopedEnvExtension_popScope___redArg(v___x_5051_, v_env_5058_);
if (v_isShared_5068_ == 0)
{
lean_ctor_set(v___x_5067_, 5, v___x_5052_);
lean_ctor_set(v___x_5067_, 0, v___x_5069_);
v___x_5071_ = v___x_5067_;
goto v_reusejp_5070_;
}
else
{
lean_object* v_reuseFailAlloc_5089_; 
v_reuseFailAlloc_5089_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_5089_, 0, v___x_5069_);
lean_ctor_set(v_reuseFailAlloc_5089_, 1, v_nextMacroScope_5059_);
lean_ctor_set(v_reuseFailAlloc_5089_, 2, v_ngen_5060_);
lean_ctor_set(v_reuseFailAlloc_5089_, 3, v_auxDeclNGen_5061_);
lean_ctor_set(v_reuseFailAlloc_5089_, 4, v_traceState_5062_);
lean_ctor_set(v_reuseFailAlloc_5089_, 5, v___x_5052_);
lean_ctor_set(v_reuseFailAlloc_5089_, 6, v_messages_5063_);
lean_ctor_set(v_reuseFailAlloc_5089_, 7, v_infoState_5064_);
lean_ctor_set(v_reuseFailAlloc_5089_, 8, v_snapshotTasks_5065_);
v___x_5071_ = v_reuseFailAlloc_5089_;
goto v_reusejp_5070_;
}
v_reusejp_5070_:
{
lean_object* v___x_5072_; lean_object* v___x_5073_; lean_object* v_mctx_5074_; lean_object* v_zetaDeltaFVarIds_5075_; lean_object* v_postponed_5076_; lean_object* v_diag_5077_; lean_object* v___x_5079_; uint8_t v_isShared_5080_; uint8_t v_isSharedCheck_5087_; 
v___x_5072_ = lean_st_ref_set(v___y_5050_, v___x_5071_);
v___x_5073_ = lean_st_ref_take(v___y_5053_);
v_mctx_5074_ = lean_ctor_get(v___x_5073_, 0);
v_zetaDeltaFVarIds_5075_ = lean_ctor_get(v___x_5073_, 2);
v_postponed_5076_ = lean_ctor_get(v___x_5073_, 3);
v_diag_5077_ = lean_ctor_get(v___x_5073_, 4);
v_isSharedCheck_5087_ = !lean_is_exclusive(v___x_5073_);
if (v_isSharedCheck_5087_ == 0)
{
lean_object* v_unused_5088_; 
v_unused_5088_ = lean_ctor_get(v___x_5073_, 1);
lean_dec(v_unused_5088_);
v___x_5079_ = v___x_5073_;
v_isShared_5080_ = v_isSharedCheck_5087_;
goto v_resetjp_5078_;
}
else
{
lean_inc(v_diag_5077_);
lean_inc(v_postponed_5076_);
lean_inc(v_zetaDeltaFVarIds_5075_);
lean_inc(v_mctx_5074_);
lean_dec(v___x_5073_);
v___x_5079_ = lean_box(0);
v_isShared_5080_ = v_isSharedCheck_5087_;
goto v_resetjp_5078_;
}
v_resetjp_5078_:
{
lean_object* v___x_5082_; 
if (v_isShared_5080_ == 0)
{
lean_ctor_set(v___x_5079_, 1, v___x_5054_);
v___x_5082_ = v___x_5079_;
goto v_reusejp_5081_;
}
else
{
lean_object* v_reuseFailAlloc_5086_; 
v_reuseFailAlloc_5086_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5086_, 0, v_mctx_5074_);
lean_ctor_set(v_reuseFailAlloc_5086_, 1, v___x_5054_);
lean_ctor_set(v_reuseFailAlloc_5086_, 2, v_zetaDeltaFVarIds_5075_);
lean_ctor_set(v_reuseFailAlloc_5086_, 3, v_postponed_5076_);
lean_ctor_set(v_reuseFailAlloc_5086_, 4, v_diag_5077_);
v___x_5082_ = v_reuseFailAlloc_5086_;
goto v_reusejp_5081_;
}
v_reusejp_5081_:
{
lean_object* v___x_5083_; lean_object* v___x_5084_; lean_object* v___x_5085_; 
v___x_5083_ = lean_st_ref_set(v___y_5053_, v___x_5082_);
v___x_5084_ = lean_box(0);
v___x_5085_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5085_, 0, v___x_5084_);
return v___x_5085_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0___boxed(lean_object* v___y_5092_, lean_object* v___x_5093_, lean_object* v___x_5094_, lean_object* v___y_5095_, lean_object* v___x_5096_, lean_object* v_a_x3f_5097_, lean_object* v___y_5098_){
_start:
{
lean_object* v_res_5099_; 
v_res_5099_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0(v___y_5092_, v___x_5093_, v___x_5094_, v___y_5095_, v___x_5096_, v_a_x3f_5097_);
lean_dec(v_a_x3f_5097_);
lean_dec(v___y_5095_);
lean_dec(v___y_5092_);
return v_res_5099_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_5100_; 
v___x_5100_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_5100_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1(void){
_start:
{
lean_object* v___x_5101_; lean_object* v___x_5102_; 
v___x_5101_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__0);
v___x_5102_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5102_, 0, v___x_5101_);
return v___x_5102_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2(void){
_start:
{
lean_object* v___x_5103_; lean_object* v___x_5104_; 
v___x_5103_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1, &lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1);
v___x_5104_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5104_, 0, v___x_5103_);
lean_ctor_set(v___x_5104_, 1, v___x_5103_);
return v___x_5104_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3(void){
_start:
{
lean_object* v___x_5105_; lean_object* v___x_5106_; 
v___x_5105_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1, &lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1_once, _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__1);
v___x_5106_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_5106_, 0, v___x_5105_);
lean_ctor_set(v___x_5106_, 1, v___x_5105_);
lean_ctor_set(v___x_5106_, 2, v___x_5105_);
lean_ctor_set(v___x_5106_, 3, v___x_5105_);
lean_ctor_set(v___x_5106_, 4, v___x_5105_);
lean_ctor_set(v___x_5106_, 5, v___x_5105_);
return v___x_5106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg(lean_object* v_t_5111_, lean_object* v___y_5112_, lean_object* v___y_5113_, lean_object* v___y_5114_, lean_object* v___y_5115_, lean_object* v___y_5116_, lean_object* v___y_5117_, lean_object* v___y_5118_, lean_object* v___y_5119_){
_start:
{
lean_object* v___x_5121_; lean_object* v_env_5122_; lean_object* v_nextMacroScope_5123_; lean_object* v_ngen_5124_; lean_object* v_auxDeclNGen_5125_; lean_object* v_traceState_5126_; lean_object* v_messages_5127_; lean_object* v_infoState_5128_; lean_object* v_snapshotTasks_5129_; lean_object* v___x_5131_; uint8_t v_isShared_5132_; uint8_t v_isSharedCheck_5196_; 
v___x_5121_ = lean_st_ref_take(v___y_5119_);
v_env_5122_ = lean_ctor_get(v___x_5121_, 0);
v_nextMacroScope_5123_ = lean_ctor_get(v___x_5121_, 1);
v_ngen_5124_ = lean_ctor_get(v___x_5121_, 2);
v_auxDeclNGen_5125_ = lean_ctor_get(v___x_5121_, 3);
v_traceState_5126_ = lean_ctor_get(v___x_5121_, 4);
v_messages_5127_ = lean_ctor_get(v___x_5121_, 6);
v_infoState_5128_ = lean_ctor_get(v___x_5121_, 7);
v_snapshotTasks_5129_ = lean_ctor_get(v___x_5121_, 8);
v_isSharedCheck_5196_ = !lean_is_exclusive(v___x_5121_);
if (v_isSharedCheck_5196_ == 0)
{
lean_object* v_unused_5197_; 
v_unused_5197_ = lean_ctor_get(v___x_5121_, 5);
lean_dec(v_unused_5197_);
v___x_5131_ = v___x_5121_;
v_isShared_5132_ = v_isSharedCheck_5196_;
goto v_resetjp_5130_;
}
else
{
lean_inc(v_snapshotTasks_5129_);
lean_inc(v_infoState_5128_);
lean_inc(v_messages_5127_);
lean_inc(v_traceState_5126_);
lean_inc(v_auxDeclNGen_5125_);
lean_inc(v_ngen_5124_);
lean_inc(v_nextMacroScope_5123_);
lean_inc(v_env_5122_);
lean_dec(v___x_5121_);
v___x_5131_ = lean_box(0);
v_isShared_5132_ = v_isSharedCheck_5196_;
goto v_resetjp_5130_;
}
v_resetjp_5130_:
{
lean_object* v___x_5133_; lean_object* v___x_5134_; lean_object* v___x_5135_; lean_object* v___x_5137_; 
v___x_5133_ = l_Lean_Meta_instanceExtension;
v___x_5134_ = l_Lean_ScopedEnvExtension_pushScope___redArg(v___x_5133_, v_env_5122_);
v___x_5135_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2, &lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__2);
if (v_isShared_5132_ == 0)
{
lean_ctor_set(v___x_5131_, 5, v___x_5135_);
lean_ctor_set(v___x_5131_, 0, v___x_5134_);
v___x_5137_ = v___x_5131_;
goto v_reusejp_5136_;
}
else
{
lean_object* v_reuseFailAlloc_5195_; 
v_reuseFailAlloc_5195_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_5195_, 0, v___x_5134_);
lean_ctor_set(v_reuseFailAlloc_5195_, 1, v_nextMacroScope_5123_);
lean_ctor_set(v_reuseFailAlloc_5195_, 2, v_ngen_5124_);
lean_ctor_set(v_reuseFailAlloc_5195_, 3, v_auxDeclNGen_5125_);
lean_ctor_set(v_reuseFailAlloc_5195_, 4, v_traceState_5126_);
lean_ctor_set(v_reuseFailAlloc_5195_, 5, v___x_5135_);
lean_ctor_set(v_reuseFailAlloc_5195_, 6, v_messages_5127_);
lean_ctor_set(v_reuseFailAlloc_5195_, 7, v_infoState_5128_);
lean_ctor_set(v_reuseFailAlloc_5195_, 8, v_snapshotTasks_5129_);
v___x_5137_ = v_reuseFailAlloc_5195_;
goto v_reusejp_5136_;
}
v_reusejp_5136_:
{
lean_object* v___x_5138_; lean_object* v___x_5139_; lean_object* v_mctx_5140_; lean_object* v_zetaDeltaFVarIds_5141_; lean_object* v_postponed_5142_; lean_object* v_diag_5143_; lean_object* v___x_5145_; uint8_t v_isShared_5146_; uint8_t v_isSharedCheck_5193_; 
v___x_5138_ = lean_st_ref_set(v___y_5119_, v___x_5137_);
v___x_5139_ = lean_st_ref_take(v___y_5117_);
v_mctx_5140_ = lean_ctor_get(v___x_5139_, 0);
v_zetaDeltaFVarIds_5141_ = lean_ctor_get(v___x_5139_, 2);
v_postponed_5142_ = lean_ctor_get(v___x_5139_, 3);
v_diag_5143_ = lean_ctor_get(v___x_5139_, 4);
v_isSharedCheck_5193_ = !lean_is_exclusive(v___x_5139_);
if (v_isSharedCheck_5193_ == 0)
{
lean_object* v_unused_5194_; 
v_unused_5194_ = lean_ctor_get(v___x_5139_, 1);
lean_dec(v_unused_5194_);
v___x_5145_ = v___x_5139_;
v_isShared_5146_ = v_isSharedCheck_5193_;
goto v_resetjp_5144_;
}
else
{
lean_inc(v_diag_5143_);
lean_inc(v_postponed_5142_);
lean_inc(v_zetaDeltaFVarIds_5141_);
lean_inc(v_mctx_5140_);
lean_dec(v___x_5139_);
v___x_5145_ = lean_box(0);
v_isShared_5146_ = v_isSharedCheck_5193_;
goto v_resetjp_5144_;
}
v_resetjp_5144_:
{
lean_object* v___x_5147_; lean_object* v___x_5149_; 
v___x_5147_ = lean_obj_once(&lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3, &lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3_once, _init_lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__3);
if (v_isShared_5146_ == 0)
{
lean_ctor_set(v___x_5145_, 1, v___x_5147_);
v___x_5149_ = v___x_5145_;
goto v_reusejp_5148_;
}
else
{
lean_object* v_reuseFailAlloc_5192_; 
v_reuseFailAlloc_5192_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_5192_, 0, v_mctx_5140_);
lean_ctor_set(v_reuseFailAlloc_5192_, 1, v___x_5147_);
lean_ctor_set(v_reuseFailAlloc_5192_, 2, v_zetaDeltaFVarIds_5141_);
lean_ctor_set(v_reuseFailAlloc_5192_, 3, v_postponed_5142_);
lean_ctor_set(v_reuseFailAlloc_5192_, 4, v_diag_5143_);
v___x_5149_ = v_reuseFailAlloc_5192_;
goto v_reusejp_5148_;
}
v_reusejp_5148_:
{
lean_object* v___x_5150_; lean_object* v___x_5151_; uint8_t v___x_5152_; lean_object* v___x_5153_; lean_object* v___x_5154_; 
v___x_5150_ = lean_st_ref_set(v___y_5117_, v___x_5149_);
v___x_5151_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___closed__5));
v___x_5152_ = 1;
v___x_5153_ = lean_unsigned_to_nat(10u);
v___x_5154_ = l_Lean_Meta_addInstance(v___x_5151_, v___x_5152_, v___x_5153_, v___y_5116_, v___y_5117_, v___y_5118_, v___y_5119_);
if (lean_obj_tag(v___x_5154_) == 0)
{
lean_object* v_r_5155_; 
lean_dec_ref_known(v___x_5154_, 1);
lean_inc(v___y_5119_);
lean_inc_ref(v___y_5118_);
lean_inc(v___y_5117_);
lean_inc_ref(v___y_5116_);
lean_inc(v___y_5115_);
lean_inc_ref(v___y_5114_);
lean_inc(v___y_5113_);
lean_inc_ref(v___y_5112_);
v_r_5155_ = lean_apply_9(v_t_5111_, v___y_5112_, v___y_5113_, v___y_5114_, v___y_5115_, v___y_5116_, v___y_5117_, v___y_5118_, v___y_5119_, lean_box(0));
if (lean_obj_tag(v_r_5155_) == 0)
{
lean_object* v_a_5156_; lean_object* v___x_5158_; uint8_t v_isShared_5159_; uint8_t v_isSharedCheck_5172_; 
v_a_5156_ = lean_ctor_get(v_r_5155_, 0);
v_isSharedCheck_5172_ = !lean_is_exclusive(v_r_5155_);
if (v_isSharedCheck_5172_ == 0)
{
v___x_5158_ = v_r_5155_;
v_isShared_5159_ = v_isSharedCheck_5172_;
goto v_resetjp_5157_;
}
else
{
lean_inc(v_a_5156_);
lean_dec(v_r_5155_);
v___x_5158_ = lean_box(0);
v_isShared_5159_ = v_isSharedCheck_5172_;
goto v_resetjp_5157_;
}
v_resetjp_5157_:
{
lean_object* v___x_5161_; 
lean_inc(v_a_5156_);
if (v_isShared_5159_ == 0)
{
lean_ctor_set_tag(v___x_5158_, 1);
v___x_5161_ = v___x_5158_;
goto v_reusejp_5160_;
}
else
{
lean_object* v_reuseFailAlloc_5171_; 
v_reuseFailAlloc_5171_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5171_, 0, v_a_5156_);
v___x_5161_ = v_reuseFailAlloc_5171_;
goto v_reusejp_5160_;
}
v_reusejp_5160_:
{
lean_object* v___x_5162_; lean_object* v___x_5164_; uint8_t v_isShared_5165_; uint8_t v_isSharedCheck_5169_; 
v___x_5162_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0(v___y_5119_, v___x_5133_, v___x_5135_, v___y_5117_, v___x_5147_, v___x_5161_);
lean_dec_ref(v___x_5161_);
v_isSharedCheck_5169_ = !lean_is_exclusive(v___x_5162_);
if (v_isSharedCheck_5169_ == 0)
{
lean_object* v_unused_5170_; 
v_unused_5170_ = lean_ctor_get(v___x_5162_, 0);
lean_dec(v_unused_5170_);
v___x_5164_ = v___x_5162_;
v_isShared_5165_ = v_isSharedCheck_5169_;
goto v_resetjp_5163_;
}
else
{
lean_dec(v___x_5162_);
v___x_5164_ = lean_box(0);
v_isShared_5165_ = v_isSharedCheck_5169_;
goto v_resetjp_5163_;
}
v_resetjp_5163_:
{
lean_object* v___x_5167_; 
if (v_isShared_5165_ == 0)
{
lean_ctor_set(v___x_5164_, 0, v_a_5156_);
v___x_5167_ = v___x_5164_;
goto v_reusejp_5166_;
}
else
{
lean_object* v_reuseFailAlloc_5168_; 
v_reuseFailAlloc_5168_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5168_, 0, v_a_5156_);
v___x_5167_ = v_reuseFailAlloc_5168_;
goto v_reusejp_5166_;
}
v_reusejp_5166_:
{
return v___x_5167_;
}
}
}
}
}
else
{
lean_object* v_a_5173_; lean_object* v___x_5174_; lean_object* v___x_5175_; lean_object* v___x_5177_; uint8_t v_isShared_5178_; uint8_t v_isSharedCheck_5182_; 
v_a_5173_ = lean_ctor_get(v_r_5155_, 0);
lean_inc(v_a_5173_);
lean_dec_ref_known(v_r_5155_, 1);
v___x_5174_ = lean_box(0);
v___x_5175_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___lam__0(v___y_5119_, v___x_5133_, v___x_5135_, v___y_5117_, v___x_5147_, v___x_5174_);
v_isSharedCheck_5182_ = !lean_is_exclusive(v___x_5175_);
if (v_isSharedCheck_5182_ == 0)
{
lean_object* v_unused_5183_; 
v_unused_5183_ = lean_ctor_get(v___x_5175_, 0);
lean_dec(v_unused_5183_);
v___x_5177_ = v___x_5175_;
v_isShared_5178_ = v_isSharedCheck_5182_;
goto v_resetjp_5176_;
}
else
{
lean_dec(v___x_5175_);
v___x_5177_ = lean_box(0);
v_isShared_5178_ = v_isSharedCheck_5182_;
goto v_resetjp_5176_;
}
v_resetjp_5176_:
{
lean_object* v___x_5180_; 
if (v_isShared_5178_ == 0)
{
lean_ctor_set_tag(v___x_5177_, 1);
lean_ctor_set(v___x_5177_, 0, v_a_5173_);
v___x_5180_ = v___x_5177_;
goto v_reusejp_5179_;
}
else
{
lean_object* v_reuseFailAlloc_5181_; 
v_reuseFailAlloc_5181_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5181_, 0, v_a_5173_);
v___x_5180_ = v_reuseFailAlloc_5181_;
goto v_reusejp_5179_;
}
v_reusejp_5179_:
{
return v___x_5180_;
}
}
}
}
else
{
lean_object* v_a_5184_; lean_object* v___x_5186_; uint8_t v_isShared_5187_; uint8_t v_isSharedCheck_5191_; 
lean_dec_ref(v_t_5111_);
v_a_5184_ = lean_ctor_get(v___x_5154_, 0);
v_isSharedCheck_5191_ = !lean_is_exclusive(v___x_5154_);
if (v_isSharedCheck_5191_ == 0)
{
v___x_5186_ = v___x_5154_;
v_isShared_5187_ = v_isSharedCheck_5191_;
goto v_resetjp_5185_;
}
else
{
lean_inc(v_a_5184_);
lean_dec(v___x_5154_);
v___x_5186_ = lean_box(0);
v_isShared_5187_ = v_isSharedCheck_5191_;
goto v_resetjp_5185_;
}
v_resetjp_5185_:
{
lean_object* v___x_5189_; 
if (v_isShared_5187_ == 0)
{
v___x_5189_ = v___x_5186_;
goto v_reusejp_5188_;
}
else
{
lean_object* v_reuseFailAlloc_5190_; 
v_reuseFailAlloc_5190_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5190_, 0, v_a_5184_);
v___x_5189_ = v_reuseFailAlloc_5190_;
goto v_reusejp_5188_;
}
v_reusejp_5188_:
{
return v___x_5189_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg___boxed(lean_object* v_t_5198_, lean_object* v___y_5199_, lean_object* v___y_5200_, lean_object* v___y_5201_, lean_object* v___y_5202_, lean_object* v___y_5203_, lean_object* v___y_5204_, lean_object* v___y_5205_, lean_object* v___y_5206_, lean_object* v___y_5207_){
_start:
{
lean_object* v_res_5208_; 
v_res_5208_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg(v_t_5198_, v___y_5199_, v___y_5200_, v___y_5201_, v___y_5202_, v___y_5203_, v___y_5204_, v___y_5205_, v___y_5206_);
lean_dec(v___y_5206_);
lean_dec_ref(v___y_5205_);
lean_dec(v___y_5204_);
lean_dec_ref(v___y_5203_);
lean_dec(v___y_5202_);
lean_dec_ref(v___y_5201_);
lean_dec(v___y_5200_);
lean_dec_ref(v___y_5199_);
return v_res_5208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0(lean_object* v_00_u03b1_5209_, lean_object* v_t_5210_, lean_object* v___y_5211_, lean_object* v___y_5212_, lean_object* v___y_5213_, lean_object* v___y_5214_, lean_object* v___y_5215_, lean_object* v___y_5216_, lean_object* v___y_5217_, lean_object* v___y_5218_){
_start:
{
lean_object* v___x_5220_; 
v___x_5220_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___redArg(v_t_5210_, v___y_5211_, v___y_5212_, v___y_5213_, v___y_5214_, v___y_5215_, v___y_5216_, v___y_5217_, v___y_5218_);
return v___x_5220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0___boxed(lean_object* v_00_u03b1_5221_, lean_object* v_t_5222_, lean_object* v___y_5223_, lean_object* v___y_5224_, lean_object* v___y_5225_, lean_object* v___y_5226_, lean_object* v___y_5227_, lean_object* v___y_5228_, lean_object* v___y_5229_, lean_object* v___y_5230_, lean_object* v___y_5231_){
_start:
{
lean_object* v_res_5232_; 
v_res_5232_ = lp_mathlib_Lean_Elab_Tactic_classical___at___00Mathlib_Tactic_Tauto_tautology_spec__0(v_00_u03b1_5221_, v_t_5222_, v___y_5223_, v___y_5224_, v___y_5225_, v___y_5226_, v___y_5227_, v___y_5228_, v___y_5229_, v___y_5230_);
lean_dec(v___y_5230_);
lean_dec_ref(v___y_5229_);
lean_dec(v___y_5228_);
lean_dec_ref(v___y_5227_);
lean_dec(v___y_5226_);
lean_dec_ref(v___y_5225_);
lean_dec(v___y_5224_);
lean_dec_ref(v___y_5223_);
return v_res_5232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0(uint8_t v___y_5234_, lean_object* v___y_5235_, lean_object* v___y_5236_, lean_object* v___y_5237_, lean_object* v___y_5238_, lean_object* v___y_5239_, lean_object* v___y_5240_, lean_object* v___y_5241_, lean_object* v___y_5242_){
_start:
{
lean_object* v___x_5244_; 
v___x_5244_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5236_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_);
if (lean_obj_tag(v___x_5244_) == 0)
{
lean_object* v_a_5245_; lean_object* v___x_5246_; uint8_t v___x_5247_; lean_object* v___x_5248_; 
v_a_5245_ = lean_ctor_get(v___x_5244_, 0);
lean_inc(v_a_5245_);
lean_dec_ref_known(v___x_5244_, 1);
v___x_5246_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___closed__0));
v___x_5247_ = 1;
v___x_5248_ = lp_mathlib_Mathlib_Tactic_constructorMatching(v_a_5245_, v___x_5246_, v___y_5234_, v___x_5247_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_);
if (lean_obj_tag(v___x_5248_) == 0)
{
lean_object* v_a_5249_; lean_object* v___x_5250_; 
v_a_5249_ = lean_ctor_get(v___x_5248_, 0);
lean_inc(v_a_5249_);
lean_dec_ref_known(v___x_5248_, 1);
v___x_5250_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v_a_5249_, v___y_5236_, v___y_5239_, v___y_5240_, v___y_5241_, v___y_5242_);
if (lean_obj_tag(v___x_5250_) == 0)
{
lean_object* v___x_5252_; uint8_t v_isShared_5253_; uint8_t v_isSharedCheck_5258_; 
v_isSharedCheck_5258_ = !lean_is_exclusive(v___x_5250_);
if (v_isSharedCheck_5258_ == 0)
{
lean_object* v_unused_5259_; 
v_unused_5259_ = lean_ctor_get(v___x_5250_, 0);
lean_dec(v_unused_5259_);
v___x_5252_ = v___x_5250_;
v_isShared_5253_ = v_isSharedCheck_5258_;
goto v_resetjp_5251_;
}
else
{
lean_dec(v___x_5250_);
v___x_5252_ = lean_box(0);
v_isShared_5253_ = v_isSharedCheck_5258_;
goto v_resetjp_5251_;
}
v_resetjp_5251_:
{
lean_object* v___x_5254_; lean_object* v___x_5256_; 
v___x_5254_ = lean_box(0);
if (v_isShared_5253_ == 0)
{
lean_ctor_set(v___x_5252_, 0, v___x_5254_);
v___x_5256_ = v___x_5252_;
goto v_reusejp_5255_;
}
else
{
lean_object* v_reuseFailAlloc_5257_; 
v_reuseFailAlloc_5257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5257_, 0, v___x_5254_);
v___x_5256_ = v_reuseFailAlloc_5257_;
goto v_reusejp_5255_;
}
v_reusejp_5255_:
{
return v___x_5256_;
}
}
}
else
{
return v___x_5250_;
}
}
else
{
lean_object* v_a_5260_; lean_object* v___x_5262_; uint8_t v_isShared_5263_; uint8_t v_isSharedCheck_5267_; 
v_a_5260_ = lean_ctor_get(v___x_5248_, 0);
v_isSharedCheck_5267_ = !lean_is_exclusive(v___x_5248_);
if (v_isSharedCheck_5267_ == 0)
{
v___x_5262_ = v___x_5248_;
v_isShared_5263_ = v_isSharedCheck_5267_;
goto v_resetjp_5261_;
}
else
{
lean_inc(v_a_5260_);
lean_dec(v___x_5248_);
v___x_5262_ = lean_box(0);
v_isShared_5263_ = v_isSharedCheck_5267_;
goto v_resetjp_5261_;
}
v_resetjp_5261_:
{
lean_object* v___x_5265_; 
if (v_isShared_5263_ == 0)
{
v___x_5265_ = v___x_5262_;
goto v_reusejp_5264_;
}
else
{
lean_object* v_reuseFailAlloc_5266_; 
v_reuseFailAlloc_5266_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5266_, 0, v_a_5260_);
v___x_5265_ = v_reuseFailAlloc_5266_;
goto v_reusejp_5264_;
}
v_reusejp_5264_:
{
return v___x_5265_;
}
}
}
}
else
{
lean_object* v_a_5268_; lean_object* v___x_5270_; uint8_t v_isShared_5271_; uint8_t v_isSharedCheck_5275_; 
v_a_5268_ = lean_ctor_get(v___x_5244_, 0);
v_isSharedCheck_5275_ = !lean_is_exclusive(v___x_5244_);
if (v_isSharedCheck_5275_ == 0)
{
v___x_5270_ = v___x_5244_;
v_isShared_5271_ = v_isSharedCheck_5275_;
goto v_resetjp_5269_;
}
else
{
lean_inc(v_a_5268_);
lean_dec(v___x_5244_);
v___x_5270_ = lean_box(0);
v_isShared_5271_ = v_isSharedCheck_5275_;
goto v_resetjp_5269_;
}
v_resetjp_5269_:
{
lean_object* v___x_5273_; 
if (v_isShared_5271_ == 0)
{
v___x_5273_ = v___x_5270_;
goto v_reusejp_5272_;
}
else
{
lean_object* v_reuseFailAlloc_5274_; 
v_reuseFailAlloc_5274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5274_, 0, v_a_5268_);
v___x_5273_ = v_reuseFailAlloc_5274_;
goto v_reusejp_5272_;
}
v_reusejp_5272_:
{
return v___x_5273_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___boxed(lean_object* v___y_5276_, lean_object* v___y_5277_, lean_object* v___y_5278_, lean_object* v___y_5279_, lean_object* v___y_5280_, lean_object* v___y_5281_, lean_object* v___y_5282_, lean_object* v___y_5283_, lean_object* v___y_5284_, lean_object* v___y_5285_){
_start:
{
uint8_t v___y_10468__boxed_5286_; lean_object* v_res_5287_; 
v___y_10468__boxed_5286_ = lean_unbox(v___y_5276_);
v_res_5287_ = lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0(v___y_10468__boxed_5286_, v___y_5277_, v___y_5278_, v___y_5279_, v___y_5280_, v___y_5281_, v___y_5282_, v___y_5283_, v___y_5284_);
lean_dec(v___y_5284_);
lean_dec_ref(v___y_5283_);
lean_dec(v___y_5282_);
lean_dec_ref(v___y_5281_);
lean_dec(v___y_5280_);
lean_dec_ref(v___y_5279_);
lean_dec(v___y_5278_);
lean_dec_ref(v___y_5277_);
return v_res_5287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1(lean_object* v___x_5288_, lean_object* v___x_5289_, lean_object* v___y_5290_, lean_object* v___y_5291_, lean_object* v___y_5292_, lean_object* v___y_5293_, lean_object* v___y_5294_, lean_object* v___y_5295_, lean_object* v___y_5296_, lean_object* v___y_5297_){
_start:
{
lean_object* v___y_5300_; lean_object* v___y_5301_; uint8_t v___y_5302_; lean_object* v___x_5307_; 
v___x_5307_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_5291_, v___y_5293_, v___y_5295_, v___y_5297_);
if (lean_obj_tag(v___x_5307_) == 0)
{
lean_object* v_a_5308_; lean_object* v___x_5309_; 
v_a_5308_ = lean_ctor_get(v___x_5307_, 0);
lean_inc(v_a_5308_);
lean_dec_ref_known(v___x_5307_, 1);
v___x_5309_ = l_Lean_Elab_Tactic_withoutRecover___redArg(v___x_5288_, v___y_5290_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_);
if (lean_obj_tag(v___x_5309_) == 0)
{
lean_dec(v_a_5308_);
lean_dec(v___x_5289_);
return v___x_5309_;
}
else
{
lean_object* v_a_5310_; uint8_t v___y_5312_; uint8_t v___x_5329_; 
v_a_5310_ = lean_ctor_get(v___x_5309_, 0);
lean_inc(v_a_5310_);
v___x_5329_ = l_Lean_Exception_isInterrupt(v_a_5310_);
if (v___x_5329_ == 0)
{
uint8_t v___x_5330_; 
v___x_5330_ = l_Lean_Exception_isRuntime(v_a_5310_);
v___y_5312_ = v___x_5330_;
goto v___jp_5311_;
}
else
{
lean_dec(v_a_5310_);
v___y_5312_ = v___x_5329_;
goto v___jp_5311_;
}
v___jp_5311_:
{
if (v___y_5312_ == 0)
{
lean_object* v___x_5313_; 
lean_dec_ref_known(v___x_5309_, 1);
v___x_5313_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_5308_, v___y_5312_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_);
if (lean_obj_tag(v___x_5313_) == 0)
{
lean_object* v___x_5314_; 
lean_dec_ref_known(v___x_5313_, 1);
v___x_5314_ = l_Lean_Elab_Tactic_saveState___redArg(v___y_5291_, v___y_5293_, v___y_5295_, v___y_5297_);
if (lean_obj_tag(v___x_5314_) == 0)
{
lean_object* v_a_5315_; lean_object* v___x_5316_; lean_object* v___x_5317_; 
v_a_5315_ = lean_ctor_get(v___x_5314_, 0);
lean_inc(v_a_5315_);
lean_dec_ref_known(v___x_5314_, 1);
v___x_5316_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_5316_, 0, v___x_5289_);
v___x_5317_ = l_Lean_Elab_Tactic_withoutRecover___redArg(v___x_5316_, v___y_5290_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_);
if (lean_obj_tag(v___x_5317_) == 0)
{
lean_dec(v_a_5315_);
return v___x_5317_;
}
else
{
lean_object* v_a_5318_; uint8_t v___x_5319_; 
v_a_5318_ = lean_ctor_get(v___x_5317_, 0);
lean_inc(v_a_5318_);
v___x_5319_ = l_Lean_Exception_isInterrupt(v_a_5318_);
if (v___x_5319_ == 0)
{
uint8_t v___x_5320_; 
v___x_5320_ = l_Lean_Exception_isRuntime(v_a_5318_);
v___y_5300_ = v_a_5315_;
v___y_5301_ = v___x_5317_;
v___y_5302_ = v___x_5320_;
goto v___jp_5299_;
}
else
{
lean_dec(v_a_5318_);
v___y_5300_ = v_a_5315_;
v___y_5301_ = v___x_5317_;
v___y_5302_ = v___x_5319_;
goto v___jp_5299_;
}
}
}
else
{
lean_object* v_a_5321_; lean_object* v___x_5323_; uint8_t v_isShared_5324_; uint8_t v_isSharedCheck_5328_; 
lean_dec(v___x_5289_);
v_a_5321_ = lean_ctor_get(v___x_5314_, 0);
v_isSharedCheck_5328_ = !lean_is_exclusive(v___x_5314_);
if (v_isSharedCheck_5328_ == 0)
{
v___x_5323_ = v___x_5314_;
v_isShared_5324_ = v_isSharedCheck_5328_;
goto v_resetjp_5322_;
}
else
{
lean_inc(v_a_5321_);
lean_dec(v___x_5314_);
v___x_5323_ = lean_box(0);
v_isShared_5324_ = v_isSharedCheck_5328_;
goto v_resetjp_5322_;
}
v_resetjp_5322_:
{
lean_object* v___x_5326_; 
if (v_isShared_5324_ == 0)
{
v___x_5326_ = v___x_5323_;
goto v_reusejp_5325_;
}
else
{
lean_object* v_reuseFailAlloc_5327_; 
v_reuseFailAlloc_5327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5327_, 0, v_a_5321_);
v___x_5326_ = v_reuseFailAlloc_5327_;
goto v_reusejp_5325_;
}
v_reusejp_5325_:
{
return v___x_5326_;
}
}
}
}
else
{
lean_dec(v___x_5289_);
return v___x_5313_;
}
}
else
{
lean_dec(v_a_5308_);
lean_dec(v___x_5289_);
return v___x_5309_;
}
}
}
}
else
{
lean_object* v_a_5331_; lean_object* v___x_5333_; uint8_t v_isShared_5334_; uint8_t v_isSharedCheck_5338_; 
lean_dec(v___x_5289_);
lean_dec_ref(v___x_5288_);
v_a_5331_ = lean_ctor_get(v___x_5307_, 0);
v_isSharedCheck_5338_ = !lean_is_exclusive(v___x_5307_);
if (v_isSharedCheck_5338_ == 0)
{
v___x_5333_ = v___x_5307_;
v_isShared_5334_ = v_isSharedCheck_5338_;
goto v_resetjp_5332_;
}
else
{
lean_inc(v_a_5331_);
lean_dec(v___x_5307_);
v___x_5333_ = lean_box(0);
v_isShared_5334_ = v_isSharedCheck_5338_;
goto v_resetjp_5332_;
}
v_resetjp_5332_:
{
lean_object* v___x_5336_; 
if (v_isShared_5334_ == 0)
{
v___x_5336_ = v___x_5333_;
goto v_reusejp_5335_;
}
else
{
lean_object* v_reuseFailAlloc_5337_; 
v_reuseFailAlloc_5337_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5337_, 0, v_a_5331_);
v___x_5336_ = v_reuseFailAlloc_5337_;
goto v_reusejp_5335_;
}
v_reusejp_5335_:
{
return v___x_5336_;
}
}
}
v___jp_5299_:
{
if (v___y_5302_ == 0)
{
lean_object* v___x_5303_; 
lean_dec_ref(v___y_5301_);
v___x_5303_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v___y_5300_, v___y_5302_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_);
if (lean_obj_tag(v___x_5303_) == 0)
{
lean_object* v___x_5304_; lean_object* v___f_5305_; lean_object* v___x_5306_; 
lean_dec_ref_known(v___x_5303_, 1);
v___x_5304_ = lean_box(v___y_5302_);
v___f_5305_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__0___boxed), 10, 1);
lean_closure_set(v___f_5305_, 0, v___x_5304_);
v___x_5306_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_5305_, v___y_5290_, v___y_5291_, v___y_5292_, v___y_5293_, v___y_5294_, v___y_5295_, v___y_5296_, v___y_5297_);
return v___x_5306_;
}
else
{
return v___x_5303_;
}
}
else
{
lean_dec_ref(v___y_5300_);
return v___y_5301_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1___boxed(lean_object* v___x_5339_, lean_object* v___x_5340_, lean_object* v___y_5341_, lean_object* v___y_5342_, lean_object* v___y_5343_, lean_object* v___y_5344_, lean_object* v___y_5345_, lean_object* v___y_5346_, lean_object* v___y_5347_, lean_object* v___y_5348_, lean_object* v___y_5349_){
_start:
{
lean_object* v_res_5350_; 
v_res_5350_ = lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1(v___x_5339_, v___x_5340_, v___y_5341_, v___y_5342_, v___y_5343_, v___y_5344_, v___y_5345_, v___y_5346_, v___y_5347_, v___y_5348_);
lean_dec(v___y_5348_);
lean_dec_ref(v___y_5347_);
lean_dec(v___y_5346_);
lean_dec_ref(v___y_5345_);
lean_dec(v___y_5344_);
lean_dec_ref(v___y_5343_);
lean_dec(v___y_5342_);
lean_dec_ref(v___y_5341_);
return v_res_5350_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6(void){
_start:
{
lean_object* v___x_5365_; 
v___x_5365_ = l_Array_mkArray0(lean_box(0));
return v___x_5365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2(lean_object* v___y_5372_, lean_object* v___y_5373_, lean_object* v___y_5374_, lean_object* v___y_5375_, lean_object* v___y_5376_, lean_object* v___y_5377_, lean_object* v___y_5378_, lean_object* v___y_5379_){
_start:
{
lean_object* v___x_5381_; 
v___x_5381_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_5373_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
if (lean_obj_tag(v___x_5381_) == 0)
{
lean_object* v_a_5382_; lean_object* v___x_5383_; 
v_a_5382_ = lean_ctor_get(v___x_5381_, 0);
lean_inc(v_a_5382_);
lean_dec_ref_known(v___x_5381_, 1);
v___x_5383_ = lp_mathlib_Mathlib_Tactic_Tauto_tautoCore(v___y_5372_, v___y_5373_, v___y_5374_, v___y_5375_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
if (lean_obj_tag(v___x_5383_) == 0)
{
lean_object* v_ref_5384_; uint8_t v___x_5385_; lean_object* v___x_5386_; lean_object* v___x_5387_; lean_object* v___x_5388_; lean_object* v___x_5389_; lean_object* v___x_5390_; lean_object* v___x_5391_; lean_object* v___x_5392_; lean_object* v___x_5393_; lean_object* v___x_5394_; lean_object* v___x_5395_; lean_object* v___x_5396_; lean_object* v___x_5397_; lean_object* v___x_5398_; lean_object* v___x_5399_; lean_object* v___x_5400_; lean_object* v___f_5401_; lean_object* v___x_5402_; lean_object* v___x_5403_; 
lean_dec_ref_known(v___x_5383_, 1);
v_ref_5384_ = lean_ctor_get(v___y_5378_, 5);
v___x_5385_ = 0;
v___x_5386_ = l_Lean_SourceInfo_fromRef(v_ref_5384_, v___x_5385_);
v___x_5387_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__1));
v___x_5388_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__2));
lean_inc_n(v___x_5386_, 5);
v___x_5389_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5389_, 0, v___x_5386_);
lean_ctor_set(v___x_5389_, 1, v___x_5388_);
v___x_5390_ = l_Lean_Syntax_node1(v___x_5386_, v___x_5387_, v___x_5389_);
v___x_5391_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__4));
v___x_5392_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__5));
v___x_5393_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5393_, 0, v___x_5386_);
lean_ctor_set(v___x_5393_, 1, v___x_5392_);
v___x_5394_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13));
v___x_5395_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6);
v___x_5396_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5396_, 0, v___x_5386_);
lean_ctor_set(v___x_5396_, 1, v___x_5394_);
lean_ctor_set(v___x_5396_, 2, v___x_5395_);
v___x_5397_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8));
lean_inc_ref_n(v___x_5396_, 4);
v___x_5398_ = l_Lean_Syntax_node1(v___x_5386_, v___x_5397_, v___x_5396_);
v___x_5399_ = l_Lean_Syntax_node6(v___x_5386_, v___x_5391_, v___x_5393_, v___x_5396_, v___x_5398_, v___x_5396_, v___x_5396_, v___x_5396_);
v___x_5400_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_evalTactic___boxed), 10, 1);
lean_closure_set(v___x_5400_, 0, v___x_5390_);
v___f_5401_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__1___boxed), 11, 2);
lean_closure_set(v___f_5401_, 0, v___x_5400_);
lean_closure_set(v___f_5401_, 1, v___x_5399_);
v___x_5402_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_iterateUntilFailure___at___00Mathlib_Tactic_Tauto_tautoCore_spec__2___boxed), 10, 1);
lean_closure_set(v___x_5402_, 0, v___f_5401_);
v___x_5403_ = lp_mathlib_Lean_Elab_Tactic_allGoals(v___x_5402_, v___y_5372_, v___y_5373_, v___y_5374_, v___y_5375_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
if (lean_obj_tag(v___x_5403_) == 0)
{
lean_object* v___x_5404_; 
lean_dec_ref_known(v___x_5403_, 1);
v___x_5404_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___y_5372_, v___y_5373_, v___y_5374_, v___y_5375_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
if (lean_obj_tag(v___x_5404_) == 0)
{
lean_object* v_a_5405_; lean_object* v___x_5407_; uint8_t v_isShared_5408_; uint8_t v_isSharedCheck_5417_; 
v_a_5405_ = lean_ctor_get(v___x_5404_, 0);
v_isSharedCheck_5417_ = !lean_is_exclusive(v___x_5404_);
if (v_isSharedCheck_5417_ == 0)
{
v___x_5407_ = v___x_5404_;
v_isShared_5408_ = v_isSharedCheck_5417_;
goto v_resetjp_5406_;
}
else
{
lean_inc(v_a_5405_);
lean_dec(v___x_5404_);
v___x_5407_ = lean_box(0);
v_isShared_5408_ = v_isSharedCheck_5417_;
goto v_resetjp_5406_;
}
v_resetjp_5406_:
{
uint8_t v___x_5409_; 
v___x_5409_ = l_List_isEmpty___redArg(v_a_5405_);
lean_dec(v_a_5405_);
if (v___x_5409_ == 0)
{
lean_object* v___x_5410_; lean_object* v___x_5411_; lean_object* v___x_5412_; 
lean_del_object(v___x_5407_);
v___x_5410_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_5411_ = lean_box(0);
v___x_5412_ = l_Lean_Meta_throwTacticEx___redArg(v___x_5410_, v_a_5382_, v___x_5411_, v___y_5376_, v___y_5377_, v___y_5378_, v___y_5379_);
return v___x_5412_;
}
else
{
lean_object* v___x_5413_; lean_object* v___x_5415_; 
lean_dec(v_a_5382_);
v___x_5413_ = lean_box(0);
if (v_isShared_5408_ == 0)
{
lean_ctor_set(v___x_5407_, 0, v___x_5413_);
v___x_5415_ = v___x_5407_;
goto v_reusejp_5414_;
}
else
{
lean_object* v_reuseFailAlloc_5416_; 
v_reuseFailAlloc_5416_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5416_, 0, v___x_5413_);
v___x_5415_ = v_reuseFailAlloc_5416_;
goto v_reusejp_5414_;
}
v_reusejp_5414_:
{
return v___x_5415_;
}
}
}
}
else
{
lean_object* v_a_5418_; lean_object* v___x_5420_; uint8_t v_isShared_5421_; uint8_t v_isSharedCheck_5425_; 
lean_dec(v_a_5382_);
v_a_5418_ = lean_ctor_get(v___x_5404_, 0);
v_isSharedCheck_5425_ = !lean_is_exclusive(v___x_5404_);
if (v_isSharedCheck_5425_ == 0)
{
v___x_5420_ = v___x_5404_;
v_isShared_5421_ = v_isSharedCheck_5425_;
goto v_resetjp_5419_;
}
else
{
lean_inc(v_a_5418_);
lean_dec(v___x_5404_);
v___x_5420_ = lean_box(0);
v_isShared_5421_ = v_isSharedCheck_5425_;
goto v_resetjp_5419_;
}
v_resetjp_5419_:
{
lean_object* v___x_5423_; 
if (v_isShared_5421_ == 0)
{
v___x_5423_ = v___x_5420_;
goto v_reusejp_5422_;
}
else
{
lean_object* v_reuseFailAlloc_5424_; 
v_reuseFailAlloc_5424_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5424_, 0, v_a_5418_);
v___x_5423_ = v_reuseFailAlloc_5424_;
goto v_reusejp_5422_;
}
v_reusejp_5422_:
{
return v___x_5423_;
}
}
}
}
else
{
lean_dec(v_a_5382_);
return v___x_5403_;
}
}
else
{
lean_dec(v_a_5382_);
return v___x_5383_;
}
}
else
{
lean_object* v_a_5426_; lean_object* v___x_5428_; uint8_t v_isShared_5429_; uint8_t v_isSharedCheck_5433_; 
v_a_5426_ = lean_ctor_get(v___x_5381_, 0);
v_isSharedCheck_5433_ = !lean_is_exclusive(v___x_5381_);
if (v_isSharedCheck_5433_ == 0)
{
v___x_5428_ = v___x_5381_;
v_isShared_5429_ = v_isSharedCheck_5433_;
goto v_resetjp_5427_;
}
else
{
lean_inc(v_a_5426_);
lean_dec(v___x_5381_);
v___x_5428_ = lean_box(0);
v_isShared_5429_ = v_isSharedCheck_5433_;
goto v_resetjp_5427_;
}
v_resetjp_5427_:
{
lean_object* v___x_5431_; 
if (v_isShared_5429_ == 0)
{
v___x_5431_ = v___x_5428_;
goto v_reusejp_5430_;
}
else
{
lean_object* v_reuseFailAlloc_5432_; 
v_reuseFailAlloc_5432_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5432_, 0, v_a_5426_);
v___x_5431_ = v_reuseFailAlloc_5432_;
goto v_reusejp_5430_;
}
v_reusejp_5430_:
{
return v___x_5431_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___boxed(lean_object* v___y_5434_, lean_object* v___y_5435_, lean_object* v___y_5436_, lean_object* v___y_5437_, lean_object* v___y_5438_, lean_object* v___y_5439_, lean_object* v___y_5440_, lean_object* v___y_5441_, lean_object* v___y_5442_){
_start:
{
lean_object* v_res_5443_; 
v_res_5443_ = lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2(v___y_5434_, v___y_5435_, v___y_5436_, v___y_5437_, v___y_5438_, v___y_5439_, v___y_5440_, v___y_5441_);
lean_dec(v___y_5441_);
lean_dec_ref(v___y_5440_);
lean_dec(v___y_5439_);
lean_dec_ref(v___y_5438_);
lean_dec(v___y_5437_);
lean_dec_ref(v___y_5436_);
lean_dec(v___y_5435_);
lean_dec_ref(v___y_5434_);
return v_res_5443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology(lean_object* v_a_5447_, lean_object* v_a_5448_, lean_object* v_a_5449_, lean_object* v_a_5450_, lean_object* v_a_5451_, lean_object* v_a_5452_, lean_object* v_a_5453_, lean_object* v_a_5454_){
_start:
{
lean_object* v___x_5456_; lean_object* v___x_5457_; 
v___x_5456_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___closed__1));
v___x_5457_ = l_Lean_Elab_Tactic_focus___redArg(v___x_5456_, v_a_5447_, v_a_5448_, v_a_5449_, v_a_5450_, v_a_5451_, v_a_5452_, v_a_5453_, v_a_5454_);
return v___x_5457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto_tautology___boxed(lean_object* v_a_5458_, lean_object* v_a_5459_, lean_object* v_a_5460_, lean_object* v_a_5461_, lean_object* v_a_5462_, lean_object* v_a_5463_, lean_object* v_a_5464_, lean_object* v_a_5465_, lean_object* v_a_5466_){
_start:
{
lean_object* v_res_5467_; 
v_res_5467_ = lp_mathlib_Mathlib_Tactic_Tauto_tautology(v_a_5458_, v_a_5459_, v_a_5460_, v_a_5461_, v_a_5462_, v_a_5463_, v_a_5464_, v_a_5465_);
lean_dec(v_a_5465_);
lean_dec_ref(v_a_5464_);
lean_dec(v_a_5463_);
lean_dec_ref(v_a_5462_);
lean_dec(v_a_5461_);
lean_dec_ref(v_a_5460_);
lean_dec(v_a_5459_);
lean_dec_ref(v_a_5458_);
return v_res_5467_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2(void){
_start:
{
lean_object* v___x_5476_; lean_object* v___x_5477_; lean_object* v___x_5478_; lean_object* v___x_5479_; 
v___x_5476_ = l_Lean_Parser_Tactic_optConfig;
v___x_5477_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__1));
v___x_5478_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_term___x3c_x3b_x3e___00__closed__3));
v___x_5479_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_5479_, 0, v___x_5478_);
lean_ctor_set(v___x_5479_, 1, v___x_5477_);
lean_ctor_set(v___x_5479_, 2, v___x_5476_);
return v___x_5479_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3(void){
_start:
{
lean_object* v___x_5480_; lean_object* v___x_5481_; lean_object* v___x_5482_; lean_object* v___x_5483_; 
v___x_5480_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2, &lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__2);
v___x_5481_ = lean_unsigned_to_nat(1022u);
v___x_5482_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0));
v___x_5483_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_5483_, 0, v___x_5482_);
lean_ctor_set(v___x_5483_, 1, v___x_5481_);
lean_ctor_set(v___x_5483_, 2, v___x_5480_);
return v___x_5483_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto(void){
_start:
{
lean_object* v___x_5484_; 
v___x_5484_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3, &lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__3);
return v___x_5484_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_5485_; lean_object* v___x_5486_; lean_object* v___x_5487_; 
v___x_5485_ = lean_box(0);
v___x_5486_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_5487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_5487_, 0, v___x_5486_);
lean_ctor_set(v___x_5487_, 1, v___x_5485_);
return v___x_5487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg(){
_start:
{
lean_object* v___x_5489_; lean_object* v___x_5490_; 
v___x_5489_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___closed__0);
v___x_5490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5490_, 0, v___x_5489_);
return v___x_5490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg___boxed(lean_object* v___y_5491_){
_start:
{
lean_object* v_res_5492_; 
v_res_5492_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg();
return v_res_5492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0(lean_object* v_00_u03b1_5493_, lean_object* v___y_5494_, lean_object* v___y_5495_, lean_object* v___y_5496_, lean_object* v___y_5497_, lean_object* v___y_5498_, lean_object* v___y_5499_, lean_object* v___y_5500_, lean_object* v___y_5501_){
_start:
{
lean_object* v___x_5503_; 
v___x_5503_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg();
return v___x_5503_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___boxed(lean_object* v_00_u03b1_5504_, lean_object* v___y_5505_, lean_object* v___y_5506_, lean_object* v___y_5507_, lean_object* v___y_5508_, lean_object* v___y_5509_, lean_object* v___y_5510_, lean_object* v___y_5511_, lean_object* v___y_5512_, lean_object* v___y_5513_){
_start:
{
lean_object* v_res_5514_; 
v_res_5514_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0(v_00_u03b1_5504_, v___y_5505_, v___y_5506_, v___y_5507_, v___y_5508_, v___y_5509_, v___y_5510_, v___y_5511_, v___y_5512_);
lean_dec(v___y_5512_);
lean_dec_ref(v___y_5511_);
lean_dec(v___y_5510_);
lean_dec_ref(v___y_5509_);
lean_dec(v___y_5508_);
lean_dec_ref(v___y_5507_);
lean_dec(v___y_5506_);
lean_dec_ref(v___y_5505_);
return v_res_5514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1(lean_object* v_x_5515_, lean_object* v_a_5516_, lean_object* v_a_5517_, lean_object* v_a_5518_, lean_object* v_a_5519_, lean_object* v_a_5520_, lean_object* v_a_5521_, lean_object* v_a_5522_, lean_object* v_a_5523_){
_start:
{
lean_object* v___x_5525_; uint8_t v___x_5526_; 
v___x_5525_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0));
lean_inc(v_x_5515_);
v___x_5526_ = l_Lean_Syntax_isOfKind(v_x_5515_, v___x_5525_);
if (v___x_5526_ == 0)
{
lean_object* v___x_5527_; 
lean_dec(v_x_5515_);
v___x_5527_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg();
return v___x_5527_;
}
else
{
lean_object* v___x_5528_; lean_object* v___x_5529_; lean_object* v___x_5530_; uint8_t v___x_5531_; 
v___x_5528_ = lean_unsigned_to_nat(1u);
v___x_5529_ = l_Lean_Syntax_getArg(v_x_5515_, v___x_5528_);
lean_dec(v_x_5515_);
v___x_5530_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__8));
lean_inc(v___x_5529_);
v___x_5531_ = l_Lean_Syntax_isOfKind(v___x_5529_, v___x_5530_);
if (v___x_5531_ == 0)
{
lean_object* v___x_5532_; 
lean_dec(v___x_5529_);
v___x_5532_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1_spec__0___redArg();
return v___x_5532_;
}
else
{
lean_object* v___x_5533_; lean_object* v___x_5534_; 
v___x_5533_ = lean_box(0);
v___x_5534_ = lp_mathlib_Mathlib_Tactic_Tauto_elabConfig___redArg(v___x_5529_, v___x_5533_, v___x_5531_, v_a_5516_, v_a_5522_, v_a_5523_);
if (lean_obj_tag(v___x_5534_) == 0)
{
lean_object* v___x_5535_; 
lean_dec_ref_known(v___x_5534_, 1);
v___x_5535_ = lp_mathlib_Mathlib_Tactic_Tauto_tautology(v_a_5516_, v_a_5517_, v_a_5518_, v_a_5519_, v_a_5520_, v_a_5521_, v_a_5522_, v_a_5523_);
return v___x_5535_;
}
else
{
lean_object* v_a_5536_; lean_object* v___x_5538_; uint8_t v_isShared_5539_; uint8_t v_isSharedCheck_5543_; 
v_a_5536_ = lean_ctor_get(v___x_5534_, 0);
v_isSharedCheck_5543_ = !lean_is_exclusive(v___x_5534_);
if (v_isSharedCheck_5543_ == 0)
{
v___x_5538_ = v___x_5534_;
v_isShared_5539_ = v_isSharedCheck_5543_;
goto v_resetjp_5537_;
}
else
{
lean_inc(v_a_5536_);
lean_dec(v___x_5534_);
v___x_5538_ = lean_box(0);
v_isShared_5539_ = v_isSharedCheck_5543_;
goto v_resetjp_5537_;
}
v_resetjp_5537_:
{
lean_object* v___x_5541_; 
if (v_isShared_5539_ == 0)
{
v___x_5541_ = v___x_5538_;
goto v_reusejp_5540_;
}
else
{
lean_object* v_reuseFailAlloc_5542_; 
v_reuseFailAlloc_5542_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5542_, 0, v_a_5536_);
v___x_5541_ = v_reuseFailAlloc_5542_;
goto v_reusejp_5540_;
}
v_reusejp_5540_:
{
return v___x_5541_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1___boxed(lean_object* v_x_5544_, lean_object* v_a_5545_, lean_object* v_a_5546_, lean_object* v_a_5547_, lean_object* v_a_5548_, lean_object* v_a_5549_, lean_object* v_a_5550_, lean_object* v_a_5551_, lean_object* v_a_5552_, lean_object* v_a_5553_){
_start:
{
lean_object* v_res_5554_; 
v_res_5554_ = lp_mathlib_Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______elabRules__Mathlib__Tactic__Tauto__tauto__1(v_x_5544_, v_a_5545_, v_a_5546_, v_a_5547_, v_a_5548_, v_a_5549_, v_a_5550_, v_a_5551_, v_a_5552_);
lean_dec(v_a_5552_);
lean_dec_ref(v_a_5551_);
lean_dec(v_a_5550_);
lean_dec_ref(v_a_5549_);
lean_dec(v_a_5548_);
lean_dec_ref(v_a_5547_);
lean_dec(v_a_5546_);
lean_dec_ref(v_a_5545_);
return v_res_5554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0(lean_object* v_name_5555_, lean_object* v_decl_5556_, lean_object* v_ref_5557_){
_start:
{
lean_object* v_defValue_5559_; lean_object* v_descr_5560_; lean_object* v_deprecation_x3f_5561_; lean_object* v___x_5562_; uint8_t v___x_5563_; lean_object* v___x_5564_; lean_object* v___x_5565_; 
v_defValue_5559_ = lean_ctor_get(v_decl_5556_, 0);
v_descr_5560_ = lean_ctor_get(v_decl_5556_, 1);
v_deprecation_x3f_5561_ = lean_ctor_get(v_decl_5556_, 2);
v___x_5562_ = lean_alloc_ctor(1, 0, 1);
v___x_5563_ = lean_unbox(v_defValue_5559_);
lean_ctor_set_uint8(v___x_5562_, 0, v___x_5563_);
lean_inc(v_deprecation_x3f_5561_);
lean_inc_ref(v_descr_5560_);
lean_inc_n(v_name_5555_, 2);
v___x_5564_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_5564_, 0, v_name_5555_);
lean_ctor_set(v___x_5564_, 1, v_ref_5557_);
lean_ctor_set(v___x_5564_, 2, v___x_5562_);
lean_ctor_set(v___x_5564_, 3, v_descr_5560_);
lean_ctor_set(v___x_5564_, 4, v_deprecation_x3f_5561_);
v___x_5565_ = lean_register_option(v_name_5555_, v___x_5564_);
if (lean_obj_tag(v___x_5565_) == 0)
{
lean_object* v___x_5567_; uint8_t v_isShared_5568_; uint8_t v_isSharedCheck_5573_; 
v_isSharedCheck_5573_ = !lean_is_exclusive(v___x_5565_);
if (v_isSharedCheck_5573_ == 0)
{
lean_object* v_unused_5574_; 
v_unused_5574_ = lean_ctor_get(v___x_5565_, 0);
lean_dec(v_unused_5574_);
v___x_5567_ = v___x_5565_;
v_isShared_5568_ = v_isSharedCheck_5573_;
goto v_resetjp_5566_;
}
else
{
lean_dec(v___x_5565_);
v___x_5567_ = lean_box(0);
v_isShared_5568_ = v_isSharedCheck_5573_;
goto v_resetjp_5566_;
}
v_resetjp_5566_:
{
lean_object* v___x_5569_; lean_object* v___x_5571_; 
lean_inc(v_defValue_5559_);
v___x_5569_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_5569_, 0, v_name_5555_);
lean_ctor_set(v___x_5569_, 1, v_defValue_5559_);
if (v_isShared_5568_ == 0)
{
lean_ctor_set(v___x_5567_, 0, v___x_5569_);
v___x_5571_ = v___x_5567_;
goto v_reusejp_5570_;
}
else
{
lean_object* v_reuseFailAlloc_5572_; 
v_reuseFailAlloc_5572_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5572_, 0, v___x_5569_);
v___x_5571_ = v_reuseFailAlloc_5572_;
goto v_reusejp_5570_;
}
v_reusejp_5570_:
{
return v___x_5571_;
}
}
}
else
{
lean_object* v_a_5575_; lean_object* v___x_5577_; uint8_t v_isShared_5578_; uint8_t v_isSharedCheck_5582_; 
lean_dec(v_name_5555_);
v_a_5575_ = lean_ctor_get(v___x_5565_, 0);
v_isSharedCheck_5582_ = !lean_is_exclusive(v___x_5565_);
if (v_isSharedCheck_5582_ == 0)
{
v___x_5577_ = v___x_5565_;
v_isShared_5578_ = v_isSharedCheck_5582_;
goto v_resetjp_5576_;
}
else
{
lean_inc(v_a_5575_);
lean_dec(v___x_5565_);
v___x_5577_ = lean_box(0);
v_isShared_5578_ = v_isSharedCheck_5582_;
goto v_resetjp_5576_;
}
v_resetjp_5576_:
{
lean_object* v___x_5580_; 
if (v_isShared_5578_ == 0)
{
v___x_5580_ = v___x_5577_;
goto v_reusejp_5579_;
}
else
{
lean_object* v_reuseFailAlloc_5581_; 
v_reuseFailAlloc_5581_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5581_, 0, v_a_5575_);
v___x_5580_ = v_reuseFailAlloc_5581_;
goto v_reusejp_5579_;
}
v_reusejp_5579_:
{
return v___x_5580_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_5583_, lean_object* v_decl_5584_, lean_object* v_ref_5585_, lean_object* v_a_5586_){
_start:
{
lean_object* v_res_5587_; 
v_res_5587_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0(v_name_5583_, v_decl_5584_, v_ref_5585_);
lean_dec_ref(v_decl_5584_);
return v_res_5587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_5602_; lean_object* v___x_5603_; lean_object* v___x_5604_; 
v___x_5602_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__3_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_));
v___x_5603_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_));
v___x_5604_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0(v___x_5602_, v___x_5603_, v___x_5602_);
return v___x_5604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4____boxed(lean_object* v_a_5605_){
_start:
{
lean_object* v_res_5606_; 
v_res_5606_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_();
return v_res_5606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg(lean_object* v___y_5607_){
_start:
{
lean_object* v___x_5609_; lean_object* v_env_5610_; lean_object* v___x_5611_; lean_object* v_mainModule_5612_; lean_object* v___x_5613_; 
v___x_5609_ = lean_st_ref_get(v___y_5607_);
v_env_5610_ = lean_ctor_get(v___x_5609_, 0);
lean_inc_ref(v_env_5610_);
lean_dec(v___x_5609_);
v___x_5611_ = l_Lean_Environment_header(v_env_5610_);
lean_dec_ref(v_env_5610_);
v_mainModule_5612_ = lean_ctor_get(v___x_5611_, 0);
lean_inc(v_mainModule_5612_);
lean_dec_ref(v___x_5611_);
v___x_5613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5613_, 0, v_mainModule_5612_);
return v___x_5613_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg___boxed(lean_object* v___y_5614_, lean_object* v___y_5615_){
_start:
{
lean_object* v_res_5616_; 
v_res_5616_ = lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg(v___y_5614_);
lean_dec(v___y_5614_);
return v_res_5616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0(lean_object* v___y_5617_, lean_object* v___y_5618_){
_start:
{
lean_object* v___x_5620_; 
v___x_5620_ = lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg(v___y_5618_);
return v___x_5620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___boxed(lean_object* v___y_5621_, lean_object* v___y_5622_, lean_object* v___y_5623_){
_start:
{
lean_object* v_res_5624_; 
v_res_5624_ = lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0(v___y_5621_, v___y_5622_);
lean_dec(v___y_5622_);
lean_dec_ref(v___y_5621_);
return v_res_5624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrind___lam__0(lean_object* v___x_5625_, lean_object* v___x_5626_, lean_object* v_x_5627_, lean_object* v_x_5628_, lean_object* v_x_5629_, lean_object* v___y_5630_, lean_object* v___y_5631_){
_start:
{
lean_object* v___x_5633_; 
v___x_5633_ = l_Lean_Elab_Command_getRef___redArg(v___y_5630_);
if (lean_obj_tag(v___x_5633_) == 0)
{
lean_object* v_a_5634_; lean_object* v___x_5635_; 
v_a_5634_ = lean_ctor_get(v___x_5633_, 0);
lean_inc(v_a_5634_);
lean_dec_ref_known(v___x_5633_, 1);
v___x_5635_ = l_Lean_Elab_Command_getCurrMacroScope___redArg(v___y_5630_);
if (lean_obj_tag(v___x_5635_) == 0)
{
lean_object* v___x_5637_; uint8_t v_isShared_5638_; uint8_t v_isSharedCheck_5658_; 
v_isSharedCheck_5658_ = !lean_is_exclusive(v___x_5635_);
if (v_isSharedCheck_5658_ == 0)
{
lean_object* v_unused_5659_; 
v_unused_5659_ = lean_ctor_get(v___x_5635_, 0);
lean_dec(v_unused_5659_);
v___x_5637_ = v___x_5635_;
v_isShared_5638_ = v_isSharedCheck_5658_;
goto v_resetjp_5636_;
}
else
{
lean_dec(v___x_5635_);
v___x_5637_ = lean_box(0);
v_isShared_5638_ = v_isSharedCheck_5658_;
goto v_resetjp_5636_;
}
v_resetjp_5636_:
{
lean_object* v_quotContext_x3f_5639_; uint8_t v___x_5640_; lean_object* v___x_5641_; 
v_quotContext_x3f_5639_ = lean_ctor_get(v___y_5630_, 5);
v___x_5640_ = 0;
v___x_5641_ = l_Lean_SourceInfo_fromRef(v_a_5634_, v___x_5640_);
lean_dec(v_a_5634_);
if (lean_obj_tag(v_quotContext_x3f_5639_) == 0)
{
lean_object* v___x_5657_; 
v___x_5657_ = lp_mathlib_Lean_getMainModule___at___00tautoToGrind_spec__0___redArg(v___y_5631_);
lean_dec_ref(v___x_5657_);
goto v___jp_5642_;
}
else
{
goto v___jp_5642_;
}
v___jp_5642_:
{
lean_object* v___x_5643_; lean_object* v___x_5644_; lean_object* v___x_5645_; lean_object* v___x_5646_; lean_object* v___x_5647_; lean_object* v___x_5648_; lean_object* v___x_5649_; lean_object* v___x_5650_; lean_object* v___x_5651_; lean_object* v___x_5652_; lean_object* v___x_5653_; lean_object* v___x_5655_; 
v___x_5643_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__0));
v___x_5644_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__1));
lean_inc_ref(v___x_5626_);
lean_inc_ref(v___x_5625_);
v___x_5645_ = l_Lean_Name_mkStr4(v___x_5643_, v___x_5644_, v___x_5625_, v___x_5626_);
lean_inc_n(v___x_5641_, 3);
v___x_5646_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_5646_, 0, v___x_5641_);
lean_ctor_set(v___x_5646_, 1, v___x_5626_);
v___x_5647_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__7));
v___x_5648_ = l_Lean_Name_mkStr4(v___x_5643_, v___x_5644_, v___x_5625_, v___x_5647_);
v___x_5649_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto___aux__Mathlib__Tactic__Tauto______macroRules____private__Mathlib__Tactic__Tauto__0__Mathlib__Tactic__Tauto__term___x3c_x3b_x3e____1___closed__13));
v___x_5650_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6, &lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6_once, _init_lp_mathlib_Mathlib_Tactic_Tauto_tautology___lam__2___closed__6);
v___x_5651_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_5651_, 0, v___x_5641_);
lean_ctor_set(v___x_5651_, 1, v___x_5649_);
lean_ctor_set(v___x_5651_, 2, v___x_5650_);
lean_inc_ref_n(v___x_5651_, 3);
v___x_5652_ = l_Lean_Syntax_node1(v___x_5641_, v___x_5648_, v___x_5651_);
v___x_5653_ = l_Lean_Syntax_node5(v___x_5641_, v___x_5645_, v___x_5646_, v___x_5652_, v___x_5651_, v___x_5651_, v___x_5651_);
if (v_isShared_5638_ == 0)
{
lean_ctor_set(v___x_5637_, 0, v___x_5653_);
v___x_5655_ = v___x_5637_;
goto v_reusejp_5654_;
}
else
{
lean_object* v_reuseFailAlloc_5656_; 
v_reuseFailAlloc_5656_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5656_, 0, v___x_5653_);
v___x_5655_ = v_reuseFailAlloc_5656_;
goto v_reusejp_5654_;
}
v_reusejp_5654_:
{
return v___x_5655_;
}
}
}
}
else
{
lean_object* v_a_5660_; lean_object* v___x_5662_; uint8_t v_isShared_5663_; uint8_t v_isSharedCheck_5667_; 
lean_dec(v_a_5634_);
lean_dec_ref(v___x_5626_);
lean_dec_ref(v___x_5625_);
v_a_5660_ = lean_ctor_get(v___x_5635_, 0);
v_isSharedCheck_5667_ = !lean_is_exclusive(v___x_5635_);
if (v_isSharedCheck_5667_ == 0)
{
v___x_5662_ = v___x_5635_;
v_isShared_5663_ = v_isSharedCheck_5667_;
goto v_resetjp_5661_;
}
else
{
lean_inc(v_a_5660_);
lean_dec(v___x_5635_);
v___x_5662_ = lean_box(0);
v_isShared_5663_ = v_isSharedCheck_5667_;
goto v_resetjp_5661_;
}
v_resetjp_5661_:
{
lean_object* v___x_5665_; 
if (v_isShared_5663_ == 0)
{
v___x_5665_ = v___x_5662_;
goto v_reusejp_5664_;
}
else
{
lean_object* v_reuseFailAlloc_5666_; 
v_reuseFailAlloc_5666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5666_, 0, v_a_5660_);
v___x_5665_ = v_reuseFailAlloc_5666_;
goto v_reusejp_5664_;
}
v_reusejp_5664_:
{
return v___x_5665_;
}
}
}
}
else
{
lean_object* v_a_5668_; lean_object* v___x_5670_; uint8_t v_isShared_5671_; uint8_t v_isSharedCheck_5675_; 
lean_dec_ref(v___x_5626_);
lean_dec_ref(v___x_5625_);
v_a_5668_ = lean_ctor_get(v___x_5633_, 0);
v_isSharedCheck_5675_ = !lean_is_exclusive(v___x_5633_);
if (v_isSharedCheck_5675_ == 0)
{
v___x_5670_ = v___x_5633_;
v_isShared_5671_ = v_isSharedCheck_5675_;
goto v_resetjp_5669_;
}
else
{
lean_inc(v_a_5668_);
lean_dec(v___x_5633_);
v___x_5670_ = lean_box(0);
v_isShared_5671_ = v_isSharedCheck_5675_;
goto v_resetjp_5669_;
}
v_resetjp_5669_:
{
lean_object* v___x_5673_; 
if (v_isShared_5671_ == 0)
{
v___x_5673_ = v___x_5670_;
goto v_reusejp_5672_;
}
else
{
lean_object* v_reuseFailAlloc_5674_; 
v_reuseFailAlloc_5674_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_5674_, 0, v_a_5668_);
v___x_5673_ = v_reuseFailAlloc_5674_;
goto v_reusejp_5672_;
}
v_reusejp_5672_:
{
return v___x_5673_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrind___lam__0___boxed(lean_object* v___x_5676_, lean_object* v___x_5677_, lean_object* v_x_5678_, lean_object* v_x_5679_, lean_object* v_x_5680_, lean_object* v___y_5681_, lean_object* v___y_5682_, lean_object* v___y_5683_){
_start:
{
lean_object* v_res_5684_; 
v_res_5684_ = lp_mathlib_tautoToGrind___lam__0(v___x_5676_, v___x_5677_, v_x_5678_, v_x_5679_, v_x_5680_, v___y_5681_, v___y_5682_);
lean_dec(v___y_5682_);
lean_dec_ref(v___y_5681_);
lean_dec(v_x_5680_);
lean_dec_ref(v_x_5679_);
lean_dec_ref(v_x_5678_);
return v_res_5684_;
}
}
static double _init_lp_mathlib_tautoToGrind___closed__2(void){
_start:
{
lean_object* v___x_5689_; double v___x_5690_; 
v___x_5689_ = lean_unsigned_to_nat(1u);
v___x_5690_ = lean_float_of_nat(v___x_5689_);
return v___x_5690_;
}
}
static lean_object* _init_lp_mathlib_tautoToGrind___closed__3(void){
_start:
{
double v___x_5691_; uint8_t v___x_5692_; uint8_t v___x_5693_; lean_object* v___f_5694_; lean_object* v___x_5695_; lean_object* v___x_5696_; lean_object* v___x_5697_; lean_object* v___x_5698_; 
v___x_5691_ = lean_float_once(&lp_mathlib_tautoToGrind___closed__2, &lp_mathlib_tautoToGrind___closed__2_once, _init_lp_mathlib_tautoToGrind___closed__2);
v___x_5692_ = 1;
v___x_5693_ = 0;
v___f_5694_ = ((lean_object*)(lp_mathlib_tautoToGrind___closed__1));
v___x_5695_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0));
v___x_5696_ = ((lean_object*)(lp_mathlib_tautoToGrind___closed__0));
v___x_5697_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_5698_ = lp_mathlib_Mathlib_TacticAnalysis_terminalReplacement(v___x_5697_, v___x_5696_, v___x_5695_, v___f_5694_, v___x_5693_, v___x_5692_, v___x_5693_, v___x_5691_);
return v___x_5698_;
}
}
static lean_object* _init_lp_mathlib_tautoToGrind(void){
_start:
{
lean_object* v___x_5699_; 
v___x_5699_ = lean_obj_once(&lp_mathlib_tautoToGrind___closed__3, &lp_mathlib_tautoToGrind___closed__3_once, _init_lp_mathlib_tautoToGrind___closed__3);
return v___x_5699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_5707_; lean_object* v___x_5708_; lean_object* v___x_5709_; 
v___x_5707_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__1_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_));
v___x_5708_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn___closed__5_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_));
v___x_5709_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4__spec__0(v___x_5707_, v___x_5708_, v___x_5707_);
return v___x_5709_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4____boxed(lean_object* v_a_5710_){
_start:
{
lean_object* v_res_5711_; 
v_res_5711_ = lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_();
return v_res_5711_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrindRegressions___lam__0(lean_object* v_x_5712_){
_start:
{
lean_object* v___x_5713_; 
v___x_5713_ = lean_box(0);
return v___x_5713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tautoToGrindRegressions___lam__0___boxed(lean_object* v_x_5714_){
_start:
{
lean_object* v_res_5715_; 
v_res_5715_ = lp_mathlib_tautoToGrindRegressions___lam__0(v_x_5714_);
lean_dec(v_x_5714_);
return v_res_5715_;
}
}
static lean_object* _init_lp_mathlib_tautoToGrindRegressions___closed__1(void){
_start:
{
double v___x_5717_; uint8_t v___x_5718_; uint8_t v___x_5719_; lean_object* v___f_5720_; lean_object* v___x_5721_; lean_object* v___x_5722_; lean_object* v___x_5723_; 
v___x_5717_ = lean_float_once(&lp_mathlib_tautoToGrind___closed__2, &lp_mathlib_tautoToGrind___closed__2_once, _init_lp_mathlib_tautoToGrind___closed__2);
v___x_5718_ = 0;
v___x_5719_ = 1;
v___f_5720_ = ((lean_object*)(lp_mathlib_tautoToGrindRegressions___closed__0));
v___x_5721_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Tauto_tauto___closed__0));
v___x_5722_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn___closed__0_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_));
v___x_5723_ = lp_mathlib_Mathlib_TacticAnalysis_grindReplacementWith(v___x_5722_, v___x_5721_, v___f_5720_, v___x_5719_, v___x_5718_, v___x_5718_, v___x_5717_);
return v___x_5723_;
}
}
static lean_object* _init_lp_mathlib_tautoToGrindRegressions(void){
_start:
{
lean_object* v___x_5724_; 
v___x_5724_ = lean_obj_once(&lp_mathlib_tautoToGrindRegressions___closed__1, &lp_mathlib_tautoToGrindRegressions___closed__1_once, _init_lp_mathlib_tautoToGrindRegressions___closed__1);
return v___x_5724_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Classical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_ConfigEval(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Classical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_initFn_00___x40_Mathlib_Tactic_Tauto_3330388757____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig = _init_lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig();
lean_mark_persistent(lp_mathlib___private_Mathlib_Tactic_Tauto_0__Mathlib_Tactic_Tauto_instEvalExprConfig);
lp_mathlib_Mathlib_Tactic_Tauto_tauto = _init_lp_mathlib_Mathlib_Tactic_Tauto_tauto();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_Tauto_tauto);
res = lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_3029345193____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_tacticAnalysis_tautoToGrind = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_tacticAnalysis_tautoToGrind);
lean_dec_ref(res);
lp_mathlib_tautoToGrind = _init_lp_mathlib_tautoToGrind();
lean_mark_persistent(lp_mathlib_tautoToGrind);
res = lp_mathlib___private_Mathlib_Tactic_Tauto_0__initFn_00___x40_Mathlib_Tactic_Tauto_2865712539____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_tacticAnalysis_regressions_tautoToGrind = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_tacticAnalysis_regressions_tautoToGrind);
lean_dec_ref(res);
lp_mathlib_tautoToGrindRegressions = _init_lp_mathlib_tautoToGrindRegressions();
lean_mark_persistent(lp_mathlib_tautoToGrindRegressions);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Classical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CasesM(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Core(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Lean_Elab_ConfigEval(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Classical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CasesM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_ConfigEval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
}
#ifdef __cplusplus
}
#endif
